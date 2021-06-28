import training_utils
from utils import load_pickle, save_obj_as_pickle
# from prepare_clip_dataset import QUERY_TITLE_DICT, LABEL_SETS
import random
import argparse
# from analyze_feature_variation import argparser, get_dataset_folder_paths
from tqdm import tqdm
import copy
import time
import numpy as np
import torch
import os

from train import NEGATIVE_LABEL, device, MODE_DICT, HyperParameter, HYPER_DICT, TrainMode, TRAIN_MODES_CATEGORY
from train import make_dataset_dict, split_dataset, get_seed_str, use_val_set, dataset_str, make_features_dict, extract_features, argparser
from train import get_loader_func, get_all_loaders_from_features_dict, get_loaders_from_features_dict, get_curr_and_prev_loaders_from_features_dict
from train import get_curr_and_random_prev_loaders_from_features_dict, get_sequential_loaders_from_features_dict
from train import MLP, make_feature_extractor, make_cnn_model, get_input_size, make_model, train, test
from train import avg_per_class_accuracy, only_positive_accuracy, run_baseline, run_single, run_single_finetune


ALPHA_VALUE_DICT = {
    'single_buffer' : {
        'fixed' : [0.5, 1.0, 2.0, 5.0],
        'dynamic' : [0.2, 0.5, 0.8, 1.0], # coeff * N/k, where k is the buffer size equal to 2 buckets
    },
    'two_buffer' : {
        'fixed' : [0.5, 1.0, 2.0, 5.0],
        'dynamic' : [0.0, 0.2, 0.5, 0.8, 1.0], # coeff * N/k, where k is the buffer size equal to 2 buckets
    }
}

ALPHA_BUFFER_DICT = ['two_buffer', # first buffer always stores current bucket, second is to keep random samples from stream
                     'single_buffer'] # only a single buffer is used, new sample will have a probably of being selected

argparser.add_argument("--alpha_value_mode",
                       default='fixed',
                    #    default='dynamic',
                       help="Whether alpha is growing as stream is moving")
argparser.add_argument("--alpha_buffer_mode",
                       default='two_buffer',
                       #    default='single_buffer',
                       help="Whether to always store current samples")

# def get_curr_and_prev_loaders_from_features_dict(features_dict, train_mode, hyperparameter, excluded_bucket_idx=0):
#     all_bucket = sorted(list(features_dict.keys()))
#     print(f"Excluding {excluded_bucket_idx} from the sequential loaders")
#     if type(excluded_bucket_idx) == int:
#         features_dict = {k: features_dict[k]
#                         for k in all_bucket if k != excluded_bucket_idx}
#     curr_and_prev_loaders_dict = {}
#     loader_func = get_loader_func(train_mode, hyperparameter.batch_size)

#     for b_idx in all_bucket:
#         curr_and_prev_loaders_dict[b_idx] = {}
#         assert 'val' not in features_dict[b_idx]
#         train_items = []
#         prev_b_idx = b_idx - 1
#         if prev_b_idx in features_dict:
#             print(f"Add {prev_b_idx} to {b_idx} train loader")
#             train_items += features_dict[prev_b_idx]['train']
#         train_items += features_dict[b_idx]['train']
#         train_loader = loader_func(train_items, True)
#         curr_and_prev_loaders_dict[b_idx]['train'] = train_loader
        
#         test_items = []
#         test_items += features_dict[b_idx]['test']
#         test_loader = loader_func(test_items, False)
#         curr_and_prev_loaders_dict[b_idx]['test'] = test_loader
#     return curr_and_prev_loaders_dict


def get_classbalanced_two_buffer_loaders_from_features_dict(alpha_value_mode, alpha_value, features_dict, train_mode, hyperparameter, excluded_bucket_idx=0):
    all_bucket = sorted(list(features_dict.keys()))
    print(f"Excluding {excluded_bucket_idx} from the sequential loaders")
    if type(excluded_bucket_idx) == int:
        features_dict = {k: features_dict[k]
                         for k in all_bucket if k != excluded_bucket_idx}
    two_buffer_loaders_dict = {}
    loader_func = get_loader_func(train_mode, hyperparameter.batch_size)
    curr_buffer = {} # key is class index, always store current samples
    memory_buffer = {} # key is class index, store random samples from previous stream up to current samples
    
    def get_class_buffer(feature_dict_copy):
        class_buffer = {}
        for feature, class_idx in feature_dict_copy:
            if not class_idx in class_buffer:
                class_buffer[class_idx] = []
            class_buffer[class_idx].append((feature, class_idx))
            
        return class_buffer
    
    def shuffle_class_buffer(class_buffer):
        for class_idx in class_buffer:
            random.shuffle(class_buffer[class_idx])
    
    def get_list_from_class_buffer(class_buffer):
        lst = []
        for class_idx in class_buffer:
            lst += class_buffer[class_idx]
        return lst

    k = None # k should be single bucket size
    n = 0. # Number of seen examples thus far
    for idx, b_idx in enumerate(all_bucket):
        two_buffer_loaders_dict[b_idx] = {}
        assert 'val' not in features_dict[b_idx]
        n += len(features_dict[b_idx]['train'])
        
        if idx == 0:
            curr_buffer = get_class_buffer(features_dict[b_idx]['train'].copy())
            k = n # This is currently bucket size
            print(f"Bucket {b_idx}: Since it is first, we put all examples in curr_buffer. And buffer size should be {n} = {k}")
        elif idx == 1:
            memory_buffer = curr_buffer.copy()
            curr_buffer = get_class_buffer(features_dict[b_idx]['train'].copy())
            print(f"Bucket {b_idx}: Since it is second, we put all examples from curr_buffer to memory_buffer. And curr_buffer is second bucket")
        else:
            previous_stream_size = n - k
            if alpha_value_mode == 'fixed':
                alpha = alpha_value
            elif alpha_value_mode == 'dynamic':
                alpha = alpha_value * previous_stream_size/k
            
            
            prob = alpha * k/previous_stream_size
            print(f'Bucket {b_idx}: First move examples from curr_buffer to memory with probability {alpha * k/previous_stream_size:.2f} = {alpha} * {k} / {previous_stream_size}')
            if prob >= 1:
                print("Since probability is greater than 1, we cap it at 1")
                prob = 1.

            # print(f"The actual implementation sample from Uniform(0, 1)")
            shuffle_class_buffer(curr_buffer)
            shuffle_class_buffer(memory_buffer)
            # Implement
            count_replace = 0
            for class_idx in curr_buffer:
                for item in curr_buffer[class_idx]:
                    rnd = random.random()
                    if rnd <= prob:
                        count_replace += 1
                        rnd_idx = int(rnd * (previous_stream_size-1)) % len(memory_buffer[class_idx])
                        memory_buffer[class_idx][rnd_idx] = item

            print(f"{count_replace} items getting replaced out of {k} samples in memory_buffer (class balanced)")
            
            curr_buffer = get_class_buffer(features_dict[b_idx]['train'].copy())
            
        train_loader = loader_func(get_list_from_class_buffer(curr_buffer.copy()) + get_list_from_class_buffer(memory_buffer.copy()), True)  # Important! TO use copy() otherwise it will be changed
        two_buffer_loaders_dict[b_idx]['train'] = train_loader
        
        test_items = []
        test_items += features_dict[b_idx]['test']
        test_loader = loader_func(test_items.copy(), False)
        two_buffer_loaders_dict[b_idx]['test'] = test_loader
        for key in features_dict:
            if not len(features_dict[key]['train']) == 2310:
                print(f"Feature dict messed up at bucket {key}")
                import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    return two_buffer_loaders_dict


def get_classbalanced_single_buffer_loaders_from_features_dict(alpha_value_mode, alpha_value, features_dict, train_mode, hyperparameter, excluded_bucket_idx=0):
    import pdb; pdb.set_trace() #TODO
    all_bucket = sorted(list(features_dict.keys()))
    print(f"Excluding {excluded_bucket_idx} from the sequential loaders")
    if type(excluded_bucket_idx) == int:
        features_dict = {k: features_dict[k]
                         for k in all_bucket if k != excluded_bucket_idx}
    single_buffer_loaders_dict = {}
    loader_func = get_loader_func(train_mode, hyperparameter.batch_size)

    train_buffer = [] # buffer is cap at size 2 bucket
    n = 0. # number of seen examples in the stream
    k = None # number of examples in two buckets. will be calculated in in first iteration. Assume each bucket has same size!
    # import pdb; pdb.set_trace()
    for idx, b_idx in enumerate(all_bucket):
            
        single_buffer_loaders_dict[b_idx] = {}
        assert 'val' not in features_dict[b_idx]
        n += len(features_dict[b_idx]['train'])
        if idx == 0:
            train_buffer = features_dict[b_idx]['train'].copy()
            k = 2 * n # This is currently 2 * bucket size
            # import pdb; pdb.set_trace()
            print(f"Bucket {b_idx}: Since it is first, we put all examples {n} in. And buffer size should be 2 * {n} = {k}")
        elif idx == 1:
            train_buffer += features_dict[b_idx]['train'].copy()
            print(f"Bucket {b_idx}: Since it is second, we still put all examples {k/2} in. ")
        else:
            if alpha_value_mode == 'fixed':
                alpha = alpha_value
            elif alpha_value_mode == 'dynamic':
                alpha = alpha_value * n/k
            print(f'Bucket {b_idx}: Select new examples with probability {alpha*k/n:.2f} = {alpha*k} / {n}')
            prob = alpha * k/n
            if prob >= 1:
                print("Since probability is greater than 1, we cap it at 1")
                prob = 1.
            new_items_to_add_to_buffer = []
            for item in features_dict[b_idx]['train']:
                if random.random() <= prob:
                    new_items_to_add_to_buffer.append(item)
            print(f"Selected {len(new_items_to_add_to_buffer)} samples out of {k/2} incoming samples")
            random.shuffle(train_buffer)
            train_buffer = train_buffer[:len(train_buffer) - len(new_items_to_add_to_buffer)]
            train_buffer += new_items_to_add_to_buffer
        
        train_loader = loader_func(train_buffer.copy(), True) # Important! TO use copy() otherwise it will be changed
        single_buffer_loaders_dict[b_idx]['train'] = train_loader
        
        test_items = []
        test_items += features_dict[b_idx]['test']
        test_loader = loader_func(test_items.copy(), False)
        single_buffer_loaders_dict[b_idx]['test'] = test_loader
        for key in features_dict:
            if not len(features_dict[key]['train']) == 2310:
                print(f"Feature dict messed up at bucket {key}")
                import pdb; pdb.set_trace()
    return single_buffer_loaders_dict

if __name__ == '__main__':
    args = argparser.parse_args()

    if args.seed == None:
        print("Not using a random seed")
    else:
        print(f"Using random seed {args.seed}")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    seed_str = get_seed_str(args.seed)

    bucket_dict = {}

    excluded_bucket_idx = args.excluded_bucket_idx
    folder_path = args.folder_path
    dataset_name = args.dataset_name
    exp_result_path = args.exp_result_path
    exp_result_save_path = os.path.join(exp_result_path, dataset_name)
    if not os.path.exists(exp_result_save_path):
        os.makedirs(exp_result_save_path)
    print(f"Working on dataset {dataset_name}")
    print(f"Dataset and result will be saved at {exp_result_save_path}")

    query_dict_path = os.path.join(folder_path, dataset_name, "query_dict.pickle")
    if not os.path.exists(query_dict_path):
        print(f"Query dict does not exist for {dataset_name}")
        exit(0)
    query_dict = load_pickle(query_dict_path)
    
    all_query = sorted(list(query_dict[list(query_dict.keys())[0]].keys()))
    print(f"We have {len(all_query)} classes.")
    print(all_query)

    ############### Create Datasets
    dataset_dict_path = os.path.join(exp_result_save_path,
                                     f"dataset_dict_{dataset_str(args.mode)}_{seed_str}.pickle")
    if os.path.exists(dataset_dict_path):
        print(f"{dataset_dict_path} already exists.")
        dataset_dict = load_pickle(dataset_dict_path)
    else:
        import pdb; pdb.set_trace()
    
    ############### Create Features
    features_dict_path = os.path.join(exp_result_save_path,
                                      f"features_dict_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}.pickle")
    if os.path.exists(features_dict_path):
        print(f"{features_dict_path} already exists.")
        features_dict = load_pickle(features_dict_path)
    else:
        import pdb; pdb.set_trace()

    if use_val_set(args.mode):
        import pdb; pdb.set_trace()

    if args.alpha_buffer_mode == 'single_buffer':
        alpha_loader_func = get_classbalanced_single_buffer_loaders_from_features_dict
    else:
        alpha_loader_func = get_classbalanced_two_buffer_loaders_from_features_dict
    
    for alpha_value in ALPHA_VALUE_DICT[args.alpha_buffer_mode][args.alpha_value_mode]:
        print(f"Generate loader for alpha value {alpha_value} in mode {args.alpha_value_mode}")        
        alpha_loaders_dict_path = os.path.join(exp_result_save_path,
                                               f"classbalanced_alpha_loaders_dict_{dataset_str(args.mode)}_{args.alpha_buffer_mode}_{args.alpha_value_mode}_{alpha_value}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
        
        if os.path.exists(alpha_loaders_dict_path):
            print(f"{alpha_loaders_dict_path} already exists.")
            alpha_loaders_dict = load_pickle(alpha_loaders_dict_path)
        else:
            alpha_loaders_dict = alpha_loader_func(
                                     args.alpha_value_mode,
                                     alpha_value,
                                     features_dict,
                                     args.train_mode,
                                     HYPER_DICT[TRAIN_MODES_CATEGORY[args.train_mode].network_type],
                                     excluded_bucket_idx=excluded_bucket_idx
                                 )
            save_obj_as_pickle(alpha_loaders_dict_path, alpha_loaders_dict)
        
        ############### Run Alpha weighting (Retrain) Experiment
        results_dict_alpha_retrain_path = os.path.join(exp_result_save_path,
                                                       f"results_dict_classbalanced_{args.alpha_buffer_mode}_{args.alpha_value_mode}_{alpha_value}_retrain_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
        
        if not os.path.exists(results_dict_alpha_retrain_path):
            results_dict_alpha_retrain = run_single(alpha_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_alpha_retrain_path, results_dict_alpha_retrain)
            print(f"Saved at {results_dict_alpha_retrain_path}")
        else:
            print(results_dict_alpha_retrain_path + " already exists")
        
        ############### Run Alpha weighting (Finetune) Experiment
        results_dict_alpha_finetune_path = os.path.join(exp_result_save_path,
                                                        f"results_dict_classbalanced_{args.alpha_buffer_mode}_{args.alpha_value_mode}_{alpha_value}_finetune_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
        
        if not os.path.exists(results_dict_alpha_finetune_path):
            results_dict_alpha_finetune = run_single_finetune(alpha_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_alpha_finetune_path, results_dict_alpha_finetune)
            print(f"Saved at {results_dict_alpha_finetune_path}")
        else:
            print(results_dict_alpha_finetune_path + " already exists")
        

    
