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
    'exponential': [1.0, 0.99, 0.95, 0.9, 0.75, 0.6, 0.5, 0.4, None],
}

ALPHA_BUFFER_DICT = ['exponential']  

argparser.add_argument("--alpha_value_mode",
                       default='exponential',
                    #    default='dynamic',
                       help="Whether alpha is growing as stream is moving")

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

def get_classbalanced_sequential_alpha_loaders_from_features_dict(alpha_value_mode, alpha_value, features_dict, train_mode, hyperparameter, excluded_bucket_idx=0):
    all_bucket = sorted(list(features_dict.keys()))
    print(f"Excluding {excluded_bucket_idx} from the sequential loaders")
    if type(excluded_bucket_idx) == int:
        features_dict = {k: features_dict[k]
                        for k in all_bucket if k != excluded_bucket_idx}
    sequential_loaders_dict = {}

    loader_func = get_loader_func(train_mode, hyperparameter.batch_size)

    if alpha_value == None:
        for b_idx in all_bucket:
            sequential_loaders_dict[b_idx] = {}
            assert 'val' not in features_dict[b_idx]

            train_items = []
            for curr_b_idx in all_bucket:
                if curr_b_idx <= b_idx:
                    train_items += features_dict[curr_b_idx]['train'].copy()
            train_loader = loader_func(train_items, True)
            sequential_loaders_dict[b_idx]['train'] = train_loader
                
            test_items = []
            test_items += features_dict[b_idx]['test'].copy()
            test_loader = loader_func(test_items, False)
            sequential_loaders_dict[b_idx]['test'] = test_loader
            # import pdb; pdb.set_trace()
    else:
        for idx, b_idx in enumerate(all_bucket):
            sequential_loaders_dict[b_idx] = {}
            assert 'val' not in features_dict[b_idx]
            n = {} # total stream - current bucket
            p = {} # probability for selecting with replacement
            alpha = 1.0
            b_idx_bucket = get_class_buffer(features_dict[b_idx]['train'].copy())
            for curr_idx, curr_b_idx in reversed(list(enumerate(all_bucket))):
                if curr_idx < idx:
                    print(f"For {idx} timestamp: Use {alpha} for {curr_idx}")
                    curr_bucket = get_class_buffer(features_dict[curr_b_idx]['train'].copy())
                    import pdb; pdb.set_trace() #TODO
                    for class_idx in curr_bucket:
                        if not class_idx in n:
                            n[class_idx] = []
                        if not class_idx in p:
                            p[class_idx] = []
                        n[class_idx] += curr_bucket[class_idx]
                        p[class_idx] += [alpha for _ in curr_bucket[class_idx]]
                    if alpha_value_mode == 'exponential':
                        alpha = alpha * alpha_value
                    else:
                        alpha = alpha_value
                else:
                    print(f"Skipping {curr_idx} for {idx} timestamp")
            
            train_items = []
            for class_idx in p:
                if len(p[class_idx]) != 0:
                    p[class_idx] = [p_i/sum(p[class_idx]) for p_i in p[class_idx]]
                    train_items_class_idx = np.random.choice(np.arange(len(n[class_idx])), size=(len(n[class_idx])), replace=True, p=p[class_idx])
                    train_items += [n[class_idx][i] for i in train_items_class_idx]
                else:
                    print(f'No train_items for {b_idx}')
                    train_items = []
            train_loader = loader_func(get_list_from_class_buffer(train_items.copy()) + get_list_from_class_buffer(b_idx_bucket), True) # Important! TO use copy() otherwise it will be changed
            sequential_loaders_dict[b_idx]['train'] = train_loader

            test_items = []
            test_items += features_dict[b_idx]['test']
            test_loader = loader_func(test_items.copy(), False)
            sequential_loaders_dict[b_idx]['test'] = test_loader
    import pdb; pdb.set_trace()
    return sequential_loaders_dict



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
    
    for alpha_value in ALPHA_VALUE_DICT[args.alpha_value_mode]:
        print(f"Generate loader for alpha value {alpha_value} in mode {args.alpha_value_mode}")        
        alpha_loaders_dict_path = os.path.join(exp_result_save_path,
                                               f"classbalanced_cumulative_alpha_loaders_dict_{dataset_str(args.mode)}_{args.alpha_value_mode}_{alpha_value}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
        
        if os.path.exists(alpha_loaders_dict_path):
            print(f"{alpha_loaders_dict_path} already exists.")
            alpha_loaders_dict = load_pickle(alpha_loaders_dict_path)
        else:
            alpha_loaders_dict = get_classbalanced_sequential_alpha_loaders_from_features_dict(
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
                                                       f"results_dict_classbalanced_cumulative_{args.alpha_value_mode}_{alpha_value}_retrain_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
        
        if not os.path.exists(results_dict_alpha_retrain_path):
            results_dict_alpha_retrain = run_single(alpha_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_alpha_retrain_path, results_dict_alpha_retrain)
            print(f"Saved at {results_dict_alpha_retrain_path}")
        else:
            print(results_dict_alpha_retrain_path + " already exists")
        
        ############### Run Alpha weighting (Finetune) Experiment
        results_dict_alpha_finetune_path = os.path.join(exp_result_save_path,
                                                        f"results_dict_classbalanced_cumulative_{args.alpha_value_mode}_{alpha_value}_finetune_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
        
        if not os.path.exists(results_dict_alpha_finetune_path):
            results_dict_alpha_finetune = run_single_finetune(alpha_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_alpha_finetune_path, results_dict_alpha_finetune)
            print(f"Saved at {results_dict_alpha_finetune_path}")
        else:
            print(results_dict_alpha_finetune_path + " already exists")
        

    
