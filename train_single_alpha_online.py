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
from train import get_loader_func, get_all_loaders_from_features_dict
from train import MLP, make_feature_extractor, make_cnn_model, get_input_size, make_model, train, test
from train import avg_per_class_accuracy, only_positive_accuracy

#TODO:
# 1 = 6 eval metrics
# 2 = test on both 30% and 100%
ALPHA_VALUE_DICT = {
    'fixed' : [0.25, 0.5, 1.0, 2.0, 5.0],
    'dynamic' : [0.25, 0.5, 0.75, 1.0], # coeff * N/k, where k is the buffer size equal to 2 buckets
}

argparser.add_argument("--alpha_value_mode",
                       default='fixed',
                    #    default='dynamic',
                       help="Whether alpha is growing as stream is moving")

def get_loaders_from_features_dict(features_dict, train_mode, hyperparameter, excluded_bucket_idx=0):
    loaders_dict = {}  # Saved the splitted loader for each bucket

    loader_func = get_loader_func(train_mode, hyperparameter.batch_size)

    for b_idx in features_dict:
        loaders_dict[b_idx] = {}
        train_items = features_dict[b_idx]['all'].copy()
        loader = loader_func(train_items, True)
        loaders_dict[b_idx]['train'] = loader

        for k_name in ['test', 'all']:
            items = features_dict[b_idx][k_name].copy()
            loader = loader_func(items, False)
            loaders_dict[b_idx][k_name] = loader
    return loaders_dict

def get_curr_and_prev_loaders_from_features_dict(features_dict, train_mode, hyperparameter, excluded_bucket_idx=0):
    all_bucket = sorted(list(features_dict.keys()))
    print(f"Excluding {excluded_bucket_idx} from the sequential loaders")
    if type(excluded_bucket_idx) == int:
        features_dict = {k: features_dict[k]
                        for k in all_bucket if k != excluded_bucket_idx}
    curr_and_prev_loaders_dict = {}
    loader_func = get_loader_func(train_mode, hyperparameter.batch_size)

    for b_idx in all_bucket:
        curr_and_prev_loaders_dict[b_idx] = {}
        assert 'val' not in features_dict[b_idx]
        train_items = []
        prev_b_idx = b_idx - 1
        if prev_b_idx in features_dict:
            print(f"Add {prev_b_idx} to {b_idx} train loader")
            train_items += features_dict[prev_b_idx]['all']
        train_items += features_dict[b_idx]['all']
        train_loader = loader_func(train_items, True)
        curr_and_prev_loaders_dict[b_idx]['train'] = train_loader
        
        test_items = []
        test_items += features_dict[b_idx]['test']
        test_loader = loader_func(test_items, False)
        curr_and_prev_loaders_dict[b_idx]['test'] = test_loader

        all_items = []
        all_items += features_dict[b_idx]['all']
        all_loader = loader_func(all_items.copy(), False)
        curr_and_prev_loaders_dict[b_idx]['all'] = all_loader
        
    return curr_and_prev_loaders_dict

def get_sequential_loaders_from_features_dict(features_dict, train_mode, hyperparameter, excluded_bucket_idx=0):
    all_bucket = sorted(list(features_dict.keys()))
    print(f"Excluding {excluded_bucket_idx} from the sequential loaders")
    if type(excluded_bucket_idx) == int:
        features_dict = {k: features_dict[k]
                        for k in all_bucket if k != excluded_bucket_idx}
    sequential_loaders_dict = {}

    loader_func = get_loader_func(train_mode, hyperparameter.batch_size)

    for b_idx in all_bucket:
        sequential_loaders_dict[b_idx] = {}
        assert 'val' not in features_dict[b_idx]

        train_items = []
        for curr_b_idx in all_bucket:
            if curr_b_idx <= b_idx:
                train_items += features_dict[curr_b_idx]['all']
        train_loader = loader_func(train_items, True)
        sequential_loaders_dict[b_idx]['train'] = train_loader
            
        test_items = []
        test_items += features_dict[b_idx]['test']
        test_loader = loader_func(test_items, False)
        sequential_loaders_dict[b_idx]['test'] = test_loader

        all_items = []
        all_items += features_dict[b_idx]['all']
        all_loader = loader_func(all_items.copy(), False)
        sequential_loaders_dict[b_idx]['all'] = all_loader
        
    return sequential_loaders_dict


def run_single(loaders_dict, all_query, train_mode):
    result_single_dict = {'models': {}, # key is bucket index
                          'b1_b2_accuracy_matrix': None,
                          'accuracy': {},  # key is bucket index
                          'b1_b2_per_class_accuracy_dict': {},  # key is bucket index
                          'only_positive_accuracy_test': None,
                          'avg_per_class_accuracy_test': None,
                          'b1_b2_accuracy_matrix_all': None, # accuracy on all training and test
                          'b1_b2_per_class_accuracy_dict_all': {},  # key is bucket index # accuracy on all training and test
                          'only_positive_accuracy_all': None,
                          'avg_per_class_accuracy_all': None,
                          'best_result_single': {},  # key is bucket index
                          'avg_results_single': {}}  # key is bucket index
    all_bucket = len(list(loaders_dict.keys()))
    sorted_buckets = sorted(list(loaders_dict.keys()))
    bucket_index_to_index = {sorted_buckets[i]: i for i in range(all_bucket)}
    print("bucket_index_to_index:")
    print(bucket_index_to_index)
    result_single_dict['bucket_index_to_index'] = bucket_index_to_index
    single_accuracy_test = np.zeros((all_bucket, all_bucket))
    only_positive_accuracy_test = np.zeros((all_bucket, all_bucket))
    avg_per_class_accuracy_test = np.zeros((all_bucket, all_bucket))
    single_accuracy_all = np.zeros((all_bucket, all_bucket))
    only_positive_accuracy_all = np.zeros((all_bucket, all_bucket))
    avg_per_class_accuracy_all = np.zeros((all_bucket, all_bucket))
    b1_b2_per_class_accuracy_dict_all = {}
    b1_b2_per_class_accuracy_dict = {}
    for b1 in sorted_buckets:
        b1_b2_per_class_accuracy_dict[b1] = {}
        b1_b2_per_class_accuracy_dict_all[b1] = {}
        single_model, single_accuracy_b1, best_result, avg_results = train(loaders_dict[b1],
                                                                           train_mode,
                                                                           len(all_query),
                                                                           epochs=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].epochs,
                                                                           lr=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].lr,
                                                                           weight_decay=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].weight_decay,
                                                                           step_size=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].step)
        result_single_dict['models'][b1] = single_model
        result_single_dict['accuracy'][b1] = single_accuracy_b1
        result_single_dict['best_result_single'][b1] = best_result
        result_single_dict['avg_results_single'][b1] = avg_results
        for b2 in sorted_buckets:
            test_loader_b2 = loaders_dict[b2]['test']
            single_accuracy_b1_b2, per_class_accuracy_b1_b2 = test(test_loader_b2, single_model, train_mode, class_names=all_query)
            b1_b2_per_class_accuracy_dict[b1][b2] = per_class_accuracy_b1_b2
            b1_idx = bucket_index_to_index[b1]
            b2_idx = bucket_index_to_index[b2]
            only_positive_accuracy_test[b1_idx][b2_idx] = only_positive_accuracy(per_class_accuracy_b1_b2)
            avg_per_class_accuracy_test[b1_idx][b2_idx] = avg_per_class_accuracy(per_class_accuracy_b1_b2)
            single_accuracy_test[b1_idx][b2_idx] = single_accuracy_b1_b2
            print(f"Train {b1}, test on {b2}: {single_accuracy_b1_b2:.4%} (per sample), {only_positive_accuracy_test[b1_idx][b2_idx]:.4%} (pos only), {avg_per_class_accuracy_test[b1_idx][b2_idx]:.4%} (per class avg)")
            all_loader_b2 = loaders_dict[b2]['all']
            single_accuracy_b1_b2_on_all, per_class_accuracy_b1_b2_on_all = test(all_loader_b2, single_model, train_mode, class_names=all_query)
            b1_b2_per_class_accuracy_dict_all[b1][b2] = per_class_accuracy_b1_b2_on_all
            only_positive_accuracy_all[b1_idx][b2_idx] = only_positive_accuracy(per_class_accuracy_b1_b2_on_all)
            avg_per_class_accuracy_all[b1_idx][b2_idx] = avg_per_class_accuracy(per_class_accuracy_b1_b2_on_all)
            single_accuracy_all[b1_idx][b2_idx] = single_accuracy_b1_b2_on_all
            print(f"Train {b1}, test on {b2} (train+test): {single_accuracy_b1_b2_on_all:.4%} (per sample), {only_positive_accuracy_all[b1_idx][b2_idx]:.4%} (pos only), {avg_per_class_accuracy_all[b1_idx][b2_idx]:.4%} (per class avg)")
    result_single_dict['b1_b2_accuracy_matrix'] = single_accuracy_test
    result_single_dict['b1_b2_per_class_accuracy_dict'] = b1_b2_per_class_accuracy_dict
    result_single_dict['only_positive_accuracy_test'] = only_positive_accuracy_test
    result_single_dict['avg_per_class_accuracy_test'] = avg_per_class_accuracy_test
    result_single_dict['b1_b2_accuracy_matrix_all'] = single_accuracy_all
    result_single_dict['b1_b2_per_class_accuracy_dict_all'] = b1_b2_per_class_accuracy_dict_all
    result_single_dict['only_positive_accuracy_all'] = only_positive_accuracy_all
    result_single_dict['avg_per_class_accuracy_all'] = avg_per_class_accuracy_all
    return result_single_dict

def run_single_finetune(loaders_dict, all_query, train_mode):
    result_single_finetune_dict = {'models': {}, # key is bucket index
                                   'b1_b2_accuracy_matrix': None,
                                   'accuracy': {},  # key is bucket index
                                   'b1_b2_per_class_accuracy_dict': {},  # key is bucket index
                                   'only_positive_accuracy_test': None,
                                   'avg_per_class_accuracy_test': None,
                                   'b1_b2_accuracy_matrix_all': None, # accuracy on all training and test
                                    'b1_b2_per_class_accuracy_dict_all': {},  # key is bucket index # accuracy on all training and test
                                    'only_positive_accuracy_all': None,
                                    'avg_per_class_accuracy_all': None,
                                   'best_result_single': {},  # key is bucket index
                                   'avg_results_single': {}}  # key is bucket index
    all_bucket = len(list(loaders_dict.keys()))
    sorted_buckets = sorted(list(loaders_dict.keys()))
    bucket_index_to_index = {sorted_buckets[i]: i for i in range(all_bucket)}
    print("bucket_index_to_index:")
    print(bucket_index_to_index)
    result_single_finetune_dict['bucket_index_to_index'] = bucket_index_to_index
    single_accuracy_test = np.zeros((all_bucket, all_bucket))
    only_positive_accuracy_test = np.zeros((all_bucket, all_bucket))
    avg_per_class_accuracy_test = np.zeros((all_bucket, all_bucket))
    b1_b2_per_class_accuracy_dict = {}
    single_accuracy_all = np.zeros((all_bucket, all_bucket))
    only_positive_accuracy_all = np.zeros((all_bucket, all_bucket))
    avg_per_class_accuracy_all = np.zeros((all_bucket, all_bucket))
    b1_b2_per_class_accuracy_dict_all = {}
    b1_b2_per_class_accuracy_dict = {}
    single_model = None
    for b1 in sorted_buckets:
        b1_b2_per_class_accuracy_dict[b1] = {}
        b1_b2_per_class_accuracy_dict_all[b1] = {}
        single_model, single_accuracy_b1, best_result, avg_results = train(loaders_dict[b1],
                                                                           train_mode,
                                                                           len(all_query),
                                                                           epochs=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].epochs,
                                                                           lr=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].lr,
                                                                           weight_decay=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].weight_decay,
                                                                           step_size=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].step,
                                                                           finetuned_model=single_model)
        result_single_finetune_dict['models'][b1] = single_model
        result_single_finetune_dict['accuracy'][b1] = single_accuracy_b1
        result_single_finetune_dict['best_result_single'][b1] = best_result
        result_single_finetune_dict['avg_results_single'][b1] = avg_results
        for b2 in sorted_buckets:
            test_loader_b2 = loaders_dict[b2]['test']
            single_accuracy_b1_b2, per_class_accuracy_b1_b2 = test(test_loader_b2, single_model, train_mode, class_names=all_query)
            b1_b2_per_class_accuracy_dict[b1][b2] = per_class_accuracy_b1_b2
            b1_idx = bucket_index_to_index[b1]
            b2_idx = bucket_index_to_index[b2]
            only_positive_accuracy_test[b1_idx][b2_idx] = only_positive_accuracy(per_class_accuracy_b1_b2)
            avg_per_class_accuracy_test[b1_idx][b2_idx] = avg_per_class_accuracy(per_class_accuracy_b1_b2)
            single_accuracy_test[b1_idx][b2_idx] = single_accuracy_b1_b2
            print(f"Train {b1}, test on {b2}: {single_accuracy_b1_b2:.4%} (per sample), {only_positive_accuracy_test[b1_idx][b2_idx]:.4%} (pos only), {avg_per_class_accuracy_test[b1_idx][b2_idx]:.4%} (per class avg)")
            all_loader_b2 = loaders_dict[b2]['all']
            single_accuracy_b1_b2_on_all, per_class_accuracy_b1_b2_on_all = test(all_loader_b2, single_model, train_mode, class_names=all_query)
            b1_b2_per_class_accuracy_dict_all[b1][b2] = per_class_accuracy_b1_b2_on_all
            only_positive_accuracy_all[b1_idx][b2_idx] = only_positive_accuracy(per_class_accuracy_b1_b2_on_all)
            avg_per_class_accuracy_all[b1_idx][b2_idx] = avg_per_class_accuracy(per_class_accuracy_b1_b2_on_all)
            single_accuracy_all[b1_idx][b2_idx] = single_accuracy_b1_b2_on_all
            print(f"Train {b1}, test on {b2} (train+test): {single_accuracy_b1_b2_on_all:.4%} (per sample), {only_positive_accuracy_all[b1_idx][b2_idx]:.4%} (pos only), {avg_per_class_accuracy_all[b1_idx][b2_idx]:.4%} (per class avg)")
    result_single_finetune_dict['b1_b2_accuracy_matrix'] = single_accuracy_test
    result_single_finetune_dict['b1_b2_per_class_accuracy_dict'] = b1_b2_per_class_accuracy_dict
    result_single_finetune_dict['only_positive_accuracy_test'] = only_positive_accuracy_test
    result_single_finetune_dict['avg_per_class_accuracy_test'] = avg_per_class_accuracy_test
    result_single_finetune_dict['b1_b2_accuracy_matrix_all'] = single_accuracy_all
    result_single_finetune_dict['b1_b2_per_class_accuracy_dict_all'] = b1_b2_per_class_accuracy_dict_all
    result_single_finetune_dict['only_positive_accuracy_all'] = only_positive_accuracy_all
    result_single_finetune_dict['avg_per_class_accuracy_all'] = avg_per_class_accuracy_all
    return result_single_finetune_dict


def get_singlememory_loaders_from_features_dict(alpha_value_mode, alpha_value, features_dict, train_mode, hyperparameter, excluded_bucket_idx=0):
    # TODO: Single memory + All loaders
    all_bucket = sorted(list(features_dict.keys()))
    print(f"Excluding {excluded_bucket_idx} from the sequential loaders")
    if type(excluded_bucket_idx) == int:
        features_dict = {k: features_dict[k]
                         for k in all_bucket if k != excluded_bucket_idx}
    single_buffer_loaders_dict = {}
    loader_func = get_loader_func(train_mode, hyperparameter.batch_size)

    train_buffer = [] # buffer is cap at size 1 bucket
    n = 0. # number of seen examples in the stream
    k = None # number of examples in two buckets. will be calculated in in first iteration. Assume each bucket has same size!
    # import pdb; pdb.set_trace()
    for idx, b_idx in enumerate(all_bucket):
        single_buffer_loaders_dict[b_idx] = {}
        assert 'val' not in features_dict[b_idx]

        test_items = []
        test_items += features_dict[b_idx]['test']
        test_loader = loader_func(test_items.copy(), False)
        single_buffer_loaders_dict[b_idx]['test'] = test_loader

        all_items = []
        all_items += features_dict[b_idx]['all']
        all_loader = loader_func(all_items.copy(), False)
        single_buffer_loaders_dict[b_idx]['all'] = all_loader
        
        if alpha_value == None:
            train_buffer = features_dict[b_idx]['all'].copy()
        else:
            n += len(features_dict[b_idx]['all'])
            if idx == 0:
                train_buffer = features_dict[b_idx]['all'].copy()
                k = n # This is currently bucket size
                # import pdb; pdb.set_trace()
                print(f"Bucket {b_idx}: Since it is first, we put all examples {n} in. And buffer size should be {n} = {k}")
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
                for item in features_dict[b_idx]['all']:
                    if random.random() <= prob:
                        new_items_to_add_to_buffer.append(item)
                print(f"Selected {len(new_items_to_add_to_buffer)} samples out of {k} incoming samples")
                random.shuffle(train_buffer)
                train_buffer = train_buffer[:len(train_buffer) - len(new_items_to_add_to_buffer)]
                train_buffer += new_items_to_add_to_buffer
            
        train_loader = loader_func(train_buffer.copy(), True) # Important! TO use copy() otherwise it will be changed
        single_buffer_loaders_dict[b_idx]['train'] = train_loader
        
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

    for alpha_value in ALPHA_VALUE_DICT[args.alpha_value_mode]:
        print(f"Generate loader for alpha value {alpha_value} in mode {args.alpha_value_mode}")        
        alpha_loaders_dict_path = os.path.join(exp_result_save_path,
                                               f"online_alpha_loaders_dict_{dataset_str(args.mode)}_singlememory_{args.alpha_value_mode}_{alpha_value}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
        
        if os.path.exists(alpha_loaders_dict_path):
            print(f"{alpha_loaders_dict_path} already exists.")
            alpha_loaders_dict = load_pickle(alpha_loaders_dict_path)
        else:
            alpha_loaders_dict = get_singlememory_loaders_from_features_dict(
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
                                                       f"results_dict_online_singlememory_{args.alpha_value_mode}_{alpha_value}_retrain_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
        
        if not os.path.exists(results_dict_alpha_retrain_path):
            results_dict_alpha_retrain = run_single(alpha_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_alpha_retrain_path, results_dict_alpha_retrain)
            print(f"Saved at {results_dict_alpha_retrain_path}")
        else:
            print(results_dict_alpha_retrain_path + " already exists")
        
        ############### Run Alpha weighting (Finetune) Experiment
        results_dict_alpha_finetune_path = os.path.join(exp_result_save_path,
                                                       f"results_dict_online_singlememory_{args.alpha_value_mode}_{alpha_value}_finetune_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
        
        if not os.path.exists(results_dict_alpha_finetune_path):
            results_dict_alpha_finetune = run_single_finetune(alpha_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_alpha_finetune_path, results_dict_alpha_finetune)
            print(f"Saved at {results_dict_alpha_finetune_path}")
        else:
            print(results_dict_alpha_finetune_path + " already exists")
        

    #### Rerun the rest for testing all loader

    loaders_dict_path = os.path.join(exp_result_save_path,
                                     f"loaders_dict_online_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

    if os.path.exists(loaders_dict_path):
        print(f"{loaders_dict_path} already exists.")
        loaders_dict = load_pickle(loaders_dict_path)
    else:
        loaders_dict = get_loaders_from_features_dict(
                           features_dict,
                           args.train_mode,
                           HYPER_DICT[TRAIN_MODES_CATEGORY[args.train_mode].network_type],
                           excluded_bucket_idx=excluded_bucket_idx
                       )
        save_obj_as_pickle(loaders_dict_path, loaders_dict)

    ############### Run Single Bucket Experiment
    results_dict_single_path = os.path.join(exp_result_save_path,
                                            f"results_dict_online_single_with_all_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

    if not os.path.exists(results_dict_single_path):
        result_single_dict = run_single(loaders_dict, all_query, args.train_mode)
        save_obj_as_pickle(results_dict_single_path, result_single_dict)
        print(f"Saved at {results_dict_single_path}")
    else:
        print(results_dict_single_path + " already exists")

    ############### Run Single Bucket (Finetune) Experiment
    results_dict_single_finetune_path = os.path.join(exp_result_save_path,
                                                     f"results_dict_online_single_finetune_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

    if not os.path.exists(results_dict_single_finetune_path):
        result_single_finetune_dict = run_single_finetune(loaders_dict, all_query, args.train_mode)
        save_obj_as_pickle(results_dict_single_finetune_path, result_single_finetune_dict)
        print(f"Saved at {results_dict_single_finetune_path}")
    else:
        print(results_dict_single_finetune_path + " already exists")

    sequential_loaders_dict_path = os.path.join(exp_result_save_path,
                                                f"sequential_loaders_dict_online_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

    if os.path.exists(sequential_loaders_dict_path):
        print(f"{sequential_loaders_dict_path} already exists.")
        sequential_loaders_dict = load_pickle(sequential_loaders_dict_path)
    else:
        sequential_loaders_dict = get_sequential_loaders_from_features_dict(
                                    features_dict,
                                    args.train_mode,
                                    HYPER_DICT[TRAIN_MODES_CATEGORY[args.train_mode].network_type],
                                    excluded_bucket_idx=excluded_bucket_idx
                                )
        save_obj_as_pickle(sequential_loaders_dict_path, sequential_loaders_dict)
    
    curr_and_prev_loaders_dict_path = os.path.join(exp_result_save_path,
                                                f"curr_and_prev_loaders_dict_online_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

    if os.path.exists(curr_and_prev_loaders_dict_path):
        print(f"{curr_and_prev_loaders_dict_path} already exists.")
        curr_and_prev_loaders_dict = load_pickle(curr_and_prev_loaders_dict_path)
    else:
        curr_and_prev_loaders_dict = get_curr_and_prev_loaders_from_features_dict(
                                        features_dict,
                                        args.train_mode,
                                        HYPER_DICT[TRAIN_MODES_CATEGORY[args.train_mode].network_type],
                                        excluded_bucket_idx=excluded_bucket_idx
                                    )
        save_obj_as_pickle(curr_and_prev_loaders_dict_path, curr_and_prev_loaders_dict)
    
    ############### Run Curr and Prev (Retrain) Experiment
    results_dict_curr_and_prev_retrain_path = os.path.join(exp_result_save_path,
                                                        f"results_dict_online_curr_and_prev_retrain_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
    
    if not os.path.exists(results_dict_curr_and_prev_retrain_path):
        results_dict_curr_and_prev_retrain = run_single(curr_and_prev_loaders_dict, all_query, args.train_mode)
        save_obj_as_pickle(results_dict_curr_and_prev_retrain_path, results_dict_curr_and_prev_retrain)
        print(f"Saved at {results_dict_curr_and_prev_retrain_path}")
    else:
        print(results_dict_curr_and_prev_retrain_path + " already exists")
    
    ############### Run Curr and Prev (Finetune) Experiment
    results_dict_curr_and_prev_finetune_path = os.path.join(exp_result_save_path,
                                                        f"results_dict_online_curr_and_prev_finetune_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

    if not os.path.exists(results_dict_curr_and_prev_finetune_path):
        results_dict_curr_and_prev_finetune = run_single_finetune(curr_and_prev_loaders_dict, all_query, args.train_mode)
        save_obj_as_pickle(results_dict_curr_and_prev_finetune_path, results_dict_curr_and_prev_finetune)
        print(f"Saved at {results_dict_curr_and_prev_finetune_path}")
    else:
        print(results_dict_curr_and_prev_finetune_path + " already exists")

    ############### Run Sequential (Retrain) Experiment
    results_dict_sequential_retrain_path = os.path.join(exp_result_save_path,
                                                        f"results_dict_online_sequential_retrain_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

    if not os.path.exists(results_dict_sequential_retrain_path):
        results_dict_sequential_retrain = run_single(sequential_loaders_dict, all_query, args.train_mode)
        save_obj_as_pickle(results_dict_sequential_retrain_path, results_dict_sequential_retrain)
        print(f"Saved at {results_dict_sequential_retrain_path}")
    else:
        print(results_dict_sequential_retrain_path + " already exists")
    
    ############### Run Sequential (Finetune) Experiment
    results_dict_sequential_finetune_path = os.path.join(exp_result_save_path,
                                                        f"results_dict_online_sequential_finetune_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

    if not os.path.exists(results_dict_sequential_finetune_path):
        results_dict_sequential_finetune = run_single_finetune(sequential_loaders_dict, all_query, args.train_mode)
        save_obj_as_pickle(results_dict_sequential_finetune_path, results_dict_sequential_finetune)
        print(f"Saved at {results_dict_sequential_finetune_path}")
    else:
        print(results_dict_sequential_finetune_path + " already exists")



    # import pdb; pdb.set_trace()

    

    
