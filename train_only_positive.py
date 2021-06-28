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
from train import get_seed_str, use_val_set, dataset_str, make_features_dict, extract_features

argparser = argparse.ArgumentParser()
argparser.add_argument("--folder_path",
                       default='/compute/autobot-1-1/zhiqiu/yfcc_dynamic_10',
                       help="The folder with the images and query_dict.pickle")
argparser.add_argument("--exp_result_path",
                       default='/project_data/ramanan/zhiqiu/yfcc_dynamic_10',
                       help="The folder with the images and query_dict.pickle")
argparser.add_argument("--dataset_name",
                       default='dynamic_300_positive_only', # best training loss
                       help="Only evaluate on this label set")
argparser.add_argument('--train_mode',
                       default='linear', choices=TRAIN_MODES_CATEGORY.keys(),
                       help='Train mode')
argparser.add_argument('--mode',
                       default='no_test_set', choices=MODE_DICT.keys(),
                       help='Mode for dataset split')
argparser.add_argument('--seed',
                       default=None, type=int,
                       help='Seed for experiment')
argparser.add_argument('--excluded_bucket_idx',
                       default=0, type=int,
                       help='Excluding this bucket from all experiments')

from train import get_loader_func, get_all_loaders_from_features_dict, get_loaders_from_features_dict, get_curr_and_prev_loaders_from_features_dict
from train import get_curr_and_random_prev_loaders_from_features_dict, get_sequential_loaders_from_features_dict
from train import MLP, make_feature_extractor, make_cnn_model, get_input_size, make_model, train, test
from train import avg_per_class_accuracy, only_positive_accuracy, run_baseline, run_single, run_single_finetune

def make_dataset_dict(query_dict, mode):
    dataset_dict = {}
    for b_idx in query_dict:
        print(f"<<<<<<<<<<<First create split the dataset for bucket {b_idx}")
        dataset_dict[b_idx] = split_dataset(query_dict[b_idx], mode)
    return dataset_dict

def split_dataset(query_dict, mode):
    dataset_dict = {}

    def gather_data(query, indices):
        return {
            'clip_features': [query_dict[query]['clip_features'][i] for i in indices],
            'metadata': [query_dict[query]['metadata'][i] for i in indices],
            'D': [query_dict[query]['D'][i] for i in indices],
        }

    for query in query_dict:
        if query == NEGATIVE_LABEL:
            print(f"SKIPPING {query}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            continue
        num_of_data = len(query_dict[query]['metadata'])
        # for query in all_query:
        #     assert num_of_data == len(query_dict[query]['metadata'])
        data_indices = list(range(num_of_data))
        random.shuffle(data_indices)
        if use_val_set(mode):
            val_set_size = int(MODE_DICT[mode]['VAL_SET_RATIO'] * num_of_data)
        else:
            val_set_size = 0
        val_set_indices = data_indices[:val_set_size]

        test_set_size = int(MODE_DICT[mode]['TEST_SET_RATIO'] * num_of_data)
        test_set_indices = data_indices[val_set_size:val_set_size+test_set_size]
        train_set_size = int(MODE_DICT[mode]['TRAIN_SET_RATIO'] * num_of_data)
        train_set_indices = data_indices[val_set_size+test_set_size:]
        total_size = len(train_set_indices) + len(val_set_indices) + len(test_set_indices)
        if not total_size == num_of_data:
            import pdb; pdb.set_trace()
        dataset_dict[query] = {}
        dataset_dict[query]['train'] = gather_data(query, train_set_indices)
        if use_val_set(mode):
            dataset_dict[query]['val'] = gather_data(query, val_set_indices)
        dataset_dict[query]['test'] = gather_data(query, test_set_indices)
        # TODO: Handle when dataset_dict has empty val set
        dataset_dict[query]['all'] = gather_data(query, data_indices)

    return dataset_dict


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
    all_query = [q for q in all_query if q != NEGATIVE_LABEL]
    print(f"We have {len(all_query)} classes.")
    print(all_query)

    ############### Create Datasets
    dataset_dict_path = os.path.join(exp_result_save_path,
                                     f"dataset_dict_{dataset_str(args.mode)}_{seed_str}.pickle")
    if os.path.exists(dataset_dict_path):
        print(f"{dataset_dict_path} already exists.")
        dataset_dict = load_pickle(dataset_dict_path)
    else:
        dataset_dict = make_dataset_dict(query_dict, args.mode) # Will save dataset_dict in file loc
        save_obj_as_pickle(dataset_dict_path, dataset_dict)
    
    ############### Create Features
    features_dict_path = os.path.join(exp_result_save_path,
                                      f"features_dict_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}.pickle")
    if os.path.exists(features_dict_path):
        print(f"{features_dict_path} already exists.")
        features_dict = load_pickle(features_dict_path)
    else:
        features_dict = make_features_dict(dataset_dict, args.train_mode)
        save_obj_as_pickle(features_dict_path, features_dict)

    ############### Create DataLoaders
    all_loaders_dict_path = os.path.join(exp_result_save_path,
                                         f"all_loaders_dict_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
    if os.path.exists(all_loaders_dict_path):
        print(f"{all_loaders_dict_path} already exists.")
        all_loaders_dict = load_pickle(all_loaders_dict_path)
    else:
        all_loaders_dict = get_all_loaders_from_features_dict(
                               features_dict,
                               args.train_mode,
                               HYPER_DICT[TRAIN_MODES_CATEGORY[args.train_mode].network_type],
                               excluded_bucket_idx=excluded_bucket_idx
                           )
        save_obj_as_pickle(all_loaders_dict_path, all_loaders_dict)

    loaders_dict_path = os.path.join(exp_result_save_path,
                                     f"loaders_dict_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

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

    if not use_val_set(args.mode):
        # get_curr_and_random_prev_loaders_from_features_dict
        print("Since not using a validation set, we can perform sequential learning experiment")
        sequential_loaders_dict_path = os.path.join(exp_result_save_path,
                                                    f"sequential_loaders_dict_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

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
                                                    f"curr_and_prev_loaders_dict_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

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
        
        curr_and_random_prev_loaders_dict_path = os.path.join(exp_result_save_path,
                                                    f"curr_and_random_prev_loaders_dict_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

        if os.path.exists(curr_and_random_prev_loaders_dict_path):
            print(f"{curr_and_random_prev_loaders_dict_path} already exists.")
            curr_and_random_prev_loaders_dict = load_pickle(curr_and_random_prev_loaders_dict_path)
        else:
            curr_and_random_prev_loaders_dict = get_curr_and_random_prev_loaders_from_features_dict(
                                                    features_dict,
                                                    args.train_mode,
                                                    HYPER_DICT[TRAIN_MODES_CATEGORY[args.train_mode].network_type],
                                                    excluded_bucket_idx=excluded_bucket_idx
                                                )
            save_obj_as_pickle(curr_and_random_prev_loaders_dict_path, curr_and_random_prev_loaders_dict)
        
        ############### Run Curr and Random Prev (Retrain) Experiment
        results_dict_curr_and_random_prev_retrain_path = os.path.join(exp_result_save_path,
                                                            f"results_dict_curr_and_random_prev_retrain_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
        
        if not os.path.exists(results_dict_curr_and_random_prev_retrain_path):
            results_dict_curr_and_random_prev_retrain = run_single(curr_and_random_prev_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_curr_and_random_prev_retrain_path, results_dict_curr_and_random_prev_retrain)
            print(f"Saved at {results_dict_curr_and_random_prev_retrain_path}")
        else:
            print(results_dict_curr_and_random_prev_retrain_path + " already exists")
        
        ############### Run Curr and Random Prev (Finetune) Experiment
        results_dict_curr_and_random_prev_finetune_path = os.path.join(exp_result_save_path,
                                                            f"results_dict_curr_and_random_prev_finetune_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

        if not os.path.exists(results_dict_curr_and_random_prev_finetune_path):
            results_dict_curr_and_random_prev_finetune = run_single_finetune(curr_and_random_prev_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_curr_and_random_prev_finetune_path, results_dict_curr_and_random_prev_finetune)
            print(f"Saved at {results_dict_curr_and_random_prev_finetune_path}")
        else:
            print(results_dict_curr_and_random_prev_finetune_path + " already exists")

        ############### Run Curr and Prev (Retrain) Experiment
        results_dict_curr_and_prev_retrain_path = os.path.join(exp_result_save_path,
                                                            f"results_dict_curr_and_prev_retrain_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
        
        if not os.path.exists(results_dict_curr_and_prev_retrain_path):
            results_dict_curr_and_prev_retrain = run_single(curr_and_prev_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_curr_and_prev_retrain_path, results_dict_curr_and_prev_retrain)
            print(f"Saved at {results_dict_curr_and_prev_retrain_path}")
        else:
            print(results_dict_curr_and_prev_retrain_path + " already exists")
        
        ############### Run Curr and Prev (Finetune) Experiment
        results_dict_curr_and_prev_finetune_path = os.path.join(exp_result_save_path,
                                                            f"results_dict_curr_and_prev_finetune_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

        if not os.path.exists(results_dict_curr_and_prev_finetune_path):
            results_dict_curr_and_prev_finetune = run_single_finetune(curr_and_prev_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_curr_and_prev_finetune_path, results_dict_curr_and_prev_finetune)
            print(f"Saved at {results_dict_curr_and_prev_finetune_path}")
        else:
            print(results_dict_curr_and_prev_finetune_path + " already exists")

        ############### Run Sequential (Retrain) Experiment
        results_dict_sequential_retrain_path = os.path.join(exp_result_save_path,
                                                            f"results_dict_sequential_retrain_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

        if not os.path.exists(results_dict_sequential_retrain_path):
            results_dict_sequential_retrain = run_single(sequential_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_sequential_retrain_path, results_dict_sequential_retrain)
            print(f"Saved at {results_dict_sequential_retrain_path}")
        else:
            print(results_dict_sequential_retrain_path + " already exists")
        
        ############### Run Sequential (Finetune) Experiment
        results_dict_sequential_finetune_path = os.path.join(exp_result_save_path,
                                                            f"results_dict_sequential_finetune_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

        if not os.path.exists(results_dict_sequential_finetune_path):
            results_dict_sequential_finetune = run_single_finetune(sequential_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_sequential_finetune_path, results_dict_sequential_finetune)
            print(f"Saved at {results_dict_sequential_finetune_path}")
        else:
            print(results_dict_sequential_finetune_path + " already exists")

    # curr_and_randomprev_loaders_dict_path = os.path.join(exp_result_save_path,
    #                                                f"curr_and_randomprev_loaders_dict_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

    # if os.path.exists(curr_and_randomprev_loaders_dict_path):
    #     print(f"{curr_and_randomprev_loaders_dict_path} already exists.")
    #     curr_and_randomprev_loaders_dict = load_pickle(curr_and_randomprev_loaders_dict_path)
    # else:
    #     curr_and_randomprev_loaders_dict = get_curr_and_randomprev_loaders_from_features_dict(
    #                                            features_dict,
    #                                            args.train_mode,
    #                                            HYPER_DICT[TRAIN_MODES_CATEGORY[args.train_mode].network_type],
    #                                            excluded_bucket_idx=excluded_bucket_idx
    #                                        )
    #     save_obj_as_pickle(curr_and_randomprev_loaders_dict_path, curr_and_randomprev_loaders_dict)
        
    ############### Run Baseline Experiment
    results_dict_all_path = os.path.join(exp_result_save_path,
                                         f"results_dict_all_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
    if not os.path.exists(results_dict_all_path):
        result_baseline_dict = run_baseline(all_loaders_dict, all_query, args.train_mode)
        save_obj_as_pickle(results_dict_all_path, result_baseline_dict)
        print(f"Saved at {results_dict_all_path}")
    else:
        print(f"Baseline result saved at {results_dict_all_path}")

    ############### Run Single Bucket Experiment
    results_dict_single_path = os.path.join(exp_result_save_path,
                                            f"results_dict_single_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

    if not os.path.exists(results_dict_single_path):
        result_single_dict = run_single(loaders_dict, all_query, args.train_mode)
        save_obj_as_pickle(results_dict_single_path, result_single_dict)
        print(f"Saved at {results_dict_single_path}")
    else:
        print(results_dict_single_path + " already exists")

    ############### Run Single Bucket (Finetune) Experiment
    results_dict_single_finetune_path = os.path.join(exp_result_save_path,
                                                     f"results_dict_single_finetune_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

    if not os.path.exists(results_dict_single_finetune_path):
        result_single_finetune_dict = run_single_finetune(loaders_dict, all_query, args.train_mode)
        save_obj_as_pickle(results_dict_single_finetune_path, result_single_finetune_dict)
        print(f"Saved at {results_dict_single_finetune_path}")
    else:
        print(results_dict_single_finetune_path + " already exists")



    # import pdb; pdb.set_trace()

    
