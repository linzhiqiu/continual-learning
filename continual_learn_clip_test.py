# python continual_learn_clip_test.py --label_set imagenet1K --use_difference_of_query --class_size 1000 --num_of_buckets 2 --train_mode nearest_nn
# python continual_learn_clip_test.py --label_set imagenet1K --use_difference_of_query --class_size 1000 --num_of_buckets 2 --train_mode nearest_nn --train_bucket future


# python continual_learn_clip_test.py --label_set imagenet1K --class_size 1000 --num_of_buckets 2 --train_mode nearest_nn
# python continual_learn_clip_test.py --label_set imagenet1K --class_size 1000 --num_of_buckets 2 --train_mode nearest_nn --train_bucket future

# python continual_learn_clip_test.py --label_set imagenet1K --use_difference_of_query --class_size 1000 --num_of_buckets 2 --train_mode nearest_center
# python continual_learn_clip_test.py --label_set imagenet1K --use_difference_of_query --class_size 1000 --num_of_buckets 2 --train_mode nearest_center --train_bucket future


# python continual_learn_clip_test.py --label_set imagenet1K --class_size 1000 --num_of_buckets 2 --train_mode nearest_center
# python continual_learn_clip_test.py --label_set imagenet1K --class_size 1000 --num_of_buckets 2 --train_mode nearest_center --train_bucket future

# python continual_learn_clip_test.py --label_set vehicle_7 --use_difference_of_query --class_size 1000 --num_of_buckets 2 --train_mode debug
# python continual_learn_clip_test.py --label_set vehicle_7 --class_size 1000 --num_of_buckets 2 --train_mode debug

# python continual_learn_clip_test.py --label_set cifar10 --use_difference_of_query --class_size 1000 --num_of_buckets 2 --train_mode debug
# python continual_learn_clip_test.py --label_set cifar10 --class_size 1000 --num_of_buckets 2 --train_mode debug

# python continual_learn_clip_test.py --label_set cifar100 --use_difference_of_query --class_size 1000 --num_of_buckets 2 --train_mode debug
# python continual_learn_clip_test.py --label_set cifar100 --class_size 1000 --num_of_buckets 2 --train_mode debug
# python continual_learn_clip_test.py --label_set cifar100 --query_title "" --label_set cifar100 --class_size 1000  --clip_dataset_path /scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_jan_31/clip_dataset_no_pre_prompt/ --train_mode debug
# python continual_learn_clip_test.py --query_title "" --label_set cifar100 --lmb 10. --use_difference_of_query --class_size 1000  --clip_dataset_path /scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_jan_31/clip_dataset_no_pre_prompt/ --train_mode debug
# python continual_learn_clip_test.py --query_title "" --label_set cifar100 --lmb 1. --use_difference_of_query --class_size 1000  --clip_dataset_path /scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_jan_31/clip_dataset_no_pre_prompt/ --train_mode debug
# python continual_learn_clip_test.py --label_set cifar100 --lmb 10. --use_difference_of_query --class_size 1000 --train_mode debug

# python continual_learn_clip_test.py --label_set imagenet1K --use_difference_of_query --class_size 1000 --num_of_buckets 2 --train_mode debug
# python continual_learn_clip_test.py --label_set imagenet1K --class_size 1000 --num_of_buckets 2 --train_mode debug
# python continual_learn_clip_test.py --label_set imagenet1K --query_title "" --label_set imagenet1K --class_size 1000  --clip_dataset_path /scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_jan_31/clip_dataset_no_pre_prompt/ --train_mode debug
# python continual_learn_clip_test.py --query_title "" --label_set imagenet1K --lmb 100. --use_difference_of_query --class_size 1000  --clip_dataset_path /scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_jan_31/clip_dataset_no_pre_prompt/ --train_mode debug
# python continual_learn_clip_test.py --query_title "" --label_set imagenet1K --lmb 1. --use_difference_of_query --class_size 1000  --clip_dataset_path /scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_jan_31/clip_dataset_no_pre_prompt/ --train_mode debug
# python continual_learn_clip_test.py --label_set imagenet1K --lmb 100. --use_difference_of_query --class_size 1000 --train_mode debug


# python continual_learn_clip_test.py --label_set imagenet1K --use_difference_of_query --class_size 1000 --num_of_buckets 2 --train_mode debug
# python continual_learn_clip_test.py --label_set imagenet1K --class_size 1000 --num_of_buckets 2 --train_mode debug


# New version
# python continual_learn_clip_test.py --label_set vehicle_7 --class_size 1000 --use_max_score --train_mode debug
# python continual_learn_clip_test.py --label_set cifar10 --class_size 1000 --use_max_score --train_mode debug
# python continual_learn_clip_test.py --label_set cifar100 --class_size 1000 --use_max_score --train_mode debug
# python continual_learn_clip_test.py --label_set imagenet1K --epoch 80 --step 25 --class_size 1000 --use_max_score --train_mode debug

# python continual_learn_clip_test.py --label_set vehicle_7 --class_size 1000 --use_max_score --query_title none --train_mode debug
# python continual_learn_clip_test.py --label_set cifar10 --class_size 1000 --use_max_score --query_title none --train_mode debug
# python continual_learn_clip_test.py --label_set cifar100 --class_size 1000 --use_max_score --query_title none --train_mode debug
# python continual_learn_clip_test.py --label_set imagenet1K  --epoch 80 --step 25 --class_size 1000 --use_max_score --query_title none --train_mode debug


import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from glob import glob
from tqdm import tqdm
import random
import numpy as np

from utils import sort_metadata_by_date, divide, save_obj_as_pickle, load_pickle
from training_utils import make_optimizer, make_scheduler, make_model, make_loader, make_clip_loader, train, test, get_imgnet_transforms
from prepare_clip_dataset import get_save_path, QUERY_TITLE_DICT, LABEL_SETS
import argparse

# For each bucket, make sure at least VAL_SET_SIZE + TEST_SET_SIZE exist
VAL_SET_SIZE = 50
TEST_SET_SIZE = 50
TRAIN_SET_SIZE = 400

argparser = argparse.ArgumentParser()
argparser.add_argument("--clip_dataset_path", 
                        # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_25',
                        default='/scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_jan_31/clip_dataset/',
                        help="The folder that will store the curated dataset")
argparser.add_argument("--label_set", 
                        default='vehicle_7', choices=LABEL_SETS,
                        help="The label sets")
argparser.add_argument("--use_difference_of_query", 
                        action='store_true',
                        help="Whether or not to use difference of query")
argparser.add_argument('--class_size', default=1000, type=int,
                       help='number of samples per class')
argparser.add_argument("--use_max_score", 
                        action='store_true',
                        help="Keep the max scoring images")
argparser.add_argument("--lmb", 
                        default=1., type=float,
                        help="The difference of query feature ratio")
argparser.add_argument("--query_title", 
                        default='photo', 
                        choices=QUERY_TITLE_DICT.keys(),
                        help="The query title")
argparser.add_argument('--num_of_buckets', default=2, type=int,
                       metavar='N',
                       help='Divide by num_of_buckets (default: 2)')
argparser.add_argument('--train_bucket', default='past', choices=['past', 'future'],
                       help='Training bucket')
argparser.add_argument('--train_mode', default='nearest_nn', choices=['nearest_nn', 'nearest_center', 'debug'],
                       help='Train mode')
argparser.add_argument('--batch_size', default=64, type=int,
                       metavar='N',
                       help='mini-batch size (default: 64)')
argparser.add_argument('--num_workers', default=4, type=int,
                       help='num_workers (default: 4)')
argparser.add_argument('--nn_size', default=10000, type=int,
                       help='number of samples per class for initial search of top NN')

def debug_overlapping(query_dict):
    def get_ID_list(meta_list):
        ID_list = [meta.get_metadata().ID for meta in meta_list]
        return ID_list
    
    ID_dict = {}
    total_ID_set = set()
    total_ID_length = 0
    for tag in query_dict:
        ID_dict[tag] = set(get_ID_list(query_dict[tag]['metadata']))
        total_ID_set = total_ID_set.union(ID_dict[tag])
        total_ID_length += len(ID_dict[tag])
    
    overlap_dict = {}
    for tag in ID_dict:
        overlap_dict[tag] = set()
        for tag_b in ID_dict:
            if tag_b != tag:
                inter = ID_dict[tag].intersection(ID_dict[tag_b])
                overlap_dict[tag] = overlap_dict[tag].union(inter)
    total_overlap_ID_set = set()
    for tag in overlap_dict:
        total_overlap_ID_set = total_overlap_ID_set.union(overlap_dict[tag])

    print(f"{len(total_ID_set)} / {total_ID_length} unique images")
    print(f"{len(total_overlap_ID_set)} out of {len(total_ID_set)} is appearing on multiple classes")

def divide_data_by_date(query_dict, num_of_buckets=2, date='date_uploaded', train_mode='finetune'):
    tag_dict_divided_by_date = []
    for b in range(num_of_buckets):
        sub_tag_dict = {}
        for tag in query_dict:
            sub_tag_dict[tag] = None
        tag_dict_divided_by_date.append(sub_tag_dict)

    for tag in query_dict:
        if train_mode == 'clip_feature':
            sorted_tag_list = sort_metadata_by_date(query_dict[tag]['metadata'], date=date, features=query_dict[tag]['features'])
        else:
            sorted_tag_list = sort_metadata_by_date(query_dict[tag]['metadata'], date=date)
        sorted_tag_chunks = divide(sorted_tag_list, num_of_buckets)
        
        for i, sorted_tag_chunk in enumerate(sorted_tag_chunks):
            random.shuffle(sorted_tag_chunk)
            val_set = sorted_tag_chunk[:VAL_SET_SIZE]
            test_set = sorted_tag_chunk[VAL_SET_SIZE:VAL_SET_SIZE+TEST_SET_SIZE]
            train_set = sorted_tag_chunk[VAL_SET_SIZE+TEST_SET_SIZE:]

            tag_dict_divided_by_date[i][tag] = {
                'train_set' : train_set,
                'val_set' : val_set,
                'test_set' : test_set,
                'all' : sorted_tag_chunk
            }
    
    return tag_dict_divided_by_date


def get_loaders_from_tag_dict(tag_dict,
                              train_transform,
                              test_transform,
                              batch_size=64, 
                              num_workers=4):
    class_names = list(tag_dict.keys())
    all_items = []
    train_set = []
    val_set = []
    test_set = []
    for idx, class_name in enumerate(class_names):
        all_items += [(meta.get_path(), idx) for meta in tag_dict[class_name]['all']]
        val_set += [(meta.get_path(), idx) for meta in tag_dict[class_name]['val_set']]
        test_set += [(meta.get_path(), idx) for meta in tag_dict[class_name]['test_set']]
        train_set += [(meta.get_path(), idx) for meta in tag_dict[class_name]['train_set']]

    train_loader = make_loader(train_set, train_transform, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = make_loader(val_set, test_transform, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    test_loader = make_loader(test_set, test_transform, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    all_loader = make_loader(all_items, test_transform, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    return {
        'all' : all_loader,
        'train' : train_loader,
        'val' : val_loader,
        'test' : test_loader
    }

from continual_learn_clip import get_clip_loaders_from_tag_dict

def get_loaders_from_tag_dicts(tag_dicts,
                               batch_size=64, 
                               num_workers=4):
    loaders = {}
    for i, tag_dict in enumerate(tag_dicts):
        loaders[i] = get_clip_loaders_from_tag_dict(
                        tag_dict,
                        batch_size=batch_size, 
                        num_workers=num_workers,
                    )
    return loaders
    
def train_nearest_model(train_loader, val_loader, test_loader, train_mode):
    network = {}

    if train_mode == 'nearest_nn':
        import copy
        network = copy.deepcopy(train_loader)
    elif train_mode == 'nearest_center':
        pbar = tqdm(train_loader)
        for batch, data in enumerate(pbar):
            inputs, labels = data
            # if train_mode == 'nearest_nn':
            #     for i in range(inputs.shape[0]):
            #         input_i = inputs[i]
            #         label_i = int(labels[i])
            #         if label_i in network:
            #             network[label_i].append(input_i.numpy())
            #         else:
            #             network[label_i] = [input_i.numpy()]
            for i in range(inputs.shape[0]):
                input_i = inputs[i]
                label_i = int(labels[i])
                if label_i in network:
                    network[label_i] = input_i.numpy() + network[label_i]
                else:
                    network[label_i] = input_i.numpy()
        
        if train_mode == 'nearest_center':
            for label_i in network:
                network[label_i] = network[label_i] / np.linalg.norm(network[label_i])
    else:
        import pdb; pdb.set_trace()
    
    # loaders = {'train' : train_loader, 'val' : val_loader, 'test' : test_loader}
    # loaders = {'test' : test_loader}
    # for phase in loaders.keys():
    #     acc = test_nearest_model(loaders[phase], network, train_mode)
    #     print(f"{phase} accuracy is {acc}")
    return network

def test_nearest_model(test_loader, network, train_mode):
    total_correct = 0.
    total_count = 0.
    pbar = tqdm(test_loader)
    for batch, data in enumerate(pbar):
        inputs, labels = data
        for i in range(inputs.shape[0]):
            input_i = inputs[i]
            label_i = int(labels[i])
            if train_mode == 'nearest_nn':
                pred_i = None
                max_score = None
                for _, data_nn in enumerate(network):
                    inputs_nn, labels_nn = data_nn
                    for i in range(inputs_nn.shape[0]):
                        input_nn_i = inputs_nn[i]
                        label_nn_i = int(labels_nn[i])
                        score = float(input_i.dot(input_nn_i))
                        if max_score == None or score > max_score:
                            max_score = score
                            pred_i = label_nn_i
            elif train_mode == 'nearest_center':
                pred_i = None
                max_score = None
                for class_i in network:
                    score = float(input_i.numpy().dot(network[class_i]))
                    if max_score == None or score > max_score:
                        max_score = score
                        pred_i = class_i
            else:
                import pdb; pdb.set_trace()
            if pred_i == label_i:
                total_correct += 1
            total_count += 1
    return total_correct/total_count

    
if __name__ == "__main__":
    args = argparser.parse_args()

    save_path = get_save_path(args)
    # lmb_str = f"lmb_{args.lmb}" if args.lmb != 1. else ""
    # save_path = os.path.join(args.clip_dataset_path, args.label_set, f"size_{args.class_size}_doq_{args.use_difference_of_query}{lmb_str}")

    query_dict_path = os.path.join(save_path, "query_dict.pickle")
    query_dict  = load_pickle(query_dict_path)
    if args.train_mode == 'debug':
        debug_overlapping(query_dict)
        exit(0)

    # clip_feature_str = "_clip"
    # tag_dict_path = os.path.join(save_path, f"tag_dict_divided_by_date{clip_feature_str}.pickle")
    tag_dict_path = os.path.join(save_path, f"tag_dict_divided_by_date.pickle")
    if os.path.exists(tag_dict_path):
        print(tag_dict_path+" exists")
        tag_dict_divided_by_date = load_pickle(tag_dict_path)
    else:
        print("First run continual_learning_clip.py!!!!")
        import pdb; pdb.set_trace()
        tag_dict_divided_by_date = divide_data_by_date(query_dict, date='date_uploaded', num_of_buckets=args.num_of_buckets, train_mode=args.train_mode)
        save_obj_as_pickle(tag_dict_path, tag_dict_divided_by_date)
        print(f"saved to {tag_dict_path}")

    loaders_divided_by_date = get_loaders_from_tag_dicts(
                                    tag_dict_divided_by_date,
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                              )


    assert args.num_of_buckets == 2
    output_size=len(list(query_dict.keys()))
    
    first_loader = loaders_divided_by_date[0]
    second_loader = loaders_divided_by_date[1]
    if args.train_bucket == 'past':
        network = train_nearest_model(first_loader['train'], first_loader['val'], first_loader['test'], args.train_mode)
    else:
        network = train_nearest_model(second_loader['train'], second_loader['val'], second_loader['test'], args.train_mode)

    print("On test set of first bucket")
    acc = test_nearest_model(first_loader['test'], network, args.train_mode)
    print(f"Accuracy is {acc}")
    # print("On all images of first bucket")
    # test_nearest_model(first_loader['all'], network, args.train_mode)
    print("On test set of second bucket")
    acc = test_nearest_model(second_loader['test'], network, args.train_mode)
    print(f"Accuracy is {acc}")
    # print("On all images of second bucket")
    # test_nearest_model(second_loader['all'], network, args.train_mode)