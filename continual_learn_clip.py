# python continual_learn.py --threshold_by_minimum --tag_group vehicle --num_of_buckets 2
# python continual_learn.py --threshold_by_fixed_length --tag_group vehicle --num_of_buckets 2
# python continual_learn.py --threshold_by_fixed_length --tag_group vehicle --num_of_buckets 2 --test
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

from utils import sort_metadata_by_date, divide
from training_utils import make_optimizer, make_scheduler, make_model, make_loader, train, test, get_imgnet_transforms
from prepare_dataset import TAG_GROUPS_DICT, get_tag_dict_for_training, argparser

TAG_GROUPS_DICT_FOR_TRAINING = TAG_GROUPS_DICT.copy()
TAG_GROUPS_DICT_FOR_TRAINING['vehicle'] = ['bike','car','motorcycle','airplane', 'bus', 'helicopter', 'train']

# For each bucket, make sure at least VAL_SET_SIZE + TEST_SET_SIZE exist
VAL_SET_SIZE = 50
TEST_SET_SIZE = 50
TRAIN_SET_SIZE = 800

# EXCLUDED_YEAR = '0-2004'
argparser.add_argument("--tag_group", 
                        default='vehicle',
                        choices=TAG_GROUPS_DICT_FOR_TRAINING.keys(),
                        help="The tag group for training")
argparser.add_argument('--num_of_buckets', default=2, type=int,
                       metavar='N',
                       help='Divide by num_of_buckets (default: 2)')
argparser.add_argument('--train_bucket', default='past', choices=['past', 'future'],
                       help='Training bucket')
argparser.add_argument('--train_mode', default='finetune', choices=['finetune', 'freeze'],
                       help='Train mode')
argparser.add_argument('--threshold_by_minimum', dest='threshold_by_minimum', action='store_true',
                       help='Each tag will have same number of images (minmal over all classes)')
argparser.add_argument('--threshold_by_fixed_length', dest='threshold_by_fixed_length', action='store_true',
                       help='Each tag will have same number of images (TRAIN_SET_SIZE)')
argparser.add_argument("--img_dir", 
                        default='/scratch/zhiqiu/small_datasets',
                        help="The image transfer location")
argparser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                       help='model architecture (default: resnet50)')
argparser.add_argument('--pretrained', dest='pretrained', action='store_true',
                       help='use imagenet pre-trained model')
argparser.add_argument('--selfsupervised', default=None, choices=['moco_v2', 'byol', 'deepcluster', 'relativeloc', 'rot'],
                       help='name of self supervised model')
argparser.add_argument('--epoch', default=450, type=int, metavar='N',
                       help='number of total epochs to run')
argparser.add_argument('--step', default=200, type=int, metavar='N',
                       help='step size for lr decay')
argparser.add_argument('--batch_size', default=64, type=int,
                       metavar='N',
                       help='mini-batch size (default: 64)')
argparser.add_argument('--num_workers', default=4, type=int,
                       help='num_workers (default: 4)')
argparser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                       metavar='LR', help='initial learning rate', dest='lr')
# argparser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# argparser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
#                     metavar='W', help='weight decay (default: 1e-5)',
#                     dest='weight_decay')

def threshold_by_fixed_length(tag_dict, train_set_size=TRAIN_SET_SIZE):
    for tag in tag_dict:
        tag_dict[tag] = tag_dict[tag][:train_set_size]
        print(f"Tag {tag} remains {len(tag_dict[tag])} number of images")
    return tag_dict

def threshold_by_minimum(tag_dict):
    minimal = None
    for tag in tag_dict:
        length = len(tag_dict[tag])
        if minimal == None or length < minimal:
            minimal = length
    
    for tag in tag_dict:
        tag_dict[tag] = tag_dict[tag][:minimal]
    
    print(f"Each tag remains {minimal} number of images")
    return tag_dict

def divide_data_by_date(tag_dict, num_of_buckets=2, date='date_uploaded'):
    tag_dict_divided_by_date = []

    for b in range(num_of_buckets):
        sub_tag_dict = {}
        for tag in tag_dict:
            sub_tag_dict[tag] = None
        tag_dict_divided_by_date.append(sub_tag_dict)

    for tag in tag_dict:
        sorted_tag_list = sort_metadata_by_date(tag_dict[tag], date=date)
        sorted_tag_chunks = divide(sorted_tag_list, num_of_buckets)
        
        for i, sorted_tag_chunk in enumerate(sorted_tag_chunks):
            tag_dict_divided_by_date[i][tag] = sorted_tag_chunk
    
    return tag_dict_divided_by_date


def get_loaders_from_tag_dict(tag_dict,
                              train_transform,
                              test_transform,
                              val_size_per_tag=VAL_SET_SIZE,
                              test_size_per_tag=TEST_SET_SIZE,
                              batch_size=64, 
                              num_workers=4):
    class_names = list(tag_dict.keys())
    all_items = []
    train_set = []
    val_set = []
    test_set = []
    for idx, class_name in enumerate(class_names):
        class_item_list = [(meta.get_path(), idx) for meta in tag_dict[class_name]]
        all_items += class_item_list
        random.shuffle(class_item_list)
        assert len(class_item_list) > VAL_SET_SIZE + TEST_SET_SIZE + 20
        val_set += class_item_list[:VAL_SET_SIZE]
        test_set += class_item_list[VAL_SET_SIZE:VAL_SET_SIZE+TEST_SET_SIZE]
        train_set += class_item_list[VAL_SET_SIZE+TEST_SET_SIZE:]

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

def get_loaders_from_tag_dicts(tag_dicts,
                               train_transform,
                               test_transform,
                               val_size_per_tag=VAL_SET_SIZE,
                               test_size_per_tag=TEST_SET_SIZE,
                               batch_size=64, 
                               num_workers=4):
    loaders = {}
    for i, tag_dict in enumerate(tag_dicts):
        loaders[i] = get_loaders_from_tag_dict(
                         tag_dict,
                         train_transform,
                         test_transform,
                         val_size_per_tag=val_size_per_tag,
                         test_size_per_tag=test_size_per_tag,
                         batch_size=batch_size, 
                         num_workers=num_workers
                     )
                     
    return loaders

if __name__ == '__main__':
    args = argparser.parse_args()
    img_transfer_dir = os.path.join(args.img_dir, args.tag_group, "images")
    if not os.path.exists(img_transfer_dir):
        os.makedirs(img_transfer_dir)
    print(f"Transfer images to {img_transfer_dir}")
    tag_dict = get_tag_dict_for_training(
                   args, args.tag_group,
                   TAG_GROUPS_DICT_FOR_TRAINING[args.tag_group],
                   img_transfer_dir=img_transfer_dir,
                   do_transfer=True
               )
            
    if args.threshold_by_minimum:          
        tag_dict = threshold_by_minimum(tag_dict)
    elif args.threshold_by_fixed_length:
        tag_dict = threshold_by_fixed_length(tag_dict)
    tag_dict_divided_by_date = divide_data_by_date(tag_dict, date='date_uploaded', num_of_buckets=args.num_of_buckets)

    train_transform, test_transform = get_imgnet_transforms()
    loaders_divided_by_date = get_loaders_from_tag_dicts(
                                  tag_dict_divided_by_date,
                                  train_transform,
                                  test_transform,
                                  val_size_per_tag=VAL_SET_SIZE,
                                  test_size_per_tag=TEST_SET_SIZE,
                                  batch_size=args.batch_size, 
                                  num_workers=args.num_workers
                              )


    assert args.num_of_buckets == 2
    network = make_model(args.arch, args.pretrained, args.selfsupervised, output_size=len(list(tag_dict.keys())), train_mode=args.train_mode)
    first_loader = loaders_divided_by_date[0]
    second_loader = loaders_divided_by_date[1]
    if args.train_bucket == 'past':
        network = train(first_loader['train'], first_loader['val'], first_loader['test'], network, epochs=args.epoch, lr=args.lr, step_size=args.step)
    else:
        network = train(second_loader['train'], second_loader['val'], second_loader['test'], network, epochs=args.epoch, lr=args.lr, step_size=args.step)
    
    save_model_dir = "./saved_models"
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    save_loc = f'{save_model_dir}{args.train_mode}_train_on_{args.train_bucket}.pickle'
    print("On test set of first bucket")
    test(first_loader['test'], network, save_loc=save_loc)
    print("On all images of first bucket")
    test(first_loader['all'], network)
    print("On test set of second bucket")
    test(second_loader['test'], network)
    print("On all images of second bucket")
    test(second_loader['all'], network)

    exit(0)
        
    
    
        