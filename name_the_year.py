# python name_the_year.py --img_dir /scratch/zhiqiu/data/fetch_by_tag/vehicle/images_minbyte_2100_totalimgs_10000/date_uploaded_by_year --pretrained
# python name_the_year.py --img_dir /scratch/zhiqiu/data/fetch_by_tag/vehicle/images_minbyte_2100_totalimgs_10000/date_uploaded_by_year

# python name_the_year.py --img_dir /scratch/zhiqiu/data/fetch_by_tag/car/images_minbyte_2100_totalimgs_10000/date_uploaded_by_year
# python name_the_year.py --img_dir /scratch/zhiqiu/data/fetch_by_tag/architecture/images_minbyte_2100_totalimgs_10000/date_uploaded_by_year
# python name_the_year.py --img_dir /scratch/zhiqiu/data/fetch_by_tag/art/images_minbyte_2100_totalimgs_10000/date_uploaded_by_year

# python name_the_year.py --img_dir /scratch/zhiqiu/data/fetch_by_tag/text/images_minbyte_2100_totalimgs_10000/date_uploaded_by_year --pretrained

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

from training_utils import make_optimizer, make_scheduler, make_model, make_loader, train, get_imgnet_transforms

from flickr_parsing import ImageByAutoTag, ImageByRandom, FlickrParser

# For each bucket, make sure at least VAL_SET_SIZE + TEST_SET_SIZE exist
VAL_SET_SIZE = 20
TEST_SET_SIZE = 20

EXCLUDED_YEAR = '0-2004'

argparser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
argparser.add_argument("--img_dir", 
                        default='/scratch/zhiqiu/data/random_subset/images_minbyte_2100_totalimgs_10000/date_uploaded_by_year',
                        help="The yfcc100M dataset store location")
argparser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='model architecture (default: resnet50)')
argparser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use imagenet pre-trained model')
argparser.add_argument('--selfsupervised', default=None, choices=['moco_v2', 'byol', 'deepcluster', 'relativeloc', 'rot'],
                    help='name of self supervised model')

argparser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
argparser.add_argument('--classes', default=1000, type=int, metavar='N',
                    help='number of total classes for the experiment')
argparser.add_argument('--epoch', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
argparser.add_argument('--step', default=20, type=int, metavar='N',
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


if __name__ == '__main__':
    args = argparser.parse_args()

    img_subdirs = glob(args.img_dir + "/*/")
    for i, img_subdir in enumerate(img_subdirs):
        if EXCLUDED_YEAR in img_subdir:
            excluded_index = i
    print(f"Deleting {excluded_index}-th item from img_subdirs. It now becomes:")
    del img_subdirs[excluded_index]
    print(img_subdirs)


    train_set = [] # list of tuples (path, label)
    val_set = [] # list of tuples (path, label)
    test_set = [] # list of tuples (path, label)

    for i, img_subdir in enumerate(img_subdirs):
        bucket_items = []
        img_paths = glob(img_subdir + '*')
        for item in img_paths:
            bucket_items.append((item, i))
        assert len(bucket_items) > VAL_SET_SIZE + TEST_SET_SIZE + 20
        val_set += bucket_items[:VAL_SET_SIZE]
        test_set += bucket_items[VAL_SET_SIZE:VAL_SET_SIZE+TEST_SET_SIZE]
        train_set += bucket_items[VAL_SET_SIZE+TEST_SET_SIZE:]
    
    print(f"Train set {len(train_set)} | Val set {len(val_set)} | Test set {len(test_set)}")

    if False:
        min_edge = None
        min_path = None

        min_w_edge = None
        min_w_path = None

        min_h_edge = None
        min_h_path = None

        smallest_area = None
        smallest_area_path = None

        not_500 = 0
        for path, _ in tqdm(train_set + val_set + test_set):
            img = default_loader(path)
            w, h = img.size
            if min_edge == None or w < min_edge or h < min_edge:
                min_edge = min(w, h)
                min_path = path
            if min_w_edge == None or w < min_w_edge:
                min_w_edge = w
                min_w_path = path
            if min_h_edge == None or h < min_h_edge:
                min_h_edge = h
                min_h_path = path
            if smallest_area == None or w*h < smallest_area:
                smallest_area = w*h
                smallest_w_h = (w, h)
                smallest_area_path = path
            
            if w != 500 and h != 500:
                not_500 += 1
        print(f"Minimum edge is {min_edge}")
        print(f"Minimum w edge is {min_w_edge}")
        print(f"Minimum w image is {min_w_path}")
        print(f"Minimum h edge is {min_h_edge}")
        print(f"Minimum h image is {min_h_path}")

        print(f"Smallest image is {smallest_w_h}")
        print(f"Smallest image is {smallest_area_path}")
        print(f"Not_500 edge is {not_500}")



    train_transform, test_transform = get_imgnet_transforms()

    train_loader = make_loader(train_set, train_transform, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = make_loader(val_set, test_transform, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = make_loader(test_set, test_transform, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    network = make_model(args.arch, args.pretrained, args.selfsupervised)
    train(train_loader, val_loader, test_loader, network, epochs=args.epoch, lr=args.lr, step_size=args.step)

        
    
    
        