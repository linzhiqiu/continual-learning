# python name_the_year.py --img_dir /scratch/zhiqiu/data/fetch_by_tag/vehicle/images_minbyte_2100_totalimgs_10000/date_uploaded_by_year --pretrained
# python name_the_year.py --img_dir /scratch/zhiqiu/data/fetch_by_tag/vehicle/images_minbyte_2100_totalimgs_10000/date_uploaded_by_year

# python name_the_year.py --img_dir /scratch/zhiqiu/data/fetch_by_tag/car/images_minbyte_2100_totalimgs_10000/date_uploaded_by_year
# python name_the_year.py --img_dir /scratch/zhiqiu/data/fetch_by_tag/architecture/images_minbyte_2100_totalimgs_10000/date_uploaded_by_year
# python name_the_year.py --img_dir /scratch/zhiqiu/data/fetch_by_tag/art/images_minbyte_2100_totalimgs_10000/date_uploaded_by_year

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

from models import self_supervised

from glob import glob
from tqdm import tqdm

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


def make_optimizer(network, lr):
    optimizer = torch.optim.SGD(network.parameters(), 
                                lr=lr,
                                weight_decay=1e-5,
                                momentum=0.9)
    return optimizer

def make_scheduler(optimizer, step_size=50):
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=step_size,
                    gamma=0.1
                )
    return scheduler

def make_model(pretrained, selfsupervised):
    if pretrained or selfsupervised:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        if args.selfsupervised:
            if args.selfsupervised == "moco_v2":
                model = self_supervised.moco_v2(model)
            elif args.selfsupervised == "byol":
                model = self_supervised.byol(model)
            elif args.selfsupervised == "rot":
                model = self_supervised.rot(model)
            elif args.selfsupervised == "deepcluster":
                model = self_supervised.deepcluster(model)
            elif args.selfsupervised == "relativeloc":
                model = self_supervised.relativeloc(model)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        print("No model checkpoint is supplied")
    return model


def train(train_loader, val_loader, test_loader, pretrained, selfsupervised, epochs=150, lr=0.1, step_size=60):
    network = make_model(pretrained, selfsupervised).cuda()
    optimizer = make_optimizer(network, lr)
    scheduler = make_scheduler(optimizer, step_size=step_size)
            
    criterion = torch.nn.NLLLoss(reduction='mean')

    avg_loss_per_epoch = []
    avg_acc_per_epoch = []
    avg_val_loss_per_epoch = []
    avg_val_acc_per_epoch = []

    avg_test_acc_per_epoch = []

    best_val_acc = 0
    best_val_epoch = None

    loaders = {'train' : train_loader, 'val' : val_loader, 'test' : test_loader}
    for epoch in range(0, epochs):
        for phase in loaders.keys():
            if phase == 'train':
                network.train()
            else:
                network.eval()

            running_loss = 0.0
            running_corrects = 0.
            count = 0

            pbar = tqdm(loaders[phase])

            for batch, data in enumerate(pbar):
                inputs, labels = data
                count += inputs.size(0)
                    
                inputs = inputs.cuda()
                labels = labels.cuda()
                # import pdb; pdb.set_trace()

                if phase == 'train': optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # import pdb; pdb.set_trace()
                    outputs = network(inputs)
                    _, preds = torch.max(outputs, 1)

                    log_probability = torch.nn.functional.log_softmax(outputs, dim=1)
                    # import pdb; pdb.set_trace()
                    loss = criterion(log_probability, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_postfix(loss=float(running_loss)/count, 
                                    acc=float(running_corrects)/count,
                                    epoch=epoch,
                                    phase=phase)
                
            avg_loss = float(running_loss)/count
            avg_acc = float(running_corrects)/count
            if phase == 'train': 
                avg_loss_per_epoch.append(avg_loss)
                avg_acc_per_epoch.append(avg_acc)
                scheduler.step()
            elif phase == 'val':
                avg_val_loss_per_epoch.append(avg_loss)
                avg_val_acc_per_epoch.append(avg_acc)
                if avg_acc > best_val_acc:
                    print(f"Best val accuracy at epoch {epoch} being {avg_acc}")
                    best_val_epoch = epoch
                    best_val_acc = avg_acc
            else:
                avg_test_acc_per_epoch.append(avg_acc)
            print(f"Average {phase} Loss {avg_loss}, Accuracy {avg_acc}")
        print()
    print(f"Best Test Accuracy (for best val model): {avg_test_acc_per_epoch[best_val_epoch]}")
    print(f"Best Test Accuracy overall: {max(avg_test_acc_per_epoch)}")



from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
class SimpleDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        path, label = self.samples[index]
        sample = default_loader(path)
        sample = self.transform(sample)
        return sample, label

def make_loader(dataset, transform, shuffle=False, batch_size=256, num_workers=0):
    return torch.utils.data.DataLoader(
        SimpleDataset(dataset, transform), 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
    )

if __name__ == '__main__':
    args = argparser.parse_args()
    # if args.fetch_by_tag:
    #     criteria = ImageByAutoTag(args)
    # elif args.random_images:
    #     criteria = ImageByRandom(args)

    # flickr_parser = FlickrParser(args, criteria)

    # # flickr_parser.group_by_month_date_taken()
    # if args.date == 'date_taken':
    #     if args.mode == 'month':
    #         raise NotImplementedError()
    #     elif args.mode == 'year':
    #         sorted_buckets_list, buckets_dict = flickr_parser.group_by_year_date_taken()
    # elif args.date == 'date_uploaded':
    #     if args.mode == 'month':
    #         raise NotImplementedError()
    #     elif args.mode == 'year':
    #         sorted_buckets_list, buckets_dict = flickr_parser.group_by_year_date_uploaded()

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


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_loader = make_loader(train_set, train_transform, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = make_loader(val_set, test_transform, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = make_loader(test_set, test_transform, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    train(train_loader, val_loader, test_loader, args.pretrained, args.selfsupervised, epochs=args.epoch, lr=args.lr, step_size=args.step)

        
    
    
        