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
argparser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
argparser.add_argument('--step', default=60, type=int, metavar='N',
                    help='step size for lr decay')
argparser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
argparser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
argparser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
argparser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
argparser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')

best_acc1 = 0

def main():
    args = argparser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    if args.pretrained or args.selfsupervised:
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

    if model.fc.weight.shape[0] != args.classes:
        print("changing the size of last layer")
        model.fc = torch.nn.Linear(model.fc.weight.shape[1], args.classes)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # optionally resume from a checkpoint for model and optimizer
    start_epoch = args.start_epoch
    if args.resume:
        start_epoch, model, optimizer = load_from_checkpoint(args.resume, model, optimizer)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        get_train_transform()
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, get_test_transform()),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    else:
        # Make sure a ckpt folder exist
        assert args.ckpt_dir != None
        if args.pretrained:
            scratch_folder = get_imgpretrained_folder_name(args)
        elif args.selfsupervised:
            scratch_folder = get_selfsupervised_folder_name(args)
        else:
            scratch_folder = get_scratch_folder_name(args)
        print(f"Saving to {scratch_folder}")
        if not os.path.exists(scratch_folder):
            os.makedirs(scratch_folder)

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args.lr, args.step)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, folder=scratch_folder)

    # validate(val_loader, model, criterion, args)

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

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

def make_loader(dataset, batch_size=256, num_workers=0):
    pass

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

    train_loader = make_loader(train_set)
    val_loader = make_loader(val_set)
    test_loader = make_loader(test_set)

        
    
    
        