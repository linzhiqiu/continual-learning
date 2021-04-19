# A script to parse flickr datasets/autotags
# Difference from large_scale_feature.py: It save numpy arrays in chunks in order to avoid memory error (MAX_SIZE = 1000000)
# Feb 18 bucket 11: 
# python large_scale_feature_chunks.py --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 \
    # --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar

# bucket 4 gpu
# python large_scale_feature_chunks.py --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 \
# --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_4/checkpoint_0199.pth.tar

# bucket 4 gpu last bucket
# python large_scale_feature_chunks.py --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 \
# --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_10_gpu_4/checkpoint_0199.pth.tar


import os
import time
import argparse

import math
import time
running_time = time.time()

from PIL import Image
from tqdm import tqdm
import pickle
from dateutil import parser
import random

import numpy as np
from training_utils import get_imgnet_transforms
from utils import save_obj_as_pickle, load_pickle, normalize, divide
from large_scale_yfcc_download import FlickrAccessor, FlickrFolder, get_flickr_accessor
from prepare_yfcc_dataset import get_bucket_folder_paths
import sys
sys.path.append("./CLIP")
import clip

import torch
from torch.utils.data import Dataset
import torchvision.models as models
from torchvision.datasets.folder import default_loader
device = "cuda"
MAX_SIZE = 500000
BATCH_SIZE = 128

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", 
                        default='RN50', choices=clip.available_models(),
                        help="The CLIP model to use")
argparser.add_argument("--folder_path",
                       # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_25',
                       # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_31',
                       # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_16',
                       default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18',
                       help="The folder with all_folders.pickle and features.pickle")
argparser.add_argument('--num_of_bucket', default=11, type=int,
                       help='number of bucket')
argparser.add_argument("--moco_model",
                       default='', 
                       help="The moco model to use")
argparser.add_argument('--arch', metavar='ARCH', default='resnet50',
                       help='model architecture: ' +
                       ' (default: resnet50)')

class MocoDataset(Dataset):
    def __init__(self, flickr_accessor, preprocess, device='cuda'):
        self.flickr_accessor = flickr_accessor
        self.device = device
        self.preprocess = preprocess

    def __len__(self):
        return len(self.flickr_accessor)

    def __getitem__(self, index):
        path = self.flickr_accessor[index].metadata.IMG_PATH
        sample = self.preprocess(default_loader(path)).to(self.device)
        return sample

class CLIPDataset(Dataset):
    def __init__(self, flickr_accessor, preprocess, device='cuda'):
        self.flickr_accessor = flickr_accessor
        self.device = device
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.flickr_accessor)
    
    def __getitem__(self,index):
        path = self.flickr_accessor[index].metadata.IMG_PATH
        sample = self.preprocess(Image.open(path)).to(self.device)
        return sample

def get_clip_loader(flickr_accessor, preprocess, batch_size=BATCH_SIZE, num_workers=0, device='cuda', dataset_class=CLIPDataset):
    return torch.utils.data.DataLoader(
        dataset_class(flickr_accessor, preprocess, device=device), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
    )

def get_clip_features(clip_loader, model):
    clip_features = []
    pbar = tqdm(clip_loader)
    with torch.no_grad():
        for batch, images in enumerate(pbar):
            image_features = model.encode_image(images)
            clip_features.append(image_features.cpu().numpy())
    return np.concatenate(clip_features, axis=0)


def get_moco_features(clip_loader, model):
    moco_features = []
    pbar = tqdm(clip_loader)
    with torch.no_grad():
        for batch, images in enumerate(pbar):
            image_features = model(images)
            moco_features.append(image_features.cpu().numpy())
    return np.concatenate(moco_features, axis=0)

def get_moco_model(moco_model, arch):
    model = models.__dict__[arch]()
    if os.path.isfile(moco_model):
        print("=> loading checkpoint '{}'".format(moco_model))
        # checkpoint = torch.load(moco_model, map_location="cpu")
        checkpoint = torch.load(moco_model)

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        model.fc = torch.nn.Identity()
        model = torch.nn.DataParallel(model).cuda()
    else:
        print("=> no checkpoint found at '{}'".format(moco_model))
    _, test_transform = get_imgnet_transforms() # Note that this is not exactly imagenet transform/moco transform for val set
    return model, test_transform


def parse_moco_model_path(moco_model, arch):
    moco_paths = moco_model.split(os.sep)
    model_configs = moco_paths[-2].split("_")
    bucket_num = model_configs[model_configs.index('bucket')+1]
    # assert int(bucket_num) == args.num_of_bucket
    
    checkpoint_num = moco_paths[-1].split(".")[0].split("_")[-1]
    model_str = moco_paths[-2] + "_checkpoint_" + checkpoint_num
    
    return "moco_features_" + model_str


def get_moco_save_location(folder_path, moco_model, arch):
    moco_name = parse_moco_model_path(moco_model, arch)
    main_moco_location = os.path.join(folder_path, f"{moco_name}.pickle")
    return main_moco_location


def divide_meta_list(bucket_dict_i, sub_folder, MAX_SIZE=MAX_SIZE):
    meta_list = bucket_dict_i['flickr_accessor']
    sub_features_size = math.ceil(len(meta_list) / MAX_SIZE)
    names = [os.path.join(sub_folder, f'features_{i}') for i in range(sub_features_size)]
    chunks = divide(meta_list, sub_features_size)
    assert len(chunks) == len(names)
    return chunks, names
    

def get_moco_sub_feature_paths(bucket_dict_i, folder_path, main_moco_location, moco_model, arch, MAX_SIZE=MAX_SIZE):
    moco_name = parse_moco_model_path(moco_model, arch)
    sub_folder = os.path.join(folder_path, moco_name)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    chunks, names = divide_meta_list(bucket_dict_i, sub_folder, MAX_SIZE=MAX_SIZE)
    path_dict_list = [{'original' : n+"_original.pickle"} for n in names]
    save_obj_as_pickle(main_moco_location, (chunks, path_dict_list))
    return chunks, path_dict_list


def get_main_save_location(folder_path, model_name):
    main_save_location = os.path.join(
        folder_path, f"features_{model_name.replace(os.sep, '_')}.pickle")
    return main_save_location


def get_sub_feature_paths(bucket_dict_i, folder_path, main_save_location, MAX_SIZE=MAX_SIZE, model_str=''):
    sub_folder = os.path.join(folder_path, f"features{model_str}")
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    chunks, names = divide_meta_list(bucket_dict_i, sub_folder, MAX_SIZE=MAX_SIZE)
    
    path_dict_list = [{'original': n+"_original.pickle",
                       "normalized": n+"_normalized.pickle"} for n in names]
    save_obj_as_pickle(main_save_location, (chunks, path_dict_list))
    return chunks, path_dict_list
    
if __name__ == "__main__":
    args = argparser.parse_args()
    start = time.time()
    
    folder_paths = get_bucket_folder_paths(args.folder_path, args.num_of_bucket)
    bucket_dict_path = os.path.join(args.folder_path, f'bucket_{args.num_of_bucket}.pickle')
    if not os.path.exists(bucket_dict_path):
        print(f"{bucket_dict_path} does not exists. Check subfolders")
        bucket_dict = {}
        for i, folder_path in enumerate(folder_paths):
            bucket_dict_i_path = os.path.join(folder_path, f'bucket_{i}.pickle')
            if not os.path.exists(bucket_dict_i_path):
                import pdb; pdb.set_trace()
            bucket_dict[i] = load_pickle(bucket_dict_i_path)
            date_str = f"For bucket {i}: Date range from {bucket_dict[i]['min_date']} to {bucket_dict[i]['max_date']}"
            print(date_str)
        save_obj_as_pickle(bucket_dict_path, bucket_dict)
            # bucket_dict[i] = {
            #     'indices' : sorted_indices_chunk,
            #     'normalized_feature_path' : clip_norm_features_location,
            #     'flickr_accessor' : meta_list,
            #     'folder_path' : folder_paths[i],
            #     'min_date': get_date_uploaded(min_date),
            #     'max_date': get_date_uploaded(max_date),
            #     'date_uploaded_list' : [date_uploaded_list[i] for i in sorted_indices_chunk]
            # }
    else:
        bucket_dict = load_pickle(bucket_dict_path)
    
    length_of_dataset = 0
    for i in bucket_dict:
        length_of_dataset += len(bucket_dict[i]['flickr_accessor'])

    end = time.time()
    print(f"{end - start} seconds are used to load all {length_of_dataset} images")
    
    for i, folder_path in enumerate(folder_paths):
        main_pickle_location = get_main_save_location(folder_path, args.model_name)
        print(main_pickle_location)
        if os.path.exists(main_pickle_location):
            chunks, path_dict_list = load_pickle(main_pickle_location)
            print(f"Loaded from {main_pickle_location}")
        else:
            model_str = '' if args.model_name == 'RN50' else "_"+args.model_name
            chunks, path_dict_list = get_sub_feature_paths(
                bucket_dict[i], folder_path, main_pickle_location, model_str=model_str)
        
        print(f"Using model {args.model_name}")
        model, preprocess = clip.load(args.model_name, device=device)
        for chunk, path_dict in zip(chunks, path_dict_list):
            if os.path.exists(path_dict['normalized']) and os.path.exists(path_dict['original']):
                print(f"Already exists: {path_dict['normalized']}")
                continue
            clip_loader = get_clip_loader(chunk, preprocess, dataset_class=CLIPDataset)
            clip_features = get_clip_features(clip_loader, model)
            
            save_obj_as_pickle(path_dict['original'], clip_features)
            print(f"Saved at {path_dict['original']}")
            
            clip_features_normalized = normalize(clip_features.astype(np.float32))
            save_obj_as_pickle(path_dict['normalized'], clip_features_normalized)
            
        main_moco_location = get_moco_save_location(
            folder_path, args.moco_model, args.arch)
        print(main_moco_location)
        if os.path.exists(main_moco_location):
            chunks, path_dict_list = load_pickle(main_moco_location)
            print(f"Loaded from {main_moco_location}")
        else:
            chunks, path_dict_list = get_moco_sub_feature_paths(
                bucket_dict[i], folder_path, main_moco_location, args.moco_model, args.arch)

        print(f"Using model {args.arch} for moco")
        
        moco_model, moco_preprocess = get_moco_model(args.moco_model, args.arch)
        for chunk, path_dict in zip(chunks, path_dict_list):
            if os.path.exists(path_dict['original']):
                print(f"Already exists: {path_dict['original']}")
                continue
            clip_loader = get_clip_loader(chunk, moco_preprocess, dataset_class=MocoDataset)
            moco_features = get_moco_features(clip_loader, moco_model)

            save_obj_as_pickle(path_dict['original'], moco_features)
            print(f"Saved at {path_dict['original']}")
