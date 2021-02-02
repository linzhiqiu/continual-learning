# A script to parse flickr datasets/autotags
# Download all: python large_scale_feature.py --all_images --img_dir /project_data/ramanan/yfcc100m_all --min_size 10 --chunk_size 10000;
from io import BytesIO
import os
import json
import time
import argparse

import time
running_time = time.time()

from PIL import Image
import requests
from tqdm import tqdm
import pickle
from datetime import datetime
from dateutil import parser
import shutil
import random
import imagesize

import matplotlib.pyplot as plt
import numpy as np
# from sklearn.preprocessing import normalize

from utils import save_obj_as_pickle, load_pickle, normalize
from large_scale_yfcc_download import FlickrAccessor, FlickrFolder, get_flickr_accessor

import sys
sys.path.append("./CLIP")
import clip

import torch
from torch.utils.data import Dataset

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", 
                        default='ViT-B/32', choices=clip.available_models(),
                        help="The CLIP model to use")
argparser.add_argument("--new_folder_path", 
                        # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_25',
                        default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_31',
                        help="It will copy all things to this folder. If None, no copying is done")
argparser.add_argument("--img_dir", 
                        default='/project_data/ramanan/yfcc100m_all',
                        help="The yfcc100M dataset store location")
argparser.add_argument("--data_file", 
                        default='./yfcc100m/yfcc100m_dataset',
                        help="The yfcc100M dataset file ")
argparser.add_argument("--auto_file",
                        default='./yfcc100m/yfcc100m_autotags-v1', 
                        help="The autotag file")
argparser.add_argument("--exif_file",
                        default='./yfcc100m/yfcc100m_exif', 
                        help="The exif file")
argparser.add_argument("--hash_file",
                        default='./yfcc100m/yfcc100m_hash', 
                        help="The hash file")
argparser.add_argument("--hash_pickle",
                        default='./yfcc100m/yfcc100m_hash.pickle', 
                        help="The hash dictionary pickle object")
argparser.add_argument("--lines_file",
                        default='./yfcc100m/yfcc100m_lines', 
                        help="The lines file")
argparser.add_argument("--size_option",
                        default='original', choices=['original'],
                        help="Whether to use the original image size (max edge has 500 px).")
argparser.add_argument("--chunk_size",
                        type=int, default=10000,
                        help="The maximum images to store")
argparser.add_argument("--min_size",
                        type=int, default=10,
                        help="Images with size smaller than min_size will be ignored.")
argparser.add_argument("--all_images",
                        action='store_true',
                        help="Store all images.")
argparser.add_argument("--use_valid_date",
                        type=bool, default=True,
                        help="Images with valid date (upload date < taken date) will be used if set true")

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

def get_clip_loader(flickr_accessor, preprocess, batch_size=1024, num_workers=0, device='cuda'):
    return torch.utils.data.DataLoader(
        CLIPDataset(flickr_accessor, preprocess, device=device), 
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

def get_feature_name(new_folder_path, model_name, normalize=False):
    if normalize:
        normalize_str = "_normalized"
    else:
        normalize_str = ""
    return os.path.join(args.new_folder_path, f"features_{model_name.replace(os.sep, '_')}{normalize_str}.pickle")

if __name__ == "__main__":
    args = argparser.parse_args()
    start = time.time()
    flickr_accessor = get_flickr_accessor(args, new_folder_path=args.new_folder_path)
    end = time.time()
    print(f"{end - start} seconds are used to load all {len(flickr_accessor)} images")
    
    clip_features_location = get_feature_name(args.new_folder_path, args.model_name, normalize=False)
    print(clip_features_location)
    if os.path.exists(clip_features_location):
        clip_features = load_pickle(clip_features_location)
        print(f"Loaded from {clip_features_location}")
    else:
        device = "cuda"
        print(f"Using model {args.model_name}")
        model, preprocess = clip.load(args.model_name, device=device)
        
        # model = torch.nn.DataParallel(model).cuda()

        clip_loader = get_clip_loader(flickr_accessor, preprocess)
        
        clip_features = get_clip_features(clip_loader, model)
        
        save_obj_as_pickle(clip_features_location, clip_features)
        print(f"Saved at {clip_features_location}")
    
    clip_norm_features_location = get_feature_name(args.new_folder_path, args.model_name, normalize=True)
    if not os.path.exists(clip_norm_features_location):
        clip_features_normalized = normalize(clip_features.astype(np.float32))
        save_obj_as_pickle(clip_norm_features_location, clip_features_normalized)
    # image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    
    # with torch.no_grad():
    #     image_features = model.encode_image(image)
    #     text_features = model.encode_text(text)
    #     print(image_features.norm())
    #     print(text_features.size())
        
    #     logits_per_image, logits_per_text = model(image, text)
    #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
    # flickr_dataset = CLIPDataset(Dataset)