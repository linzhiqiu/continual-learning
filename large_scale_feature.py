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

from utils import save_obj_as_pickle, load_pickle
from large_scale_yfcc_download import FlickrAccessor, FlickrFolder

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_accessor_path", 
                        default='/project_data/ramanan/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_25.pickle',
                        help="The yfcc100M dataset store location")
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

def get_flickr_accessor(args, data_accessor_path=None):
    if data_accessor_path:
        print(f"Using saved pickle file at {data_accessor_path}")
        flickr_accessor = load_pickle(data_accessor_path)
        if flickr_accessor:
            return flickr_accessor
        else:
            flickr_accessor = FlickrAccessor(args)
            save_obj_as_pickle(data_accessor_path, flickr_accessor)
    else:
        return FlickrAccessor(args)
   
if __name__ == "__main__":
    args = argparser.parse_args()
    start = time.time()
    flickr_accessor = get_flickr_accessor(args, data_accessor_path=args.data_accessor_path)
    end = time.time()
    print(f"{end - start} seconds are used to load all {len(flickr_accessor)} images")
    import pdb; pdb.set_trace()