# A script to parse flickr datasets/autotags
# Download all: python large_scale_yfcc_download.py --all_images --img_dir /project_data/ramanan/yfcc100m_all --min_size 10 --chunk_size 100000;
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

from flickr_parsing import Metadata, Criteria, fetch_and_save_image
from utils import save_obj_as_pickle, load_pickle
# CKPT_POINT_LENGTH = 10000 # Save every CKPT_POINT_LENGTH metadata

argparser = argparse.ArgumentParser()
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

class AllValidDate(Criteria):
    """Return all valid images
    """
    def __init__(self, args):
        super().__init__(args)
        self.size_option = args.size_option
        self.min_size = args.min_size
        self.use_valid_date = args.use_valid_date

        self.auxilary_info_str = ""
        if self.size_option != 'original':
            self.auxilary_info_str += f"_size_{self.size_option}"
            raise NotImplementedError()
        if self.min_size != 0:
            self.auxilary_info_str += f"_minbyte_{self.min_size}"
        if self.use_valid_date != 0:
            self.auxilary_info_str += f"_valid_uploaded_date"
            
        self.save_folder = os.path.join(args.img_dir, f'images{self.auxilary_info_str}')
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def is_valid(self, metadata : Metadata):
        metadata_obj = metadata.get_metadata()
        if self.use_valid_date:
            if metadata.date_taken() == None or metadata.date_taken() >= metadata.date_uploaded():
                return False
        if metadata.is_img():
            fetch_success = fetch_and_save_image(
                metadata.get_path(),
                metadata_obj.DOWNLOAD_URL,
                MIN_EDGE=0,
                MIN_IMAGE_SIZE=self.min_size
            )
            if fetch_success:
                width, height = imagesize.get(metadata.get_path())
                metadata_obj.WIDTH, metadata_obj.HEIGHT = width, height
                metadata_obj.ASPECT_RATIO = max(width, height) / min(width, height)
                return True
        return False

    def make_metadata(self, data_line, auto_line, line_num, hash_dict, exif_line, save_folder):
        # Overwrite so no exif is saved
        return Metadata(data_line, auto_line, line_num, hash_dict, save_folder, exif_line=None)
    
    def has_enough(self, metadata_list):
        return False

class FlickrFolder():
    def __init__(self, idx, folder_location, num_images=10000, last_index=None):
        self.last_index = last_index
        self.folder = os.path.join(folder_location, str(idx))
        self.num_images = num_images
        self.metadata_location = os.path.join(self.folder, "metadata.pickle")
        self.image_folder = os.path.join(self.folder, "images")
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
    
    def has_enough(self):
        metadata_list = self.load_metadata()
        return len(metadata_list) == self.num_images

    def load_metadata(self):
        return load_pickle(self.metadata_location)
    
    def save_metadata(self, metadata_list):
        print(f"Save at {self.metadata_location}")
        save_obj_as_pickle(self.metadata_location, metadata_list)

class FlickrFolderAccessor():
    def __init__(self, flickr_folder):
        self.flickr_folder = flickr_folder
        self.metadata_list = flickr_folder.load_metadata()

    def __getitem__(self, idx):
        return self.metadata_list[idx]
    
    def __len__(self):
        return len(self.metadata_list)

class FlickrAccessor():
    def __init__(self, args, data_accessor_path=None):
        if args.all_images:
            criteria = AllValidDate(args)
    
        flickr_parser = FlickrParserBuckets(args, criteria)
        folders = flickr_parser.load_images()
        
        self.flickr_folders = [FlickrFolderAccessor(f) for f in folders]

        self.total_length = 0
        self.num_images = len(self.flickr_folders[0])
        for f in self.flickr_folders:
            self.total_length += len(f)

    def __getitem__(self, idx):
        f_idx = idx / self.num_images
        i_idx = idx % self.num_images
        return self.flickr_folders[f_idx][i_idx]
    
    def __len__(self):
        return self.total_length
    
class FlickrParserBuckets():
    def __init__(self, args, criteria : Criteria):
        self.chunk_size = args.chunk_size
        self.data_file = args.data_file
        self.auto_file = args.auto_file
        self.exif_file = args.exif_file
        self.hash_file = args.hash_file
        self.hash_pickle = args.hash_pickle
        self.lines_file = args.lines_file

        self.criteria = criteria
        self.save_folder = criteria.get_save_folder()
        self.main_pickle_location = os.path.join(self.save_folder, "all_folders.pickle")
        
        self.flickr_folders = load_pickle(self.main_pickle_location, default_obj=[])
        self._check(self.flickr_folders)

    def load_images(self):
        return self.flickr_folders

    def _check(self, flickr_folders):
        lind_idx = 0
        for flickr_folder in flickr_folders:
            if not flickr_folder.has_enough():
                import pdb; pdb.set_trace()
            elif flickr_folder.num_images != self.chunk_size:
                import pdb; pdb.set_trace()
            else:
                lind_idx += flickr_folder.num_images
        
        folder_idx = int(lind_idx / self.chunk_size)
        assert folder_idx == len(self.flickr_folders)
    
    def fetch_images(self):
        if len(self.flickr_folders) > 0:
            print("Continue fetching images")
            last_index = self.flickr_folders[-1].last_index
        else:
            print("Start fetching images")
            last_index = 0

        if os.path.exists(self.hash_pickle):
            hash_dict = pickle.load(open(self.hash_pickle, 'rb'))
            print(f"Load Hash dictionary at {self.hash_pickle}")
        else:
            hash_dict = {}
            with open(self.hash_file, 'r') as hash_f:
                for hash_line in tqdm(hash_f):
                    hash_id, hash_value = hash_line.strip().split("\t")
                    hash_dict[hash_id] = hash_value
                    # print(hash_id)
            pickle.dump(hash_dict, open(self.hash_pickle, "wb+"))
            print(f"Saved Hash dictionary at {self.hash_pickle}")

                
        with open(self.data_file, "r") as f, \
             open(self.auto_file, "r") as auto_f, \
             open(self.lines_file, "r") as line_f, \
             open(self.exif_file, 'r') as exif_f:

            zip_object = zip(f, auto_f, line_f, exif_f)
            zip_index = list(range(len(hash_dict.keys())))
            
            metadata_list = []
            flickr_folder = FlickrFolder(len(self.flickr_folders), self.save_folder, num_images=self.chunk_size)
            
            
            for i, (data_line, auto_line, line_num, exif_line) in tqdm(enumerate(zip_object)):
                if i < last_index:
                    continue
                else:
                    meta = self.criteria.make_metadata(data_line, auto_line, line_num, hash_dict, exif_line, flickr_folder.image_folder)
                    if self.criteria.is_valid(meta):
                        metadata_list.append(meta)

                    if len(metadata_list) == self.chunk_size:
                        print(f"Save the metadata list for {i - self.chunk_size + 1} to {i + 1} images at {flickr_folder.metadata_location}")
                        self._save_flickr_folder(flickr_folder, metadata_list, i)
                        metadata_list = []
                        flickr_folder = FlickrFolder(len(self.flickr_folders), self.save_folder, num_images=self.chunk_size)

            print(f"Finished all media objects.")
            self._save_flickr_folder(flickr_folder, metadata_list, i)
    
    def _save_flickr_folder(self, flickr_folder, metadata_list, i):
        flickr_folder.save_metadata(metadata_list)
        flickr_folder.last_index = i + 1
        self.flickr_folders.append(flickr_folder)
        save_obj_as_pickle(self.main_pickle_location, self.flickr_folders)
        print(f"Updated at {self.main_pickle_location}")


if __name__ == "__main__":
    args = argparser.parse_args()
    if args.all_images:
        criteria = AllValidDate(args)
    
    flickr_parser = FlickrParserBuckets(args, criteria)
    flickr_parser.fetch_images()