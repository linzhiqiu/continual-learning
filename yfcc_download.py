# A script to parse flickr datasets/autotags
# Example: python yfcc_download.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2;
# Test: python yfcc_download.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_nov_27 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2;

from io import BytesIO
import os
import re
from urllib.parse import unquote_plus
import time
import logging
import argparse
import time
from PIL import Image
import subprocess
import requests
from tqdm import tqdm
from datetime import datetime
from dateutil import parser
import shutil
import random
import imagesize
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import save_as_json, load_json
import threading
import sys
import time
from temp import FlickrFolder, Metadata, MetadataObject

argparser = argparse.ArgumentParser()
argparser.add_argument("--img_dir", 
                        default='/data3/zhiqiul/yfcc100m_all_new/',
                        help="Yfcc100M dataset will be downloaded at this location")
argparser.add_argument("--metadata_dir",
                       default='./',
                       help='The directory with all downloaded metadata files.')
argparser.add_argument("--size_option",
                        default='original', choices=['original'],
                        help="Whether to use the original image size (max edge has 500 px).")
argparser.add_argument("--chunk_size",
                        type=int, default=50000,
                        help="The number of images to store in each subfolder")
argparser.add_argument("--max_aspect_ratio",
                       type=float, default=0,
                       help="If not 0: Images with aspect ratio larger than max_aspect_ratio will be ignored.")
argparser.add_argument("--min_edge",
                       type=int, default=0,
                       help="If not 0: Images with edge shorter than min_edge will be ignored.")
argparser.add_argument("--min_size",
                        type=int, default=10,
                        help="If not 0: Images with size (in bytes) smaller than min_size will be ignored.")
argparser.add_argument("--use_valid_date",
                        type=bool, default=True,
                        help="Images with valid date (upload date < taken date) will be used if set true")   
argparser.add_argument("--max_workers",
                        type=int, default=128,
                        help="The number of parallel workers for image download.")

# The index of metadata field for dataset file
IDX_LIST = [
    "ID", # Unique photo/video identifier
    "USER_ID", # User ID
    "NICKNAME", # User nickname
    "DATE_TAKEN", # Date taken
    "DATE_UPLOADED", # Date uploaded
    "DEVICE", # Capture device
    "TITLE", # Photo title
    "DESCRIPTION", # Image Description
    "USER_TAGS", # User tags (comma-separated)
    "MACHINE_TAGS", # Machine (SVM) tags (comma-separated)
    "LON", # Longitude
    "LAT", # Latitude
    "GEO_ACCURACY", # Accuracy of the longitude and latitude coordinates (1=world level accuracy, ..., 16=street level accuracy)
    "PAGE_URL", # Photo/video page URL
    "DOWNLOAD_URL", # Photo/video download URL
    "LICENSE_NAME", # License name
    "LICENSE_URL", # License URL
    "SERVER_ID", # Photo/video server identifier
    "FARM_ID", # Photo/video farm identifier
    "SECRET", # Photo/video secret
    "SECRET_ORIGINAL", # Photo/video secret original
    "EXT", # Extension of the original photo
    "IMG_OR_VIDEO", # (type = int) Photos/video marker (0 = photo, 1 = video)
]

IDX_TO_NAME = {i : IDX_LIST[i] for i in range(len(IDX_LIST))}
# Other metadata that will be included in the dictionary:
# AUTO_TAG_SCORES (dict): A dictionary of autotag_scores (key = category, value = confidence score in float)
# LINE_NUM (int): Line Number
# HASH_VALUE (str):  Hash value
# EXIF (str) : EXIF
# IMG_PATH (str): Image path
# IMG_DIR (str): Image folder (default: None)

def date_uploaded(metadata):
    return datetime.utcfromtimestamp(int(metadata['DATE_UPLOADED']))

def date_taken(metadata):
    try:
        return parser.isoparse(metadata['DATE_TAKEN'])
    except:
        return None

def _parse_line(line):
    entries = line.strip().split("\t")
    if len(entries) == 1:
        return entries[0], None
    else:
        return entries[0], entries[1]

def _parse_autotags(line):
    entries = line.strip().split("\t")
    if len(entries) == 1:
        tag_scores = {}
    else:
        tags = entries[1].split(",")
        if len(tags) > 0:
            tag_scores = {t.split(":")[0] : float(t.split(":")[1]) for t in tags}
        else:
            tag_scores = {}
    return entries[0], tag_scores

def _parse_metadata(data, autotag, line_num, hash_dict, save_folder, exif_line=None):
    """Parse the metadata and return MetadataObject
    """
    metadata = {}

    entries = data.strip().split("\t")
    for i, entry in enumerate(entries):
        metadata[IDX_TO_NAME[i]] = entry
        # self.__setattr__(IDX_TO_NAME[i], entry)
    metadata['IMG_OR_VIDEO'] = int(metadata['IMG_OR_VIDEO'])

    # get autotag scores in a dict
    tag_ID, autotag_scores = _parse_autotags(autotag)
    assert metadata['ID'] == tag_ID, "AUTOTAG ID != Photo ID"
    
    line_number, line_ID = _parse_line(line_num)
    assert metadata['ID'] == line_ID, "LINE ID != Photo ID"
    
    if exif_line != None:
        exif_ID, exif_number = _parse_line(exif_line)
        assert metadata['ID'] == exif_ID, "EXIF ID != Photo ID"
    else:
        exif_number = None
    
    hash_value = hash_dict[metadata['ID']]
    img_dir = os.path.abspath(save_folder)
    img_path = f"{metadata['ID']}.{metadata['EXT']}"

    metadata['AUTO_TAG_SCORES'] = autotag_scores
    metadata['LINE_NUM'] = line_number
    metadata['EXIF'] = exif_number
    metadata['HASH_VALUE'] = hash_value
    metadata['IMG_DIR'] = img_dir
    metadata['IMG_PATH'] = img_path

    return metadata

def fetch_and_save_image(img_path, url, MIN_EDGE=0, MAX_ASPECT_RATIO=None, MAX_NUM_OF_TRAILS=3, MIN_IMAGE_SIZE=2100):
    """Return true if image is valid and successfully downloaded
    """
    trials = 0
    max_trails = MAX_NUM_OF_TRAILS
    sleep_time = 1
    while True:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            if img.size[0] < MIN_EDGE or img.size[1] < MIN_EDGE:
                return False
            max_edge = max(img.size[0], img.size[1])
            min_edge = min(img.size[0], img.size[1])
            ratio = max_edge / min_edge
            if MAX_ASPECT_RATIO and ratio > MAX_ASPECT_RATIO:
                return False
            img.save(img_path)
            if os.path.getsize(img_path) < MIN_IMAGE_SIZE:
                return False
            return True
        except:
            if trials < max_trails:
                time.sleep(sleep_time)
                trials += 1
                continue
            else:
                return False
          
def get_flickr_image_folder(folder_location, idx):
    folder = os.path.join(folder_location, str(idx))
    return os.path.join(folder, "images")

def get_flickr_metadata_json_path(folder_location, idx):
    folder = os.path.join(folder_location, str(idx))
    return os.path.join(folder, "metadata.json")

def get_flickr_folder_dict(idx, folder_location, num_images=10000):
    flickr_folder_dict = {
        'folder' : os.path.join(folder_location, str(idx)),
        'num_images' : num_images,
        'image_folder' : get_flickr_image_folder(folder_location, idx),
        'metadata_location' : get_flickr_metadata_json_path(folder_location, idx),
    }
    if not os.path.exists(flickr_folder_dict['image_folder']):
        print(f"make dir at {flickr_folder_dict['image_folder']}")
        os.makedirs(flickr_folder_dict['image_folder'])
    
    return flickr_folder_dict 

def get_main_folder_json_location(save_folder):
    return os.path.join(save_folder, "all_folders.json")

def _get_info_str(size_option,
                  min_size,
                  use_valid_date,
                  min_edge,
                  max_aspect_ratio):
    info_str = ""
    if size_option != 'original':
        info_str += f"_size_{size_option}"
        raise NotImplementedError()
    if min_size != 0:
        info_str += f"_minbyte_{min_size}"
    if use_valid_date != 0:
        info_str += f"_valid_uploaded_date"
    if min_edge != 0:
        info_str += f"_minedge_{min_edge}"
    if max_aspect_ratio != 0:
        info_str += f"_maxratio_{max_aspect_ratio}"
    return info_str

def get_save_folder(img_dir,
                    size_option,
                    min_size,
                    use_valid_date,
                    min_edge,
                    max_aspect_ratio):
    info_str = _get_info_str(
        size_option,
        min_size,
        use_valid_date,
        min_edge,
        max_aspect_ratio
    )
        
    save_folder = os.path.join(img_dir, f'images{info_str}')
    return save_folder
    
class FlickrDownloader():
    """
    Parse Flickr dataset files, in order to download images
    Only download images that pass the is_valid() test
    """
    def __init__(self, args):
        self.chunk_size = args.chunk_size

        self.size_option = args.size_option
        self.min_size = args.min_size
        self.min_edge = args.min_edge
        self.max_aspect_ratio = args.max_aspect_ratio
        self.use_valid_date = args.use_valid_date

        self.save_folder = get_save_folder(
            args.img_dir,
            self.size_option,
            self.min_size,
            self.use_valid_date,
            self.min_edge,
            self.max_aspect_ratio
        )
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.main_folder_json_location = get_main_folder_json_location(self.save_folder)
        
        self.flickr_folders = load_json(self.main_folder_json_location, default_obj={})

        self.metadata_dir = args.metadata_dir
        self.data_file = os.path.join(args.metadata_dir, 'yfcc100m', 'yfcc100m_dataset')
        self.auto_file = os.path.join(args.metadata_dir, 'yfcc100m', 'yfcc100m_autotags-v1')
        self.exif_file = os.path.join(args.metadata_dir, 'yfcc100m', 'yfcc100m_exif')
        self.hash_file = os.path.join(args.metadata_dir, 'yfcc100m', 'yfcc100m_hash')
        self.hash_json_location = os.path.join(args.metadata_dir, 'yfcc100m', 'yfcc100m_hash.json')
        self.lines_file = os.path.join(args.metadata_dir, 'yfcc100m', 'yfcc100m_lines')

        self.max_workers = args.max_workers
    
    def is_valid(self, metadata):
        if self.use_valid_date:
            if date_taken(metadata) == None or date_taken(metadata) >= date_uploaded(metadata):
                return False
        if metadata['IMG_OR_VIDEO'] == 0:
            return True
        return False

    def fetch_one(self, metadata):
        fetch_success = fetch_and_save_image(
            os.path.join(metadata['IMG_DIR'], metadata['IMG_PATH']),
            metadata['DOWNLOAD_URL'],
            MIN_EDGE=self.min_edge,
            MIN_IMAGE_SIZE=self.min_size,
            MAX_ASPECT_RATIO=self.max_aspect_ratio,
        )
        return fetch_success

    def fetch_images(self):
        if len(self.flickr_folders) > 0:
            print("Continue fetching images")
            last_index = max(self.flickr_folders.keys()) * self.chunk_size
        else:
            print("Start fetching images")
            last_index = 0

        if os.path.exists(self.hash_json_location):
            hash_dict = load_json(self.hash_json_location)
            print(f"Load Hash dictionary at {self.hash_json_location}")
        else:
            hash_dict = {}
            with open(self.hash_file, 'r') as hash_f:
                for hash_line in tqdm(hash_f):
                    hash_id, hash_value = hash_line.strip().split("\t")
                    hash_dict[hash_id] = hash_value
                    # print(hash_id)
            save_as_json(self.hash_json_location, hash_dict)
            print(f"Saved Hash dictionary at {self.hash_json_location}")

                
        with open(self.data_file, "r") as f, \
             open(self.auto_file, "r") as auto_f, \
             open(self.lines_file, "r") as line_f, \
             open(self.exif_file, 'r') as exif_f:

            zip_object = zip(f, auto_f, line_f, exif_f)
            zip_index = list(range(len(hash_dict.keys())))
            metadata_lists = {} # list of metadata
            metadata_counts = {} # list of counts of downloaded (or attempted) metadata

            lock = threading.Lock()
            
            def download_image(i, data_line, auto_line, line_num, exif_line, lock, metadata_lists, metadata_counts):
                try:
                    folder_idx = int(i / self.chunk_size)
                    
                    with lock:
                        if not folder_idx in self.flickr_folders:
                            self.flickr_folders[folder_idx] = get_flickr_folder_dict(folder_idx, self.save_folder, num_images=self.chunk_size)
                    
                    meta = _parse_metadata(data_line, auto_line, line_num, hash_dict, get_flickr_image_folder(self.save_folder, folder_idx), exif_line=None)
                    if self.is_valid(meta):
                        fetch_success = self.fetch_one(meta)
                        if fetch_success:
                            with lock:
                                if not folder_idx in metadata_lists:
                                    metadata_lists[folder_idx] = []
                                metadata_lists[folder_idx].append(meta)
                    
                    with lock:
                        if not folder_idx in metadata_counts:
                            metadata_counts[folder_idx] = 0
                        metadata_counts[folder_idx] += 1
                            
                        if metadata_counts[folder_idx] == self.chunk_size:
                            assert folder_idx in self.flickr_folders
                            print(f"Save the metadata list (successfully download ({len(metadata_lists[folder_idx])})) for {folder_idx * self.chunk_size} to {(1+folder_idx) * self.chunk_size} images at {self.flickr_folders[folder_idx]['image_folder']}")
                            save_as_json(self.flickr_folders[folder_idx]['metadata_location'], metadata_lists[folder_idx])
                            save_as_json(self.main_folder_json_location, self.flickr_folders)
                            print(f"Updated at {self.main_folder_json_location}")
                            metadata_lists[folder_idx] = None
                            del metadata_lists[folder_idx]
                except Exception as e:
                    print(e)
                return

            from iteration_utilities import grouper
            chunk_idx = 0
            images_per_chunk = 1000000
            for chunk in grouper(enumerate(zip_object), images_per_chunk):
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    results = []
                    for i, (data_line, auto_line, line_num, exif_line) in chunk:
                        if i < last_index:
                            continue
                        else:
                            results += [executor.submit(download_image, i, data_line, auto_line, line_num, exif_line, lock, metadata_lists, metadata_counts)]
                        # executor.submit(download_image, i, data_line, auto_line, line_num, exif_line, lock, metadata_lists, metadata_counts).result()
                    for cur_result in as_completed(results):
                    # for cur_result in tqdm(as_completed(results), total=len(results)):
                        cur_result.result()
                    print(f"Finish the {chunk_idx+1} chunk = {(chunk_idx+1)*images_per_chunk} images.")
                    chunk_idx += 1

            print(f"Finished all media objects.")

def _metadata_of_single_folder(flickr_folder):
    if os.path.exists(flickr_folder['metadata_location']):
        return load_json(flickr_folder['metadata_location'])
    return None

def _metadata_of_all_folders(flickr_folder_dicts):
    all_metadata_lists = [_metadata_of_single_folder(flickr_folder_dicts[f_idx])
                          for f_idx in sorted(flickr_folder_dicts.keys())]
    all_metadata = []
    for lst in all_metadata_lists:
        if lst:
            all_metadata += lst
    return all_metadata

def get_all_metadata(save_folder):
    main_folder_json_location = get_main_folder_json_location(save_folder)
    flickr_folder_dicts = load_json(main_folder_json_location, default_obj={})
    return _metadata_of_all_folders(flickr_folder_dicts)

if __name__ == "__main__":
    args = argparser.parse_args()
    flickr_downloader = FlickrDownloader(args)
    flickr_downloader.fetch_images()
