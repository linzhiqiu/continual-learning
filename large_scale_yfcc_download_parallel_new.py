# A script to parse flickr datasets/autotags
# python large_scale_yfcc_download_parallel_new.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2;
from io import BytesIO
import os
import re
from urllib.parse import unquote_plus
import json
import time
import logging
import argparse
from dataclasses import dataclass
import time
from PIL import Image
import subprocess
import requests
from tqdm import tqdm
import pickle
from datetime import datetime
from dateutil import parser
import shutil
import random
import imagesize
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import save_obj_as_pickle, load_pickle
import threading

argparser = argparse.ArgumentParser()
argparser.add_argument("--img_dir", 
                        default='/data3/zhiqiul/yfcc100m_all_new/',
                        help="Yfcc100M dataset will be downloaded at this location")
argparser.add_argument("--data_file", 
                        default='./yfcc100m/yfcc100m_dataset',
                        help="The yfcc100M dataset file")
argparser.add_argument("--auto_file",
                        default='./yfcc100m/yfcc100m_autotags-v1', 
                        help="The yfcc100M autotag file")
argparser.add_argument("--exif_file",
                        default='./yfcc100m/yfcc100m_exif', 
                        help="The yfcc100M exif file")
argparser.add_argument("--hash_file",
                        default='./yfcc100m/yfcc100m_hash', 
                        help="The yfcc100M hash file")
argparser.add_argument("--hash_pickle",
                        default='./yfcc100m/yfcc100m_hash.pickle', 
                        help="Save yfcc100M hash dictionary pickle object to this location")
argparser.add_argument("--lines_file",
                        default='./yfcc100m/yfcc100m_lines', 
                        help="The lines file")
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

# The index for dataset file
IDX_LIST = [
    "ID",
    "USER_ID",
    "NICKNAME",
    "DATE_TAKEN",
    "DATE_UPLOADED",
    "DEVICE",
    "TITLE",
    "DESCRIPTION",
    "USER_TAGS",
    "MACHINE_TAGS",
    "LON",
    "LAT",
    "GEO_ACCURACY",
    "PAGE_URL",
    "DOWNLOAD_URL",
    "LICENSE_NAME",
    "LICENSE_URL",
    "SERVER_ID",
    "FARM_ID",
    "SECRET",
    "SECRET_ORIGINAL",
    "EXT",
    "IMG_OR_VIDEO",
]


IDX_TO_NAME = {i : IDX_LIST[i] for i in range(len(IDX_LIST))}

# TARGET_LINE_NUMBERS = set(pickle.load(open("line_numbers.pkl", "rb")))

@dataclass
class MetadataObject():
    """Represent Metadata for a single media object
    Args:
        num_init_classes (int): Number of initial labeled (discovered) classes
        sample_per_class (int): Number of sample per discovered class
        num_open_classes (int): Number of classes hold out as open set
        ID (str): Photo/video identifier
        USER_ID (str): User NSID
        NICKNAME (str): User nickname
        DATE_TAKEN (str): Date taken
        DATE_UPLOADED (str): Date uploaded
        DEVICE (str): Capture device
        TITLE (str): Title
        DESCRIPTION (str): Description
        USER_TAGS (str): User tags (comma-separated)
        MACHINE_TAGS (str): Machine tags (comma-separated)
        LON (str): Longitude
        LAT (str): Latitude
        GEO_ACCURACY (str): Accuracy of the longitude and latitude coordinates (1=world level accuracy, ..., 16=street level accuracy)
        PAGE_URL (str): Photo/video page URL
        DOWNLOAD_URL (str): Photo/video download URL
        LICENSE_NAME (str): License name
        LICENSE_URL (str): License URL
        SERVER_ID (str): Photo/video server identifier
        FARM_ID (str): Photo/video farm identifier
        SECRET (str): Photo/video secret
        SECRET_ORIGINAL (str): Photo/video secret original
        EXT (str): Extension of the original photo
        IMG_OR_VIDEO (int): Photos/video marker (0 = photo, 1 = video)
        AUTO_TAG_SCORES (dict): A dictionary of autotag_scores (key = category, value = confidence score in float)
        LINE_NUM (int): Line Number
        HASH_VALUE (str):  Hash value
        EXIF (str) : EXIF
        IMG_PATH (str): Image path
        IMG_DIR (str): Image folder (default: None)
    """
    ID : str
    USER_ID : str
    NICKNAME : str
    DATE_TAKEN : str
    DATE_UPLOADED : str
    DEVICE : str
    TITLE : str
    DESCRIPTION : str
    USER_TAGS : str
    MACHINE_TAGS : str
    LON : str
    LAT : str
    GEO_ACCURACY : str
    PAGE_URL : str
    DOWNLOAD_URL : str
    LICENSE_NAME : str
    LICENSE_URL : str
    SERVER_ID : str
    FARM_ID : str
    SECRET : str
    SECRET_ORIGINAL : str
    EXT : str
    IMG_OR_VIDEO : int
    AUTO_TAG_SCORES : dict
    LINE_NUM : int
    HASH_VALUE : str
    EXIF : str
    IMG_PATH : str
    IMG_DIR : str = None


class Metadata(object):
    """A class for metadata verfication/manipulation
    """
    def __init__(self, data, autotag, line_num, hash_dict, save_folder, exif_line=None):
        self.metadata = self._parse_metadata(data, autotag, line_num, hash_dict, save_folder, exif_line=exif_line)
    
    def get_metadata(self):
        return self.metadata

    def is_img(self):
        return self.metadata.IMG_OR_VIDEO == 0
    
    def get_path(self):
        if self.metadata.IMG_DIR == None:
            return self.metadata.IMG_PATH
        else:
            return os.path.join(self.metadata.IMG_DIR, self.metadata.IMG_PATH)

    def date_uploaded(self):
        # Attributes: year, month, day, hour, minute, second, microsecond, and tzinfo.
        return datetime.utcfromtimestamp(int(self.metadata.DATE_UPLOADED))
    
    def date_taken(self):
        try:
            return parser.isoparse(self.metadata.DATE_TAKEN)
        except:
            return None
    
    def _parse_line(self, line):
        entries = line.strip().split("\t")
        if len(entries) == 1:
            return entries[0], None
        else:
            return entries[0], entries[1]

    def _parse_metadata(self, data, autotag, line_num, hash_dict, save_folder, exif_line=None):
        """Parse the metadata and return MetadataObject
        """
        entries = data.strip().split("\t")
        for i, entry in enumerate(entries):
            self.__setattr__(IDX_TO_NAME[i], entry)

        metadict = {IDX_TO_NAME[i] : entries[i] for i in range(len(entries))}

        # get autotag scores in a dict
        tag_ID, autotag_scores = self._parse_autotags(autotag)
        if not metadict['ID'] == tag_ID:
            print("AUTOTAG ID != Photo ID")
            import pdb; pdb.set_trace()
        
        line_number, line_ID = self._parse_line(line_num)
        if not metadict['ID'] == line_ID:
            print("LINE ID != Photo ID")
            import pdb; pdb.set_trace()
        
        if exif_line != None:
            exif_ID, exif_number = self._parse_line(exif_line)
            if not metadict['ID'] == exif_ID:
                print("EXIF ID != Photo ID")
                import pdb; pdb.set_trace()
        else:
            exif_number = None
        
        hash_value = hash_dict[metadict['ID']]
        # import pdb; pdb.set_trace()
        img_dir = os.path.abspath(save_folder)
        img_path = f"{metadict['ID']}.{metadict['EXT']}"

        metadata = MetadataObject(
            ID = metadict['ID'],
            USER_ID = metadict['USER_ID'],
            NICKNAME = metadict['NICKNAME'], 
            DATE_TAKEN = metadict['DATE_TAKEN'],
            DATE_UPLOADED = metadict['DATE_UPLOADED'],
            DEVICE = metadict['DEVICE'],
            TITLE = metadict['TITLE'],
            DESCRIPTION = metadict['DESCRIPTION'],
            USER_TAGS = metadict['USER_TAGS'],
            MACHINE_TAGS = metadict['MACHINE_TAGS'],
            LON = metadict['LON'],
            LAT = metadict['LAT'],
            GEO_ACCURACY = metadict['GEO_ACCURACY'],
            PAGE_URL = metadict['PAGE_URL'],
            DOWNLOAD_URL = metadict['DOWNLOAD_URL'],
            LICENSE_NAME = metadict['LICENSE_NAME'],
            LICENSE_URL = metadict['LICENSE_URL'],
            SERVER_ID = metadict['SERVER_ID'],
            FARM_ID = metadict['FARM_ID'],
            SECRET = metadict['SECRET'],
            SECRET_ORIGINAL = metadict['SECRET_ORIGINAL'],
            EXT = metadict['EXT'],
            IMG_OR_VIDEO = int(metadict['IMG_OR_VIDEO']),
            AUTO_TAG_SCORES = autotag_scores,
            LINE_NUM = line_number,
            HASH_VALUE = hash_value,
            EXIF = exif_number,
            IMG_PATH = img_path,
            IMG_DIR = img_dir,
        )
        return metadata
        
    def _parse_autotags(self, line):
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

class Criteria():
    """Whether or not a media file is valid
    """
    def __init__(self, args):
        self.args = args
        self.save_folder = None
        self.pickle_location = None
    
    def is_valid(self, metadata : Metadata):
        raise NotImplementedError()
    
    def make_metadata(self, data_line, auto_line, line_num, hash_dict, exif_line, save_folder, absolute_path=True):
        return Metadata(data_line, auto_line, line_num, hash_dict, save_folder, exif_line=exif_line, absolute_path=absolute_path)
    
import time
running_time = time.time()
def fetch_and_save_image(img_path, url, MIN_EDGE=0, MAX_ASPECT_RATIO=None, MAX_NUM_OF_TRAILS=3, RUN_TIME=1000, MIN_IMAGE_SIZE=2100):
    """Return true if image is valid and successfully downloaded
    """
    global running_time
    trials = 0
    max_trails = 3
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
                print("Sleep for {:d} secs".format(sleep_time))
                time.sleep(sleep_time)
                trials += 1
                # running_time = time.time()
                continue
            # if time.time() - running_time < RUN_TIME:
            else:
                print("Cannot fetch {:s}".format(url))
                return False
          
class AllValidDate(Criteria):
    """Return all valid images
    """
    def __init__(self, args):
        super().__init__(args)
        self.size_option = args.size_option
        self.min_size = args.min_size
        self.min_edge = args.min_edge
        self.max_aspect_ratio = args.max_aspect_ratio
        self.use_valid_date = args.use_valid_date

        self.auxilary_info_str = ""
        if self.size_option != 'original':
            self.auxilary_info_str += f"_size_{self.size_option}"
            raise NotImplementedError()
        if self.min_size != 0:
            self.auxilary_info_str += f"_minbyte_{self.min_size}"
        if self.use_valid_date != 0:
            self.auxilary_info_str += f"_valid_uploaded_date"
        if self.min_edge != 0:
            self.auxilary_info_str += f"_minedge_{self.min_edge}"
        if self.max_aspect_ratio != 0:
            self.auxilary_info_str += f"_maxratio_{self.max_aspect_ratio}"
            
        self.save_folder = os.path.join(args.img_dir, f'images{self.auxilary_info_str}')
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def pre_valid_check(self, metadata : Metadata):
        metadata_obj = metadata.get_metadata()
        if self.use_valid_date:
            if metadata.date_taken() == None or metadata.date_taken() >= metadata.date_uploaded():
                return False
        if metadata.is_img():
            return True
        return False
    
    def post_valid_check(self, metadata: Metadata):
        metadata_obj = metadata.get_metadata()
        width, height = imagesize.get(metadata.get_path())
        metadata_obj.WIDTH, metadata_obj.HEIGHT = width, height
        metadata_obj.ASPECT_RATIO = max(width, height) / min(width, height)
        return True

    def fetch_one(self, metadata : Metadata):
        metadata_obj = metadata.get_metadata()
        fetch_success = fetch_and_save_image(
            metadata.get_path(),
            metadata_obj.DOWNLOAD_URL,
            MIN_EDGE=self.min_edge,
            MIN_IMAGE_SIZE=self.min_size,
            MAX_ASPECT_RATIO=self.max_aspect_ratio,
        )
        return fetch_success

    def make_metadata(self, data_line, auto_line, line_num, hash_dict, exif_line, save_folder):
        # Overwrite so no exif is saved
        return Metadata(data_line, auto_line, line_num, hash_dict, save_folder, exif_line=None)
    
    def has_enough(self, metadata_list):
        return False

def get_flickr_folder(folder_location, idx):
    return os.path.join(folder_location, str(idx))

def get_flickr_image_folder(folder_location, idx):
    folder = get_flickr_folder(folder_location, idx)
    return os.path.join(folder, "images")

def get_flickr_metadata_pickle_path(folder_location, idx):
    folder = get_flickr_folder(folder_location, idx)
    return os.path.join(folder, "metadata.pickle")

class FlickrFolder():
    def __init__(self, idx, folder_location, num_images=10000):
        # num_images is not the actual size but rather the split index
        self.folder = get_flickr_folder(folder_location, idx)
        self.num_images = num_images
        self.metadata_location = get_flickr_metadata_pickle_path(folder_location, idx)
        self.image_folder = get_flickr_image_folder(folder_location, idx)
        if not os.path.exists(self.image_folder):
            print(f"make dir at {self.image_folder}")
            os.makedirs(self.image_folder)
    
    def copy_to_new_save_folder(self, idx, new_folder_location):
        new_folder = os.path.join(new_folder_location, str(idx))
        if os.path.exists(new_folder):
            # print(f"{new_folder} already exists")
            # import pdb; pdb.set_trace()
            return

        os.makedirs(new_folder)

        if not self.has_enough():
            import pdb; pdb.set_trace()
            return
        
        new_metadata_location = os.path.join(new_folder, "metadata.pickle")
        
        new_image_folder = os.path.join(new_folder, "images")
        if os.path.exists(new_image_folder):
            print(f"{new_image_folder} already exists")
            import pdb; pdb.set_trace()
            return

        cp_script = f'cp -r {self.image_folder}/. {new_image_folder}'
        # cp_script = f'rsync -a {self.image_folder}/. {new_image_folder}'
        subprocess.call(cp_script, shell=True)
        print(f"Done {cp_script}")

        old_metadata_list = self.load_metadata()
        for meta in old_metadata_list:
            img_name = meta.metadata.IMG_PATH.split(os.sep)[-1]
            new_img_name = os.path.join(new_image_folder, img_name)
            meta.metadata.IMG_PATH = new_img_name
        save_obj_as_pickle(new_metadata_location, old_metadata_list)
    
    # def has_enough(self):
    #     metadata_list = self.load_metadata()
    #     return len(metadata_list) == self.num_images

    def load_metadata(self):
        return load_pickle(self.metadata_location)
    
    def save_metadata(self, metadata_list):
        print(f"Save at {self.metadata_location}")
        save_obj_as_pickle(self.metadata_location, metadata_list)
    
    def get_folder_path(self):
        return self.folder
    
class FlickrFolderAccessor():
    def __init__(self, flickr_folder):
        self.flickr_folder = flickr_folder
        self.metadata_list = flickr_folder.load_metadata()

    def __getitem__(self, idx):
        return self.metadata_list[idx]
    
    def __len__(self):
        return len(self.metadata_list)

class FlickrAccessor():
    """
    Wrap around a list of FlickrFolder object in order to access image metadata as a single list
    """
    def __init__(self, folders):
        self.flickr_folders = [FlickrFolderAccessor(folders[f_idx]) for f_idx in sorted(folders.keys())]

        self.total_length = 0
        self.num_images = len(self.flickr_folders[0])
        for f in self.flickr_folders:
            self.total_length += len(f)

    def __getitem__(self, idx):
        f_idx = int(idx / self.num_images)
        i_idx = idx % self.num_images
        return self.flickr_folders[f_idx][i_idx]
    
    def __len__(self):
        return self.total_length

def get_flickr_accessor(args, new_folder_path=None):
    flickr_parser = get_flickr_parser(args)
    if new_folder_path == None:
        folders = flickr_parser.load_folders()
    else:
        print("Copying all content of flickr folders")
        folders = flickr_parser.copy_to_new_save_folder(new_folder_path)
        
    return FlickrAccessor(folders)

def get_flickr_parser(args):
    criteria = AllValidDate(args)
    flickr_parser = FlickrParser(args, criteria)
    return flickr_parser

def get_flickr_folder_location(args, new_folder_path=None):
    if new_folder_path == None:
        return get_flickr_parser(args).save_folder
    else:
        return new_folder_path

class FlickrParser():
    """
    Parse Flickr dataset files, in order to download images
    """
    def __init__(self, args, criteria : Criteria):
        self.chunk_size = args.chunk_size
        self.data_file = args.data_file
        self.auto_file = args.auto_file
        self.exif_file = args.exif_file
        self.hash_file = args.hash_file
        self.hash_pickle = args.hash_pickle
        self.lines_file = args.lines_file

        self.criteria = criteria
        self.save_folder = criteria.save_folder
        self.main_pickle_location = os.path.join(self.save_folder, "all_folders.pickle")
        
        self.flickr_folders = load_pickle(self.main_pickle_location, default_obj={})

    def copy_to_new_save_folder(self, new_save_folder):
        new_main_pickle_location = os.path.join(new_save_folder, "all_folders.pickle")
        loaded_result = load_pickle(new_main_pickle_location)
        if loaded_result:
            return loaded_result

        new_flickr_folders = {}
        for folder_idx in enumerate(self.flickr_folders):
            folder = self.flickr_folders[folder_idx]
            folder.copy_to_new_save_folder(folder_idx, new_save_folder)
            new_folder = FlickrFolder(folder_idx, new_save_folder, num_images=folder.num_images)
            new_flickr_folders[folder_idx] = new_folder
        save_obj_as_pickle(new_main_pickle_location, new_flickr_folders)
        return new_flickr_folders

    def load_folders(self):
        return self.flickr_folders

    def fetch_images(self):
        if len(self.flickr_folders) > 0:
            print("Continue fetching images")
            last_index = max(self.flickr_folders.keys()) * self.chunk_size
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
            # format = "%(asctime)s: %(message)s"
            # logging.basicConfig(format=format, level=logging.INFO,
            #                     datefmt="%H:%M:%S")
            metadata_lists = {} # list of metadata
            metadata_counts = {} # list of counts of downloaded (or attempted) metadata
            # metadata_list = []
            # flickr_folder = [FlickrFolder(len(self.flickr_folders), self.save_folder, num_images=self.chunk_size)]
            lock = threading.Lock()
            
            def download_image(i, data_line, auto_line, line_num, exif_line, lock, metadata_lists, metadata_counts):
                try:
                    # print(f"Size current {len(metadata_list[0])}")
                    if i % 5000 == 0:
                        logging.info("Thread %i: curr iter", i)
                    folder_idx = int(i / self.chunk_size)
                    
                    with lock:
                        if not folder_idx in self.flickr_folders:
                            self.flickr_folders[folder_idx] = FlickrFolder(folder_idx, self.save_folder, num_images=self.chunk_size)
                    
                    meta = self.criteria.make_metadata(data_line, auto_line, line_num, hash_dict, exif_line, get_flickr_image_folder(self.save_folder, folder_idx))
                    if self.criteria.pre_valid_check(meta):
                        fetch_success = self.criteria.fetch_one(meta)
                        if fetch_success and self.criteria.post_valid_check(meta):
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
                            print(f"Save the metadata list (successful download ({len(metadata_lists[folder_idx])})) for {folder_idx * self.chunk_size} to {(1+folder_idx) * self.chunk_size} images at {self.flickr_folders[folder_idx].image_folder}")
                            self.flickr_folders[folder_idx].save_metadata(metadata_lists[folder_idx])
                            save_obj_as_pickle(self.main_pickle_location, self.flickr_folders)
                            print(f"Updated at {self.main_pickle_location}")
                            del metadata_lists[folder_idx]
                except Exception as e:
                    print(e)

            with ThreadPoolExecutor(max_workers=128) as executor:
                results = []            
                for i, (data_line, auto_line, line_num, exif_line) in tqdm(enumerate(zip_object)):
                    if i < last_index:
                        continue
                    else:
                        results += [executor.submit(download_image, i, data_line, auto_line, line_num, exif_line, lock, metadata_lists, metadata_counts)]
                for cur_result in tqdm(as_completed(results), total=len(results)):
                    cur_result.result()

            print(f"Finished all media objects.")
    


if __name__ == "__main__":
    args = argparser.parse_args()
    criteria = AllValidDate(args)
    
    flickr_parser = FlickrParser(args, criteria)
    flickr_parser.fetch_images()
