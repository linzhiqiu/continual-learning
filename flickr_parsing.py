# A script to parse flickr datasets/autotags
# python flickr_parsing.py --fetch_by_tag head; python flickr_parsing.py --fetch_by_tag indoor; python flickr_parsing.py --fetch_by_tag text; python flickr_parsing.py --fetch_by_tag vehicle; python flickr_parsing.py --fetch_by_tag water; python flickr_parsing.py --fetch_by_tag architecture; python flickr_parsing.py --fetch_by_tag art; python flickr_parsing.py --fetch_by_tag car; python flickr_parsing.py --fetch_by_tag blue; python flickr_parsing.py --fetch_by_tag groupshot; python flickr_parsing.py --fetch_by_tag indoor; python flickr_parsing.py --fetch_by_tag nature; python flickr_parsing.py --fetch_by_tag outdoor; python flickr_parsing.py --fetch_by_tag vehicle; python flickr_parsing.py --fetch_by_tag water;
# Download all: python flickr_parsing.py --all_images --img_dir /project_data/ramanan/yfcc100m;
# Download all: python flickr_parsing.py --all_images --img_dir /project_data/ramanan/yfcc100m_v2;
# Download all: python flickr_parsing.py --all_images --img_dir /project_data/ramanan/yfcc100m_v3; No sleep
# Download all: python flickr_parsing.py --all_images --img_dir /project_data/ramanan/yfcc100m --min_size 10; No checking of broken paths
from io import BytesIO
import os
import json
import time
import argparse
from dataclasses import dataclass

from PIL import Image
import requests
from tqdm import tqdm
import pickle
from datetime import datetime
from dateutil import parser
import shutil
import random

import matplotlib.pyplot as plt
import numpy as np

CKPT_POINT_LENGTH = 10000 # Save every CKPT_POINT_LENGTH metadata

argparser = argparse.ArgumentParser()
argparser.add_argument("--img_dir", 
                        default='./yfcc100m/data',
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
argparser.add_argument("--max_images",
                        type=int, default=10000,
                        help="The maximum images to store")
argparser.add_argument("--original_size",
                        action='store_true',
                        help="Whether to use the original image size.")
argparser.add_argument("--min_edge",
                        type=int, default=0,
                        help="Images with edge shorter than min_edge will be ignored.")
argparser.add_argument("--min_size",
                        type=int, default=2100,
                        help="Images with size smaller than min_size will be ignored.")
argparser.add_argument("--fetch_by_tag",
                        type=str, default=None,
                        help="Images with tag fetch_by_tag .")
argparser.add_argument("--random_images",
                        action='store_true',
                        help="Random subset of images .")
argparser.add_argument("--all_images",
                        action='store_true',
                        help="Store all images.")



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

    def has_exif(self):
        return self.metadata.EXIF != None
    
    def date_uploaded(self):
        # Attributes: year, month, day, hour, minute, second, microsecond, and tzinfo.
        return datetime.utcfromtimestamp(int(self.metadata.DATE_UPLOADED))
    
    def date_taken(self):
        try:
            return parser.isoparse(self.metadata.DATE_TAKEN)
        except:
            # import pdb; pdb.set_trace()
            return None
    
    def _parse_line(self, line):
        entries = line.strip().split("\t")
        if len(entries) == 1:
            return entries[0], None
        else:
            return entries[0], entries[1]

    def _parse_metadata(self, data, autotag, line_num, hash_dict, save_folder, exif_line=None):
        """Store all meta data in self attributes
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
                import pdb; pdb.set_trace()
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
    
    def get_save_folder(self):
        return self.save_folder
    
    def make_metadata(self, data_line, auto_line, line_num, hash_dict, exif_line, save_folder, absolute_path=True):
        return Metadata(data_line, auto_line, line_num, hash_dict, save_folder, exif_line=exif_line, absolute_path=absolute_path)
    
    def save_metadata_list_as_pickle(self, metadata_list):
        pickle.dump(metadata_list, open(self.pickle_location, 'wb+'))
        print(f"Save metadata list as a pickle at {self.pickle_location}")

    def get_metadata_pickle(self):
        if os.path.exists(self.pickle_location):
            return pickle.load(open(self.pickle_location, 'rb'))
        else:
            return None

class ImageByRandom(Criteria):
    """Return all valid images
    """
    def __init__(self, args):
        super().__init__(args)
        self.size_option = args.size_option
        self.min_edge = args.min_edge
        self.min_size = args.min_size
        self.max_images = args.max_images

        self.auxilary_info_str = ""
        if self.size_option != 'original':
            self.auxilary_info_str += f"_size_{self.size_option}"
            raise NotImplementedError()
        if self.min_edge > 0:
            self.auxilary_info_str += f"_minedge_{self.min_edge}"
            raise NotImplementedError()
        if self.min_size != 0:
            self.auxilary_info_str += f"_minbyte_{self.min_size}"
        if self.max_images:
            self.auxilary_info_str += f"_totalimgs_{self.max_images}"
            
        self.save_folder = os.path.join(args.img_dir, "random_subset", f'images{self.auxilary_info_str}')
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.pickle_location = self.save_folder + ".pickle"

    def is_valid(self, metadata : Metadata):
        metadata_obj = metadata.get_metadata()
        if metadata.is_img():
            return fetch_and_save_image(
                metadata_obj.IMG_PATH,
                metadata_obj.DOWNLOAD_URL,
                MIN_EDGE=self.min_edge,
                MIN_IMAGE_SIZE=self.min_size
            )

    def has_enough(self, metadata_list):
        return len(metadata_list) >= self.max_images

class AllImages(Criteria):
    """Return all valid images
    """
    def __init__(self, args):
        super().__init__(args)
        self.size_option = args.size_option
        self.min_edge = args.min_edge
        self.min_size = args.min_size

        self.auxilary_info_str = ""
        if self.size_option != 'original':
            self.auxilary_info_str += f"_size_{self.size_option}"
            raise NotImplementedError()
        if self.min_edge > 0:
            self.auxilary_info_str += f"_minedge_{self.min_edge}"
            raise NotImplementedError()
        if self.min_size != 0:
            self.auxilary_info_str += f"_minbyte_{self.min_size}"
            
        self.save_folder = os.path.join(args.img_dir, "all_images", f'images{self.auxilary_info_str}')
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.pickle_location = self.save_folder + ".pickle"

    def is_valid(self, metadata : Metadata):
        metadata_obj = metadata.get_metadata()
        if metadata.is_img():
            return fetch_and_save_image(
                metadata_obj.IMG_PATH,
                metadata_obj.DOWNLOAD_URL,
                MIN_EDGE=self.min_edge,
                MIN_IMAGE_SIZE=self.min_size
            )

    def make_metadata(self, data_line, auto_line, line_num, hash_dict, exif_line, save_folder):
        # Overwrite so no exif is saved
        return Metadata(data_line, auto_line, line_num, hash_dict, save_folder, exif_line=None)
    
    def has_enough(self, metadata_list):
        return False

class ImageByAutoTag(Criteria):
    """Return all valid images with autotag
    """
    def __init__(self, args):
        super().__init__(args)
        self.fetch_by_tag = args.fetch_by_tag

        self.size_option = args.size_option
        self.min_edge = args.min_edge
        self.min_size = args.min_size
        self.max_images = args.max_images

        self.auxilary_info_str = ""
        if self.size_option != 'original':
            self.auxilary_info_str += f"_size_{self.size_option}"
            raise NotImplementedError()
        if self.min_edge > 0:
            self.auxilary_info_str += f"_minedge_{self.min_edge}"
            raise NotImplementedError()
        if self.min_size != 0:
            self.auxilary_info_str += f"_minbyte_{self.min_size}"
        if self.max_images:
            self.auxilary_info_str += f"_totalimgs_{self.max_images}"
            
        self.save_folder = os.path.join(args.img_dir, "fetch_by_tag", self.fetch_by_tag, f'images{self.auxilary_info_str}')
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.pickle_location = self.save_folder + ".pickle"

    def is_valid(self, metadata : Metadata):
        metadata_obj = metadata.get_metadata()
        if self.fetch_by_tag not in metadata_obj.AUTO_TAG_SCORES:
            return False
        else:
            if metadata.is_img():
                return fetch_and_save_image(
                    metadata_obj.IMG_PATH,
                    metadata_obj.DOWNLOAD_URL,
                    MIN_EDGE=self.min_edge,
                    MIN_IMAGE_SIZE=self.min_size
                )
            else:
                return False

    def has_enough(self, metadata_list):
        return len(metadata_list) >= self.max_images

# def fetch_and_save_image(img_path, url, MIN_EDGE=0, MAX_NUM_OF_TRAILS=3, SLEEP_TIME=3, MIN_IMAGE_SIZE=2100):
#     """Return true if image is valid and successfully downloaded
#     """
#     number_of_trails = 0
#     sleep_time = SLEEP_TIME
#     while True:
#         try:
#             # print(1)
#             response = requests.get(url)
#             img = Image.open(BytesIO(response.content))
#             if img.size[0] < MIN_EDGE or img.size[1] < MIN_EDGE:
#                 return False
#             img.save(img_path)
#             is_removed = remove_broken_path(img_path)
#             if is_removed:
#                 return False
#             else:
#                 if os.path.getsize(img_path) < MIN_IMAGE_SIZE:
#                     return False
#                 return True
#         except:
#             # Sleep for a while and try again
#             number_of_trails += 1
#             sleep_time += SLEEP_TIME
#             if number_of_trails >= MAX_NUM_OF_TRAILS:
#                 print("Cannot fetch {:s} after {:d} trails".format(url, MAX_NUM_OF_TRAILS))
#                 return False
#             print("Sleep for {:d} secs".format(SLEEP_TIME))
#             time.sleep(SLEEP_TIME)
#             continue

import time
running_time = time.time()
def fetch_and_save_image(img_path, url, MIN_EDGE=0, MAX_ASPECT_RATIO=None, MAX_NUM_OF_TRAILS=3, RUN_TIME=1000, MIN_IMAGE_SIZE=2100):
    """Return true if image is valid and successfully downloaded
    """
    global running_time
    while True:
        try:
            # print(1)
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
            if time.time() - running_time < RUN_TIME:
                print("Cannot fetch {:s}".format(url))
                return False
            else:
                print("Sleep for a while and try again")
                print("Sleep for {:d} secs".format(1))
                time.sleep(1)
                running_time = time.time()
                continue

def remove_broken_path(img_path):
    """Remove an image locally if it's broken. Return True if the image is removed
    """
    try:
        a = Image.open(img_path)
    except:
        print(img_path + " is broken, so we delete it")
        os.remove(img_path)
        return True
    return False
        

class FlickrParser():
    def __init__(self, args, criteria : Criteria):
        self.args = args
        self.data_file = args.data_file
        self.auto_file = args.auto_file
        self.exif_file = args.exif_file
        self.hash_file = args.hash_file
        self.hash_pickle = args.hash_pickle
        self.lines_file = args.lines_file

        self.criteria = criteria
        self.save_folder = criteria.get_save_folder()

        self.ckpt_pickle_location = self.save_folder + "_ckpt.pickle" # for intermediate ckpt
        print(f"Intermediate ckpt location: {self.ckpt_pickle_location}")
        self.metadata_list = self._load_files()

    def _load_files(self):
        metadata_list = self.criteria.get_metadata_pickle()
        if metadata_list == None or len(metadata_list) == 0:
            metadata_list = []
        elif not self.criteria.has_enough(metadata_list):
            print("Continue fetching images")
        else:
            # TODO: Remove this after run for all subfolders
            min_size = None
            min_path = None
            wrong_taken_date = 0
            invalid_date = 0
            after_2015 = 0
            before_1998 = 0
            # need_resave = False
            if not len(metadata_list) == self.criteria.max_images:
                import pdb; pdb.set_trace()
            for meta in metadata_list:
                # s = os.path.getsize(meta.get_path())
                # if min_size == None or s < min_size:
                #     min_size = s
                #     min_path = meta.get_path()
                #     print(f"Min path is {min_path} with size {min_size}")
                if meta.date_taken() == None:
                    invalid_date += 1
                elif meta.date_taken() >= meta.date_uploaded():
                    wrong_taken_date += 1
                else:
                    if meta.date_taken().year == 2013:
                        print(meta.get_path())
                        print(meta.metadata)
                        import pdb; pdb.set_trace()
                
                if meta.date_taken() != None and meta.date_taken().year >= 2015:
                    after_2015 += 1

                if meta.date_taken() != None and meta.date_taken().year <= 1998:
                    before_1998 += 1

            print(f"Wrong taken date {wrong_taken_date}/{len(metadata_list)}")
            print(f"Invalid taken date {invalid_date}/{len(metadata_list)}")
            print(f"In or After 2015 taken date {after_2015}/{len(metadata_list)}")
            print(f"In or Before 1998 taken date {before_1998}/{len(metadata_list)}")
                # if meta.metadata.IMG_PATH[0] == ".":
                #     meta.metadata.IMG_PATH = os.path.abspath(meta.metadata.IMG_PATH)
                #     need_resave = True
            # if need_resave:
            #     print("Changing relative paths to absolute paths")
            #     self.criteria.save_metadata_list_as_pickle(metadata_list)
            return metadata_list
        

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
            if type(self.criteria) in [ImageByRandom, AllImages]:
                zip_index = list(range(len(hash_dict.keys())))
                
                if os.path.exists(self.ckpt_pickle_location) and os.path.getsize(self.ckpt_pickle_location) > 0:
                    sorted_zip_index, curr_index = pickle.load(open(self.ckpt_pickle_location, 'rb'))
                else:
                    metadata_list = []
                    print(f"Save metadata list as a pickle at {self.criteria.pickle_location}")
                    if type(self.criteria) == ImageByRandom:
                        print(f"Randomly shuffled the data and only taking {2*self.criteria.max_images} of all images")
                        random.shuffle(zip_index)
                        zip_index = zip_index[:2*self.criteria.max_images]
                    sorted_zip_index = sorted(zip_index)
                    curr_index = sorted_zip_index[0]
                    pickle.dump((sorted_zip_index, curr_index), open(self.ckpt_pickle_location, 'wb+'))
                
                for i, (data_line, auto_line, line_num, exif_line) in tqdm(enumerate(zip_object)):
                    if i == curr_index:
                        meta = self.criteria.make_metadata(data_line, auto_line, line_num, hash_dict, exif_line, self.save_folder)
                        if self.criteria.is_valid(meta):
                            # import pdb; pdb.set_trace()
                            metadata_list.append(meta)
                            if self.criteria.has_enough(metadata_list):
                                break
                        if len(sorted_zip_index) > 0:
                            del sorted_zip_index[0]
                            curr_index = sorted_zip_index[0]
                        else:
                            break
                    if len(metadata_list) % CKPT_POINT_LENGTH == 0:
                        print(f"Update the metadata list for {len(metadata_list)} images")
                        pickle.dump((sorted_zip_index, curr_index), open(self.ckpt_pickle_location, 'wb+'))
                        self.criteria.save_metadata_list_as_pickle(metadata_list)
                self.criteria.save_metadata_list_as_pickle(metadata_list)
            else:
                for data_line, auto_line, line_num, exif_line in tqdm(zip_object):
                    meta = self.criteria.make_metadata(data_line, auto_line, line_num, hash_dict, exif_line, self.save_folder)
                    if self.criteria.is_valid(meta):
                        # import pdb; pdb.set_trace()
                        metadata_list.append(meta)
                        if self.criteria.has_enough(metadata_list):
                            self.criteria.save_metadata_list_as_pickle(metadata_list)
                            break
                        elif len(metadata_list) % CKPT_POINT_LENGTH == 0:
                            print(f"Saving the metadata list for {len(metadata_list)} images")
                            self.criteria.save_metadata_list_as_pickle(metadata_list)
        
        print(f"Finished parsing {len(metadata_list)} media objects.")
        return metadata_list
    
    def group_by_year_date_taken(self):
        return self._group(mode='year', date='date_taken')

    def group_by_month_date_taken(self):
        return self._group(mode='month', date='date_taken')

    def group_by_year_date_uploaded(self):
        self._group(mode='year', date='date_uploaded')

    def group_by_month_date_uploaded(self):
        self._group(mode='month', date='date_uploaded')

    def _group(self, mode='month', date='date_taken'): 
        save_folder = os.path.join(self.save_folder, f"{date}_by_{mode}")
        sub_folder_name_func =  lambda idx, bucket_name : f"{idx}-{bucket_name[0]}-{bucket_name[1]}" if mode == 'month' else f"{idx}-{bucket_name}"
        save_pickle = os.path.join(save_folder, "dataset.pickle")
        if os.path.exists(save_pickle):
            sorted_buckets_list, buckets_dict = pickle.load(open(save_pickle, 'rb'))
            print(f"Load from pickle: {save_pickle}")
        else:
            sorted_buckets_list, buckets_dict = create_sorted_date_buckets(
                                                    self.metadata_list,
                                                    mode=mode,
                                                    date=date
                                                )
            for idx, bucket_name in enumerate(sorted_buckets_list):
                # year, month = bucket_name
                save_subfolder = os.path.join(save_folder, sub_folder_name_func(idx, bucket_name))
                os.makedirs(save_subfolder)
                for meta in buckets_dict[bucket_name]:
                    shutil.copy(meta.get_path(), save_subfolder)
            pickle.dump((sorted_buckets_list, buckets_dict), open(save_pickle, 'wb+'))
            print(f"Save folder: {save_folder}")
            print(f"Save pickle: {save_pickle}")
        
        plot_buckets(save_folder, sorted_buckets_list, buckets_dict, mode=mode, date=date)
        return sorted_buckets_list, buckets_dict
    
def plot_buckets(save_folder, sorted_buckets_list, buckets_dict, mode='month', date='date_taken', optional_name=""):
    save_png_barchart = os.path.join(save_folder, f"dataset_bucket_{date}{optional_name}.png")
    save_png_freqchart = os.path.join(save_folder, f"dataset_frequency_{date}{optional_name}.png")
    min_count = None
    max_count = None
    avg_count = 0
    for b in sorted_buckets_list:
        count = len(buckets_dict[b])
        if not min_count or count < min_count:
            min_count = count
        if not max_count or count > max_count:
            max_count = count
        avg_count += count
    avg_count = avg_count / len(sorted_buckets_list)
    
    print(f"Min/Max number of images {optional_name} per bucket: {min_count}/{max_count}. Average is {avg_count}")
    plt.figure(figsize=(8,8))
    axes = plt.gca()
    axes.set_ylim([0,max_count])
    plt.title(f'Number of samples per {mode}.')
    x = [str(a) for a in sorted_buckets_list]
    y = [len(buckets_dict[b]) for b in sorted_buckets_list]
    plt.bar(x, y, align='center')
    plt.axhline(y=avg_count, label=f"Mean Number of Samples {avg_count}", linestyle='--', color='black')
    x_tick = [str(a) for a in sorted_buckets_list]
    plt.xticks(x, x_tick)
    plt.xlabel('Date')
    plt.ylabel(f'Number of samples for each {mode}')
    plt.setp(axes.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='large')
    # plt.legend()
    plt.savefig(save_png_barchart)
    plt.close('all')

    
    bins = np.linspace(min_count, max_count, 15)
    plt.figure(figsize=(8,8))
    plt.hist(y, bins, alpha=0.5, label='Number of samples')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_png_freqchart)
    print(f"Saving image at {save_png_freqchart}")

def create_sorted_date_buckets(metadata_list, mode='year', date='date_taken'):
    buckets_dict = {}
    for meta in metadata_list:
        if mode == 'year':
            time_meta = getattr(meta, date)().year
        elif mode == 'month':
            time_meta = (getattr(meta, date)().year, getattr(meta, date)().month)
        
        if time_meta in buckets_dict:
            buckets_dict[time_meta].append(meta)
        else:
            buckets_dict[time_meta] = [meta]
        
    sorted_buckets_list = sorted(buckets_dict.keys())
    return sorted_buckets_list, buckets_dict

def plot_time_buckets(metadata_list, save_folder, mode='year', date='date_uploaded', optional_name=""):
    buckets_dict = {}

    for meta in metadata_list:
        if date == 'date_uploaded':
            date_obj = datetime.utcfromtimestamp(int(meta.DATE_UPLOADED))
        else:
            raise NotImplementedError()

        if mode == 'year':
            time_meta = date_obj.year
        elif mode == 'month':
            time_meta = (date_obj.year, date_obj.month)
        
        if time_meta in buckets_dict:
            buckets_dict[time_meta].append(meta)
        else:
            buckets_dict[time_meta] = [meta]
        
    sorted_buckets_list = sorted(buckets_dict.keys())
    
    plot_buckets(save_folder, sorted_buckets_list, buckets_dict, mode=mode, date=date, optional_name=optional_name)

def plot_scores_jupyter(score_list, plot_mean=False):
    min_count = None
    max_count = None
    avg_count = 0
    for s in score_list:
        if not min_count or count < min_count:
            min_count = s
        if not max_count or count > max_count:
            max_count = s
        avg_count += s
    avg_count = avg_count / len(score_list)

    bins = np.linspace(min_count, max_count, 15)
    plt.figure(figsize=(8,8))
    if plot_mean:
        plt.axhline(y=avg_count, label=f"Mean scores of Samples {avg_count}", linestyle='--', color='black')
    plt.hist(y, bins, alpha=0.5, label='Number of samples')
    plt.legend(loc='upper right')
    plt.tight_layout()


def plot_time_jupyter(metadata_list, mode='year', date='date_uploaded', plot_mean=False):
    buckets_dict = {}

    for metadata in metadata_list:
        meta = metadata.get_metadata()
        if date == 'date_uploaded':
            date_obj = datetime.utcfromtimestamp(int(meta.DATE_UPLOADED))
        else:
            raise NotImplementedError()

        if mode == 'year':
            time_meta = date_obj.year
        elif mode == 'month':
            time_meta = (date_obj.year, date_obj.month)
        
        if time_meta in buckets_dict:
            buckets_dict[time_meta].append(meta)
        else:
            buckets_dict[time_meta] = [meta]
    
    if mode == 'year' and not 2004 in buckets_dict:
        buckets_dict[2004] = []
        
    sorted_buckets_list = sorted(buckets_dict.keys())
    min_count = None
    max_count = None
    avg_count = 0
    for b in sorted_buckets_list:
        count = len(buckets_dict[b])
        if not min_count or count < min_count:
            min_count = count
        if not max_count or count > max_count:
            max_count = count
        avg_count += count
    avg_count = avg_count / len(sorted_buckets_list)
    
    # print(f"Min/Max number of images {optional_name} per bucket: {min_count}/{max_count}. Average is {avg_count}")
    plt.figure(figsize=(8,8))
    axes = plt.gca()
    axes.set_ylim([0,max_count])
    plt.title(f'Number of samples per {mode}.')
    x = [str(a) for a in sorted_buckets_list]
    y = [len(buckets_dict[b]) for b in sorted_buckets_list]
    plt.bar(x, y, align='center')
    if plot_mean:
        plt.axhline(y=avg_count, label=f"Mean Number of Samples {avg_count}", linestyle='--', color='black')
    x_tick = [str(a) for a in sorted_buckets_list]
    plt.xticks(x, x_tick)
    plt.xlabel('Date')
    plt.ylabel(f'Number of samples for each {mode}')
    plt.setp(axes.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='large')
    plt.legend()
    plt.show()
    plt.close('all')

    

if __name__ == "__main__":
    args = argparser.parse_args()
    if args.fetch_by_tag:
        criteria = ImageByAutoTag(args)
    elif args.random_images:
        criteria = ImageByRandom(args)
    elif args.all_images:
        criteria = AllImages(args)
    
        
    flickr_parser = FlickrParser(args, criteria)

    # flickr_parser.group_by_month_date_taken()
    if not args.all_images:
        flickr_parser.group_by_month_date_uploaded()
        flickr_parser.group_by_year_date_uploaded()

        flickr_parser.group_by_month_date_taken()
        flickr_parser.group_by_year_date_taken()

    
        
    