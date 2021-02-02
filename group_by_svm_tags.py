# A script to parse flickr datasets/autotags

# python group_by_svm_tags.py --min_size 10 --svm_tag_group tech; 
# python group_by_svm_tags.py --min_size 10 --svm_tag_group text; 
# python group_by_svm_tags.py --min_size 10 --svm_tag_group vehicle; 
# python group_by_svm_tags.py --min_size 10 --svm_tag_group ballgame; 
# python group_by_svm_tags.py --min_size 10 --svm_tag_group fashion; 
# python group_by_svm_tags.py --min_size 10 --svm_tag_group event; 
# python group_by_svm_tags.py --min_size 10 --svm_tag_group sports; 
# python group_by_svm_tags.py --min_size 10 --svm_tag_group people; 
# python group_by_svm_tags.py --min_size 10 --svm_tag_group indoor; 
# python group_by_svm_tags.py --min_size 10 --svm_tag_group urban; 
# python group_by_svm_tags.py --min_size 10 --svm_tag_group architecture; 
# python group_by_svm_tags.py --min_size 10 --svm_tag_group time; 
# python group_by_svm_tags.py --min_size 10 --svm_tag_group outdoor; 
# python group_by_svm_tags.py --min_size 10 --svm_tag_group weather; 
# python group_by_svm_tags.py --min_size 10 --svm_tag_group animal;
# python group_by_svm_tags.py --min_size 10 --svm_tag_group things; 

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
import imagesize

from tag_analysis import TagParser

import matplotlib.pyplot as plt
import numpy as np

CKPT_POINT_LENGTH = 1000 # Save every CKPT_POINT_LENGTH metadata

TEXT_TAGS = ['Sign','Screen_shot','Gravestone','Poster','Magazine','Document','Banner','Map','Scoreboard','Dashboard']
TECH_TAGS = ['Monitor','Camera','Laptop','Computer_Keyboard','Television','Disk','Telephone','Computer_Mouse']
VEHICLE_TAGS = ['Bike','Car','Van','Motorcycle','Airplane','Ship','Tramline','Train','Truck','School_Bus','Sail_boat','Bus','Military_Vehicle','Carriage','Motor_Scooter','Helicopter','Snowmobile','Yacht']
FASHION_TAGS = ['Swimsuit','Dress','Shirt','Coat','Handbag','Sportswear','Skirt','Tattoo','Shoe']
PEOPLE_TAGS = ['Athlete','Cosplay','Demonstrator','Speaker','Dancer','Student','Clown','Soldier','Diver']
INDOOR_TAGS = ['Office','Library','Closet','Classroom','Concert','Bar','Shop','Dining_room','Meeting_Room','Ice_rink','Restaurant','Bedroom','Bathroom','Warehouse','Art_Gallery','Stairwell']
FOOD_TAGS = ['Coffee','Tomato','Sandwich','Burrito','Cake','Soup','Alcohol','Tea','Sushi','Barbecue','Dessert','Salad','Pasta','Egg','Candy','Chocolate','Dumpling','Sashimi','Grape','Strawberry','Apple','Juice']
THINGS_TAGS = ['Balloon','Violin','Billiards','Bonfire','Fireworks','Architecture','Chair','Flowers','Tree','Water','Art','Animal','Food','Piano','Bed','Boat','Table','Window','Door','Book','Golf_Club','Hat','Painting','Flag','Candle','Kitchen_ware','Lamp','Drum','Tent','Guitar','Jewelry','Bag','Toy','Mushroom']
ARCHITECTURE_TAGS = ['Skyscraper','Church','Temple','Tower','Bridge','Gravestone','Skyscraper','Mansion','Sculpture','Fountain','Temple','Obelisk']
URBAN_TAGS = ['Casino','Concert','Amusement_Park','Shop','Alley','Tower','Harbor','Garden','Bridge','Neon_light','Highway','Windmill','Palace','Park','Walkway','Fence','Railroad','Bazaar','Mansion','Stadium','Sculpture','Streetlight','Fountain','Playground','Pool']
EVENT_TAGS = ['Graduation','Sunbath','Camping','Music','Parade','Dining','Fieldwork']
SPORTS_TAGS = ['Bungee','Snorkel','Kayak','Snowboard','Judo','Rock_Climbing','Archery','Running','Racing','Swim','Cycling','Gymnastics','Ski','Fishing','Ice_skating','Roller_skating','Riding','Surfing','Skateboarding','Boxing','Water_Skiing']
BALLGAME_TAGS = ['Soccer','Hockey','Field_Hockey','Baseball','Football','Tennis','Volleyball','Basketball','Lacrosse','Rugby','Golf','Badminton']
NATURE_TAGS = ['Gravel','Wave','Rainbow','Sky','Soil','Sand','Rock','Snow','Cloud','Fog','Grass','Ice','Moon']
TIME_TAGS = ['Dusk','Night','Sunset','Sunrise']
OUTDOOR_TAGS = ['Cave','Mountain','Reef','Lake','River','Forest','Beach','Ocean','Plain','Waterfall','Dune','Glacier','Underwater']
WEATHER_TAGS = ['Rain','Cloud','Snow','Sun','Fog','Storm',]
ANIMAL_TAGS = []

TAG_GROUPS_DICT = {
    'vehicle' : VEHICLE_TAGS ,
    'text' : TEXT_TAGS,
    'tech' : TECH_TAGS,
    # 'fashion' : FASHION_TAGS ,
    # 'people' : PEOPLE_TAGS,
    # 'indoor' : INDOOR_TAGS,
    # 'food' : FOOD_TAGS,
    # 'things' : THINGS_TAGS,
    # 'architecture' : ARCHITECTURE_TAGS,
    # 'urban' : URBAN_TAGS,
    # 'event' : EVENT_TAGS,
    # 'sports' : SPORTS_TAGS,
    # 'ballgame' : BALLGAME_TAGS,
    # 'nature' : NATURE_TAGS,
    # 'time' : TIME_TAGS,
    # 'outdoor' : OUTDOOR_TAGS,
    # 'weather' : WEATHER_TAGS,
    # 'animal' : ANIMAL_TAGS,
}

for group_name in TAG_GROUPS_DICT:
    TAG_GROUPS_DICT[group_name] = [t.lower() for t in TAG_GROUPS_DICT[group_name]]

# |-- ./dataset/small_datasets
#   |-- args.svm_tag_group
#     |-- info.txt (about the parameter used)
#     |-- metadata_dict.pickle (all the parameter/path used + images with each tag)
#     |-- images
#       |-- image_1
#       |-- image_2
#       |-- ...
#     |-- html
#       

argparser = argparse.ArgumentParser()
argparser.add_argument("--download_dir", 
                        default='./dataset/small_datasets',
                        help="The download location")
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
argparser.add_argument("--min_prob",
                        type=float, default=0.75,
                        help="Images with probability score smaller than args.min_prob associated with the tag will be ignored.")
argparser.add_argument("--max_images_per_tag",
                        type=int, default=1350,
                        help="The maximum images to store for each tag")
argparser.add_argument("--max_aspect_ratio",
                        type=float, default=2.0,
                        help="Images with max/min edge greater than max_aspect_ratio will be ignored.")
argparser.add_argument("--min_size",
                        type=int, default=10,
                        help="Images with size smaller than min_size will be ignored.")
argparser.add_argument("--use_valid_date",
                        type=bool, default=True,
                        help="Images with valid date (upload date < taken date) will be used if set true")
argparser.add_argument("--svm_tag_group",
                        type=str, default=None, choices=TAG_GROUPS_DICT.keys(),
                        help="A SVM tag group.")

from flickr_parsing import FlickrParser, Criteria, Metadata, MetadataObject, IDX_TO_NAME, IDX_LIST, fetch_and_save_image

class ImageBySVMTagList(Criteria):
    """Return all valid images in each SVM tag from the group
    """
    def __init__(self, args, svm_tag_group, save_info=False):
        super().__init__(args)
        self.svm_tag_group = svm_tag_group
        self.tag_groups = TAG_GROUPS_DICT[svm_tag_group]

        self.min_prob = args.min_prob
        self.max_images_per_tag = args.max_images_per_tag
        self.min_size = args.min_size
        self.max_aspect_ratio = args.max_aspect_ratio
        self.use_valid_date = args.use_valid_date

        self.save_main_folder = os.path.join(args.download_dir, svm_tag_group)
        self.info_location = os.path.join(self.save_main_folder, "info.txt")
        self.save_folder = os.path.join(self.save_main_folder, "images")
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.pickle_location = os.path.join(self.save_main_folder, "images.pickle")
        self.ckpt_pickle_location = os.path.join(self.save_main_folder, "download_ckpt.pickle")
        
        self.auxilary_info_str = ""
        self.auxilary_info_str += f"Minimum Probability: {self.min_prob}\n"
        self.auxilary_info_str += f"Max Aspect Ratio: {self.max_aspect_ratio}\n"
        self.auxilary_info_str += f"Valid date: {self.use_valid_date}\n"
        self.auxilary_info_str += f"Minimum byte size: {self.min_size}\n"
        self.auxilary_info_str += f"Maximum number of images per tag: {self.max_images_per_tag}\n"
        
        if save_info:
            with open(self.info_location, "w+") as f:
                f.write(self.auxilary_info_str)
        else:
            print("Please check whether the two info are the same.")
            print(self.auxilary_info_str)
            print("________________________")
            with open(self.info_location, "r") as f:
                print(f.read())
        
        self.tag_location = os.path.join(self.save_main_folder, "tags.pickle")
        self.tag_dict = {}
        for tag in self.tag_groups:
            self.tag_dict[tag] = []

    def is_valid(self, metadata : Metadata):
        metadata_obj = metadata.get_metadata()
        if self.use_valid_date:
            try:
                date_taken = parser.isoparse(metadata_obj.DATE_TAKEN)
                date_uploaded = datetime.utcfromtimestamp(int(metadata_obj.DATE_UPLOADED))
                if date_taken >= date_uploaded:
                    return False
            except:
                return False

        matched_tag = None
        for tag in self.tag_groups:
            if tag in metadata_obj.AUTO_TAG_SCORES:
                if metadata_obj.AUTO_TAG_SCORES[tag] > self.min_prob \
                    and tag in self.tag_groups \
                    and len(self.tag_dict[tag]) < self.max_images_per_tag:
                    matched_tag = tag
        
        if matched_tag:
            if metadata.is_img():
                fetch_success = fetch_and_save_image(
                    os.path.join(self.save_folder, metadata_obj.IMG_PATH),
                    metadata_obj.DOWNLOAD_URL,
                    MAX_ASPECT_RATIO=self.max_aspect_ratio,
                    MIN_IMAGE_SIZE=self.min_size
                )
                if fetch_success:
                    width, height = imagesize.get(os.path.join(self.save_folder, metadata_obj.IMG_PATH))
                    metadata_obj.WIDTH, metadata_obj.HEIGHT = width, height
                    metadata_obj.ASPECT_RATIO = max(width, height) / min(width, height)
                    self.tag_dict[matched_tag].append(metadata)
                return fetch_success
            else:
                return False

    def has_enough(self):
        for tag in self.tag_dict:
            if len(self.tag_dict[tag]) < self.max_images_per_tag:
                return False
        return True
    
    def sync(self):
        # This function exists because of a bug in code.
        # tag_dict = self.load_tag_dict()
        # print("Loaded tag_dict")
        # print_tag_dict(tag_dict)
        metadata_list = self.get_metadata_pickle()
        computed_tag_dict = {tag : [] for tag in self.tag_groups}
        for meta in metadata_list:
            metadata_obj = meta.get_metadata()
            matched_tag = None
            for tag in self.tag_groups:
                if tag in metadata_obj.AUTO_TAG_SCORES:
                    if metadata_obj.AUTO_TAG_SCORES[tag] > self.min_prob \
                        and len(computed_tag_dict[tag]) < self.max_images_per_tag:
                        matched_tag = tag
                        continue
            
            if matched_tag:
                width, height = imagesize.get(os.path.join(self.save_folder, metadata_obj.IMG_PATH))
                metadata_obj.WIDTH, metadata_obj.HEIGHT = width, height
                metadata_obj.ASPECT_RATIO = max(width, height) / min(width, height)
                computed_tag_dict[matched_tag].append(meta)
        
        print()
        print("reverse tag_dict")
        print_tag_dict(computed_tag_dict)
        self.tag_dict = computed_tag_dict
        # self.save_tag_pickle()
        return computed_tag_dict
    

    def save_tag_pickle(self):
        pickle.dump(self.tag_dict, open(self.tag_location, 'wb+'))
        print(f"Save tag dictionary as a pickle at {self.tag_location}")
    
    def load_tag_dict(self):
        if os.path.exists(self.tag_location):
            return pickle.load(open(self.tag_location, 'rb'))
        else:
            return None

def print_tag_dict(tag_dict):
    for tag in tag_dict:
        print(f"{tag} has {len(tag_dict[tag])} images")

class FlickrTagGroupParser(FlickrParser):
    def __init__(self, args, criteria : Criteria):
        super().__init__(args, criteria)
        self.ckpt_pickle_location = self.criteria.ckpt_pickle_location

    def _load_files(self):
        metadata_list = self.criteria.get_metadata_pickle()
        if metadata_list == None or len(metadata_list) == 0:
            metadata_list = []
        elif not self.criteria.has_enough():
            print("continue fetching images")
            self.criteria.tag_dict = self.criteria.sync()
        else:
            raise NotImplementedError()

        if os.path.exists(self.hash_pickle):
            hash_dict = pickle.load(open(self.hash_pickle, 'rb'))
            print(f"Load Hash dictionary at {self.hash_pickle}")
        else:
            hash_dict = {}
            with open(self.hash_file, 'r') as hash_f:
                for hash_line in tqdm(hash_f):
                    hash_id, hash_value = hash_line.strip().split("\t")
                    hash_dict[hash_id] = hash_value
            pickle.dump(hash_dict, open(self.hash_pickle, "wb+"))
            print(f"Saved Hash dictionary at {self.hash_pickle}")

                
        with open(self.data_file, "r") as f, \
             open(self.auto_file, "r") as auto_f, \
             open(self.lines_file, "r") as line_f, \
             open(self.exif_file, 'r') as exif_f:

            zip_object = zip(f, auto_f, line_f, exif_f)
            if type(self.criteria) in [ImageBySVMTagList]:
                zip_index = list(range(len(hash_dict.keys())))
                
                if os.path.exists(self.ckpt_pickle_location) and os.path.getsize(self.ckpt_pickle_location) > 0:
                    sorted_zip_index, curr_index = pickle.load(open(self.ckpt_pickle_location, 'rb'))
                else:
                    metadata_list = []
                    print(f"Save metadata list as a pickle at {self.criteria.pickle_location}")
                    sorted_zip_index = sorted(zip_index)
                    curr_index = sorted_zip_index[0]
                    pickle.dump((sorted_zip_index, curr_index), open(self.ckpt_pickle_location, 'wb+'))
                
                self.criteria.load_tag_dict()
                for i, (data_line, auto_line, line_num, exif_line) in tqdm(enumerate(zip_object)):
                    if i == curr_index:
                        meta = self.criteria.make_metadata(data_line, auto_line, line_num, hash_dict, exif_line, self.save_folder, absolute_path=False)
                        if self.criteria.is_valid(meta):
                            metadata_list.append(meta)
                            if self.criteria.has_enough():
                                break
                        if len(sorted_zip_index) > 0:
                            del sorted_zip_index[0]
                            curr_index = sorted_zip_index[0]
                        else:
                            break
                    if i % CKPT_POINT_LENGTH == 0:
                        print(f"Update the metadata list for {i} iterations")
                        pickle.dump((sorted_zip_index, curr_index), open(self.ckpt_pickle_location, 'wb+'))
                        self.criteria.save_metadata_list_as_pickle(metadata_list)
                        self.criteria.save_tag_pickle()
                        self._print_ckpt()
                self.criteria.save_metadata_list_as_pickle(metadata_list)
                self.criteria.save_tag_pickle()
        
        print(f"Finished parsing {len(metadata_list)} media objects.")
        return metadata_list
    
    def _print_ckpt(self):
        for tag in self.criteria.tag_dict:
            print(f"Tag {tag} has {len(self.criteria.tag_dict[tag])}/{self.criteria.max_images_per_tag} images")
    
    # def group_by_year_date_taken(self):
    #     return self._group(mode='year', date='date_taken')

    # def group_by_month_date_taken(self):
    #     return self._group(mode='month', date='date_taken')

    # def group_by_year_date_uploaded(self):
    #     self._group(mode='year', date='date_uploaded')

    # def group_by_month_date_uploaded(self):
    #     self._group(mode='month', date='date_uploaded')

    # def _group(self, mode='month', date='date_taken'): 
        # save_folder = os.path.join(self.save_folder, f"{date}_by_{mode}")
        # sub_folder_name_func =  lambda idx, bucket_name : f"{idx}-{bucket_name[0]}-{bucket_name[1]}" if mode == 'month' else f"{idx}-{bucket_name}"
        # save_pickle = os.path.join(save_folder, "dataset.pickle")
        # if os.path.exists(save_pickle):
        #     sorted_buckets_list, buckets_dict = pickle.load(open(save_pickle, 'rb'))
        #     print(f"Load from pickle: {save_pickle}")
        # else:
        #     sorted_buckets_list, buckets_dict = create_sorted_date_buckets(
        #                                             self.metadata_list,
        #                                             mode=mode,
        #                                             date=date
        #                                         )
        #     for idx, bucket_name in enumerate(sorted_buckets_list):
        #         # year, month = bucket_name
        #         save_subfolder = os.path.join(save_folder, sub_folder_name_func(idx, bucket_name))
        #         os.makedirs(save_subfolder)
        #         for meta in buckets_dict[bucket_name]:
        #             shutil.copy(meta.get_path(), save_subfolder)
        #     pickle.dump((sorted_buckets_list, buckets_dict), open(save_pickle, 'wb+'))
        #     print(f"Save folder: {save_folder}")
        #     print(f"Save pickle: {save_pickle}")
        
        # self._plot_buckets(save_folder, sorted_buckets_list, buckets_dict, mode=mode, date=date)
        # return sorted_buckets_list, buckets_dict
    
#     def _plot_buckets(self, save_folder, sorted_buckets_list, buckets_dict, mode='month', date='date_taken'):
#         save_png_barchart = os.path.join(save_folder, f"dataset_bucke_{date}.png")
#         save_png_freqchart = os.path.join(save_folder, f"dataset_frequency_{date}.png")
#         min_count = None
#         max_count = None
#         avg_count = 0
#         for b in sorted_buckets_list:
#             count = len(buckets_dict[b])
#             if not min_count or count < min_count:
#                 min_count = count
#             if not max_count or count > max_count:
#                 max_count = count
#             avg_count += count
#         avg_count = avg_count / len(sorted_buckets_list)
        
#         print(f"Min/Max number of images per bucket: {min_count}/{max_count}. Average is {avg_count}")
#         plt.figure(figsize=(8,8))
#         axes = plt.gca()
#         axes.set_ylim([0,max_count])
#         plt.title(f'Number of samples per {mode}.')
#         x = [str(a) for a in sorted_buckets_list]
#         y = [len(buckets_dict[b]) for b in sorted_buckets_list]
#         plt.bar(x, y, align='center')
#         plt.axhline(y=avg_count, label=f"Mean Number of Samples {avg_count}", linestyle='--', color='black')
#         x_tick = [str(a) for a in sorted_buckets_list]
#         plt.xticks(x, x_tick)
#         plt.xlabel('Date')
#         plt.ylabel(f'Number of samples for each {mode}')
#         plt.setp(axes.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='large')
#         # plt.legend()
#         plt.savefig(save_png_barchart)
#         plt.close('all')

        
#         bins = np.linspace(min_count, max_count, 15)
#         plt.figure(figsize=(8,8))
#         plt.hist(y, bins, alpha=0.5, label='Number of samples')
#         plt.legend(loc='upper right')
#         plt.tight_layout()
#         plt.savefig(save_png_freqchart)
#         print(f"Saving image at {save_png_freqchart}")

# def create_sorted_date_buckets(metadata_list, mode='year', date='date_taken'):
#     buckets_dict = {}
#     for meta in metadata_list:
#         if mode == 'year':
#             time_meta = getattr(meta, date)().year
#         elif mode == 'month':
#             time_meta = (getattr(meta, date)().year, getattr(meta, date)().month)
        
#         if time_meta in buckets_dict:
#             buckets_dict[time_meta].append(meta)
#         else:
#             buckets_dict[time_meta] = [meta]
        
#     sorted_buckets_list = sorted(buckets_dict.keys())
#     return sorted_buckets_list, buckets_dict

if __name__ == "__main__":
    args = argparser.parse_args()
    tag_groups = TAG_GROUPS_DICT[args.svm_tag_group]

    criteria = ImageBySVMTagList(args, args.svm_tag_group, save_info=True)
        
    flickr_parser = FlickrTagGroupParser(args, criteria)

    tag_parser = TagParser(args, criteria)

    tag_parser.generate_img_html()

    # # flickr_parser.group_by_month_date_taken()
    # if not args.all_images:
    #     flickr_parser.group_by_month_date_uploaded()
    #     flickr_parser.group_by_year_date_uploaded()

    #     flickr_parser.group_by_month_date_taken()
    #     flickr_parser.group_by_year_date_taken()

    
        
    