# A script to parse flickr datasets/autotags
# python prepare_dataset.py --fix_bug True;
# python prepare_dataset.py --print_html True;
# python prepare_dataset.py --min_size 10 --svm_tag_group tech; 
# python prepare_dataset.py --min_size 10 --svm_tag_group text; 
# python prepare_dataset.py --min_size 10 --svm_tag_group vehicle; 
# python prepare_dataset.py --min_size 10 --svm_tag_group ballgame; 
# python prepare_dataset.py --min_size 10 --svm_tag_group fashion; 
# python prepare_dataset.py --min_size 10 --svm_tag_group event; 
# python prepare_dataset.py --min_size 10 --svm_tag_group sports; 
# python prepare_dataset.py --min_size 10 --svm_tag_group people; 
# python prepare_dataset.py --min_size 10 --svm_tag_group indoor; 
# python prepare_dataset.py --min_size 10 --svm_tag_group urban; 
# python prepare_dataset.py --min_size 10 --svm_tag_group time; 
# python prepare_dataset.py --min_size 10 --svm_tag_group outdoor; 
# python prepare_dataset.py --min_size 10 --svm_tag_group weather; 
# python prepare_dataset.py --min_size 10 --svm_tag_group animal;
# python prepare_dataset.py --min_size 10 --svm_tag_group things;


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


import matplotlib.pyplot as plt
import numpy as np

CKPT_POINT_LENGTH = 10000 # Save every CKPT_POINT_LENGTH metadata

# |-- ./dataset/small_datasets
#   |-- args.svm_tag_group
#     |-- info.txt (about the parameter used)
#     |-- images.pickle (all images metadata)
#     |-- tags.pickle (all images with each tag)
#     |-- images
#       |-- image_1
#       |-- image_2
#       |-- ...
#     |-- html
#       
from tag_analysis import TagParser
from group_by_svm_tags import ImageBySVMTagList, TAG_GROUPS_DICT, argparser


argparser.add_argument("--excluded_id", 
                        default='excluded_id',
                        help="The filename under download_dir with image ID to be excluded")
# argparser.add_argument("--download_dir", 
#                         default='./dataset/small_datasets',
#                         help="The download location")
# argparser.add_argument("--svm_tag_group",
#                         type=str, default=None, choices=TAG_GROUPS_DICT.keys(),
#                         help="A SVM tag group.")
# argparser.add_argument("--min_prob",
#                         type=float, default=0.75,
#                         help="Images with probability score smaller than args.min_prob associated with the tag will be ignored.")
# argparser.add_argument("--max_images_per_tag",
#                         type=int, default=800,
#                         help="The maximum images to store for each tag")
# argparser.add_argument("--max_aspect_ratio",
#                         type=float, default=2.0,
#                         help="Images with max/min edge greater than max_aspect_ratio will be ignored.")
# argparser.add_argument("--min_size",
#                         type=int, default=10,
#                         help="Images with size smaller than min_size will be ignored.")
# argparser.add_argument("--use_valid_date",
#                         type=bool, default=True,
#                         help="Images with valid date (upload date < taken date) will be used if set true")
argparser.add_argument("--print_html",
                        type=bool, default=False,
                        help="HTML to print.")
argparser.add_argument("--fix_bug",
                        type=bool, default=False,
                        help="HTML to print.")
                        
                        
def load_id(filename):
    with open(filename) as f:
        lines = [l for l in f.read().splitlines()]
        ids = []
        for l in lines:
            try:
                ids.append(int(l))
            except:
                continue
    return ids

def remove_excluded_id_from_tag_dict(tag_dict, excluded_ids):
    for idx in excluded_ids:
        for tag in tag_dict:
            new_tag_list = []
            for meta in tag_dict[tag]:
                if int(meta.get_metadata().ID) != idx:
                    new_tag_list.append(meta)
            tag_dict[tag] = new_tag_list
    return tag_dict

def get_tag_dict_for_training(args, tag_group, tag_keys, img_transfer_dir="/scratch/zhiqiu/", do_transfer=True):
    criteria = ImageBySVMTagList(args, tag_group)
    excluded_ids = load_id(os.path.join(criteria.save_main_folder, args.excluded_id))
    computed_tag_dict = criteria.sync()
    computed_tag_dict = remove_excluded_id_from_tag_dict(computed_tag_dict, excluded_ids)
    tag_dict = {}
    transferred_count = 0
    for tag in tag_keys:
        tag_dict[tag] = computed_tag_dict[tag]
        if do_transfer:
            for meta in tag_dict[tag]:
                img_path = os.path.abspath(os.path.join(criteria.save_folder, meta.get_path()))
                shutil.copy(img_path, img_transfer_dir)
                transferred_count += 1
                transferred_path = os.path.join(img_transfer_dir, meta.get_path())
                meta.get_metadata().IMG_PATH = transferred_path
    if do_transfer:
        print(f"Transferred {transferred_count} images to {img_transfer_dir}")
    return tag_dict

if __name__ == "__main__":
    args = argparser.parse_args()
    if args.print_html:
        for tag_group in TAG_GROUPS_DICT:
            try:
                criteria = ImageBySVMTagList(args, tag_group)
                try:
                    excluded_ids = load_id(os.path.join(criteria.save_main_folder, args.excluded_id))
                except:
                    excluded_ids = None
                    print(f"No excluded ids for {tag_group}")
            except:
                print(f"{tag_group} has nothing.")
                continue
            tag_groups = TAG_GROUPS_DICT[tag_group]
            try:
                tag_parser = TagParser(args, criteria, absolute_path=False, all_svm_tag_names=tag_groups)
            except NotImplementedError:
                print(f"No images downloaded yet for {tag_group}")
                continue
            computed_tag_dict = criteria.sync()
            tag_parser.generate_tag_dict_html(computed_tag_dict, conf_threshold=criteria.min_prob, excluded_ids=excluded_ids)
    elif args.fix_bug:
        for tag_group in TAG_GROUPS_DICT:
            try:
                criteria = ImageBySVMTagList(args, tag_group)
            except:
                print(f"{tag_group} has nothing.")
                continue
            if criteria.get_metadata_pickle() == []:
                tag_dict = criteria.load_tag_dict()
                if tag_dict != None:
                    metadata_list = []
                    for tag in tag_dict:
                        metadata_list += tag_dict[tag]
                    print(f"{tag_group} Meta data list has length {len(metadata_list)}")
                    
                    from tag_analysis import remove_duplicate
                    metadata_list = remove_duplicate(metadata_list)
                    print(f"After removing duplicate: length is {len(metadata_list)}")
                    criteria.save_metadata_list_as_pickle(metadata_list)
                    # import pdb; pdb.set_trace() 
    else:
        for tag_group in TAG_GROUPS_DICT:
            try:
                criteria = ImageBySVMTagList(args, tag_group)
            except:
                print(f"{tag_group} has nothing.")
                continue
            tag_dict = criteria.load_tag_dict()
        
            excluded_ids = load_id(os.path.join(criteria.get_save_folder(), args.excluded_id))

            tag_dict = remove_excluded_id_from_tag_dict(tag_dict, excluded_ids)

            for tag in tag_dict:
                print(f"Tag {tag} has {len(tag_dict[tag])}/{criteria.max_images_per_tag} images")
            
            plot_dir = os.path.join(criteria.get_save_folder(), "date_plots")
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
    
    

    # # flickr_parser.group_by_month_date_taken()
    # if not args.all_images:
    #     flickr_parser.group_by_month_date_uploaded()
    #     flickr_parser.group_by_year_date_uploaded()

    #     flickr_parser.group_by_month_date_taken()
    #     flickr_parser.group_by_year_date_taken()

    
        
    