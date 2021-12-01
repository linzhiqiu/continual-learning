# A script to prepare labeled/unlabeled dataset files (in csv format)
from io import BytesIO
import os
import json
import time
import argparse
from dataclasses import dataclass
import time
from tqdm import tqdm
import pickle
from datetime import datetime
from dateutil import parser
import shutil
import csv
from utils import save_obj_as_pickle, load_pickle

argparser = argparse.ArgumentParser()
argparser.add_argument("--folder_path", 
                        default='/data3/zhiqiul/images_minbyte_10_valid_uploaded_date_feb_18',
                        help="The image folder with bucket_{num_of_bucket}.pickle")
argparser.add_argument("--labeled_data_path", 
                        default='/data3/zhiqiul/yfcc_dynamic_10/dynamic_300/query_dict.pickle',
                        help="The query_dict pickle file for labeled datasets")
argparser.add_argument("--save_folder_path", 
                        default='/data3/zhiqiul/clear_10_public_new/',
                        help="The folder to save the output csv files")

# All metadata entries
METADATA = [
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
    "AUTO_TAG_SCORES",
]

# The additional column list in the output csv files (for unlabeled data)
UNLABELED_CSV_ENTRIES = ["BUCKET_INDEX"]

# The additional column list in the output csv files (for labeled data)
LABELED_CSV_ENTRIES = ['VISUAL_CONCEPT', "BUCKET_INDEX"]

def write_to_csv(save_path,
                 metadata_lst,
                 prefix_lst,
                 list_of_prefix_names=[],
                 list_of_metadata=METADATA):
    assert len(metadata_lst) == len(prefix_lst)
    assert len(prefix_lst[0]) == len(list_of_prefix_names)
    
    with open(save_path, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list_of_prefix_names+list_of_metadata)
        writer.writeheader()
        for metadata, prefix in zip(metadata_lst, prefix_lst):
            metadata_as_dict = {name : metadata.__dict__[name] for name in list_of_metadata}
            prefix_as_dict = {list_of_prefix_names[i] : prefix[i] for i in range(len(list_of_prefix_names))}
            prefix_as_dict.update(metadata_as_dict)
            writer.writerow(prefix_as_dict)
    print(f"Finish writing {len(metadata_lst)} rows to {save_path}")

def prepare_csv(args):
    query_dict = load_pickle(args.labeled_data_path)
    num_of_bucket = len(query_dict.keys()) + 1
    bucket_dict_path = os.path.join(args.folder_path, f"bucket_{num_of_bucket}.pickle")
    assert os.path.exists(bucket_dict_path)
    bucket_dict = load_pickle(bucket_dict_path)
    assert len(bucket_dict.keys()) == num_of_bucket
    
    if not os.path.exists(args.save_folder_path):
        os.makedirs(args.save_folder_path)

    for bucket_idx in range(num_of_bucket):
        save_dir = os.path.join(args.save_folder_path, str(bucket_idx))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # work on all data
        all_save_path = os.path.join(save_dir, 'all.csv')
        unlabeled_prefix_lst = [[str(bucket_idx)] for i in bucket_dict[bucket_idx]['flickr_accessor']]
        write_to_csv(
            all_save_path,
            bucket_dict[bucket_idx]['flickr_accessor'],
            unlabeled_prefix_lst,
            list_of_prefix_names=UNLABELED_CSV_ENTRIES
        )

        if bucket_idx in query_dict:
            # work on labeled data
            labeled_save_path = os.path.join(save_dir, "labeled.csv")

            labeled_prefix_lst = []
            labeled_metadata_lst = []
            for class_name in query_dict[bucket_idx]:
                labeled_metadata_lst += query_dict[bucket_idx][class_name]['metadata']
                labeled_prefix_lst += [[class_name, str(bucket_idx)] for _ in query_dict[bucket_idx][class_name]['metadata']]
            
            write_to_csv(
                labeled_save_path,
                labeled_metadata_lst,
                labeled_prefix_lst,
                list_of_prefix_names=LABELED_CSV_ENTRIES
            )

            

if __name__ == '__main__':
    args = argparser.parse_args()
    prepare_csv(args)