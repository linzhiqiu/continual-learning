# After running verify_query_dict_inclusive.py to get the large dataset, this is to prepare
# the csv file only for small query dict

import os
import time
import argparse
import shutil

import math
import time
running_time = time.time()

import csv
from PIL import Image
from tqdm import tqdm
import pickle
import random

import numpy as np
from training_utils import get_imgnet_transforms, get_unnormalize_func
from utils import save_obj_as_pickle, load_pickle, normalize, divide

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import torchvision
device = "cpu"
BATCH_SIZE = 1

argparser = argparse.ArgumentParser()
argparser.add_argument("--folder_path",
                       default='/compute/autobot-1-1/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/clip_dataset_negative',
                       help="The folder with query_dict.pickle")
argparser.add_argument("--save_path",
                       default='/project_data/ramanan/zhiqiu/yfcc_aws_feb_18',
                       help="The folder to save the images locally")
argparser.add_argument("--s3_save_path",
                       default='https://yfcc-clip-dynamic-10.s3.us-east-2.amazonaws.com',
                       help="The link in csv to show the images on s3")
argparser.add_argument('--num_of_bucket', default=11, type=int,
                       help='number of bucket')
argparser.add_argument("--smaller_dataset",
                       default='dynamic_negative_300',
                       help="The smaller version of the dataset")
argparser.add_argument("--larger_dataset",
                       default='dynamic_negative_600_v2',
                       help="The larger version of the dataset")
argparser.add_argument("--csv_length",
                       default=None, type=int,
                       help="If None, save all; else save this many")


def get_query_dict(folder_path, dataset_name, bucket_num):
    d = load_pickle(os.path.join(folder_path, f"bucket_{bucket_num}", dataset_name, 'query_dict.pickle'))
    return d

if __name__ == "__main__":
    args = argparser.parse_args()
    start = time.time()

    small_query_dict = get_query_dict(args.folder_path, args.smaller_dataset, args.num_of_bucket)
    large_query_dict = get_query_dict(args.folder_path, args.larger_dataset, args.num_of_bucket)
    
    
    inv_normalize = get_unnormalize_func()
    _, preprocess = get_imgnet_transforms()
    main_save_dir_large = os.path.join(args.save_path, args.larger_dataset)

    cropped_dir_large = os.path.join(main_save_dir_large, "cropped")
    original_dir_large = os.path.join(main_save_dir_large, "original")
    if not os.path.exists(main_save_dir_large):
        exit(0)
    
    query_dict_large = {'cropped' : {}, 'original' : {}}
    
    if os.path.exists(os.path.join(main_save_dir_large, "cropped.pickle")) and os.path.exists(os.path.join(main_save_dir_large, "original.pickle")):
        query_dict_large['cropped'] = load_pickle(os.path.join(main_save_dir_large, "cropped.pickle"))
        query_dict_large['original'] = load_pickle(os.path.join(main_save_dir_large, "original.pickle"))
    else:
        exit(0)
    
    # import pdb; pdb.set_trace()
    # store small query dict
    query_dict_small = {'cropped' : {}, 'original' : {}}
    for b_idx in range(args.num_of_bucket):
        cropped_dir_b = os.path.join(cropped_dir_large, str(b_idx))
        original_dir_b = os.path.join(original_dir_large, str(b_idx))
        for query in large_query_dict[b_idx]:
            cropped_dir_b_query = os.path.join(cropped_dir_b, query)
            original_dir_b_query = os.path.join(original_dir_b, query)
            assert os.path.exists(cropped_dir_b_query)
            assert os.path.exists(original_dir_b_query)
            for item_idx, item in enumerate(small_query_dict[b_idx][query]['metadata']):
                ID = item.metadata.ID
                if ID in query_dict_large['cropped']:
                    query_dict_small['cropped'][ID] = query_dict_large['cropped'][ID] 
                    query_dict_small['original'][ID] = query_dict_large['original'][ID]
                else:
                    print(f"{ID} not found.")
    
    # save_obj_as_pickle(os.path.join(main_save_dir, "cropped.pickle"), query_dict['cropped'])
    # save_obj_as_pickle(os.path.join(main_save_dir, "original.pickle"), query_dict['original'])
    # print(f"saved at {main_save_dir}")

    total_length = list(query_dict_small['original'].keys()).__len__()
    indices = [i for i in range(total_length)]
    random.shuffle(indices)
    if args.csv_length:
        csv_length = args.csv_length
    else:
        csv_length = total_length
    len_str = str(csv_length)
    indices = indices[:csv_length]
    csv_path_cropped = os.path.join(main_save_dir_large, f"cropped_{len_str}_300_per_bucket.csv")
    csv_path_original = os.path.join(main_save_dir_large, f"original_{len_str}_300_per_bucket.csv")

    headers = ['image_url', 'ID', 'bucket_index', 'query']
    for csv_path, q_dict_key in [(csv_path_cropped, 'cropped'), (csv_path_original, 'original')]:
        q_dict = query_dict_small[q_dict_key]
        s3_save_dir = args.s3_save_path + "/" + args.larger_dataset + "/" + q_dict_key
        csv_dicts = []
        for ID in q_dict:
            b_idx, query, item_idx = q_dict[ID]['key']
            path = q_dict[ID]['path']
            file_name = path[path.rfind(os.sep)+1:]
            s3_path = s3_save_dir + "/" + str(b_idx) + "/" + query + "/" + file_name
            csv_dicts.append({
                'image_url' : s3_path.replace(" ", "+"),
                'ID' : str(ID),
                'bucket_index' : str(b_idx),
                'query' : query,
            })
        
        csv_dicts = [csv_dicts[i] for i in indices]
        with open(csv_path, 'w', newline='\n') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for csv_dict in csv_dicts:
                writer.writerow(csv_dict)
        print(f"Write at {csv_path}")
