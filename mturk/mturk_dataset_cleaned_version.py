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
                       help="The original folder with query_dict.pickle")
argparser.add_argument("--new_folder_path",
                       default='/compute/autobot-1-1/zhiqiu/yfcc_dynamic_10',
                       help="The folder to save the images locally")
argparser.add_argument('--num_of_bucket', default=11, type=int,
                       help='number of bucket')
argparser.add_argument("--original_dataset",
                       default='dynamic_negative_600_dress_soccer',
                       help="The larger version of the dataset")
argparser.add_argument("--cleaned_dataset",
                    #    default='dynamic_negative_300_cleaned',
                    #    default='dynamic_negative_300_cleaned_bucket_1_only',
                    #    default='dynamic_negative_300_cleaned_bucket_1_and_10',
                    #    default='dynamic_negative_300_cleaned_bucket_1_2_10',
                    #    default='dynamic_negative_300_cleaned_bucket_temp',
                    #    default='dynamic_300_cleaned', # Result dict are still using best test model
                    #    default='dynamic_300', # Using best training loss model
                    #    default='dynamic_300_lbfgs',  # Using best training loss model
                       default='dynamic_300_positive_only',
                       help="The cleaned version of the dataset")
argparser.add_argument("--samples_per_class",
                       default=300,
                       type=int,
                       help="To pick this many random samples")


def get_csv_files(new_folder_path, num_of_bucket):
    csv_file_dict = {}
    for b_idx in range(num_of_bucket):
        csv_path = os.path.join(new_folder_path, "csv_files", f"final_cleaned_bucket_{b_idx}.csv")
        if os.path.exists(csv_path):
            csv_file_dict[b_idx] = csv_path
    return csv_file_dict

def get_query_dict_path(folder_path, dataset_name, bucket_num):
    return os.path.join(folder_path, f"bucket_{bucket_num}", dataset_name, 'query_dict.pickle')

def get_info_dict(folder_path, dataset_name, bucket_num):
    p = os.path.join(folder_path, f"bucket_{bucket_num}", dataset_name, 'info_dict.pickle')
    d = load_pickle(p)
    return d

def get_query_dict(folder_path, dataset_name, bucket_num):
    d = load_pickle(get_query_dict_path(folder_path, dataset_name, bucket_num))
    return d

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(path + " already exists.")


def get_query_dict_index_by_ID(query_dict):
    # Index by ID
    query_dict_index_by_ID = {b_idx : 
                            {query : {} for query in query_dict[b_idx]} 
                            for b_idx in query_dict}
    for b_idx in query_dict:
        for query in query_dict[b_idx]:
            for idx in range(len(query_dict[b_idx][query]['D'])):
                meta = query_dict[b_idx][query]['metadata'][idx]
                ID = meta.metadata.ID
                if ID in query_dict_index_by_ID[b_idx][query]:
                    import pdb; pdb.set_trace()
                query_dict_index_by_ID[b_idx][query][ID] = {
                    'clip_feature': query_dict[b_idx][query]['clip_features'][idx],
                    'metadata': query_dict[b_idx][query]['metadata'][idx],
                    'D': query_dict[b_idx][query]['D'][idx]
                }
    return query_dict_index_by_ID


def gather_new_query_dict(csv_file_dict, query_dict_index_by_ID, samples_per_class, image_folder_path):
    for b_idx in query_dict_index_by_ID:
        if not b_idx in csv_file_dict:
            print(f"{b_idx} bucket not in csv_file_dict!!!!")
        else:
            print(f"{b_idx} bucket exists!")
    new_query_dict = {b_idx:
                        {query: 
                            {'clip_features' : [], 'metadata' : [], 'D' : []} 
                        for query in query_dict_index_by_ID[b_idx]}
                      for b_idx in csv_file_dict}
    for b_idx in csv_file_dict:
        with open(csv_file_dict[b_idx], newline='\n') as cleaned_file:
            csv_reader = csv.DictReader(cleaned_file)
            headers = csv_reader.fieldnames
            assert 'ID' in headers and 'query' in headers and 'bucket_index' in headers
            for row in csv_reader:
                query = row['query']
                ID = row['ID']
                assert int(row['bucket_index']) == b_idx
                new_query_dict[b_idx][query]['clip_features'].append(query_dict_index_by_ID[b_idx][query][ID]['clip_feature'])
                new_query_dict[b_idx][query]['metadata'].append(query_dict_index_by_ID[b_idx][query][ID]['metadata'])
                new_query_dict[b_idx][query]['D'].append(query_dict_index_by_ID[b_idx][query][ID]['D'])
        for query in new_query_dict[b_idx]:
            if len(new_query_dict[b_idx][query]['metadata']) < samples_per_class:
                import pdb; pdb.set_trace()
            sorted_indices = [idx for idx, score in sorted(enumerate(new_query_dict[b_idx][query]['D']), key=lambda x : x[1], reverse=True)]
            new_query_dict[b_idx][query]['metadata'] = [new_query_dict[b_idx][query]['metadata'][idx] for idx in sorted_indices[:samples_per_class]]
            new_query_dict[b_idx][query]['D'] = [new_query_dict[b_idx][query]['D'][idx] for idx in sorted_indices[:samples_per_class]]
            new_query_dict[b_idx][query]['clip_features'] = [new_query_dict[b_idx][query]['clip_features'][idx].reshape(1,-1) for idx in sorted_indices[:samples_per_class]]
            new_query_dict[b_idx][query]['clip_features'] = np.concatenate(new_query_dict[b_idx][query]['clip_features'], axis=0)
    
    for b_idx in new_query_dict:
        image_folder_b = os.path.join(image_folder_path, f"bucket_{b_idx}")
        makedirs(image_folder_b)
        for query in new_query_dict[b_idx]:
            image_folder_b_query = os.path.join(image_folder_b, query)
            makedirs(image_folder_b_query)
            for item_idx, item in enumerate(new_query_dict[b_idx][query]['metadata']):
                old_path = item.metadata.IMG_PATH
                ID = item.metadata.ID
                # if ID == '11427206726':
                #     print(
                #          f" 11427206726 is at (idx {item_idx})?")
                #     import pdb; pdb.set_trace()
                EXT = item.metadata.EXT
                new_name = str(ID) + "." + EXT
                new_path = os.path.join(image_folder_b_query, new_name)
                try:
                    shutil.copy(old_path, new_path)
                except:
                    # print(old_path + f" is the same (idx {item_idx})?")
                    import pdb; pdb.set_trace()
                item.metadata.IMG_PATH = new_path
                item.metadata.IMG_DIR = image_folder_b_query
    return new_query_dict

if __name__ == "__main__":
    args = argparser.parse_args()
    start = time.time()

    original_query_dict = get_query_dict(args.folder_path, args.original_dataset, args.num_of_bucket)
    # original_query_dict is a dict with: 
    #   original_query_dict[bucket_index][query]['clip_features'] is a list of clip features in numpy
    #   original_query_dict[bucket_index][query]['metadata'] is a list of Metadata object
    #   original_query_dict[bucket_index][query]['D'] is a list of scores
    original_query_dict_index_by_ID = get_query_dict_index_by_ID(original_query_dict)
    # original_query_dict_index_by_ID[bucket_index][query][ID] is a dict with:
    #   'clip_features' is a clip features in numpy
    #   'metadata' is a Metadata object
    #   'D' is a float scores

    # original_info_dict = get_info_dict(args.folder_path, args.original_dataset, args.num_of_bucket)
    new_dataset_path = os.path.join(args.new_folder_path, args.cleaned_dataset)
    new_image_folder_path = os.path.join(new_dataset_path, "images")
    new_query_dict_path = os.path.join(new_dataset_path, "query_dict.pickle")
    
    makedirs(new_dataset_path)

    csv_file_dict = get_csv_files(args.new_folder_path, args.num_of_bucket)
    new_query_dict = gather_new_query_dict(csv_file_dict, original_query_dict_index_by_ID, args.samples_per_class, new_image_folder_path)
    save_obj_as_pickle(new_query_dict_path, new_query_dict)
