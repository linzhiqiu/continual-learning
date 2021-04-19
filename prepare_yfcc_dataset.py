# The script to run before moco/main_yfcc.py
# python prepare_yfcc_dataset.py --num_of_bucket 11 --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18 

import sys
import os
from faiss_utils import get_flickr_folder
import time
from datetime import datetime

from large_scale_feature import FlickrAccessor, FlickrFolder, get_flickr_accessor
import argparse
import importlib
from utils import divide, load_pickle, save_obj_as_pickle

argparser = argparse.ArgumentParser()
argparser.add_argument("--folder_path", 
                        # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_25',
                        # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_31',
                        # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_16',
                        default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18',
                        help="The folder with all_folders.pickle and features.pickle")
argparser.add_argument('--num_of_bucket', default=11, type=int,
                       help='number of bucket')

def get_bucket_folder_paths(folder_path, num_of_bucket):
    sub_folder_paths = []
    for b_idx in range(num_of_bucket):
        sub_folder_path = os.path.join(folder_path, f'bucket_{num_of_bucket}', f'{b_idx}')
        sub_folder_paths.append(sub_folder_path)
        if not os.path.exists(sub_folder_path):
            os.makedirs(sub_folder_path)
    return sub_folder_paths

def get_date_uploaded(DATE_UPLOADED):
    return datetime.utcfromtimestamp(int(DATE_UPLOADED))

if __name__ == '__main__':
    args = argparser.parse_args()

    start = time.time()
    flickr_folders_path = get_flickr_folder(args.folder_path)
    flickr_folders = load_pickle(flickr_folders_path)
    flickr_accessor = FlickrAccessor(flickr_folders)
    print(f"Size of dataset is {len(flickr_accessor)}")
    end = time.time()
    print(f"{end - start} seconds are used to load all {len(flickr_accessor)} image metadata")
    date_uploaded_list = []
    for meta in flickr_accessor:
        metadata_obj = meta.get_metadata()
        date_uploaded_list.append(metadata_obj.DATE_UPLOADED)
    date_uploaded_indices = [i[0] for i in sorted(enumerate(date_uploaded_list), key=lambda x : x[1])]
    sorted_indices_chunks = divide(date_uploaded_indices, args.num_of_bucket)
    
    bucket_dict = {}
    folder_paths = get_bucket_folder_paths(args.folder_path, args.num_of_bucket)
    for i, sorted_indices_chunk in enumerate(sorted_indices_chunks):
        min_date, max_date = date_uploaded_list[sorted_indices_chunk[0]], date_uploaded_list[sorted_indices_chunk[-1]]
        date_str = f"For bucket {i}: Date range from {get_date_uploaded(min_date)} to {get_date_uploaded(max_date)}"
        print(date_str)
        
        meta_list = [flickr_accessor[i] for i in sorted_indices_chunk]
        
        bucket_dict[i] = {
            'indices' : sorted_indices_chunk,
            # 'normalized_feature_path' : clip_norm_features_location,
            'flickr_accessor' : meta_list,
            'folder_path' : folder_paths[i],
            'min_date': get_date_uploaded(min_date),
            'max_date': get_date_uploaded(max_date),
            'date_uploaded_list' : [date_uploaded_list[i] for i in sorted_indices_chunk]
        }
        save_obj_as_pickle(os.path.join(folder_paths[i], f'bucket_{i}.pickle'), bucket_dict[i])
    
    save_obj_as_pickle(os.path.join(args.folder_path, f'bucket_{args.num_of_bucket}.pickle'), bucket_dict)
