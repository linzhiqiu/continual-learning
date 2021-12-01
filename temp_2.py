# Description:
# (1) Then split the image stream to buckets sorted by upload time
# (2) Finally generate and save the CLIP features (normalized + unnormalized)

# Version 1 - Split by year, e.g., 2004, 2005, 2006,..., 2014
#   Split the data stream in original --img_dir
    #   python temp_2.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --split_by_year True --model_name RN50 
    #   python temp_2.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --split_by_year True --model_name RN50x4
    #   python temp_2.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --split_by_year True --model_name RN101
    #   python temp_2.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --num_of_bucket 11 --model_name RN50 
    #   python temp_2.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --num_of_bucket 11 --model_name RN50x4
    #   python temp_2.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --num_of_bucket 11 --model_name RN101

    #   python temp_2.py --b /scratch/zhiqiu/yfcc100m_all_new_sep_21/images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0 --split_by_year True --model_name RN50 
    #   python temp_2.py --b /scratch/zhiqiu/yfcc100m_all_new_sep_21/images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0 --split_by_year True --model_name RN50x4
    #   python temp_2.py --b /scratch/zhiqiu/yfcc100m_all_new_sep_21/images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0 --split_by_year True --model_name RN101
    #   python temp_2.py --b /scratch/zhiqiu/yfcc100m_all_new_sep_21/images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0 --num_of_bucket 11 --model_name RN50 
    #   python temp_2.py --b /scratch/zhiqiu/yfcc100m_all_new_sep_21/images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0 --num_of_bucket 11 --model_name RN50x4
    #   python temp_2.py --b /scratch/zhiqiu/yfcc100m_all_new_sep_21/images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0 --num_of_bucket 11 --model_name RN101

# Version 2 - Split to equal-sized buckets
#   Split the data stream in original --img_dir
    #   python temp_2.py --b /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_31/ --num_of_bucket 11 --model_name RN50 
    #   python temp_2.py --b /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_31/ --num_of_bucket 11 --model_name RN50x4
    #   python temp_2.py --b /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_31/ --num_of_bucket 11 --model_name RN101
    #   python temp_2.py --b /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --model_name RN50 
    #   python temp_2.py --b /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --model_name RN50x4
    #   python temp_2.py --b /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --model_name RN101


import os
import time

import argparse

from datetime import datetime
from PIL import Image
from tqdm import tqdm
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import sys
sys.path.append("./CLIP")
from faiss_utils import KNearestFaissFeatureChunks
import clip
from yfcc_download import get_save_folder, get_all_metadata
from temp import argparser, get_flickr_accessor, FlickrFolder, Metadata, MetadataObject
from utils import divide, load_pickle, save_obj_as_pickle, normalize, save_as_json
import prepare_dataset

MAX_SIZE = 500000 # The maximum number of features to store in a single file. You may adjust it according to your CPU memory.
BATCH_SIZE = 128 # The batch size used for extracting features. Adjust it according to your GPU memory
device = "cuda"

# # Continue the argparser in yfcc_download
# /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18
argparser.add_argument('--b', default=None)
argparser.add_argument('--num_of_bucket', default=11, type=int,
                       help='If not split the bucket according to year, then split image into this many (equal size) bucket, sorted by time. ')
argparser.add_argument('--split_by_year', default=False, type=bool,
                       help='if set to True, ignore --num_of_bucket and sort by year. ')
argparser.add_argument("--model_name",
                       default='RN50', choices=clip.available_models(),
                       help="The CLIP model architecture to use")

def get_knearest_models_func(folder_path, clip_model_name, num_of_bucket, model, preprocess):
    folder_paths = get_bucket_folder_paths(folder_path, num_of_bucket)
    def knearest_func(bucket_idx):
        clip_features_normalized_paths = get_clip_features_normalized_paths(folder_paths[bucket_idx], clip_model_name)
        k_near_faiss = KNearestFaissFeatureChunks(clip_features_normalized_paths, model, preprocess)
        return k_near_faiss
    return knearest_func

def get_clip_features_normalized_paths(f_path, model_name):
    """A helper function to return paths to normalized ·clip features
    """
    clip_features_normalized_paths = []
    main_pickle_location = get_main_save_location(f_path, model_name)
    print(main_pickle_location)
    if os.path.exists(main_pickle_location):
        chunks, path_dict_list = load_pickle(main_pickle_location)
        print(f"Loaded from {main_pickle_location}")
    else:
        import pdb; pdb.set_trace()

    for chunk, path_dict in zip(chunks, path_dict_list):
        if os.path.exists(path_dict['normalized']):
            print(f"Already exists: {path_dict['normalized']}")
            clip_features_normalized_paths.append(path_dict['normalized'])
        else:
            import pdb; pdb.set_trace()
    return clip_features_normalized_paths

def get_bucket_folder_paths(folder_path, num_of_bucket, split_by_year=False):
    sub_folder_paths = []
    if split_by_year:
        for year_idx in range(2004, 2015):
            year_str = str(year_idx)
            sub_folder_path = os.path.join(folder_path, f'bucket_by_year', f'{year_str}')
            sub_folder_paths.append(sub_folder_path)
            if not os.path.exists(sub_folder_path):
                os.makedirs(sub_folder_path)
    else:
        for b_idx in range(num_of_bucket):
            sub_folder_path = os.path.join(folder_path, f'bucket_{num_of_bucket}', f'{b_idx}')
            sub_folder_paths.append(sub_folder_path)
            if not os.path.exists(sub_folder_path):
                os.makedirs(sub_folder_path)
    return sub_folder_paths

def get_main_save_location(folder_path, model_name):
    main_save_location = os.path.join(
        folder_path, f"features_{model_name.replace(os.sep, '_')}.pickle")
    return main_save_location


def _get_date_uploaded(date_uploaded):
    return datetime.utcfromtimestamp(int(date_uploaded))

def _divide_meta_list(bucket_dict_i, sub_folder, MAX_SIZE=MAX_SIZE):
    # Divide the metadata list into chunks with maximum size being MAX_SIZE
    # Return a list of chunks and save path for each dict
    meta_list = bucket_dict_i['flickr_accessor']
    sub_features_size = math.ceil(len(meta_list) / MAX_SIZE)
    names = [os.path.join(sub_folder, f'features_{i}') for i in range(sub_features_size)]
    chunks = divide(meta_list, sub_features_size)
    assert len(chunks) == len(names)
    return chunks, names


def _get_sub_feature_paths(bucket_dict_i, folder_path, main_save_location, model_name, MAX_SIZE=MAX_SIZE):
    sub_folder = os.path.join(folder_path, f"features_{model_name.replace(os.sep, '_')}")
    # if not os.path.exists(sub_folder):
    #     os.makedirs(sub_folder)
    chunks, names = _divide_meta_list(bucket_dict_i, sub_folder, MAX_SIZE=MAX_SIZE)
    
    path_dict_list = [{'original': n+"_original.pickle",
                       "normalized": n+"_normalized.pickle"} for n in names]
    # save_obj_as_pickle(main_save_location, (chunks, path_dict_list))
    return chunks, path_dict_list

def _new_divide_meta_list(bucket_dict_i, sub_folder, MAX_SIZE=MAX_SIZE):
    # Divide the metadata list into chunks with maximum size being MAX_SIZE
    # Return a list of chunks and save path for each dict
    meta_list = bucket_dict_i['all_metadata']
    sub_features_size = math.ceil(len(meta_list) / MAX_SIZE)
    names = [os.path.join(sub_folder, f'features_{i}') for i in range(sub_features_size)]
    chunks = divide(meta_list, sub_features_size)
    assert len(chunks) == len(names)
    return chunks, names

def _new_get_sub_feature_paths(bucket_dict_i, folder_path, main_save_location, model_name, MAX_SIZE=MAX_SIZE):
    sub_folder = os.path.join(folder_path, f"features_{model_name.replace(os.sep, '_')}")
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    chunks, names = _new_divide_meta_list(bucket_dict_i, sub_folder, MAX_SIZE=MAX_SIZE)
    
    path_dict_list = [{'original': n+"_original.npy",
                       "normalized": n+"_normalized.npy"} for n in names]
    save_as_json(main_save_location, (chunks, path_dict_list))
    return chunks, path_dict_list

class CLIPDataset(Dataset):
    def __init__(self, flickr_accessor, preprocess, device='cuda'):
        self.flickr_accessor = flickr_accessor
        self.device = device
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.flickr_accessor)
    
    def __getitem__(self,index):
        meta = self.flickr_accessor[index]
        path = os.path.join(meta['IMG_DIR'], meta['IMG_PATH'])
        sample = self.preprocess(Image.open(path)).to(self.device)
        return sample

def get_clip_loader(flickr_accessor, preprocess, batch_size=BATCH_SIZE, num_workers=0, device='cuda', dataset_class=CLIPDataset):
    return torch.utils.data.DataLoader(
        dataset_class(flickr_accessor, preprocess, device=device), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
    )

def get_clip_features(clip_loader, model):
    clip_features = []
    pbar = tqdm(clip_loader)
    with torch.no_grad():
        for batch, images in enumerate(pbar):
            image_features = model.encode_image(images)
            clip_features.append(image_features.cpu().numpy())
    return np.concatenate(clip_features, axis=0)

def load_bucket_dict(flickr_folder_location, num_of_bucket, split_by_year=False):
    if split_by_year:
        bucket_dict_path = os.path.join(flickr_folder_location, f'bucket_by_year.pickle')
    else:
        bucket_dict_path = os.path.join(flickr_folder_location, f'bucket_{num_of_bucket}.pickle')
    bucket_dict = load_pickle(bucket_dict_path)
    return bucket_dict

def save_bucket_dict(flickr_folder_location, flickr_accessor, folder_paths, num_of_bucket, split_by_year):
    assert num_of_bucket == len(folder_paths)
    # Sort images by time, and then split into buckets
    date_uploaded_list = []
    for meta in flickr_accessor:
        date_uploaded_list.append(meta.DATE_UPLOADED)
    indices_sorted_by_upload = [i[0] for i in sorted(enumerate(date_uploaded_list), key=lambda x : x[1])]
    # import pdb; pdb.set_trace()
    if split_by_year:
        bucket_dict_path = os.path.join(flickr_folder_location, f'bucket_by_year.pickle')
        chunks_of_indices = []
        curr_chunk = []
        year_idx = 2004
        for sorted_idx in indices_sorted_by_upload:
            if _get_date_uploaded(date_uploaded_list[sorted_idx]).year < year_idx:
                continue
            if _get_date_uploaded(date_uploaded_list[sorted_idx]).year > year_idx:
                year_idx += 1
                chunks_of_indices.append(curr_chunk)
                curr_chunk = []
            if not _get_date_uploaded(date_uploaded_list[sorted_idx]).year == year_idx:
                import pdb; pdb.set_trace()
            curr_chunk.append(sorted_idx)
        chunks_of_indices.append(curr_chunk)
    else:
        bucket_dict_path = os.path.join(flickr_folder_location, f'bucket_{num_of_bucket}.pickle')
        chunks_of_indices = divide(indices_sorted_by_upload, num_of_bucket)
    
    bucket_dict = {}
    for i, chunk in enumerate(chunks_of_indices):
        bucket_dict_i_path = os.path.join(folder_paths[i], f'bucket_{i}.pickle')
        if os.path.exists(bucket_dict_i_path):
            bucket_dict[i] = load_pickle(bucket_dict_i_path)
        else:
            meta_list = [flickr_accessor[i] for i in chunk]
            date_uploaded_list_i = [date_uploaded_list[i] for i in chunk]
            bucket_dict[i] = {
                'indices' : chunk,
                'flickr_accessor' : meta_list,
                'folder_path' : folder_paths[i],
                'min_date': _get_date_uploaded(date_uploaded_list_i[0]),
                'max_date': _get_date_uploaded(date_uploaded_list_i[-1]),
                'date_uploaded_list' : date_uploaded_list_i
            }
            # save_obj_as_pickle(bucket_dict_i_path, bucket_dict[i])
        min_date, max_date = bucket_dict[i]['min_date'], bucket_dict[i]['max_date']
        date_str = f"For bucket {i}: Date range from {min_date} to {max_date}"
        print(date_str)
    # if not os.path.exists(bucket_dict_path):    
    # save_obj_as_pickle(bucket_dict_path, bucket_dict)
    # else:
    #     print(f"{bucket_dict_path} already exists")
    return bucket_dict

if __name__ == '__main__':

    args = argparser.parse_args()
    start = time.time()
    if args.b:
        flickr_folder_location = args.b
        flickr_accessor = get_all_metadata(flickr_folder_location)
        # import pdb; pdb.set_trace()
        # Save each bucket into a pickle object
        old_folder_paths = prepare_dataset.get_bucket_folder_paths(flickr_folder_location, args.num_of_bucket, args.split_by_year)
        old_bucket_dict = prepare_dataset.save_bucket_dict(flickr_folder_location, flickr_accessor, old_folder_paths, args.num_of_bucket, args.split_by_year)
        length_of_dataset = 0
        for i in old_bucket_dict:
            print(f"{i}-th bucket has {len(old_bucket_dict[i]['all_metadata'])} images")
            length_of_dataset += len(old_bucket_dict[i]['all_metadata'])
            old_bucket_dict[i]['flickr_accessor'] = old_bucket_dict[i]['all_metadata']
            del old_bucket_dict[i]['all_metadata']
    else:
        flickr_folder_location = get_save_folder(
                                    args.img_dir,
                                    args.size_option,
                                    args.min_size,
                                    args.use_valid_date,
                                    args.min_edge,
                                    args.max_aspect_ratio
                                 )
    
        flickr_accessor = get_flickr_accessor(args)
        # Save each bucket into a pickle object
        old_folder_paths = get_bucket_folder_paths(flickr_folder_location, args.num_of_bucket, args.split_by_year)
        old_bucket_dict = save_bucket_dict(flickr_folder_location, flickr_accessor, old_folder_paths, args.num_of_bucket, args.split_by_year)
        length_of_dataset = 0
        for i in old_bucket_dict:
            print(f"{i}-th bucket has {len(old_bucket_dict[i]['flickr_accessor'])} images")
            length_of_dataset += len(old_bucket_dict[i]['flickr_accessor'])
    end = time.time()
    print(f"{end - start} seconds are used to load all old {length_of_dataset} images")

    print(f"{end - start} seconds are used to load all {len(flickr_accessor)} images")
    print(f"Size of dataset is {len(flickr_accessor)}")
    

    new_folder_paths = prepare_dataset.get_bucket_folder_paths(flickr_folder_location, args.num_of_bucket, args.split_by_year)
    new_bucket_dict = prepare_dataset.save_bucket_dict(flickr_folder_location, flickr_accessor, new_folder_paths, args.num_of_bucket, args.split_by_year)
    # If you want to load the bucket_dict, use load_bucket_dict(flickr_folder_location, args.num_of_bucket, args.split_by_year)
    
    length_of_dataset = 0
    for i in new_bucket_dict:
        print(f"{i}-th bucket has {len(new_bucket_dict[i]['all_metadata'])} images")
        length_of_dataset += len(new_bucket_dict[i]['all_metadata'])
    print(f"{end - start} seconds are used to load all new {length_of_dataset} images")

    # Extract and save the CLIP features
    # print(f"Using CLIP pre-trained model {args.model_name}")
    # model, preprocess = clip.load(args.model_name, device=device)
    for i, old_folder_path in enumerate(old_folder_paths):
        main_pickle_location = get_main_save_location(old_folder_path, args.model_name)
        main_save_location = prepare_dataset.get_main_save_location(new_folder_paths[i], args.model_name)
        print(main_pickle_location)
        chunks, old_path_dict_list = _get_sub_feature_paths(
            old_bucket_dict[i], old_folder_path, main_pickle_location, model_name=args.model_name)
        _, new_path_dict_list = _new_get_sub_feature_paths(
          new_bucket_dict[i], new_folder_paths[i], main_save_location, model_name=args.model_name)
        # import pdb; pdb.set_trace()
        for chunk, old_path_dict, new_path_dict in zip(chunks, old_path_dict_list, new_path_dict_list):
            if not (os.path.exists(old_path_dict['normalized']) and os.path.exists(old_path_dict['original'])):
                print(f"Not exists: {old_path_dict['normalized']}")
                import pdb; pdb.set_trace()
            
            clip_features = load_pickle(old_path_dict['original'])
            print(f"Load from {old_path_dict['original']}")

            clip_features_normalized = load_pickle(old_path_dict['normalized'])
            print(f"Load from {old_path_dict['normalized']}")

            with open(new_path_dict['original'], 'wb') as f:
                np.save(f, clip_features)
            print(f"Saved at {new_path_dict['original']}")
            
            with open(new_path_dict['normalized'], 'wb') as f:
                np.save(f, clip_features_normalized)
            print(f"Saved at {new_path_dict['normalized']}")