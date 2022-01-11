# Description:
# (1) Split the image stream to buckets sorted by upload time
# (2) Then generate and save the CLIP features (normalized + unnormalized)

# Version 1 - Split by year, e.g., 2004, 2005, 2006,..., 2014
#   python prepare_dataset.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --split_by_year True --model_name RN50 
#   python prepare_dataset.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --split_by_year True --model_name RN50x4
#   python prepare_dataset.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --split_by_year True --model_name RN101

# Version 2 - Split to equal-sized buckets
#   python prepare_dataset.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --num_of_bucket 11 --model_name RN50 
#   python prepare_dataset.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --num_of_bucket 11 --model_name RN50x4
#   python prepare_dataset.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --num_of_bucket 11 --model_name RN101

# Version 3 - Split by precise time
#   python prepare_dataset.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --split_by_time ./clear_10_time.json --model_name RN50 
#   python prepare_dataset.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --split_by_time ./clear_10_time.json --model_name RN50x4
#   python prepare_dataset.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --split_by_time ./clear_10_time.json --model_name RN101


import os
import time

import argparse

from datetime import datetime
from dateutil import parser
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
from yfcc_download import argparser, get_all_metadata, get_save_folder
from utils import divide, load_json, save_as_json, normalize

MAX_SIZE = 500000 # The maximum number of features to store in a single file. You may adjust it according to your CPU memory.
BATCH_SIZE = 128 # The batch size used for extracting features. Adjust it according to your GPU memory
device = "cuda"
# MIN_LINE_NUM = 11000000 # Minimal Line Num
MIN_LINE_NUM = None

# Continue the argparser in yfcc_download
argparser.add_argument('--split_by_time', default=None, type=str,
                    #    default='./clear_10_time.json',
                       help='if set to a json file path, will split image according to time period specified in the json file. Check out ./clear_10_time.json for an example.')
argparser.add_argument('--split_by_year', default=False, type=bool,
                       help='if set to True, ignore --num_of_bucket and sort by year. ')
argparser.add_argument('--num_of_bucket', default=11, type=int,
                       help='If not split the bucket according to year, then split image into this many (equal size) bucket, sorted by time. ')
argparser.add_argument("--model_name",
                       default='RN50', choices=clip.available_models(),
                       help="The CLIP model architecture to use")

def get_knearest_models_func(bucket_dict, clip_model_name, device='cpu'):
    """Return a function knearest_func: bucket_index (int) -> KNearestFaissFeatureChunks (for CLIP-based retrieval)
    """
    model, preprocess = clip.load(clip_model_name, device=device)
    def knearest_func(bucket_idx):
        clip_features_normalized_paths = get_clip_features_normalized_paths(bucket_dict[bucket_idx]['folder_path'], clip_model_name)
        k_near_faiss = KNearestFaissFeatureChunks(clip_features_normalized_paths, model, preprocess, device=device)
        return k_near_faiss
    return knearest_func

def get_clip_features_normalized_paths(f_path, model_name):
    """A helper function to return paths to normalized ·clip features
    """
    clip_features_normalized_paths = []
    main_save_location = get_main_save_location(f_path, model_name)
    print(main_save_location)
    if os.path.exists(main_save_location):
        chunks, path_dict_list = load_json(main_save_location)
        print(f"Loaded from {main_save_location}")
    else:
        print(f"{main_save_location} not exists.")
        import pdb; pdb.set_trace()

    for chunk, path_dict in zip(chunks, path_dict_list):
        if os.path.exists(path_dict['normalized']):
            # print(f"Already exists: {path_dict['normalized']}")
            clip_features_normalized_paths.append(path_dict['normalized'])
        else:
            print(f"{path_dict['normalized']} not exists.")
            import pdb; pdb.set_trace()
    return clip_features_normalized_paths

def _get_split_by_time_name(split_by_time):
    return split_by_time[split_by_time.rfind(os.sep)+1:-5]

def get_bucket_folder_paths(folder_path, num_of_bucket, split_by_year=False, split_by_time=None):
    sub_folder_paths = []
    if split_by_time:
        if not os.path.exists(split_by_time):
            print(f"{split_by_time} not exists.")
            exit(0)
        split_by_time_list = load_json(split_by_time)
        split_by_time_name = _get_split_by_time_name(split_by_time)
        for idx, time_dict in enumerate(split_by_time_list):
            sub_folder_path = os.path.join(folder_path, f'bucket_by_{split_by_time_name}', f'{idx}')
            sub_folder_paths.append(sub_folder_path)
            if not os.path.exists(sub_folder_path):
                os.makedirs(sub_folder_path)
    elif split_by_year:
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
        folder_path, f"features_{model_name.replace(os.sep, '_')}.json")
    return main_save_location


def _get_date_uploaded(date_uploaded):
    return datetime.utcfromtimestamp(int(date_uploaded))

def _get_date_uploaded_from_str(date_str):
    return parser.isoparse(date_str)

def _divide_meta_list(bucket_dict_i, sub_folder, MAX_SIZE=MAX_SIZE):
    # Divide the metadata list into chunks with maximum size being MAX_SIZE
    # Return a list of chunks and save path for each dict
    meta_list = bucket_dict_i['all_metadata']
    sub_features_size = math.ceil(len(meta_list) / MAX_SIZE)
    names = [os.path.join(sub_folder, f'features_{i}') for i in range(sub_features_size)]
    chunks = divide(meta_list, sub_features_size)
    assert len(chunks) == len(names)
    return chunks, names


def _get_sub_feature_paths(bucket_dict_i, folder_path, main_save_location, model_name, MAX_SIZE=MAX_SIZE):
    sub_folder = os.path.join(folder_path, f"features_{model_name.replace(os.sep, '_')}")
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    chunks, names = _divide_meta_list(bucket_dict_i, sub_folder, MAX_SIZE=MAX_SIZE)
    
    path_dict_list = [{'original': n+"_original.npy",
                       "normalized": n+"_normalized.npy"} for n in names]
    save_as_json(main_save_location, (chunks, path_dict_list))
    return chunks, path_dict_list
        

class CLIPDataset(Dataset):
    def __init__(self, all_metadata, preprocess, device='cuda'):
        self.all_metadata = all_metadata
        self.device = device
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.all_metadata)
    
    def __getitem__(self,index):
        meta = self.all_metadata[index]
        path = os.path.join(meta['IMG_DIR'], meta['IMG_PATH'])
        sample = self.preprocess(Image.open(path)).to(self.device)
        return sample

def get_clip_loader(all_metadata, preprocess, batch_size=BATCH_SIZE, num_workers=0, device='cuda', dataset_class=CLIPDataset):
    return torch.utils.data.DataLoader(
        dataset_class(all_metadata, preprocess, device=device), 
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

def save_bucket_dict(flickr_folder_location, all_metadata, folder_paths, num_of_bucket, split_by_year, split_by_time=None):
    assert num_of_bucket == len(folder_paths)
    # Sort images by time, and then split into buckets
    date_uploaded_list = []
    for meta in all_metadata:
        date_uploaded_list.append(meta['DATE_UPLOADED'])
    indices_sorted_by_upload = [i[0] for i in sorted(enumerate(date_uploaded_list), key=lambda x : x[1])]
    
    if split_by_time:
        if not os.path.exists(split_by_time):
            print(f"{split_by_time} not exists.")
            exit(0)
        split_by_time_list = load_json(split_by_time)
        split_by_time_name = _get_split_by_time_name(split_by_time)
        bucket_dict_path = os.path.join(flickr_folder_location, f'bucket_by_{split_by_time_name}.json')
        chunks_of_indices = []
        curr_chunk = []
        bucket_idx = 0
        end_timestamp = _get_date_uploaded_from_str(split_by_time_list[bucket_idx]['end'])
        start_timestamp = _get_date_uploaded_from_str(split_by_time_list[bucket_idx]['start'])
        for sorted_idx in indices_sorted_by_upload:
            curr_timestamp = _get_date_uploaded(date_uploaded_list[sorted_idx])
            if curr_timestamp < start_timestamp:
                continue
            if curr_timestamp > end_timestamp:
                bucket_idx += 1
                chunks_of_indices.append(curr_chunk)
                curr_chunk = []
                if len(split_by_time_list) == bucket_idx:
                    break
                else:
                    end_timestamp = _get_date_uploaded_from_str(split_by_time_list[bucket_idx]['end'])
                    start_timestamp = _get_date_uploaded_from_str(split_by_time_list[bucket_idx]['start'])
            curr_chunk.append(sorted_idx)
        if len(curr_chunk) > 0:
            chunks_of_indices.append(curr_chunk)
        if not len(chunks_of_indices) == len(split_by_time_list):
            import pdb; pdb.set_trace()
    elif split_by_year:
        bucket_dict_path = os.path.join(flickr_folder_location, f'bucket_by_year.json')
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
        bucket_dict_path = os.path.join(flickr_folder_location, f'bucket_{num_of_bucket}.json')
        chunks_of_indices = divide(indices_sorted_by_upload, num_of_bucket)
    
    bucket_dict = {}
    for i, chunk in enumerate(chunks_of_indices):
        bucket_dict_i_path = os.path.join(folder_paths[i], f'bucket_{i}.json')
        if MIN_LINE_NUM:
            print(f"Before filtering by MIN_LINE_NUM: chunk size = {len(chunk)}")
            chunk = [i for i in chunk if int(all_metadata[i]['LINE_NUM']) > MIN_LINE_NUM]
            print(f"After filtering by MIN_LINE_NUM: chunk size = {len(chunk)}")
        meta_list = [all_metadata[i] for i in chunk]
        date_uploaded_list_i = [date_uploaded_list[i] for i in chunk]
        if os.path.exists(bucket_dict_i_path):
            bucket_dict[i] = load_json(bucket_dict_i_path)
        else:
            bucket_dict[i] = {
                'indices' : chunk,
                'all_metadata' : meta_list,
                'folder_path' : folder_paths[i],
                'min_date': str(_get_date_uploaded(date_uploaded_list_i[0])),
                'max_date': str(_get_date_uploaded(date_uploaded_list_i[-1])),
                'date_uploaded_list' : date_uploaded_list_i
            }
            save_as_json(bucket_dict_i_path, bucket_dict[i])
        min_date, max_date = bucket_dict[i]['min_date'], bucket_dict[i]['max_date']
        date_str = f"For bucket {i}: Date range from {min_date} to {max_date}"
        print(date_str)
        line_num_list = [int(meta['LINE_NUM']) for meta in meta_list]
        min_line, max_line = min(line_num_list), max(line_num_list)
        print(f"For bucket {i}: Line number range from {min_line} to {max_line}")
    
    save_as_json(bucket_dict_path, bucket_dict)
    return bucket_dict_path, bucket_dict

if __name__ == '__main__':
    args = argparser.parse_args()
    
    if MIN_LINE_NUM != None:
        print(f"Only select images with line number greater than {MIN_LINE_NUM}")
        import pdb; pdb.set_trace()
    
    start = time.time()
    flickr_folder_location = get_save_folder(
                                 args.img_dir,
                                 args.size_option,
                                 args.min_size,
                                 args.use_valid_date,
                                 args.min_edge,
                                 args.max_aspect_ratio
                             )
    
    all_metadata = get_all_metadata(flickr_folder_location)
    end = time.time()
    print(f"{end - start} seconds are used to load all {len(all_metadata)} images")
    print(f"Size of dataset is {len(all_metadata)}")
    
    # Save each bucket into a json object
    folder_paths = get_bucket_folder_paths(flickr_folder_location, args.num_of_bucket, args.split_by_year, args.split_by_time)
    bucket_dict_path, bucket_dict = save_bucket_dict(flickr_folder_location, all_metadata, folder_paths, args.num_of_bucket, args.split_by_year, args.split_by_time)
    # If you want to load the bucket_dict, use load_bucket_dict(flickr_folder_location, args.num_of_bucket, args.split_by_year)
    
    length_of_dataset = 0
    for i in bucket_dict:
        print(f"{i}-th bucket has {len(bucket_dict[i]['all_metadata'])} images")
        length_of_dataset += len(bucket_dict[i]['all_metadata'])
    print(f"{end - start} seconds are used to load all {length_of_dataset} images")

    # Extract and save the CLIP features
    print(f"Using CLIP pre-trained model {args.model_name}")
    model, preprocess = clip.load(args.model_name, device=device)
    for i, folder_path in enumerate(folder_paths):
        main_save_location = get_main_save_location(folder_path, args.model_name)
        print(main_save_location)
        chunks, path_dict_list = _get_sub_feature_paths(
            bucket_dict[i], folder_path, main_save_location, model_name=args.model_name)
        # import pdb; pdb.set_trace()
        for chunk, path_dict in zip(chunks, path_dict_list):
            if os.path.exists(path_dict['normalized']) and os.path.exists(path_dict['original']):
                print(f"Already exists: {path_dict['normalized']}")
                continue
            else:
                print(f"Save to {path_dict['normalized']}")
            clip_loader = get_clip_loader(chunk, preprocess, dataset_class=CLIPDataset)
            clip_features = get_clip_features(clip_loader, model)
            
            with open(path_dict['original'], 'wb') as f:
                np.save(f, clip_features)
            print(f"Saved at {path_dict['original']}")
            
            clip_features_normalized = normalize(clip_features.astype(np.float32))
            with open(path_dict['normalized'], 'wb') as f:
                np.save(f, clip_features_normalized)

    print(f"Finished extracting the CLIP features. You should replace the bucket_dict_path in CLIP-PromptEngineering.ipynb with {bucket_dict_path} to use this dataset.")