# First transfer the folder to another location if needed
# Next generate the CLIP features (normalized + unnormalized)
# Next split the dataset to buckets sorted by time

# Example: python prepare_dataset.py --img_dir /data3/zhiqiul/yfcc100m_all_new --min_size 10 --chunk_size 10000 --min_edge 120 --max_aspect_ratio 2 --new_folder_path images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0_july_5;
import os
import time
from faiss_utils import get_flickr_folder

import argparse

from datetime import datetime
from PIL import Image
from tqdm import tqdm
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.models as models
from torchvision.datasets.folder import default_loader
import sys
sys.path.append("./CLIP")

import clip
from large_scale_yfcc_download import Metadata, Criteria, MetadataObject, argparser, FlickrAccessor, FlickrFolder, get_flickr_accessor
from utils import divide, load_pickle, save_obj_as_pickle, normalize
from training_utils import get_imgnet_transforms


# Continue the argparser in large_scale_yfcc_download
argparser.add_argument("--new_folder_path", 
                       default=None,
                    #    default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0_july_5',
                       help="Set to a new path instead of None if you want to transfer the files (you cannot simply cp the all_folders.pickle since it has absolute path linked to image file)")
argparser.add_argument('--num_of_bucket', default=11, type=int,
                       help='number of bucket (sorted by time)')
argparser.add_argument("--model_name", 
                       default='RN50', choices=clip.available_models(),
                       help="The CLIP model architecture to use")


def get_bucket_folder_paths(folder_path, num_of_bucket):
    sub_folder_paths = []
    for b_idx in range(num_of_bucket):
        sub_folder_path = os.path.join(folder_path, f'bucket_{num_of_bucket}', f'{b_idx}')
        sub_folder_paths.append(sub_folder_path)
        if not os.path.exists(sub_folder_path):
            os.makedirs(sub_folder_path)
    return sub_folder_paths

def get_date_uploaded(date_uploaded):
    return datetime.utcfromtimestamp(int(date_uploaded))

def divide_meta_list(bucket_dict_i, sub_folder, MAX_SIZE=MAX_SIZE):
    # Divide the metadata list into chunks with maximum size being MAX_SIZE
    # Return a list of chunks and save path for each dict
    meta_list = bucket_dict_i['flickr_accessor']
    sub_features_size = math.ceil(len(meta_list) / MAX_SIZE)
    names = [os.path.join(sub_folder, f'features_{i}') for i in range(sub_features_size)]
    chunks = divide(meta_list, sub_features_size)
    assert len(chunks) == len(names)
    return chunks, names

def get_main_save_location(folder_path, model_name):
    main_save_location = os.path.join(
        folder_path, f"features_{model_name.replace(os.sep, '_')}.pickle")
    return main_save_location

def get_sub_feature_paths(bucket_dict_i, folder_path, main_save_location, model_name, MAX_SIZE=MAX_SIZE):
    sub_folder = os.path.join(folder_path, f"features_{model_name.replace(os.sep, '_')}")
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    chunks, names = divide_meta_list(bucket_dict_i, sub_folder, MAX_SIZE=MAX_SIZE)
    
    path_dict_list = [{'original': n+"_original.pickle",
                       "normalized": n+"_normalized.pickle"} for n in names]
    save_obj_as_pickle(main_save_location, (chunks, path_dict_list))
    return chunks, path_dict_list
        

class CLIPDataset(Dataset):
    def __init__(self, flickr_accessor, preprocess, device='cuda'):
        self.flickr_accessor = flickr_accessor
        self.device = device
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.flickr_accessor)
    
    def __getitem__(self,index):
        path = self.flickr_accessor[index].metadata.IMG_PATH
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

if __name__ == '__main__':
    args = argparser.parse_args()
    start = time.time()
    flickr_accessor = get_flickr_accessor(args, new_folder_path=args.new_folder_path)
    end = time.time()
    print(f"{end - start} seconds are used to load all {len(flickr_accessor)} images")
    print(f"Size of dataset is {len(flickr_accessor)}")
    
    # Sort images by time, and then split into buckets
    date_uploaded_list = []
    for meta in flickr_accessor:
        metadata_obj = meta.get_metadata()
        date_uploaded_list.append(metadata_obj.DATE_UPLOADED)
    date_uploaded_indices = [i[0] for i in sorted(enumerate(date_uploaded_list), key=lambda x : x[1])]
    sorted_indices_chunks = divide(date_uploaded_indices, args.num_of_bucket)
    
    # Save each bucket into a pickle object
    bucket_dict = {}
    folder_paths = get_bucket_folder_paths(args.folder_path, args.num_of_bucket)
    for i, sorted_indices_chunk in enumerate(sorted_indices_chunks):
        min_date, max_date = date_uploaded_list[sorted_indices_chunk[0]], date_uploaded_list[sorted_indices_chunk[-1]]
        date_str = f"For bucket {i}: Date range from {get_date_uploaded(min_date)} to {get_date_uploaded(max_date)}"
        print(date_str)
        
        meta_list = [flickr_accessor[i] for i in sorted_indices_chunk]
        
        bucket_dict[i] = {
            'indices' : sorted_indices_chunk,
            'flickr_accessor' : meta_list,
            'folder_path' : folder_paths[i],
            'min_date': get_date_uploaded(min_date),
            'max_date': get_date_uploaded(max_date),
            'date_uploaded_list' : [date_uploaded_list[i] for i in sorted_indices_chunk]
        }
        save_obj_as_pickle(os.path.join(folder_paths[i], f'bucket_{i}.pickle'), bucket_dict[i])
    
    bucket_dict_path = os.path.join(args.folder_path, f'bucket_{args.num_of_bucket}.pickle')
    save_obj_as_pickle(bucket_dict_path, bucket_dict)
    
    length_of_dataset = 0
    for i in bucket_dict:
        length_of_dataset += len(bucket_dict[i]['flickr_accessor'])
    print(f"{end - start} seconds are used to load all {length_of_dataset} images")

    # Extract and save the CLIP features
    print(f"Using CLIP pre-trained model {args.model_name}")
    for i, folder_path in enumerate(folder_paths):
        main_pickle_location = get_main_save_location(folder_path, args.model_name)
        print(main_pickle_location)
        if os.path.exists(main_pickle_location):
            chunks, path_dict_list = load_pickle(main_pickle_location)
            print(f"Loaded from {main_pickle_location}")
        else:
            chunks, path_dict_list = get_sub_feature_paths(
                bucket_dict[i], folder_path, main_pickle_location, model_name=args.model_name)
        
        model, preprocess = clip.load(args.model_name, device=device)
        for chunk, path_dict in zip(chunks, path_dict_list):
            if os.path.exists(path_dict['normalized']) and os.path.exists(path_dict['original']):
                print(f"Already exists: {path_dict['normalized']}")
                continue
            clip_loader = get_clip_loader(chunk, preprocess, dataset_class=CLIPDataset)
            clip_features = get_clip_features(clip_loader, model)
            
            save_obj_as_pickle(path_dict['original'], clip_features)
            print(f"Saved at {path_dict['original']}")
            
            clip_features_normalized = normalize(clip_features.astype(np.float32))
            save_obj_as_pickle(path_dict['normalized'], clip_features_normalized)