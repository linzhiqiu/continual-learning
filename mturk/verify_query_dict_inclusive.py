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

class MocoDataset(Dataset):
    def __init__(self, flickr_accessor, preprocess, device='cuda'):
        self.flickr_accessor = flickr_accessor
        self.device = device
        self.preprocess = preprocess

    def __len__(self):
        return len(self.flickr_accessor)

    def __getitem__(self, index):
        path = self.flickr_accessor[index].metadata.IMG_PATH
        sample = self.preprocess(default_loader(path)).to(self.device)
        return sample

class TempDataset(Dataset):
    def __init__(self, lst_of_paths, preprocess, device='cuda'):
        self.lst_of_paths = lst_of_paths
        self.device = device
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.lst_of_paths)
    
    def __getitem__(self,index):
        path = self.lst_of_paths[index]
        sample = self.preprocess(default_loader(path)).to(self.device)
        return sample

def get_image_loader(lst_of_paths, preprocess, batch_size=BATCH_SIZE, num_workers=0, device='cuda', dataset_class=TempDataset):
    return torch.utils.data.DataLoader(
        dataset_class(lst_of_paths, preprocess, device=device), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
    )

def get_image_features(clip_loader, model):
    clip_features = []
    pbar = tqdm(clip_loader)
    with torch.no_grad():
        for batch, images in enumerate(pbar):
            image_features = model.encode_image(images)
            clip_features.append(image_features.cpu().numpy())
    return np.concatenate(clip_features, axis=0)


def get_query_dict(folder_path, dataset_name, bucket_num):
    d = load_pickle(os.path.join(folder_path, f"bucket_{bucket_num}", dataset_name, 'query_dict.pickle'))
    return d


def save_csv(query_dict, main_save_dir, k='cropped'):
    csv_path = os.path.join(main_save_dir, k+'.csv')
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    args = argparser.parse_args()
    start = time.time()

    small_query_dict = get_query_dict(args.folder_path, args.smaller_dataset, args.num_of_bucket)
    large_query_dict = get_query_dict(args.folder_path, args.larger_dataset, args.num_of_bucket)
    
    ID_lost = []
    for b_idx in range(args.num_of_bucket):
        for query in small_query_dict[b_idx]:
            large_ids = set()
            for item in large_query_dict[b_idx][query]['metadata']:
                large_ids.add(item.ID)
            
            for item in small_query_dict[b_idx][query]['metadata']:
                small_id = item.ID
                if not small_id in large_ids:
                    print(f"Bucket {b_idx} {query}: {small_id} not in larger bucket")
                    # import pdb; pdb.set_trace()
                    ID_lost.append((small_id, query, b_idx))
    print(f"IDs lost: {len(ID_lost)}")
    for small_id, small_query, small_b_idx in ID_lost:
        for b_idx in range(args.num_of_bucket):
            for query in small_query_dict[b_idx]:
                for item in large_query_dict[b_idx][query]['metadata']:
                    large_id = item.ID
                    if large_id == small_id:
                        print(f"Bucket {small_b_idx} {small_query}: {small_id} found in larger dataset with bucket {b_idx} {query}")
    
    inv_normalize = get_unnormalize_func()
    _, preprocess = get_imgnet_transforms()
    # lst_of_paths = [large_query_dict[b_idx][query]['metadata'][300].metadata.IMG_PATH]
    # if not os.path.exists('temp'): os.makedirs("temp")
    # shutil.copy(lst_of_paths[0], 'temp/original.png')
    # loader = get_image_loader(lst_of_paths, preprocess, device='cpu')
    # for inputs in loader:
    #     inputs = inv_normalize(inputs[0])
    #     torchvision.utils.save_image(inputs, "temp/torchvision.png", normalize=False)

    # import pdb; pdb.set_trace()
    main_save_dir = os.path.join(args.save_path, args.larger_dataset)

    cropped_dir = os.path.join(main_save_dir, "cropped")
    original_dir = os.path.join(main_save_dir, "original")
    if not os.path.exists(main_save_dir):
        os.makedirs(main_save_dir)
        os.makedirs(cropped_dir)
        os.makedirs(original_dir)
    
    query_dict = {'cropped' : {}, 'original' : {}}
    
    if os.path.exists(os.path.join(main_save_dir, "cropped.pickle")) and os.path.exists(os.path.join(main_save_dir, "original.pickle")):
        query_dict['cropped'] = load_pickle(os.path.join(main_save_dir, "cropped.pickle"))
        query_dict['original'] = load_pickle(os.path.join(main_save_dir, "original.pickle"))
    else:
        for b_idx in range(args.num_of_bucket):
            cropped_dir_b = os.path.join(cropped_dir, str(b_idx))
            original_dir_b = os.path.join(original_dir, str(b_idx))
            for query in large_query_dict[b_idx]:
                cropped_dir_b_query = os.path.join(cropped_dir_b, query)
                original_dir_b_query = os.path.join(original_dir_b, query)
                if not os.path.exists(cropped_dir_b_query):
                    os.makedirs(cropped_dir_b_query)
                if not os.path.exists(original_dir_b_query):
                    os.makedirs(original_dir_b_query)
                for item_idx, item in enumerate(large_query_dict[b_idx][query]['metadata']):
                    old_path = item.metadata.IMG_PATH
                    ID = item.metadata.ID
                    EXT = item.metadata.EXT
                    new_name = str(ID) + "." + EXT
                    cropped_path = os.path.join(cropped_dir, str(b_idx), query, new_name)
                    original_path = os.path.join(original_dir, str(b_idx), query, new_name)
                    shutil.copy(old_path, original_path)
                    img = default_loader(old_path)
                    # max_size, min_size = max(img.size), min(img.size)
                    # if max_size/min_size > 2:
                    #     print(cropped_path)
                    #     print(original_path)
                    #     print()
                    cropped_img = preprocess(img)
                    cropped_img = inv_normalize(cropped_img)
                    torchvision.utils.save_image(cropped_img, cropped_path, normalize=False)
                    query_dict['cropped'][ID] = {'metadata' : item.metadata, 'path' : cropped_path, 'key' : (b_idx, query, item_idx)}
                    query_dict['original'][ID] = {'metadata' : item.metadata, 'path' : original_path, 'key' : (b_idx, query, item_idx)}
        
        save_obj_as_pickle(os.path.join(main_save_dir, "cropped.pickle"), query_dict['cropped'])
        save_obj_as_pickle(os.path.join(main_save_dir, "original.pickle"), query_dict['original'])
        print(f"saved at {main_save_dir}")

    # {'ID': '108650319', 'USER_ID': '12832970@N00', 
    # 'NICKNAME': 'naotakem', 'DATE_TAKEN': '2006-02-28 20:33:06.0', 
    # 'DATE_UPLOADED': '1141639556', 'DEVICE': 'Panasonic+DMC-FZ5', 
    # 'TITLE': 'MacBook+Pro+Demo+2', 'DESCRIPTION': 'Side+by+side+with+the+Powerbook+%28right%29.', 
    # 'USER_TAGS': 'apple', 'MACHINE_TAGS': '', 'LON': '', 'LAT': '', 'GEO_ACCURACY': '', 
    # 'PAGE_URL': 'http://www.flickr.com/photos/12832970@N00/108650319/', 
    # 'DOWNLOAD_URL': 'http://farm1.staticflickr.com/44/108650319_9c6785fc85.jpg', 'LICENSE_NAME': 'Attribution License', 'LICENSE_URL': 'http://creativecommons.org/licenses/by/2.0/', 'SERVER_ID': '44', 'FARM_ID': '1', 'SECRET': '9c6785fc85', 'SECRET_ORIGINAL': '9c6785fc85', 'EXT': 'jpg', 'IMG_OR_VIDEO': '0', 'metadata': MetadataObject(ID='108650319', USER_ID='12832970@N00', NICKNAME='naotakem', DATE_TAKEN='2006-02-28 20:33:06.0', DATE_UPLOADED='1141639556', DEVICE='Panasonic+DMC-FZ5', TITLE='MacBook+Pro+Demo+2', DESCRIPTION='Side+by+side+with+the+Powerbook+%28right%29.', USER_TAGS='apple', MACHINE_TAGS='', LON='', LAT='', GEO_ACCURACY='', PAGE_URL='http://www.flickr.com/photos/12832970@N00/108650319/', DOWNLOAD_URL='http://farm1.staticflickr.com/44/108650319_9c6785fc85.jpg', LICENSE_NAME='Attribution License', LICENSE_URL='http://creativecommons.org/licenses/by/2.0/', SERVER_ID='44', FARM_ID='1', SECRET='9c6785fc85', SECRET_ORIGINAL='9c6785fc85', EXT='jpg', IMG_OR_VIDEO=0, AUTO_TAG_SCORES={'computer keyboard': 0.815, 'computer monitor': 0.889, 'computer mouse': 0.762, 'computer screen': 0.891, 'computer': 0.944, 'desk': 0.561, 'display screen': 0.891, 'electronics': 0.944, 'furniture': 0.561, 'indoor': 0.998, 'keyboard': 0.547, 'laptop': 0.899, 'monitor': 0.889, 'portable computer': 0.944, 'screen': 0.891, 'table': 0.561}, LINE_NUM='6906188', HASH_VALUE='bfb5ecf1dbb492e7976d90ba3cf8c73e', EXIF=None, IMG_PATH='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/539/images/108650319.jpg', IMG_DIR='/project_data/ramanan/yfcc100m_all/images_minbyte_10_valid_uploaded_date/539/images')}
    # save_csv(query_dict, main_save_dir, 'cropped')
    # save_csv(query_dict, main_save_dir, 'original')
    total_length = list(query_dict['original'].keys()).__len__()
    indices = [i for i in range(total_length)]
    random.shuffle(indices)
    if args.csv_length:
        csv_length = args.csv_length
    else:
        csv_length = total_length
    len_str = str(csv_length)
    indices = indices[:csv_length]
    csv_path_cropped = os.path.join(main_save_dir, f"cropped_{len_str}.csv")
    csv_path_original = os.path.join(main_save_dir, f"original_{len_str}.csv")

    headers = ['image_url', 'ID', 'bucket_index', 'query']
    for csv_path, q_dict_key in [(csv_path_cropped, 'cropped'), (csv_path_original, 'original')]:
        q_dict = query_dict[q_dict_key]
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
