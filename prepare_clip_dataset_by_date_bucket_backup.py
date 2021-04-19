# Assume clip features are stored at args.folder_path
# python prepare_clip_dataset_by_date_bucket_backup.py --model_name RN50 --class_size 1000 --num_of_bucket 3 --query_title none 
# python prepare_clip_dataset_by_date_bucket_backup.py --model_name RN50 --class_size 1000 --num_of_bucket 3 --query_title photo 

# python prepare_clip_dataset_by_date_bucket_backup.py --model_name RN50 --class_size 1000 --num_of_bucket 20 --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18 --clip_dataset_path /scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_feb_18/clip_dataset_bucketed/
# python prepare_clip_dataset_by_date_bucket_backup.py --model_name RN50 --class_size 1000 --num_of_bucket 20 --query_title photo --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18 --clip_dataset_path /scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_feb_18/clip_dataset_bucketed/

import sys
sys.path.append("./CLIP")
import os
import clip
import torch
from PIL import Image
from IPython.display import Image as ImageDisplay
import faiss_utils
from faiss_utils import KNearestFaiss
import numpy as np
import time
from datetime import datetime

from large_scale_feature import argparser, get_clip_loader, get_clip_features, get_feature_name, FlickrAccessor, FlickrFolder, get_flickr_accessor
import argparse
import importlib
from prepare_clip_dataset import QUERY_TITLE_DICT, LABEL_SETS
from utils import divide, normalize, load_pickle, save_obj_as_pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", 
                        default='RN50', choices=clip.available_models(),
                        help="The CLIP model to use")
argparser.add_argument("--folder_path", 
                        # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_25',
                        default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_31',
                        # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_16',
                        # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18',
                        help="The folder with all the computed+normalized CLIP features")
argparser.add_argument("--clip_dataset_path", 
                        # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_25',
                        default='/scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_jan_31/clip_dataset_bucketed/',
                        help="The folder that will store the curated dataset")
argparser.add_argument("--query_title", 
                        default='photo', 
                        choices=QUERY_TITLE_DICT.keys(),
                        help="The query title")
argparser.add_argument('--class_size', default=1000, type=int,
                       help='number of samples per class per bucket')
argparser.add_argument('--num_of_bucket', default=3, type=int,
                       help='number of bucket')
argparser.add_argument("--chunk_size",
                        type=int, default=10000,
                        help="The maximum images to store")
argparser.add_argument("--min_size",
                        type=int, default=10,
                        help="Images with size smaller than min_size will be ignored.")

def get_clip_feature_folder_paths(folder_path, num_of_bucket):
    sub_folder_paths = []
    for b_idx in range(num_of_bucket):
        sub_folder_path = os.path.join(folder_path, f'bucket_{num_of_bucket}', f'{b_idx}')
        sub_folder_paths.append(sub_folder_path)
        if not os.path.exists(sub_folder_path):
            os.makedirs(sub_folder_path)
    return sub_folder_paths

def get_save_paths_by_bucket(clip_dataset_path, label_set, query_title, class_size, num_of_bucket):
    save_path = os.path.join(clip_dataset_path, label_set, f"query_{query_title}_size_{class_size}_bucket_{num_of_bucket}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sub_save_paths = []
    for b_idx in range(num_of_bucket):
        sub_save_path = os.path.join(save_path, f'{b_idx}')
        sub_save_paths.append(sub_save_path)
        if not os.path.exists(sub_save_path):
            os.makedirs(sub_save_path)
    return save_path, sub_save_paths

def get_date_uploaded(DATE_UPLOADED):
    return datetime.utcfromtimestamp(int(DATE_UPLOADED))

if __name__ == '__main__':
    args = argparser.parse_args()

    start = time.time()
    from faiss_utils import get_feature_name, get_flickr_folder
    feature_name = get_feature_name(args.folder_path, args.model_name, normalize=True)
    normalize_features = load_pickle(feature_name)
    assert normalize_features.dtype == np.float32
    flickr_folders_path = get_flickr_folder(args.folder_path)
    flickr_folders = load_pickle(flickr_folders_path)
    flickr_accessor = FlickrAccessor(flickr_folders)
    assert normalize_features.shape[0] == len(flickr_accessor)
    print(f"Size of dataset is {normalize_features.shape[0]}")
    end = time.time()
    print(f"{end - start} seconds are used to load all {len(flickr_accessor)} images")
    date_uploaded_list = []
    for meta in flickr_accessor:
        metadata_obj = meta.get_metadata()
        date_uploaded_list.append(metadata_obj.DATE_UPLOADED)
    date_uploaded_indices = [i[0] for i in sorted(enumerate(date_uploaded_list), key=lambda x : x[1])]
    sorted_indices_chunks = divide(date_uploaded_indices, args.num_of_bucket)
    
    bucket_dict = {}
    folder_paths = get_clip_feature_folder_paths(args.folder_path, args.num_of_bucket)
    for i, sorted_indices_chunk in enumerate(sorted_indices_chunks):
        min_date, max_date = date_uploaded_list[sorted_indices_chunk[0]], date_uploaded_list[sorted_indices_chunk[-1]]
        date_str = f"For bucket {i}: Date range from {get_date_uploaded(min_date)} to {get_date_uploaded(max_date)}"
        print(date_str)
        
        clip_norm_features_location = get_feature_name(folder_paths[i], args.model_name, normalize=True)
        print(clip_norm_features_location)
        # if os.path.exists(clip_norm_features_location):
        #     clip_features_normalized = load_pickle(clip_norm_features_location)
        #     print(f"Exists already: {clip_norm_features_location}")
        # else:
        meta_list = [flickr_accessor[i] for i in sorted_indices_chunk]
        feature_list = [normalize_features[i] for i in sorted_indices_chunk]
        clip_features_normalized = np.concatenate([f.reshape(1,-1) for f in feature_list], axis=0)
        save_obj_as_pickle(clip_norm_features_location, clip_features_normalized)
        print(f"Saved at {clip_norm_features_location}")
        
        bucket_dict[i] = {
            'indices' : sorted_indices_chunk,
            'normalized_feature_path' : clip_norm_features_location,
            'flickr_accessor' : meta_list,
            'folder_path' : folder_paths[i],
            'min_date' : min_date,
            'max_date' : max_date,
            'date_uploaded_list' : [date_uploaded_list[i] for i in sorted_indices_chunk]
        }
        save_obj_as_pickle(os.path.join(folder_paths[i], f'bucket_{i}.pickle'), bucket_dict[i])

    for label_set in LABEL_SETS:
        print(f"Processing {label_set}.. ")
        label_set_module_name = "label_sets." + label_set

        # The file gets executed upon import, as expected.
        label_set_module = importlib.import_module(label_set_module_name)

        labels = label_set_module.labels

        queries = [QUERY_TITLE_DICT[args.query_title] + label for label in labels]

        print(queries)
        print(f"We have {len(queries)} queries.")

        save_path, sub_save_paths = get_save_paths_by_bucket(args.clip_dataset_path, label_set, args.query_title, args.class_size, args.num_of_bucket)
        info_dict = {
            'query_title' : args.query_title,
            'query_title_name' : QUERY_TITLE_DICT[args.query_title],
            'model_name'  : args.model_name,
            'folder_path' : args.folder_path,
            'clip_dataset_path' : args.clip_dataset_path,
            'save_path': save_path,
            'sub_save_paths' : sub_save_paths,
            'label_set' : label_set,
            'class_size' : args.class_size,
        }


        info_dict_path = os.path.join(save_path, "info_dict.pickle")
        if os.path.exists(info_dict_path):
            print(f"{info_dict_path} already exists.")
            saved_info_dict = load_pickle(info_dict_path)
            for k in saved_info_dict:
                if not saved_info_dict[k] == info_dict[k]:
                    print(f"{k} does not match.")
                    import pdb; pdb.set_trace()
            print("All matched. Continue? >>")
            # import pdb; pdb.set_trace()
        else:
            print(f"Save info dict at {info_dict_path}")
            save_obj_as_pickle(info_dict_path, info_dict)

        query_dict = {}
        for b_idx in bucket_dict:
            query_dict[b_idx] = {}
            
            k_near_faiss = KNearestFaiss(folder_paths[b_idx], args.model_name, flickr_accessor=bucket_dict[b_idx]['flickr_accessor'])

            def grab_top_query_images(query, start_idx=0, end_idx=40, diff_queries=[], lmb=0.):
                start = time.time()
                if len(diff_queries) != 0:
                    print("Use difference of queries")
                    normalize_text_feature = k_near_faiss.get_text_difference_feature(query, diff_queries=diff_queries, lmb=lmb)
                else:
                    normalize_text_feature = k_near_faiss.get_normalized_text_feature(query)
                end_feature = time.time()
                D, meta_list, clip_features = k_near_faiss.k_nearest_meta_and_clip_feature(normalize_text_feature, k=end_idx)
                end_search = time.time()
                print(f"{end_feature-start:.4f} for querying {query}. {end_search-end_feature} for computing KNN.")
                return D[start_idx:end_idx], meta_list[start_idx:end_idx], clip_features[start_idx:end_idx], normalize_text_feature

            for idx, query in enumerate(queries):
                # if args.use_difference_of_query:
                #     diff_queries = queries[:idx] + queries[idx+1:]
                #     D, query_meta_list, query_clip_features, text_feature = grab_top_query_images(query, diff_queries=diff_queries, lmb=args.lmb, end_idx=args.class_size)
                # else:
                D, query_meta_list, query_clip_features, text_feature = grab_top_query_images(query, end_idx=args.class_size)
                query_dict[b_idx][query] = {
                    'features' : query_clip_features,
                    'features_mean' : query_clip_features.mean(axis=0),
                    'metadata' : query_meta_list,
                    'D' : D,
                    'text_feature' : text_feature,
                }
        query_dict_path = os.path.join(save_path, "query_dict.pickle")
        if os.path.exists(query_dict_path):
            print(f"Overwrite {query_dict_path}")
        save_obj_as_pickle(query_dict_path, query_dict)
        print(f"Save at {query_dict_path}")
        
        def gather_distances(distances_list):
            distances = {}
            for idx, f in enumerate(distances_list):
                if idx == len(distances_list) - 1:
                    break
                for second_idx in range(idx+1, len(distances_list)):
                    f_2 = distances_list[second_idx]
                    d = f.dot(f_2)
                    distances[(idx, second_idx)] = d
            return distances
        
        def gather_distances_first_last(distances_list):
            distances = {}
            f = distances_list[0]
            f_2 = distances_list[-1]
            d = f.dot(f_2)
            return d
        
        def print_distances(query, distance_dict_query):
            print(query)
            ds = []
            for (idx, idx_2) in distance_dict_query:
                d = distance_dict_query[(idx, idx_2)]
                ds.append(d)
                print(f"   {idx}-to-{idx_2}: {d}")
            print(f"{sum(ds)} for {query}")
            distance_mean = sum(ds)/len(ds)
            return distance_mean
        
        def print_all(distances_dict, key='distance_mean'):
            distances_dict_summary = []
            for query in distances_dict:
                # ds = []
                # for item in distances_dict[query]:
                #     ds.append(distances_dict[query][item]/len(list(distances_dict[query].keys())))
                distances_dict_summary.append((distances_dict[query][key], query))
            distances_dict_summary = sorted(distances_dict_summary, key=lambda x : x[0], reverse=True)
            print(f"{key} for {label_set}. Top are:")
            for i in range(6):
                d, q = distances_dict_summary[i]
                print(f"{q}: {d}")
            print(f"{key} for {label_set}. Bottom are:")
            for i in range(-1, -6, -1):
                d, q = distances_dict_summary[i]
                print(f"{q}: {d}")
            

        distance_pickle_path = os.path.join(save_path, "distances.pickle")
        distances_dict = {}
        for query in queries:
            distances_dict[query] = {'features_mean' : []}
            for b_idx in bucket_dict:
                distances_dict[query]['features_mean'].append(query_dict[b_idx][query]['features_mean'])
            distance_dict_query = gather_distances(distances_dict[query]['features_mean'])
            distance_mean = print_distances(query, distance_dict_query)
            distance_first_last = gather_distances_first_last(distances_dict[query]['features_mean'])
            distances_dict[query]['distance_mean'] = distance_mean
            distances_dict[query]['distance_first_last'] = distance_first_last
        
        print_all(distances_dict)
        print_all(distances_dict, key='distance_first_last')
        save_obj_as_pickle(distance_pickle_path, distances_dict)
        print(f"Saved to {distance_pickle_path}")