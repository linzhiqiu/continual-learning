# python prepare_clip_dataset_by_date_bucket.py --model_name RN50 --class_size 1000 --num_of_bucket 4 --query_title none --all_images --img_dir /project_data/ramanan/yfcc100m_all --min_size 10 --chunk_size 10000
# python prepare_clip_dataset_by_date_bucket.py --model_name RN50 --class_size 1000 --num_of_bucket 4 --query_title photo --all_images --img_dir /project_data/ramanan/yfcc100m_all --min_size 10 --chunk_size 10000

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
                        # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_31',
                        default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_16',
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
argparser.add_argument('--num_of_bucket', default=4, type=int,
                       help='number of bucket')
argparser.add_argument("--img_dir", 
                        default='/project_data/ramanan/yfcc100m_all',
                        help="The yfcc100M dataset store location")
argparser.add_argument("--data_file", 
                        default='./yfcc100m/yfcc100m_dataset',
                        help="The yfcc100M dataset file ")
argparser.add_argument("--auto_file",
                        default='./yfcc100m/yfcc100m_autotags-v1', 
                        help="The autotag file")
argparser.add_argument("--exif_file",
                        default='./yfcc100m/yfcc100m_exif', 
                        help="The exif file")
argparser.add_argument("--hash_file",
                        default='./yfcc100m/yfcc100m_hash', 
                        help="The hash file")
argparser.add_argument("--hash_pickle",
                        default='./yfcc100m/yfcc100m_hash.pickle', 
                        help="The hash dictionary pickle object")
argparser.add_argument("--lines_file",
                        default='./yfcc100m/yfcc100m_lines', 
                        help="The lines file")
argparser.add_argument("--size_option",
                        default='original', choices=['original'],
                        help="Whether to use the original image size (max edge has 500 px).")
argparser.add_argument("--chunk_size",
                        type=int, default=10000,
                        help="The maximum images to store")
argparser.add_argument("--min_size",
                        type=int, default=10,
                        help="Images with size smaller than min_size will be ignored.")
argparser.add_argument("--all_images",
                        action='store_true',
                        help="Store all images.")
argparser.add_argument("--use_valid_date",
                        type=bool, default=True,
                        help="Images with valid date (upload date < taken date) will be used if set true")

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
    flickr_accessor = get_flickr_accessor(args, new_folder_path=args.folder_path)
    end = time.time()
    print(f"{end - start} seconds are used to load all {len(flickr_accessor)} images")
    date_uploaded_list = []
    for meta in flickr_accessor:
        metadata_obj = meta.get_metadata()
        date_uploaded_list.append(metadata_obj.DATE_UPLOADED)
    date_uploaded_indices = [i[0] for i in sorted(enumerate(date_uploaded_list), key=lambda x : x[1])]
    sorted_indices_chunks = divide(date_uploaded_indices, args.num_of_bucket)
    
    bucket_dict = {}
    device = "cuda"
    print(f"Using model {args.model_name}")
    model, preprocess = clip.load(args.model_name, device=device)
    folder_paths = get_clip_feature_folder_paths(args.folder_path, args.num_of_bucket)
    for i, sorted_indices_chunk in enumerate(sorted_indices_chunks):
        min_date, max_date = date_uploaded_list[sorted_indices_chunk[0]], date_uploaded_list[sorted_indices_chunk[-1]]
        date_str = f"For bucket {i}: Date range from {get_date_uploaded(min_date)} to {get_date_uploaded(max_date)}"
        print(date_str)
        
        clip_features_location = get_feature_name(folder_paths[i], args.model_name, normalize=False)
        clip_norm_features_location = get_feature_name(folder_paths[i], args.model_name, normalize=True)
        print(clip_features_location)
        print(clip_norm_features_location)
        if os.path.exists(clip_features_location):
            # clip_features = load_pickle(clip_features_location)
            print(f"Exists already: {clip_features_location}")
        else:
            meta_list = [flickr_accessor[i] for i in sorted_indices_chunk]
            clip_loader = get_clip_loader(meta_list, preprocess)
            clip_features = get_clip_features(clip_loader, model)
            save_obj_as_pickle(clip_features_location, clip_features)
            print(f"Saved at {clip_features_location}")

            clip_features_normalized = normalize(clip_features.astype(np.float32))
            save_obj_as_pickle(clip_norm_features_location, clip_features_normalized)
            print(f"Saved at {clip_norm_features_location}")
        
        bucket_dict[i] = {
            'indices' : sorted_indices_chunk,
            'normalized_feature_path' : clip_norm_features_location,
            'feature_path' : clip_features_location,
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

        info_dict = {
            'query_title' : args.query_title,
            'query_title_name' : QUERY_TITLE_DICT[args.query_title],
            'model_name'  : args.model_name,
            'folder_path' : args.folder_path,
            'clip_dataset_path' : args.clip_dataset_path,
            'save_path': save_path,
            'sub_save_paths' : sub_save_paths,
            'label_set' : args.label_set,
            'class_size' : args.class_size,
        }

        save_path, sub_save_paths = get_save_paths_by_bucket(args.clip_dataset_path, label_set, args.query_title, args.class_size, args.num_of_bucket)

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

        for b_idx in bucket_dict:
            query_dict_path = os.path.join(sub_save_paths[b_idx], "query_dict.pickle")
            if os.path.exists(query_dict_path):
                print(f"Overwrite {query_dict_path}")
                # import pdb; pdb.set_trace()
            k_near_faiss = KNearestFaiss(folder_paths[b_idx], args.model_name)

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


    # if args.use_max_score:
    #     assert args.class_size < args.nn_size
    #     meta_dict = {}
    #     for idx, query in enumerate(queries):
    #         D, query_meta_list, query_clip_features, text_feature = grab_top_query_images(query, end_idx=args.nn_size)
    #         for meta_idx, meta in enumerate(query_meta_list):
    #             ID = meta.get_metadata().ID
    #             if ID in meta_dict:
    #                 meta_dict[ID]['features'].append(query_clip_features[meta_idx])
    #                 meta_dict[ID]['D'].append(D[meta_idx])
    #                 meta_dict[ID]['query'].append(query)
    #             else:
    #                 meta_dict[ID] = {
    #                     'metadata' : meta,
    #                     'features' : [query_clip_features[meta_idx]],
    #                     'D' : [D[meta_idx]],
    #                     'query' : [query]
    #                 }

    #         query_dict[query] = {
    #             'features' : [],
    #             'metadata' : [],
    #             'D' : [],
    #             'text_feature' : text_feature,
    #         }

    #     for ID in meta_dict:
    #         meta = meta_dict[ID]['metadata']
    #         max_idx, max_D = max(enumerate(meta_dict[ID]['D']), key=lambda x: x[1])
    #         max_query = meta_dict[ID]['query'][max_idx]
    #         max_feature = meta_dict[ID]['features'][max_idx]
    #         if max_query not in query_dict:
    #             import pdb; pdb.set_trace()
    #         else:
    #             query_dict[max_query]['metadata'].append(meta)
    #             query_dict[max_query]['features'].append(max_feature)
    #             query_dict[max_query]['D'].append(max_D)
        
    #     for query in query_dict:
    #         if len(query_dict[query]['metadata']) < args.class_size:
    #             import pdb; pdb.set_trace()
    #         else:
    #             sorted_indices = [idx for idx, score in sorted(enumerate(query_dict[query]['D']), key=lambda x : x[1], reverse=True)]
    #             query_dict[query]['metadata'] = [query_dict[query]['metadata'][idx] for idx in sorted_indices][:args.class_size]
    #             query_dict[query]['features'] = [query_dict[query]['features'][idx] for idx in sorted_indices][:args.class_size]
    #             query_dict[query]['D'] = [query_dict[query]['D'][idx] for idx in sorted_indices][:args.class_size]
    # else:
            for idx, query in enumerate(queries):
                # if args.use_difference_of_query:
                #     diff_queries = queries[:idx] + queries[idx+1:]
                #     D, query_meta_list, query_clip_features, text_feature = grab_top_query_images(query, diff_queries=diff_queries, lmb=args.lmb, end_idx=args.class_size)
                # else:
                D, query_meta_list, query_clip_features, text_feature = grab_top_query_images(query, end_idx=args.class_size)
                query_dict[query] = {
                    'features' : query_clip_features,
                    'metadata' : query_meta_list,
                    'D' : D,
                    'text_feature' : text_feature,
                }

            save_obj_as_pickle(query_dict_path, query_dict)
            print(f"Save at {query_dict_path}")
