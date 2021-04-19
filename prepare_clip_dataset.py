# python prepare_clip_dataset.py --model_name RN50 --label_set vehicle_7 --lmb 1. --use_difference_of_query --class_size 1000
# python prepare_clip_dataset.py --model_name RN50 --label_set vehicle_7 --class_size 1000

# python prepare_clip_dataset.py --model_name RN50 --label_set imagenet1K --lmb 1. --use_difference_of_query --class_size 1000
# python prepare_clip_dataset.py --model_name RN50 --label_set imagenet1K --class_size 1000

# python prepare_clip_dataset.py --model_name RN50 --label_set cifar10 --lmb 1. --use_difference_of_query --class_size 1000
# python prepare_clip_dataset.py --model_name RN50 --label_set cifar10 --class_size 1000

# python prepare_clip_dataset.py --model_name RN50 --label_set cifar100 --lmb 1. --use_difference_of_query --class_size 1000
# python prepare_clip_dataset.py --model_name RN50 --label_set cifar100 --class_size 1000

# python prepare_clip_dataset.py --model_name RN50 --label_set cifar100 --lmb 10. --use_difference_of_query --class_size 1000
# python prepare_clip_dataset.py --model_name RN50 --label_set imagenet1K --lmb 100. --use_difference_of_query --class_size 1000

# python prepare_clip_dataset.py --model_name RN50 --query_title "" --label_set cifar100 --lmb 10. --use_difference_of_query --class_size 1000  --clip_dataset_path /scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_jan_31/clip_dataset_no_pre_prompt/
# python prepare_clip_dataset.py --model_name RN50 --query_title "" --label_set imagenet1K  --lmb 100. --use_difference_of_query --class_size 1000 --clip_dataset_path /scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_jan_31/clip_dataset_no_pre_prompt/

# python prepare_clip_dataset.py --model_name RN50 --query_title "" --label_set cifar100 --use_difference_of_query --class_size 1000  --clip_dataset_path /scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_jan_31/clip_dataset_no_pre_prompt/
# python prepare_clip_dataset.py --model_name RN50 --query_title "" --label_set imagenet1K  --use_difference_of_query --class_size 1000 --clip_dataset_path /scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_jan_31/clip_dataset_no_pre_prompt/

# python prepare_clip_dataset.py --model_name RN50 --query_title "" --label_set cifar100 --class_size 1000  --clip_dataset_path /scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_jan_31/clip_dataset_no_pre_prompt/
# python prepare_clip_dataset.py --model_name RN50 --query_title "" --label_set imagenet1K  --class_size 1000 --clip_dataset_path /scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_jan_31/clip_dataset_no_pre_prompt/


# New version with max score
# python prepare_clip_dataset.py --model_name RN50 --label_set tech_7 --class_size 1000 --use_max_score
# python prepare_clip_dataset.py --model_name RN50 --label_set tech_7 --class_size 1000 --use_max_score --query_title none
# python prepare_clip_dataset.py --model_name RN50 --label_set vehicle_7 --class_size 1000 --use_max_score
# python prepare_clip_dataset.py --model_name RN50 --label_set vehicle_7 --class_size 1000 --use_max_score --query_title none
# python prepare_clip_dataset.py --model_name RN50 --label_set cifar10 --class_size 1000 --use_max_score
# python prepare_clip_dataset.py --model_name RN50 --label_set cifar10 --class_size 1000 --use_max_score --query_title none

# python prepare_clip_dataset.py --model_name RN50 --label_set cifar100 --class_size 1000 --use_max_score
# python prepare_clip_dataset.py --model_name RN50 --label_set cifar100 --class_size 1000 --use_max_score --query_title none
# python prepare_clip_dataset.py --model_name RN50 --label_set sports_30 --class_size 1000 --use_max_score
# python prepare_clip_dataset.py --model_name RN50 --label_set sports_30 --class_size 1000 --use_max_score --query_title none
# python prepare_clip_dataset.py --model_name RN50 --label_set imagenet1K --class_size 1000 --use_max_score
# python prepare_clip_dataset.py --model_name RN50 --label_set imagenet1K --class_size 1000 --use_max_score --query_title none


import sys
sys.path.append("./CLIP")
import os
import clip
import torch
from PIL import Image
from IPython.display import Image as ImageDisplay
from utils import load_pickle, save_obj_as_pickle
import faiss_utils
from faiss_utils import KNearestFaiss
import numpy as np
import time

import large_scale_feature
from large_scale_feature import argparser
import argparse
import importlib


device = "cuda" if torch.cuda.is_available() else "cpu"

QUERY_TITLE_DICT = {
    'none' : "",
    'photo' : "A photo of a ",
}

# LABEL_SETS = ['dynamic', 'random', 'random2', 'vehicle_7', 'cifar10',
#               'tech_7_new', 'sports_30', 'fashion_25', 'cifar100', 'imagenet1K']
LABEL_SETS = ['dynamic', 'vehicle_7', 'cifar10',
              'tech_7_new', 'sports_30', 'fashion_25', 'cifar100', 'imagenet1K']

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", 
                        default='RN50', choices=clip.available_models(),
                        help="The CLIP model to use")
argparser.add_argument("--folder_path", 
                        # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_25',
                        default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_31',
                        help="The folder with all the computed+normalized CLIP features")
argparser.add_argument("--clip_dataset_path", 
                        # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_25',
                        default='/scratch/zhiqiu/yfcc100m_all/clip_dataset/images_minbyte_10_valid_uploaded_date_jan_31/clip_dataset/',
                        help="The folder that will store the curated dataset")
argparser.add_argument("--label_set", 
                        default='vehicle_7', choices=LABEL_SETS,
                        help="The label sets")
argparser.add_argument("--use_difference_of_query", 
                        action='store_true',
                        help="Whether or not to use difference of query")
argparser.add_argument("--use_max_score", 
                        action='store_true',
                        help="Keep the max scoring images")
argparser.add_argument("--query_title", 
                        default='photo', 
                        choices=QUERY_TITLE_DICT.keys(),
                        help="The query title")
argparser.add_argument('--class_size', default=1000, type=int,
                       help='number of samples per class')
argparser.add_argument('--nn_size', default=10000, type=int,
                       help='number of samples per class for initial search of top NN')
argparser.add_argument("--lmb", 
                        default=1., type=float,
                        help="The difference of query feature ratio")

def get_save_path(args):
    lmb_str = f"_lmb_{args.lmb}" if args.use_difference_of_query else ""
    if args.use_difference_of_query:
        detail_str = f"_doq_{lmb_str}"
    elif args.use_max_score:
        detail_str = f"_maxscore"
    else:
        detail_str = ""
    save_path = os.path.join(args.clip_dataset_path, args.label_set, f"query_{args.query_title}_size_{args.class_size}_nnsize_{args.nn_size}{detail_str}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path
    

if __name__ == '__main__':
    args = argparser.parse_args()

    # Contrived example of generating a module named as a string
    label_set_module_name = "label_sets." + args.label_set

    # The file gets executed upon import, as expected.
    label_set_module = importlib.import_module(label_set_module_name)

    labels = label_set_module.labels

    queries = [QUERY_TITLE_DICT[args.query_title] + label for label in labels]

    print(queries)
    print(f"We have {len(queries)} queries.")

    # lmb_str = f"lmb_{args.lmb}" if args.lmb != 1. else ""
    # save_path = os.path.join(args.clip_dataset_path, args.label_set, f"size_{args.class_size}_doq_{args.use_difference_of_query}{lmb_str}")
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    save_path = get_save_path(args)

    info_dict = {
        'query_title' : args.query_title,
        'query_title_name' : QUERY_TITLE_DICT[args.query_title],
        'model_name'  : args.model_name,
        'folder_path' : args.folder_path,
        'clip_dataset_path' : args.clip_dataset_path,
        'save_path': save_path,
        'label_set' : args.label_set,
        'class_size' : args.class_size,
        'use_difference_of_query': args.use_difference_of_query,
        'use_max_score': args.use_max_score,
        'lmb' : args.lmb
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
        import pdb; pdb.set_trace()
    else:
        print(f"Save info dict at {info_dict_path}")
        save_obj_as_pickle(info_dict_path, info_dict)

    query_dict_path = os.path.join(save_path, "query_dict.pickle")
    if os.path.exists(query_dict_path):
        print(f"Overwrite {query_dict_path}?")
        import pdb; pdb.set_trace()

    k_near_faiss = KNearestFaiss(args.folder_path, args.model_name)

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


    query_dict = {}
    if args.use_max_score:
        assert args.class_size < args.nn_size
        meta_dict = {}
        for idx, query in enumerate(queries):
            D, query_meta_list, query_clip_features, text_feature = grab_top_query_images(query, end_idx=args.nn_size)
            for meta_idx, meta in enumerate(query_meta_list):
                ID = meta.get_metadata().ID
                if ID in meta_dict:
                    meta_dict[ID]['features'].append(query_clip_features[meta_idx])
                    meta_dict[ID]['D'].append(D[meta_idx])
                    meta_dict[ID]['query'].append(query)
                else:
                    meta_dict[ID] = {
                        'metadata' : meta,
                        'features' : [query_clip_features[meta_idx]],
                        'D' : [D[meta_idx]],
                        'query' : [query]
                    }

            query_dict[query] = {
                'features' : [],
                'metadata' : [],
                'D' : [],
                'text_feature' : text_feature,
            }

        for ID in meta_dict:
            meta = meta_dict[ID]['metadata']
            max_idx, max_D = max(enumerate(meta_dict[ID]['D']), key=lambda x: x[1])
            max_query = meta_dict[ID]['query'][max_idx]
            max_feature = meta_dict[ID]['features'][max_idx]
            if max_query not in query_dict:
                import pdb; pdb.set_trace()
            else:
                query_dict[max_query]['metadata'].append(meta)
                query_dict[max_query]['features'].append(max_feature)
                query_dict[max_query]['D'].append(max_D)
        
        for query in query_dict:
            if len(query_dict[query]['metadata']) < args.class_size:
                import pdb; pdb.set_trace()
            else:
                sorted_indices = [idx for idx, score in sorted(enumerate(query_dict[query]['D']), key=lambda x : x[1], reverse=True)]
                query_dict[query]['metadata'] = [query_dict[query]['metadata'][idx] for idx in sorted_indices][:args.class_size]
                query_dict[query]['features'] = [query_dict[query]['features'][idx] for idx in sorted_indices][:args.class_size]
                query_dict[query]['D'] = [query_dict[query]['D'][idx] for idx in sorted_indices][:args.class_size]
    else:
        for idx, query in enumerate(queries):
            if args.use_difference_of_query:
                diff_queries = queries[:idx] + queries[idx+1:]
                D, query_meta_list, query_clip_features, text_feature = grab_top_query_images(query, diff_queries=diff_queries, lmb=args.lmb, end_idx=args.class_size)
            else:
                D, query_meta_list, query_clip_features, text_feature = grab_top_query_images(query, end_idx=args.class_size)
            query_dict[query] = {
                'features' : query_clip_features,
                'metadata' : query_meta_list,
                'D' : D,
                'text_feature' : text_feature,
            }

    save_obj_as_pickle(query_dict_path, query_dict)
    print(f"Save at {query_dict_path}")
