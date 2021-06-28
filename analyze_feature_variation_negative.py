# Run after analyze_feature_variation.py
# python analyze_feature_variation_negative.py --negative_label_set dynamic_negative_300 --model_name RN50 --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_feature_variation_negative.py --negative_label_set dynamic_negative_3000 --model_name RN50 --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_feature_variation_negative.py --negative_label_set dynamic_negative_600_v2 --model_name RN50 --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_feature_variation_negative.py --negative_label_set dynamic_negative_600_dress --model_name RN50 --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_feature_variation_negative.py --negative_label_set dynamic_negative_600_hoodies --model_name RN50 --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_feature_variation_negative.py --negative_label_set dynamic_negative_600_dress_football --model_name RN50 --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar

import sys
sys.path.append("./CLIP")
import os
import clip
import torch
import faiss_utils
from faiss_utils import KNearestFaissFeatureChunks
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

# from large_scale_feature import argparser, get_clip_loader, get_clip_features, get_feature_name, FlickrAccessor, FlickrFolder, get_flickr_accessor
from large_scale_feature_chunks import argparser
import large_scale_feature_chunks
from analyze_feature_variation import get_dataset_folder_paths
import argparse
import importlib
from prepare_clip_dataset import QUERY_TITLE_DICT, LABEL_SETS
from utils import divide, normalize, load_pickle, save_obj_as_pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

NEGATIVE_LABEL = 'NEGATIVE'

NEG_DICT = {
    'dynamic_negative_300' : {
        'positive' : {
            'label_set' : 'dynamic',
            'model_name' : 'RN50',
            'query_title' : 'none',
            'class_size' : 300,
            'avoid_multiple_class' : True,
            'nn_size' : 8000,
            'reverse_order':  False
        },
        'negative' : {
            'label_set' : 'dynamic',
            'model_name' : 'RN50',
            'query_title' : 'none',
            'class_size' : 300,
            'avoid_multiple_class' : True,
            'nn_size' : 8000,
            'reverse_order':  True
        },
        'single_negative_class' : True,# put all negative classes into one class
        'negative_ratio' : 0.1, # The ratio of negative samples per class to keep
    },
    'dynamic_negative_600_v2': {
        'positive': {
            'label_set': 'dynamic',
            'model_name': 'RN50',
            'query_title': 'none',
            'class_size': 600,
            'avoid_multiple_class': True,
            'nn_size': 16000,
            'reverse_order':  False
        },
        'negative': {
            'label_set': 'dynamic',
            'model_name': 'RN50',
            'query_title': 'none',
            'class_size': 600,
            'avoid_multiple_class': True,
            'nn_size': 16000,
            'reverse_order':  True
        },
        'single_negative_class': True,  # put all negative classes into one class
        'negative_ratio': 0.1,  # The ratio of negative samples per class to keep
        'discard_overlap' : True,
    },
    'dynamic_negative_600_dress': {
        'positive': {
            'label_set': 'dynamic_dress',
            'model_name': 'RN50',
            'query_title': 'none',
            'class_size': 600,
            'avoid_multiple_class': True,
            'nn_size': 16000,
            'reverse_order':  False
        },
        'negative': {
            'label_set': 'dynamic_dress',
            'model_name': 'RN50',
            'query_title': 'none',
            'class_size': 600,
            'avoid_multiple_class': True,
            'nn_size': 16000,
            'reverse_order':  True
        },
        'single_negative_class': True,  # put all negative classes into one class
        'negative_ratio': 0.1,  # The ratio of negative samples per class to keep
        'discard_overlap': True,
    },
    # 'dynamic_negative_600_hoodies': {
    #     'positive': {
    #         'label_set': 'dynamic_hoodies',
    #         'model_name': 'RN50',
    #         'query_title': 'none',
    #         'class_size': 600,
    #         'avoid_multiple_class': True,
    #         'nn_size': 16000,
    #         'reverse_order':  False
    #     },
    #     'negative': {
    #         'label_set': 'dynamic_hoodies',
    #         'model_name': 'RN50',
    #         'query_title': 'none',
    #         'class_size': 600,
    #         'avoid_multiple_class': True,
    #         'nn_size': 16000,
    #         'reverse_order':  True
    #     },
    #     'single_negative_class': True,  # put all negative classes into one class
    #     'negative_ratio': 0.1,  # The ratio of negative samples per class to keep
    #     'discard_overlap': True,
    # },
    'dynamic_negative_600_dress_soccer': {
        'positive': {
            'label_set': 'dynamic_dress_soccer',
            'model_name': 'RN50',
            'query_title': 'none',
            'class_size': 600,
            'avoid_multiple_class': True,
            'nn_size': 16000,
            'reverse_order':  False
        },
        'negative': {
            'label_set': 'dynamic_dress_soccer',
            'model_name': 'RN50',
            'query_title': 'none',
            'class_size': 600,
            'avoid_multiple_class': True,
            'nn_size': 16000,
            'reverse_order':  True
        },
        'single_negative_class': True,  # put all negative classes into one class
        'negative_ratio': 0.1,  # The ratio of negative samples per class to keep
        'discard_overlap': True,
    },
    # 'dynamic_negative_3000' : {
    #     'positive' : {
    #         'label_set' : 'dynamic',
    #         'model_name' : 'RN50',
    #         'query_title' : 'none',
    #         'class_size' : 300,
    #         'avoid_multiple_class' : True,
    #         'nn_size' : 8000,
    #         'reverse_order':  False
    #     },
    #     'negative' : {
    #         'label_set' : 'dynamic',
    #         'model_name' : 'RN50',
    #         'query_title' : 'none',
    #         'class_size' : 300,
    #         'avoid_multiple_class' : True,
    #         'nn_size' : 8000,
    #         'reverse_order':  True
    #     },
    #     'single_negative_class' : True, # put all negative classes into one class
    #     'negative_ratio' : 1.0, # The ratio of negative samples per class to keep
    # },
}

NEGATIVE_LABEL_SETS = list(NEG_DICT.keys())

# argparser = argparse.ArgumentParser()
# Below are in large_scale_feature_chunks already
# argparser.add_argument("--model_name", 
#                         default='RN50', choices=clip.available_models(),
#                         help="The CLIP model to use")
# argparser.add_argument("--folder_path", 
#                         default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18',
#                         help="The folder with all the computed+normalized CLIP + Moco features got by large_scale_features_chunk.py")
# argparser.add_argument('--num_of_bucket', default=11, type=int,
#                        help='number of bucket')
# argparser.add_argument("--moco_model",
#                        default='',
#                        help="The moco model to use")
# argparser.add_argument('--arch', metavar='ARCH', default='resnet50',
#                        help='model architecture: ' +
#                        ' (default: resnet50)')
# argparser.add_argument("--query_title", 
#                         default='photo', 
#                         choices=QUERY_TITLE_DICT.keys(),
#                         help="The query title")
# argparser.add_argument('--class_size', default=2000, type=int,
#                        help='number of (max score) samples per class per bucket')
# argparser.add_argument("--avoid_multiple_class",
#                        action='store_true',
#                        help="Only keep the max scoring images if set True")
# argparser.add_argument("--reverse_order",
#                        action='store_true',
#                        help="Use the min scoring images if set True")
# argparser.add_argument("--nn_size",
#                        default=2048, type=int,
#                        help="If avoid_multiple_class set to True, then first retrieve this number of top score images, and filter out duplicate")
argparser.add_argument("--negative_label_set", 
                        default='dynamic_negative_300', 
                        choices=NEG_DICT.keys(),
                        help="The name of this composed dataset with negative samples")


def get_negative_dataset_folder_paths(folder_path, num_of_bucket):
    main_dataset_dir = os.path.join(
        folder_path, "clip_dataset_negative", f'bucket_{num_of_bucket}')
    
    dataset_folder_paths_dict = {}
    for negative_label_set in NEGATIVE_LABEL_SETS:
        sub_folder_paths = []
        main_label_set_path = os.path.join(main_dataset_dir, negative_label_set)
        for b_idx in range(num_of_bucket):
            sub_folder_path = os.path.join(main_label_set_path, f'{b_idx}')
            sub_folder_paths.append(sub_folder_path)
            if not os.path.exists(sub_folder_path):
                os.makedirs(sub_folder_path)
        dataset_folder_paths_dict[negative_label_set] = {
            'sub_folder_paths': sub_folder_paths,
            'main_label_set_path' : main_label_set_path,
        }
    return dataset_folder_paths_dict

def check_info_dict_aligned(save_dict, new_dict):
    if save_dict['query_title'] != new_dict['query_title']:
        import pdb; pdb.set_trace()
    if save_dict['query_title_name'] != QUERY_TITLE_DICT[new_dict['query_title']]:
        import pdb; pdb.set_trace()
    if save_dict['model_name'] != new_dict['model_name']:
        import pdb; pdb.set_trace()
    if save_dict['label_set'] != new_dict['label_set']:
        import pdb; pdb.set_trace()
    if save_dict['class_size'] != new_dict['class_size']:
        import pdb; pdb.set_trace()
    if save_dict['avoid_multiple_class'] != new_dict['avoid_multiple_class']:
        import pdb; pdb.set_trace()
    if save_dict['nn_size'] != new_dict['nn_size']:
        import pdb; pdb.set_trace()
    # Didn't check for
    # 'folder_path': args.folder_path,
    # 'clip_dataset_paths': dataset_paths_dict[label_set],

if __name__ == '__main__':
    args = argparser.parse_args()

    folder_paths = large_scale_feature_chunks.get_bucket_folder_paths(args.folder_path, args.num_of_bucket)
    dataset_paths_dict = get_negative_dataset_folder_paths(args.folder_path, args.num_of_bucket)

    for negative_label_set in dataset_paths_dict.keys():
        bucket_dict = {}
        main_label_set_path = dataset_paths_dict[negative_label_set]['main_label_set_path']
        sub_folder_paths = dataset_paths_dict[negative_label_set]['sub_folder_paths']
        info_dict = NEG_DICT[negative_label_set]
        info_dict_path = os.path.join(main_label_set_path, "info_dict.pickle")
        
        if os.path.exists(info_dict_path):
            saved_info_dict = load_pickle(info_dict_path)
            if not saved_info_dict == info_dict:
                print("Info dict does not align")
                import pdb; pdb.set_trace()
        else:
            save_obj_as_pickle(info_dict_path, info_dict)
        
        print(f"Processing {negative_label_set}.. ")

        pos_dict = info_dict['positive']
        neg_dict = info_dict['negative']
        positive_label_set_name = pos_dict['label_set']
        negative_label_set_name = neg_dict['label_set']
        positive_label_set_module_name = "label_sets." + positive_label_set_name
        negative_label_set_module_name = "label_sets." + negative_label_set_name

        # The file gets executed upon import, as expected.
        positive_label_set_module = importlib.import_module(positive_label_set_module_name)
        negative_label_set_module = importlib.import_module(negative_label_set_module_name)

        positive_labels = positive_label_set_module.labels
        negative_labels = negative_label_set_module.labels

        positive_queries = [QUERY_TITLE_DICT[pos_dict['query_title']] + label for label in positive_labels]
        negative_queries = [QUERY_TITLE_DICT[neg_dict['query_title']] + label for label in negative_labels]

        query_dict = {} 
        query_dict_path = os.path.join(main_label_set_path, "query_dict.pickle")
        if os.path.exists(query_dict_path):
            print(f"Query dict already exists for {negative_label_set}")
            continue
        
        positive_dataset_paths_dict = get_dataset_folder_paths(
            args.folder_path, args.num_of_bucket, pos_dict['query_title'], 
            pos_dict['class_size'], pos_dict['avoid_multiple_class'], nn_size=pos_dict['nn_size'], reverse_order=pos_dict['reverse_order'])
        
        negative_dataset_paths_dict = get_dataset_folder_paths(
            args.folder_path, args.num_of_bucket, neg_dict['query_title'], 
            neg_dict['class_size'], neg_dict['avoid_multiple_class'], nn_size=neg_dict['nn_size'], reverse_order=neg_dict['reverse_order'])
        
        positive_main_label_set_path = positive_dataset_paths_dict[positive_label_set_name]['main_label_set_path']
        positive_sub_folder_paths = positive_dataset_paths_dict[positive_label_set_name]['sub_folder_paths']
        positive_info_dict_path = os.path.join(positive_main_label_set_path, "info_dict.pickle")
        negative_main_label_set_path = negative_dataset_paths_dict[negative_label_set_name]['main_label_set_path']
        negative_sub_folder_paths = negative_dataset_paths_dict[negative_label_set_name]['sub_folder_paths']
        negative_info_dict_path = os.path.join(negative_main_label_set_path, "info_dict.pickle")
        if os.path.exists(positive_info_dict_path) and os.path.exists(negative_info_dict_path):
            positive_saved_info_dict = load_pickle(positive_info_dict_path)
            negative_saved_info_dict = load_pickle(negative_info_dict_path)
            check_info_dict_aligned(positive_saved_info_dict, pos_dict)
            check_info_dict_aligned(negative_saved_info_dict, neg_dict)
        else:
            print("Info dict does not exist. ")
            import pdb; pdb.set_trace()
            exit(0)
        
        positive_query_dict_path = os.path.join(positive_main_label_set_path, "query_dict.pickle")
        negative_query_dict_path = os.path.join(negative_main_label_set_path, "query_dict.pickle")
        if not os.path.exists(positive_query_dict_path) or not os.path.exists(negative_query_dict_path):
            print(f"Query dict does not exist {positive_query_dict_path} or {negative_query_dict_path}")
            exit(0)
        else:
            positive_query_dict = load_pickle(positive_query_dict_path)
            negative_query_dict = load_pickle(negative_query_dict_path)
            
        if not info_dict['single_negative_class']:
            import pdb; pdb.set_trace()

        for b_idx, folder_path in enumerate(folder_paths):
            overlap_ID = [] # All ID that occur in more than one class
            query_dict[b_idx] = {}
            
            sub_folder_path = sub_folder_paths[b_idx]
            
            query_dict_i_path = os.path.join(sub_folder_path, f"query_dict_{b_idx}.pickle")
            
            if os.path.exists(query_dict_i_path):
                print(f"Exists: {query_dict_i_path}")
                continue
            else:
                print(f"Starting composing for bucket {b_idx} to create {query_dict_i_path}")
            
            positive_sub_folder_path = positive_sub_folder_paths[b_idx]
            positive_query_dict_i_path = os.path.join(positive_sub_folder_path, f"query_dict_{b_idx}.pickle")
            if not os.path.exists(positive_query_dict_i_path):
                print(f"Does not exists: {positive_query_dict_i_path}")
                import pdb; pdb.set_trace()
            else:
                positive_query_dict_i = load_pickle(positive_query_dict_i_path)
            
            negative_sub_folder_path = negative_sub_folder_paths[b_idx]
            negative_query_dict_i_path = os.path.join(negative_sub_folder_path, f"query_dict_{b_idx}.pickle")
            if not os.path.exists(negative_query_dict_i_path):
                print(f"Does not exists: {negative_query_dict_i_path}")
                import pdb; pdb.set_trace()
            else:
                negative_query_dict_i = load_pickle(negative_query_dict_i_path)
            
            indices_dict = {}
            # Positive 
            for positive_query in positive_query_dict_i:
                if sorted(positive_query_dict_i[positive_query]['D'], reverse=True) != positive_query_dict_i[positive_query]['D']:
                    import pdb; pdb.set_trace()
                query_dict[b_idx][positive_query] = {
                    'clip_features': [],
                    'metadata': [],
                    'D' : [],
                }
                for i in range(len(positive_query_dict_i[positive_query]['D'])):
                    score = positive_query_dict_i[positive_query]['D'][i]
                    clip_feature = positive_query_dict_i[positive_query]['clip_features'][i]
                    meta = positive_query_dict_i[positive_query]['metadata'][i]
                    ID = meta.get_metadata().ID
                    if ID in indices_dict:
                        import pdb; pdb.set_trace()
                    indices_dict[ID] = {
                        'D' : score,
                        'query' : positive_query,
                        'clip_feature' : clip_feature,
                        'metadata' : meta
                    }
            
            query_dict[b_idx][NEGATIVE_LABEL] = {
                'clip_features': [],
                'metadata': [],
                'D' : [],
            }
            for negative_query in negative_query_dict_i:
                if sorted(negative_query_dict_i[negative_query]['D'], reverse=True) != negative_query_dict_i[negative_query]['D']:
                    import pdb; pdb.set_trace()
                
                length_of_bucket = int(len(negative_query_dict_i[negative_query]['D']) * info_dict['negative_ratio'])
                print(f"For {negative_query} we only keep {length_of_bucket}/{len(negative_query_dict_i[negative_query]['D'])} samples")
                if 'discard_overlap' in info_dict and info_dict['discard_overlap']:
                    uniques = 0
                    print("Discard overlapping IDs from negative set..")

                    for i in tqdm(range(len(negative_query_dict_i[negative_query]['D']))):
                        if uniques >= length_of_bucket:
                            print(f"Got {uniques} IDs from {negative_query}")
                            break
                        score = negative_query_dict_i[negative_query]['D'][i]
                        clip_feature = negative_query_dict_i[negative_query]['clip_features'][i]
                        meta = negative_query_dict_i[negative_query]['metadata'][i]
                        ID = meta.get_metadata().ID
                        if ID in indices_dict:
                            continue
                        else:
                            indices_dict[ID] = {
                                'D': score,
                                'query': NEGATIVE_LABEL,
                                'clip_feature': clip_feature,
                                'metadata': meta
                            }
                            uniques += 1
                else:
                    for i in range(length_of_bucket):
                        score = negative_query_dict_i[negative_query]['D'][i]
                        clip_feature = negative_query_dict_i[negative_query]['clip_features'][i]
                        meta = negative_query_dict_i[negative_query]['metadata'][i]
                        ID = meta.get_metadata().ID
                        if ID in indices_dict:
                            overlap_ID.append(ID)
                        else:
                            indices_dict[ID] = {
                                'D' : score,
                                'query' : NEGATIVE_LABEL,
                                'clip_feature' : clip_feature,
                                'metadata' : meta
                            }
            
            for unique_idx in indices_dict:
                metadata = indices_dict[unique_idx]['metadata']
                D = indices_dict[unique_idx]['D']
                query = indices_dict[unique_idx]['query']
                clip_feature = indices_dict[unique_idx]['clip_feature']
                if query not in query_dict[b_idx]:
                    import pdb; pdb.set_trace()
                else:
                    query_dict[b_idx][query]['metadata'].append(metadata)
                    query_dict[b_idx][query]['clip_features'].append(clip_feature)
                    query_dict[b_idx][query]['D'].append(D)
                
            for query in query_dict[b_idx]:
                sorted_indices = [idx for idx, score in sorted(enumerate(query_dict[b_idx][query]['D']), key=lambda x : x[1], reverse=True)]
                query_dict[b_idx][query]['metadata'] = [query_dict[b_idx][query]['metadata'][idx] for idx in sorted_indices]
                query_dict[b_idx][query]['D'] = [query_dict[b_idx][query]['D'][idx] for idx in sorted_indices]
                query_dict[b_idx][query]['clip_features'] = [query_dict[b_idx][query]['clip_features'][idx].reshape(1,-1) for idx in sorted_indices]
                query_dict[b_idx][query]['clip_features'] = np.concatenate(query_dict[b_idx][query]['clip_features'], axis=0)

            save_obj_as_pickle(query_dict_i_path, query_dict[b_idx])
            print(f"Save at {query_dict_i_path}")
            if len(overlap_ID) > 0:
                print(f"IDs that occur in both positive and negative set: {len(overlap_ID)}")
        
        save_obj_as_pickle(query_dict_path, query_dict)
        print(f"Save at {query_dict_path}")
        
