# Run after large_scale_feature_chunks.py

# Run 23: python analyze_feature_variation.py --model_name RN50 --class_size 2000 --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# Run 1-1: python analyze_feature_variation.py --model_name RN50 --class_size 2000 --query_title photo --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar

# Run 0-23: python analyze_feature_variation.py --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# Run 0-19: python analyze_feature_variation.py --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title photo --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar

# Above avoid_multiple_class does not work for imagenet 1K: 183 classes did not have more than 100 samples.
# (Pdb) min([(len(query_dict[b_idx][query]['metadata']), query) for query in query_dict[b_idx]])
# (16, 'A photo of a grey wolf')
# (Pdb) max([(len(query_dict[b_idx][query]['metadata']), query) for query in query_dict[b_idx]])
# (1830, 'A photo of a alp')
# Run 0-19: python analyze_feature_variation.py --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# Run 0-23: python analyze_feature_variation.py --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title photo --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar

# reverse: 
# python analyze_feature_variation.py --reverse_order --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_feature_variation.py  --reverse_order --model_name RN50 --class_size 2000 --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_feature_variation.py --reverse_order --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar

# python analyze_feature_variation.py --model_name RN50 --class_size 600 --nn_size 16000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_feature_variation.py --reverse_order --model_name RN50 --class_size 600 --nn_size 16000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
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

# from large_scale_feature import argparser, get_clip_loader, get_clip_features, get_feature_name, FlickrAccessor, FlickrFolder, get_flickr_accessor
from large_scale_feature_chunks import argparser
import large_scale_feature_chunks
import argparse
import importlib
from prepare_clip_dataset import QUERY_TITLE_DICT, LABEL_SETS
from utils import divide, normalize, load_pickle, save_obj_as_pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

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
argparser.add_argument("--query_title", 
                        default='none', 
                        choices=QUERY_TITLE_DICT.keys(),
                        help="The query title")
argparser.add_argument('--class_size', default=2000, type=int,
                       help='number of (max score) samples per class per bucket')
argparser.add_argument("--avoid_multiple_class",
                       action='store_true',
                       help="Only keep the max scoring images if set True")
argparser.add_argument("--reverse_order",
                       action='store_true',
                       help="Use the min scoring images if set True")
argparser.add_argument("--nn_size",
                       default=2048, type=int,
                       help="If avoid_multiple_class set to True, then first retrieve this number of top score images, and filter out duplicate")


def get_dataset_folder_paths(folder_path, num_of_bucket, query_title, class_size, avoid_multiple_class, nn_size=2048, reverse_order=False):
    # here the folder_path is args.folder_path, not the subfolders
    main_dataset_dir = os.path.join(
        folder_path, "clip_dataset", f'bucket_{num_of_bucket}')
    
    if avoid_multiple_class:
        multiple_class_str = f"_nodup_nnsize_{nn_size}"
    else:
        multiple_class_str = ""
    
    if reverse_order:
        reverse_order_str = "_reverse"
    else:
        reverse_order_str = ""
    
    query_detail_str = f"query_{query_title}_size_{class_size}{multiple_class_str}{reverse_order_str}"
    
    dataset_folder_paths_dict = {}
    for label_set in LABEL_SETS:
        sub_folder_paths = []
        main_label_set_path = os.path.join(main_dataset_dir, query_detail_str, label_set)
        for b_idx in range(num_of_bucket):
            sub_folder_path = os.path.join(main_label_set_path, f'{b_idx}')
            sub_folder_paths.append(sub_folder_path)
            if not os.path.exists(sub_folder_path):
                os.makedirs(sub_folder_path)
        dataset_folder_paths_dict[label_set] = {
            'sub_folder_paths': sub_folder_paths,
            'main_label_set_path' : main_label_set_path,
        }
    return dataset_folder_paths_dict

def get_clip_features_normalized_paths(folder_path, model_name):
    clip_features_normalized_paths = []
    main_pickle_location = large_scale_feature_chunks.get_main_save_location(folder_path, model_name)
    print(main_pickle_location)
    if os.path.exists(main_pickle_location):
        chunks, path_dict_list = load_pickle(main_pickle_location)
        print(f"Loaded from {main_pickle_location}")
    else:
        import pdb; pdb.set_trace()

    for chunk, path_dict in zip(chunks, path_dict_list):
        if os.path.exists(path_dict['normalized']):
            print(f"Already exists: {path_dict['normalized']}")
            clip_features_normalized_paths.append(
                path_dict['normalized'])
        else:
            import pdb; pdb.set_trace()
    return clip_features_normalized_paths

def get_moco_features_paths(folder_path, moco_model, arch):
    moco_features_paths = []  
    main_moco_location = large_scale_feature_chunks.get_moco_save_location(folder_path, moco_model, arch)
    if os.path.exists(main_moco_location):
        chunks, path_dict_list = load_pickle(main_moco_location)
        print(f"Loaded from {main_moco_location}")
    else:
        import pdb; pdb.set_trace()
        
    for chunk, path_dict in zip(chunks, path_dict_list):
        if os.path.exists(path_dict['original']):
            print(f"Already exists: {path_dict['original']}")
            moco_features_paths.append(path_dict['original'])
        else:
            import pdb; pdb.set_trace()
    return moco_features_paths

if __name__ == '__main__':
    args = argparser.parse_args()

    start = time.time()
    
    folder_paths = large_scale_feature_chunks.get_bucket_folder_paths(args.folder_path, args.num_of_bucket)
    dataset_paths_dict = get_dataset_folder_paths(
        args.folder_path, args.num_of_bucket, args.query_title, 
        args.class_size, args.avoid_multiple_class, nn_size=args.nn_size, reverse_order=args.reverse_order)
    bucket_dict = {}

    for label_set in dataset_paths_dict.keys():
        main_label_set_path = dataset_paths_dict[label_set]['main_label_set_path']
        sub_folder_paths = dataset_paths_dict[label_set]['sub_folder_paths']
        info_dict = {
            'query_title': args.query_title,
            'query_title_name': QUERY_TITLE_DICT[args.query_title],
            'model_name': args.model_name,
            'folder_path': args.folder_path,
            'clip_dataset_paths': dataset_paths_dict[label_set],
            'label_set': label_set,
            'class_size': args.class_size,
            'avoid_multiple_class': args.avoid_multiple_class,
            'nn_size': args.nn_size,
            'num_of_bucket' : args.num_of_bucket,
            'moco_model': args.moco_model,
            'arch' : args.arch,
        }
        info_dict_path = os.path.join(main_label_set_path, "info_dict.pickle")
        if os.path.exists(info_dict_path):
            saved_info_dict = load_pickle(info_dict_path)
            if not saved_info_dict == info_dict:
                print("Info dict does not align")
                import pdb; pdb.set_trace()
        else:
            save_obj_as_pickle(info_dict_path, info_dict)
        
        print(f"Processing {label_set}.. ")
        label_set_module_name = "label_sets." + label_set

        # The file gets executed upon import, as expected.
        label_set_module = importlib.import_module(label_set_module_name)

        labels = label_set_module.labels

        queries = [QUERY_TITLE_DICT[args.query_title] + label for label in labels]

        print(queries)
        print(f"We have {len(queries)} queries.")
        model, preprocess = clip.load(args.model_name, device='cpu')
        
        query_dict = {} # Saved the query results for each data bucket
        query_dict_path = os.path.join(main_label_set_path, "query_dict.pickle")
        if os.path.exists(query_dict_path) and not args.avoid_multiple_class:
            print(f"Query dict already exists for {label_set}")
            continue
        
        for b_idx, folder_path in enumerate(folder_paths):
            query_dict[b_idx] = {}
            if not b_idx in bucket_dict:
                bucket_dict_i_path = os.path.join(folder_path, f'bucket_{b_idx}.pickle')
                if not os.path.exists(bucket_dict_i_path):
                    import pdb; pdb.set_trace()
                bucket_dict[b_idx] = load_pickle(bucket_dict_i_path)
            
            sub_folder_path = sub_folder_paths[b_idx]
            
            query_dict_i_path = os.path.join(sub_folder_path, f"query_dict_{b_idx}.pickle")
            if os.path.exists(query_dict_i_path): # TODO
                print(f"Exists: {query_dict_i_path}")
                continue
            else:
                print(f"Starting querying for bucket {b_idx} to create {query_dict_i_path}")
                
            clip_features_normalized_paths = get_clip_features_normalized_paths(folder_path, args.model_name)
            
            moco_features_paths = get_moco_features_paths(folder_path, args.moco_model, args.arch)
            
            k_near_faiss = KNearestFaissFeatureChunks(clip_features_normalized_paths, model, preprocess)
            
            if args.avoid_multiple_class:
                assert args.class_size < args.nn_size
                indices_dict = {}
                for query in queries:
                    if args.reverse_order:
                        print("Using min score instead of max score!!!!!!!!!!!!!!!")
                        D, indices, text_feature = k_near_faiss.grab_bottom_query_indices(query, end_idx=args.nn_size)
                    else:
                        D, indices, text_feature = k_near_faiss.grab_top_query_indices(query, end_idx=args.nn_size)
                        
                    selected_clip_features = faiss_utils.aggregate_for_numpy(clip_features_normalized_paths, indices)
                    selected_moco_features = faiss_utils.aggregate_for_numpy(moco_features_paths, indices)
                    selected_metadata = [bucket_dict[b_idx]['flickr_accessor'][i] for i in indices]
                    for d_idx, unique_idx in enumerate(indices):
                        if unique_idx in indices_dict:
                            indices_dict[unique_idx]['D'].append(D[d_idx])
                            indices_dict[unique_idx]['query'].append(query)
                        else:
                            indices_dict[unique_idx] = {
                                'D': [D[d_idx]],
                                'query': [query],
                                'clip_feature' : selected_clip_features[d_idx],
                                'moco_feature' : selected_moco_features[d_idx],
                                'metadata' : selected_metadata[d_idx]
                            }
                    query_dict[b_idx][query] = {
                        'clip_features': [],
                        'clip_features_mean': None,
                        'moco_features': [],
                        'moco_features_mean': None,
                        'metadata': [],
                        'D' : [],
                        'text_feature' : text_feature,
                    }

                for unique_idx in indices_dict:
                    metadata = indices_dict[unique_idx]['metadata']
                    max_idx, max_D = max(enumerate(indices_dict[unique_idx]['D']), key=lambda x: x[1])
                    max_query = indices_dict[unique_idx]['query'][max_idx]
                    clip_feature = indices_dict[unique_idx]['clip_feature']
                    moco_feature = indices_dict[unique_idx]['moco_feature']
                    if max_query not in query_dict[b_idx]:
                        import pdb; pdb.set_trace()
                    else:
                        query_dict[b_idx][max_query]['metadata'].append(metadata)
                        query_dict[b_idx][max_query]['clip_features'].append(clip_feature)
                        query_dict[b_idx][max_query]['moco_features'].append(moco_feature)
                        query_dict[b_idx][max_query]['D'].append(max_D)
                
                for query in query_dict[b_idx]:
                    if len(query_dict[b_idx][query]['metadata']) < args.class_size:
                        import pdb; pdb.set_trace()
                    else:
                        sorted_indices = [idx for idx, score in sorted(enumerate(query_dict[b_idx][query]['D']), key=lambda x : x[1], reverse=True)]
                        query_dict[b_idx][query]['metadata'] = [query_dict[b_idx][query]['metadata'][idx] for idx in sorted_indices[:args.class_size]]
                        query_dict[b_idx][query]['D'] = [query_dict[b_idx][query]['D'][idx] for idx in sorted_indices[:args.class_size]]
                        query_dict[b_idx][query]['clip_features'] = [query_dict[b_idx][query]['clip_features'][idx].reshape(1,-1) for idx in sorted_indices[:args.class_size]]
                        query_dict[b_idx][query]['moco_features'] = [query_dict[b_idx][query]['moco_features'][idx].reshape(1,-1) for idx in sorted_indices[:args.class_size]]
                        query_dict[b_idx][query]['clip_features'] = np.concatenate(query_dict[b_idx][query]['clip_features'], axis=0)
                        query_dict[b_idx][query]['moco_features'] = np.concatenate(query_dict[b_idx][query]['moco_features'], axis=0)
                        query_dict[b_idx][query]['clip_features_mean'] = query_dict[b_idx][query]['clip_features'].mean(axis=0)
                        query_dict[b_idx][query]['moco_features_mean'] = query_dict[b_idx][query]['moco_features'].mean(axis=0)
            else:
                for query in queries:
                    if args.reverse_order:
                        print("Using min score instead of max score!!!!!!!!!!!!!!!")
                        D, indices, text_feature = k_near_faiss.grab_bottom_query_indices(query, end_idx=args.nn_size)
                    else:
                        D, indices, text_feature = k_near_faiss.grab_top_query_indices(query, end_idx=args.class_size)
                    selected_clip_features = faiss_utils.aggregate_for_numpy(clip_features_normalized_paths, indices)
                    selected_moco_features = faiss_utils.aggregate_for_numpy(moco_features_paths, indices)
                    selected_metadata = [bucket_dict[b_idx]['flickr_accessor'][i] for i in indices]
                    assert selected_clip_features.shape[0] == len(selected_metadata) == selected_moco_features.shape[0]
                    query_dict[b_idx][query] = {
                        'clip_features': selected_clip_features,
                        'clip_features_mean': selected_clip_features.mean(axis=0),
                        'moco_features': selected_moco_features,
                        'moco_features_mean': selected_moco_features.mean(axis=0),
                        'metadata': selected_metadata,
                        'D' : D,
                        'text_feature' : text_feature,
                    }
                
            save_obj_as_pickle(query_dict_i_path, query_dict[b_idx])
            print(f"Save at {query_dict_i_path}")
        
        save_obj_as_pickle(query_dict_path, query_dict)
        print(f"Save at {query_dict_path}")
        
