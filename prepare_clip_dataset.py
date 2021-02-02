import sys
sys.path.append("./CLIP")
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

argparser = argparse.ArgumentParser()
argparser.
argparser.add_argument("--model_name", 
                        default='ViT-B/32', choices=clip.available_models(),
                        help="The CLIP model to use")
argparser.add_argument("--folder_path", 
                        # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_25',
                        default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_31',
                        help="The folder with all the computed+normalized CLIP features")
argparser.add_argument("--clip_dataset_path", 
                        # default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_25',
                        default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_jan_31',
                        help="The folder with all the computed+normalized CLIP features")
argparser.add_argument("--label_set", 
                        default='vehicle_7', choices=['imagenet1K', 'vehicle_7'],
                        help="The label sets")
argparser.add_argument('--class_size', default=2000, type=int,
                       help='number of samples per class')
argparser.add_argument("--query_title", 
                        default='A photo of a ', 
                        help="The query title")

# Contrived example of generating a module named as a string
label_set_module_name = "label_sets." + args.label_set

# The file gets executed upon import, as expected.
label_set_module = importlib.import_module(label_set_module_name)

labels = label_set_module_name.labels

queries = [args.query_title + label for label in labels]

print(queries)
print(f"We have {len(queries)} queries.")

k_near_faiss = KNearestFaiss(args.folder_path, args.model_name)

def grab_top_query_images(query, start_idx=0, end_idx=40):
    start = time.time()
    normalize_text_feature = k_near_faiss.get_normalized_text_feature(query)
    end_feature = time.time()
    _, meta_list, clip_features = k_near_faiss.k_nearest_meta_and_clip_feature(normalize_text_feature, k=end_idx)
    end_search = time.time()
    print(f"{end_feature-start:.4f} for querying {query}. {end_search-end_feature} for computing KNN.")
    return meta_list[start_idx:end_idx], clip_features[start_idx:end_idx]

save_path = os.path.join(args.clip_dataset_path, args.label_set, f"size_{args.class_size}")
info_dict = {
    'query_title' : args.query_title,
    'model_name'  : args.model_name,
    'folder_path' : args.folder_path,
    'clip_dataset_path' : args.clip_dataset_path,
    'save_path', save_path,
    'label_set' : args.label_set,
    'class_size' : args.class_size
}

info_dict_path = os.path.join(save_path, "details.pickle")
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
feature_dict_path = os.path.join(save_path, "feature_dict.pickle")
if os.path.exists(query_dict_path):
    print(f"Overwrite {query_dict_path}?")
    import pdb; pdb.set_trace()

if os.path.exists(feature_dict_path):
    print(f"Overwrite {feature_dict_path}?")
    import pdb; pdb.set_trace()

query_dict = {}
feature_dict = {}
for query in queries:
    query_meta_list, query_clip_features = grab_top_query_images(query, end_idx=args.class_size)
    query_dict[query] = query_meta_list
    feature_dict[query] = query_clip_features

save_obj_as_pickle(query_dict_path, query_dict)
save_obj_as_pickle(feature_dict_path, feature_dict)
print(f"Save at {query_dict_path} and {feature_dict_path}")
    
