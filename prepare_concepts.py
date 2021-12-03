# For collecting a group of visual concepts. 
# You should run CLIP-ConceptGroups.ipynb first to generate the dataset information
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
import shutil

import argparse
import prepare_dataset
from utils import divide, normalize, load_json, save_as_json

device = "cuda" if torch.cuda.is_available() else "cpu"

argparser = argparse.ArgumentParser()
argparser.add_argument("--concept_group_dict",
                       default="./clear_10_config.json", type=str,
                       help="You should run CLIP-ConceptGroups.ipynb to generate a concept_group_dict (in json format)")

def get_dataset_dict_path(concept_group_dict):
    """Save dataset metadata + features at this path
    """
    save_path = get_save_path(concept_group_dict)
    return os.path.join(save_path, "dataset.json")

def get_save_path(concept_group_dict):
    dataset_name = concept_group_dict['NAME']
    save_path = os.path.join(concept_group_dict['SAVE_PATH'], dataset_name)
    return save_path
    
def get_concept_group_dict_path(concept_group_dict):
    """Save concept_group_dict at this path
    """
    save_path = get_save_path(concept_group_dict)
    return os.path.join(save_path, "concept_group_dict.json")

def prepare_dataset_folder(concept_group_dict, bucket_indices):
    save_path = get_save_path(concept_group_dict)
    concept_group_dict_path = get_concept_group_dict_path(concept_group_dict)
    if os.path.exists(save_path) and os.path.exists(concept_group_dict_path):
        concept_group_dict_saved = load_json(concept_group_dict_path)
        if concept_group_dict_saved != concept_group_dict:
            print(f'Dataset already exists at {save_path} and has conflicting options. Please double check.')
            import pdb; pdb.set_trace()
    else:
        os.makedirs(save_path)
        save_as_json(concept_group_dict_path, concept_group_dict)
        for bucket_idx in bucket_indices:
            for concept in concept_group_dict['GROUP']:
                os.makedirs(os.path.join(save_path, bucket_idx, concept))
            if concept_group_dict['BACKGROUND']:
                os.makedirs(os.path.join(save_path, bucket_idx, 'BACKGROUND'))
        print(f"Save dataset folder at {save_path}")

def retrieve_examples(prompts, # a dictionary of key (label) and value (prompt)
                      retrieval_func,
                      clip_features_normalized_paths,
                      bucket_dict_b_idx,
                      allow_overlap=False,
                      class_size=600, # Num of class size
                      nn_size=16000, # Num of Nearest neighbor
                      ):
    dataset_dict_b_idx = {}
    assert class_size < nn_size
    indices_dict = {}
    if not allow_overlap:  
        for label in prompts:
            prompt = prompts[label] 
            # D is cosine scores
            D, indices, text_feature = retrieval_func(prompt, end_idx=nn_size)
            selected_clip_features = faiss_utils.aggregate_for_numpy(clip_features_normalized_paths, indices)
            selected_metadata = [bucket_dict_b_idx['all_metadata'][i] for i in indices]
            for d_idx, unique_idx in enumerate(indices):
                if unique_idx in indices_dict:
                    indices_dict[unique_idx]['D'].append(D[d_idx])
                    indices_dict[unique_idx]['label'].append(label)
                else:
                    indices_dict[unique_idx] = {
                        'D': [D[d_idx]],
                        'label': [label],
                        # 'clip_feature' : selected_clip_features[d_idx],
                        'metadata' : selected_metadata[d_idx]
                    }
            dataset_dict_b_idx[label] = {
                # 'clip_features': [],
                'metadata': [],
                'D' : [],
                # 'text_feature' : text_feature,
            }

        for unique_idx in indices_dict:
            metadata = indices_dict[unique_idx]['metadata']
            max_idx, max_D = max(enumerate(indices_dict[unique_idx]['D']), key=lambda x: x[1])
            max_label = indices_dict[unique_idx]['label'][max_idx]
            clip_feature = indices_dict[unique_idx]['clip_feature']
            if max_label not in dataset_dict_b_idx:
                import pdb; pdb.set_trace()
            else:
                dataset_dict_b_idx[max_label]['metadata'].append(metadata)
                # dataset_dict_b_idx[max_label]['clip_features'].append(clip_feature)
                dataset_dict_b_idx[max_label]['D'].append(max_D)
        
        for label in dataset_dict_b_idx:
            if len(dataset_dict_b_idx[label]['metadata']) < class_size:
                import pdb; pdb.set_trace()
            else:
                sorted_indices = [idx for idx, score in sorted(enumerate(dataset_dict_b_idx[label]['D']), key=lambda x : x[1], reverse=True)]
                dataset_dict_b_idx[label]['metadata'] = [dataset_dict_b_idx[label]['metadata'][idx] for idx in sorted_indices[:class_size]]
                dataset_dict_b_idx[label]['D'] = [dataset_dict_b_idx[label]['D'][idx] for idx in sorted_indices[:class_size]]
                # dataset_dict_b_idx[label]['clip_features'] = [dataset_dict_b_idx[label]['clip_features'][idx].reshape(1,-1) for idx in sorted_indices[:class_size]]
                # dataset_dict_b_idx[label]['clip_features'] = np.concatenate(dataset_dict_b_idx[label]['clip_features'], axis=0)
    else:
        # Allow over lap
        for label in prompts:
            prompt = prompts[label]
            D, indices, text_feature = retrieval_func(prompt, end_idx=class_size)
            selected_clip_features = faiss_utils.aggregate_for_numpy(clip_features_normalized_paths, indices)
            selected_metadata = [bucket_dict_b_idx['all_metadata'][i] for i in indices]
            for d_idx, unique_idx in enumerate(indices):
                if unique_idx in indices_dict:
                    indices_dict[unique_idx]['D'].append(D[d_idx])
                    indices_dict[unique_idx]['label'].append(label)
                else:
                    indices_dict[unique_idx] = {
                        'D': [D[d_idx]],
                        'label': [label],
                        # 'clip_feature' : selected_clip_features[d_idx],
                        'metadata' : selected_metadata[d_idx]
                    }
            dataset_dict_b_idx[label] = {
                # 'clip_features': selected_clip_features,
                'metadata': selected_metadata,
                'D' : D,
                # 'text_feature' : text_feature,
            }
    return dataset_dict_b_idx

def compose_pos_neg_dataset_dict(positive_dataset_dict, negative_dataset_dict, negative_ratio=0.1):
    indices_dict = {}
    dataset_dict = {}
    for label in positive_dataset_dict:
        if sorted(positive_dataset_dict[label]['D'], reverse=True) != positive_dataset_dict[label]['D']:
            import pdb; pdb.set_trace()
        dataset_dict[label] = {
            # 'clip_features': [],
            'metadata': [],
            'D' : [],
        }
        for i in range(len(positive_dataset_dict[label]['D'])):
            score = positive_dataset_dict[label]['D'][i]
            # clip_feature = positive_dataset_dict[label]['clip_features'][i]
            meta = positive_dataset_dict[label]['metadata'][i]
            ID = meta['ID']
            if ID in indices_dict:
                import pdb; pdb.set_trace()
            indices_dict[ID] = {
                'D' : score,
                'label' : label,
                # 'clip_feature' : clip_feature,
                'metadata' : meta
            }
    
    dataset_dict['BACKGROUND'] = {
        # 'clip_features': [],
        'metadata': [],
        'D' : [],
    }
    for label in negative_dataset_dict:
        if sorted(negative_dataset_dict[label]['D'], reverse=True) != negative_dataset_dict[label]['D']:
            import pdb; pdb.set_trace()
        
        length_of_bucket = int(len(negative_dataset_dict[label]['D']) * negative_ratio)
        print(f"For {label} we only keep {length_of_bucket}/{len(negative_dataset_dict[label]['D'])} samples")
        
        uniques = 0
        print("Discard overlapping IDs from negative set..")

        for i in tqdm(range(len(negative_dataset_dict[label]['D']))):
            if uniques >= length_of_bucket:
                print(f"Got {uniques} IDs from {label}")
                break
            score = negative_dataset_dict[label]['D'][i]
            # clip_feature = negative_dataset_dict[label]['clip_features'][i]
            meta = negative_dataset_dict[label]['metadata'][i]
            ID = meta['ID']
            if ID in indices_dict:
                continue
            else:
                indices_dict[ID] = {
                    'D': score,
                    'label': 'BACKGROUND',
                    # 'clip_feature': clip_feature,
                    'metadata': meta
                }
                uniques += 1

    for unique_idx in indices_dict:
        metadata = indices_dict[unique_idx]['metadata']
        D = indices_dict[unique_idx]['D']
        label = indices_dict[unique_idx]['label']
        # clip_feature = indices_dict[unique_idx]['clip_feature']
        if label not in dataset_dict:
            import pdb; pdb.set_trace()
        else:
            dataset_dict[label]['metadata'].append(metadata)
            # dataset_dict[label]['clip_features'].append(clip_feature)
            dataset_dict[label]['D'].append(D)
        
    for label in dataset_dict:
        sorted_indices = [idx for idx, score in sorted(enumerate(dataset_dict[label]['D']), key=lambda x : x[1], reverse=True)]
        dataset_dict[label]['metadata'] = [dataset_dict[label]['metadata'][idx] for idx in sorted_indices]
        dataset_dict[label]['D'] = [dataset_dict[label]['D'][idx] for idx in sorted_indices]
        # dataset_dict[label]['clip_features'] = [dataset_dict[label]['clip_features'][idx].reshape(1,-1) for idx in sorted_indices]
        # dataset_dict[label]['clip_features'] = np.concatenate(dataset_dict[label]['clip_features'], axis=0)
    return dataset_dict

if __name__ == '__main__':
    args = argparser.parse_args()

    start = time.time()
    cg = load_json(args.concept_group_dict)
    if cg == None:
        print("Concept group json file does not exist.")
    
    # sub_folder_paths = [os.path.join(save_path, f'{b_idx}') for b_idx in cg['NUM_OF_BUCKETS']] # Each bucket has a subfolder
    
    dataset_dict_save_path = get_dataset_dict_path(cg)
    if os.path.exists(dataset_dict_save_path):
        print(f"{dataset_dict_save_path} already exists.")
        dataset_dict = load_json(dataset_dict_save_path)
    else:
        bucket_dict = load_json(cg['BUCKET_DICT_PATH'])
        bucket_indices = sorted(list(bucket_dict.keys()), key=lambda idx: int(idx))
        save_path = get_save_path(cg) # The main save folder
        
        dataset_dict = {}
        print(f"Collecting images for {cg['NAME']}.. ")
        labels = cg['GROUP']
        num_of_labels = len(labels)
        print(f"We have {num_of_labels} classes in total.")
        if not cg['PREFIX'] == "":
            print(f"Adding prefix {cg['PREFIX']} to all classes")
        prompts = {label: cg['PREFIX'] + label for label in labels}

        prepare_dataset_folder(cg, bucket_indices) # prepare dataset folder if not exists

        dataset_dict = {}
        model, preprocess = clip.load(cg['CLIP_MODEL'], device=device)
        k_nearest_func = prepare_dataset.get_knearest_models_func(
                             bucket_dict,
                             model,
                             preprocess
                         )

        folder_paths = [bucket_dict[bucket_idx]['folder_path'] for bucket_idx in bucket_indices]
        for b_idx, folder_path in zip(bucket_indices, folder_paths):
            clip_features_normalized_paths = prepare_dataset.get_clip_features_normalized_paths(
                                                folder_path,
                                                cg['CLIP_MODEL']
                                            )
            save_folder_path = os.path.join(save_path, b_idx)
            dataset_dict[b_idx] = {}
            dataset_dict_i_path = os.path.join(save_path, f"dataset_dict_{b_idx}.json")
            if os.path.exists(dataset_dict_i_path):
                print(f"Exists: {dataset_dict_i_path}")
                dataset_dict[b_idx] = load_json(dataset_dict_i_path)
                continue
            else:
                print(f"Starting querying for bucket {b_idx}. Result will be saved at {dataset_dict_i_path}")
            
            k_near_faiss = k_nearest_func(b_idx)

            positive_dataset_dict_b_idx = retrieve_examples(
                prompts,
                k_near_faiss.grab_top_query_indices,
                clip_features_normalized_paths,
                bucket_dict[b_idx],
                allow_overlap=cg['ALLOW_OVERLAP'],
                class_size=cg['NUM_OF_IMAGES_PER_CLASS_PER_BUCKET'], # Num of class size
                nn_size=cg['NUM_OF_IMAGES_PER_CLASS_PER_BUCKET_TO_QUERY'], # Num of Nearest neighbor
            )
            
            if cg['BACKGROUND']:
                negative_dataset_dict_b_idx = retrieve_examples(
                    prompts,
                    k_near_faiss.grab_bottom_query_indices,
                    clip_features_normalized_paths,
                    bucket_dict[b_idx],
                    allow_overlap=False,
                    class_size=cg['NUM_OF_IMAGES_PER_CLASS_PER_BUCKET'], # Num of class size
                    nn_size=cg['NUM_OF_IMAGES_PER_CLASS_PER_BUCKET_TO_QUERY'], # Num of Nearest neighbor
                )

                dataset_dict[b_idx] = compose_pos_neg_dataset_dict(
                    positive_dataset_dict_b_idx,
                    negative_dataset_dict_b_idx,
                    negative_ratio=cg['NEGATIVE_RATIO']
                )
            else:
                dataset_dict[b_idx] = positive_dataset_dict_b_idx

            if len(dataset_dict[b_idx].keys()) == 0:
                import pdb; pdb.set_trace()
            save_as_json(dataset_dict_i_path, dataset_dict[b_idx])
            print(f"Save at {dataset_dict_i_path}")
        
        save_as_json(dataset_dict_save_path, dataset_dict)
        print(f"Save at {dataset_dict_save_path}")

    for b_idx, folder_path in enumerate(folder_paths):
        save_folder_path = os.path.join(save_path, f'{b_idx}')
        
        for label in dataset_dict[b_idx]:
            save_folder_path_label = os.path.join(save_folder_path, label)
            if not os.path.exists(save_folder_path_label):
                os.makedirs(save_folder_path_label)
            for meta in dataset_dict[b_idx][label]['metadata']:
                original_path = os.path.join(meta['IMG_DIR'], meta['IMG_PATH'])
                ID = meta['ID']
                EXT = meta['EXT']
                transfer_path = os.path.join(save_folder_path_label, f"{ID}.{EXT}")
                shutil.copy(original_path, transfer_path)
        print(f"Finish transferring images to {save_folder_path}")
    
