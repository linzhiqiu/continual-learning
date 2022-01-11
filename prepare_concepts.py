# For collecting a group of visual concepts. 
import sys
sys.path.append("./CLIP")
import os
import torch
import time
from tqdm import tqdm
import shutil

from pathlib import Path
import argparse
import prepare_dataset
from utils import divide, normalize, load_json, save_as_json

device = "cuda" if torch.cuda.is_available() else "cpu"

argparser = argparse.ArgumentParser()
argparser.add_argument("--concept_group_dict",
                       default="./clear_10_config.json", type=str,
                       help="You can specify the various configs for data collection (in json format)")
argparser.add_argument("--save_all_images",
                       default=False, type=bool,
                       help="Image files for all images will be saved under 'SAVE_PATH/NAME/all_images/'")
argparser.add_argument("--save_all_metadata",
                       default=True, type=bool,
                       help="Metadata files for all images will be saved under 'SAVE_PATH/NAME/all_metadata/'")


def get_save_path(concept_group_dict):
    dataset_name = concept_group_dict['NAME']
    save_path = Path(concept_group_dict['SAVE_PATH']) / dataset_name
    return save_path
    
def get_concept_group_dict_path(concept_group_dict):
    """Copy concept_group_dict to this path
    """
    save_path = get_save_path(concept_group_dict)
    return save_path / "concept_group_dict.json"

def retrieve_examples(prompts, # a dictionary of key (label) and value (prompt)
                      retrieval_func,
                      clip_features_normalized_paths,
                      bucket_dict_b_idx,
                      allow_overlap=False,
                      class_size=600, # Num of class size
                      nn_size=16000, # Num of Nearest neighbor
                      ):
    clip_result_b_idx = {}
    assert class_size < nn_size
    indices_dict = {}
    if not allow_overlap:  
        for label in prompts:
            prompt = prompts[label] 
            # D is cosine scores
            D, indices, text_feature = retrieval_func(prompt, end_idx=nn_size)
            selected_metadata = [bucket_dict_b_idx['all_metadata'][i] for i in indices]
            for d_idx, unique_idx in enumerate(indices):
                if unique_idx in indices_dict:
                    indices_dict[unique_idx]['D'].append(D[d_idx])
                    indices_dict[unique_idx]['label'].append(label)
                else:
                    indices_dict[unique_idx] = {
                        'D': [D[d_idx]],
                        'label': [label],
                        'metadata' : selected_metadata[d_idx]
                    }
            clip_result_b_idx[label] = {
                'metadata': [],
                'D' : [],
            }

        for unique_idx in indices_dict:
            metadata = indices_dict[unique_idx]['metadata']
            max_idx, max_D = max(enumerate(indices_dict[unique_idx]['D']), key=lambda x: x[1])
            max_label = indices_dict[unique_idx]['label'][max_idx]
            if max_label not in clip_result_b_idx:
                import pdb; pdb.set_trace()
            else:
                clip_result_b_idx[max_label]['metadata'].append(metadata)
                clip_result_b_idx[max_label]['D'].append(max_D)
        
        for label in clip_result_b_idx:
            if len(clip_result_b_idx[label]['metadata']) < class_size:
                import pdb; pdb.set_trace()
            else:
                sorted_indices = [idx for idx, score in sorted(enumerate(clip_result_b_idx[label]['D']), key=lambda x : x[1], reverse=True)]
                clip_result_b_idx[label]['metadata'] = [clip_result_b_idx[label]['metadata'][idx] for idx in sorted_indices[:class_size]]
                clip_result_b_idx[label]['D'] = [clip_result_b_idx[label]['D'][idx] for idx in sorted_indices[:class_size]]
    else:
        # Allow over lap
        for label in prompts:
            prompt = prompts[label]
            D, indices, text_feature = retrieval_func(prompt, end_idx=class_size)
            selected_metadata = [bucket_dict_b_idx['all_metadata'][i] for i in indices]
            for d_idx, unique_idx in enumerate(indices):
                if unique_idx in indices_dict:
                    indices_dict[unique_idx]['D'].append(D[d_idx])
                    indices_dict[unique_idx]['label'].append(label)
                else:
                    indices_dict[unique_idx] = {
                        'D': [D[d_idx]],
                        'label': [label],
                        'metadata' : selected_metadata[d_idx]
                    }
            clip_result_b_idx[label] = {
                'metadata': selected_metadata,
                'D' : D,
            }
    return clip_result_b_idx

def compose_pos_neg_clip_result(positive_clip_result, negative_clip_result, negative_ratio=0.1):
    indices_dict = {}
    clip_result = {}
    for label in positive_clip_result:
        if sorted(positive_clip_result[label]['D'], reverse=True) != positive_clip_result[label]['D']:
            raise ValueError('The images are not sorted yet.')
        clip_result[label] = {
            'metadata': [],
            'D' : [],
        }
        for i in range(len(positive_clip_result[label]['D'])):
            score = positive_clip_result[label]['D'][i]
            meta = positive_clip_result[label]['metadata'][i]
            ID = meta['ID']
            if ID in indices_dict:
                import pdb; pdb.set_trace()
            indices_dict[ID] = {
                'D' : score,
                'label' : label,
                'metadata' : meta
            }
    
    clip_result['BACKGROUND'] = {
        'metadata': [],
        'D' : [],
    }
    for label in negative_clip_result:
        if sorted(negative_clip_result[label]['D'], reverse=True) != negative_clip_result[label]['D']:
            import pdb; pdb.set_trace()
        
        length_of_bucket = int(len(negative_clip_result[label]['D']) * negative_ratio)
        print(f"For {label} we only keep {length_of_bucket}/{len(negative_clip_result[label]['D'])} samples")
        
        uniques = 0
        print("Discard overlapping IDs from negative set..")

        for i in tqdm(range(len(negative_clip_result[label]['D']))):
            if uniques >= length_of_bucket:
                print(f"Got {uniques} IDs from {label}")
                break
            score = negative_clip_result[label]['D'][i]
            meta = negative_clip_result[label]['metadata'][i]
            ID = meta['ID']
            if ID in indices_dict:
                continue
            else:
                indices_dict[ID] = {
                    'D': score,
                    'label': 'BACKGROUND',
                    'metadata': meta
                }
                uniques += 1

    for unique_idx in indices_dict:
        metadata = indices_dict[unique_idx]['metadata']
        D = indices_dict[unique_idx]['D']
        label = indices_dict[unique_idx]['label']
        if label not in clip_result:
            import pdb; pdb.set_trace()
        else:
            clip_result[label]['metadata'].append(metadata)
            clip_result[label]['D'].append(D)
        
    for label in clip_result:
        sorted_indices = [idx for idx, score in sorted(enumerate(clip_result[label]['D']), key=lambda x : x[1], reverse=True)]
        clip_result[label]['metadata'] = [clip_result[label]['metadata'][idx] for idx in sorted_indices]
        clip_result[label]['D'] = [clip_result[label]['D'][idx] for idx in sorted_indices]
    return clip_result

if __name__ == '__main__':
    args = argparser.parse_args()
    start = time.time()
    cg = load_json(args.concept_group_dict)
    if cg == None:
        print("Concept group json file does not exist.")
    
    bucket_dict = load_json(cg['BUCKET_DICT_PATH'])

    bucket_indices = sorted(list(bucket_dict.keys()), key=lambda idx: int(idx))
    '''A sorted list of bucket indices
    '''

    folder_paths = [bucket_dict[bucket_idx]['folder_path'] for bucket_idx in bucket_indices]
    save_path = get_save_path(cg) # The main save folder
    
    class_names_path = save_path / 'class_names.txt'
    clip_result_save_path = save_path / 'clip_result.json'
    filelists_json_path = save_path / 'filelists.json'
    filelists_path = save_path / 'filelists'
    labeled_images_path = save_path / 'labeled_images'
    labeled_metadata_json_path = save_path / 'labeled_metadata.json'
    labeled_metadata_path = save_path / 'labeled_metadata'
    
    # optional
    all_images_path = save_path / 'all_images'
    all_metadata_path = save_path / 'all_metadata'
    all_metadata_json_path = save_path / 'all_metadata.json'
    
    clip_result = {} # key is bucket index (str), value is information for retrieved example

    labels = cg['GROUP']
    num_of_labels = len(labels)
    print(f"We have {num_of_labels} classes in total.")
    if not cg['PREFIX'] == "":
        print(f"Adding prefix {cg['PREFIX']} to all classes")
    prompts = {label: cg['PREFIX'] + label for label in labels}

    # prepare main folder if not exist already, and check whether 
    # concept_group_dict (if saved already) is aligned with current options
    concept_group_dict_path = get_concept_group_dict_path(cg)
    if save_path.exists() and concept_group_dict_path.exists():
        concept_group_dict_saved = load_json(concept_group_dict_path)
        if concept_group_dict_saved != cg:
            print(f'Dataset already exists at {save_path} and has conflicting options. Please double check.')
            raise ValueError(f'Dataset already exists at {save_path} and has conflicting options. Please double check.')
    else:
        save_path.mkdir()
        save_as_json(concept_group_dict_path, cg)

    # Write class names in sorted order to class_names_path
    sorted_prompts = [prompts[k] for k in prompts]
    if cg['BACKGROUND']:
        sorted_prompts += ['BACKGROUND']
    sorted_prompts = sorted(sorted_prompts)
    class_names_str = "\n".join(sorted_prompts)
    if class_names_path.exists():
        old_class_names_str = class_names_path.read_text()
        if not class_names_str == old_class_names_str:
            raise ValueError(f"Old class names do not match with current")
    else:
        with open(class_names_path, 'w+') as f:
            f.write(class_names_str)
    
    if os.path.exists(clip_result_save_path):
        print(f"{clip_result_save_path} already exists.")
        clip_result = load_json(clip_result_save_path)
    else:
        print(f"Collecting images for {cg['NAME']} then save to {clip_result_save_path}..")
        
        k_nearest_func = prepare_dataset.get_knearest_models_func(
                             bucket_dict,
                             cg['CLIP_MODEL'],
                             device=device
                         )

        for b_idx, folder_path in zip(bucket_indices, folder_paths):
            clip_features_normalized_paths = prepare_dataset.get_clip_features_normalized_paths(
                                                folder_path,
                                                cg['CLIP_MODEL']
                                             )
            save_folder_path = os.path.join(save_path, b_idx)
            clip_result[b_idx] = {}
            print(f"Starting querying for bucket {b_idx}.")
            
            k_near_faiss = k_nearest_func(b_idx)

            positive_clip_result_b_idx = retrieve_examples(
                prompts,
                k_near_faiss.grab_top_query_indices,
                clip_features_normalized_paths,
                bucket_dict[b_idx],
                allow_overlap=cg['ALLOW_OVERLAP'],
                class_size=cg['NUM_OF_IMAGES_PER_CLASS_PER_BUCKET'], # Num of class size
                nn_size=cg['NUM_OF_IMAGES_PER_CLASS_PER_BUCKET_TO_QUERY'], # Num of Nearest neighbor
            )
            
            if cg['BACKGROUND']:
                negative_clip_result_b_idx = retrieve_examples(
                    prompts,
                    k_near_faiss.grab_bottom_query_indices,
                    clip_features_normalized_paths,
                    bucket_dict[b_idx],
                    allow_overlap=False,
                    class_size=cg['NUM_OF_IMAGES_PER_CLASS_PER_BUCKET'], # Num of class size
                    nn_size=cg['NUM_OF_IMAGES_PER_CLASS_PER_BUCKET_TO_QUERY'], # Num of Nearest neighbor
                )

                clip_result[b_idx] = compose_pos_neg_clip_result(
                    positive_clip_result_b_idx,
                    negative_clip_result_b_idx,
                    negative_ratio=cg['NEGATIVE_RATIO']
                )
            else:
                clip_result[b_idx] = positive_clip_result_b_idx

        save_as_json(clip_result_save_path, clip_result)
        print(f"Save at {clip_result_save_path}")
    
    
    # prepare misc. folders if not exist already
    if not labeled_metadata_path.exists() or \
       not labeled_images_path.exists() or \
       not filelists_path.exists():
        labeled_metadata_path.mkdir(exist_ok=True)
        labeled_images_path.mkdir(exist_ok=True)
        filelists_path.mkdir(exist_ok=True)
    
    filelists_dict = {}
    labeled_metadata_dict = {}
    
    for b_idx in bucket_indices:
        labeled_metadata_path_i = labeled_metadata_path / b_idx
        labeled_metadata_path_i.mkdir(exist_ok=True)
        labeled_metadata_dict[b_idx] = {} # key is label, value is json path

        labeled_images_path_i = labeled_images_path / b_idx
        labeled_images_path_i.mkdir(exist_ok=True)
        
        filelists_path_i = filelists_path / (b_idx + ".txt")
        filelists_dict[b_idx] = str(Path('filelists') / (b_idx + ".txt"))
        filelist_strs_list_i = []
        for label in clip_result[b_idx]:
            label_index = sorted_prompts.index(label)
            labeled_images_path_i_label = labeled_images_path_i / label
            labeled_images_path_i_label.mkdir(exist_ok=True)

            labeled_metadata_path_i_label = labeled_metadata_path_i / (label + ".json")
            labeled_metadata_dict[b_idx][label] = str(Path('labeled_metadata') / b_idx / (label + ".json"))
            labeled_metadata_i_label = {} # key is flickr ID (str), value is metadata dict for this image
            for meta in clip_result[b_idx][label]['metadata']:
                original_path = Path(meta['IMG_DIR']) / meta['IMG_PATH']
                ID = meta['ID']
                EXT = meta['EXT']
                img_name = f"{ID}.{EXT}"
                transfer_path = labeled_images_path_i_label / img_name
                shutil.copy(original_path, transfer_path)
                meta['IMG_DIR'] = str(save_path)
                meta['IMG_PATH'] = str(Path("labeled_images") / b_idx / label / img_name)
                labeled_metadata_i_label[ID] = meta
                filelist_strs_list_i.append(f"{meta['IMG_PATH']} {label_index}")

            save_as_json(labeled_metadata_path_i_label, labeled_metadata_i_label)
        filelist_str = "\n".join(filelist_strs_list_i)
        with open(filelists_path_i, "w+") as f:
            f.write(filelist_str)
            
    save_as_json(filelists_json_path, filelists_dict)
    save_as_json(labeled_metadata_json_path, labeled_metadata_dict)
    
    if args.save_all_images and not args.save_all_metadata:
        raise ValueError("Save all images but without saving metadata?")
    
    if args.save_all_metadata:
        all_metadata_dict = {}
        if not all_metadata_path.exists():
            all_metadata_path.mkdir()
        for b_idx in bucket_indices:
            all_metadata_path_i = all_metadata_path / (b_idx + ".json")
            all_metadata_dict[b_idx] = str(Path('all_metadata') / (b_idx + ".json"))
            all_metadata_i = {} # key is flickr ID, value is metadata dict

            all_images_path_i = all_images_path / b_idx
            if args.save_all_images:
                all_images_path_i.mkdir(exist_ok=True)
        
            for meta in bucket_dict[b_idx]['all_metadata']:
                original_path = Path(meta['IMG_DIR']) / meta['IMG_PATH']
                transfer_path = all_images_path_i / img_name
                ID = meta['ID']
                EXT = meta['EXT']
                img_name = f"{ID}.{EXT}"
                meta['IMG_DIR'] = str(save_path)
                meta['IMG_PATH'] = str(Path("all_images") / b_idx / img_name)
                all_metadata_i[ID] = meta
                if args.save_all_images:
                    shutil.copy(original_path, transfer_path)

            save_as_json(all_metadata_path_i, all_metadata_i)
        save_as_json(all_metadata_json_path, all_metadata_dict)