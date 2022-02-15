import argparse
from tqdm import tqdm
import copy
import time
import numpy as np
import torch
import os
from pathlib import Path
import shutil
from utils import load_json, save_as_json


device = "cuda" if torch.cuda.is_available() else "cpu"

argparser = argparse.ArgumentParser()
argparser.add_argument("--bucket_dict_path",
                       default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/bucket_11.json',
                       help="The folder with the images and query_dict.pickle")
argparser.add_argument("--new_folder_path",
                       default='/data3/zhiqiul/clear_datasets/CLEAR-10-PUBLIC',
                       help="Where the public folder will reside")
def get_save_path(concept_group_dict):
    dataset_name = concept_group_dict['NAME']
    save_path = Path(concept_group_dict['SAVE_PATH']) / dataset_name
    return save_path
if __name__ == '__main__':
    args = argparser.parse_args()
    start = time.time()
    cg = load_json("./clear_10_config.json")
    labels = cg['GROUP']
    num_of_labels = len(labels)
    print(f"We have {num_of_labels} classes in total.")
    if not cg['PREFIX'] == "":
        print(f"Adding prefix {cg['PREFIX']} to all classes")
    prompts = {label: cg['PREFIX'] + label for label in labels}

    bucket_dict = load_json(args.bucket_dict_path)
    end = time.time()
    print(f"{start - end} seconds for loading")
    # excluded_bucket_idx = args.excluded_bucket_idx
    bucket_indices = sorted(list(bucket_dict.keys()), key=lambda idx: int(idx))
    '''A sorted list of bucket indices
    '''
    bucket_indices = bucket_indices[1:]
    folder_paths = [bucket_dict[bucket_idx]['folder_path'] for bucket_idx in bucket_indices]
    save_path = Path(args.new_folder_path)
    
    class_names_path = save_path / 'class_names.txt'
    assert class_names_path.exists()
    # clip_result_save_path = save_path / 'clip_result.json'
    # assert clip_result_save_path.exists()
    labeled_images_path = save_path / 'labeled_images'
    labeled_metadata_json_path = save_path / 'labeled_metadata.json'
    labeled_metadata_path = save_path / 'labeled_metadata'
    assert labeled_metadata_path.exists()
    assert labeled_images_path.exists()
    
    # optional
    all_images_path = save_path / 'all_images'
    all_metadata_path = save_path / 'all_metadata'
    all_metadata_json_path = save_path / 'all_metadata.json'
    
    # clip_result = load_json(clip_result_save_path)
    # Write class names in sorted order to class_names_path
    label_map_dict = None
    sorted_prompts = [prompts[k] for k in prompts]

    sorted_prompts += ['BACKGROUND']
    sorted_prompts = sorted(sorted_prompts)
    
    
    
    old_labeled_metadata_dict = load_json(labeled_metadata_json_path)
    for b_idx in bucket_indices:
        print(f"Checking on {b_idx}")
        labeled_metadata_path_i = labeled_metadata_path / b_idx
        assert labeled_metadata_path_i.exists()

        labeled_images_path_i = labeled_images_path / b_idx
        assert labeled_images_path_i.exists()
        
        for label in sorted_prompts:
            labeled_images_path_i_label = labeled_images_path_i / label
            if not labeled_images_path_i_label.exists():
                import pdb; pdb.set_trace()

            labeled_metadata_path_i_label = labeled_metadata_path_i / (label + ".json")
            old_labeled_metadata_i_label = load_json(labeled_metadata_path_i_label)
            assert old_labeled_metadata_dict[b_idx][label] == str(Path('labeled_metadata') / b_idx / (label + ".json"))
            # for meta in clip_result[b_idx][label]['metadata']:
            #     ID = meta['ID']
            #     EXT = meta['EXT']
            #     img_name = f"{ID}.{EXT}"
            #     assert meta['IMG_DIR'] == str(save_path)
            #     assert meta['IMG_PATH'] == str(Path("labeled_images") / b_idx / label / img_name)
            #     assert old_labeled_metadata_i_label[ID] == meta
                # meta['IMG_PATH'] = str(Path("labeled_images") / b_idx / label / img_name)
                # labeled_metadata_i_label[ID] = meta


    all_metadata_dict = {}
    if not all_metadata_path.exists():
        all_metadata_path.mkdir()
    for b_idx in bucket_indices:
        all_metadata_path_i = all_metadata_path / (b_idx + ".json")
        all_metadata_dict[b_idx] = str(Path('all_metadata') / (b_idx + ".json"))
        all_metadata_i = {} # key is flickr ID, value is metadata dict

        
        for meta in bucket_dict[b_idx]['all_metadata']:
            # original_path = Path(meta['IMG_DIR']) / meta['IMG_PATH']
            # transfer_path = all_images_path_i / img_name
            ID = meta['ID']
            EXT = meta['EXT']
            img_name = f"{ID}.{EXT}"
            meta['IMG_DIR'] = str(save_path)
            meta['IMG_PATH'] = str(Path("all_images") / b_idx / img_name)
            all_metadata_i[ID] = meta
            # if args.save_all_images:
            #     shutil.copy(original_path, transfer_path)
            # # original_path = Path(meta['IMG_DIR']) / meta['IMG_PATH']
            # # if args.save_all_images:
            # #     transfer_path = all_images_path_i / img_name
            # #     shutil.copy(original_path, transfer_path)
            # #     ID = meta['ID']
            # #     EXT = meta['EXT']
            # #     img_name = f"{ID}.{EXT}"
            # meta['IMG_DIR'] = str(new_folder_path)
            # meta['IMG_PATH'] = str(Path("all_images") / b_idx / img_name)
            # all_metadata_i[ID] = meta

        save_as_json(all_metadata_path_i, all_metadata_i)
    save_as_json(all_metadata_json_path, all_metadata_dict)
    
    