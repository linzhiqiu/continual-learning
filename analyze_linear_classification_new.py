# Negative
#  300+3000 negative per bucket
# python analyze_linear_classification_new.py --use_negative_samples --train_mode cnn_scratch --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification_new.py --use_negative_samples --train_mode cnn_imgnet --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification_new.py --use_negative_samples --train_mode cnn_byol --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification_new.py --use_negative_samples --train_mode cnn_moco --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_linear_classification_new.py --use_negative_samples --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0 --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification_new.py --use_negative_samples --train_mode linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification_new.py --use_negative_samples --train_mode moco_v2_imgnet_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification_new.py --use_negative_samples --train_mode byol_imgnet_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification_new.py --use_negative_samples --train_mode imgnet_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification_new.py --use_negative_samples --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar

# Negative no test set
#  300+3000 negative per bucket
# python analyze_linear_classification_new.py --mode no_test_set --use_negative_samples --train_mode cnn_scratch --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# CUDA_VISIBLE_DEVICES=3 python analyze_linear_classification_new.py --mode no_test_set --use_negative_samples --train_mode cnn_imgnet --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# CUDA_VISIBLE_DEVICES=3 python analyze_linear_classification_new.py --mode no_test_set --use_negative_samples --train_mode cnn_byol --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# CUDA_VISIBLE_DEVICES=3 python analyze_linear_classification_new.py --mode no_test_set --use_negative_samples --train_mode cnn_moco --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_linear_classification_new.py --mode no_test_set --use_negative_samples --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0 --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_linear_classification_new.py --mode no_test_set --use_negative_samples --train_mode linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_linear_classification_new.py --mode no_test_set --use_negative_samples --train_mode moco_v2_imgnet_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_linear_classification_new.py --mode no_test_set --use_negative_samples --train_mode byol_imgnet_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_linear_classification_new.py --mode no_test_set --use_negative_samples --train_mode imgnet_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_linear_classification_new.py --mode no_test_set --use_negative_samples --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar

# MLP
# Negative no test set
# python analyze_linear_classification_new.py --mode no_test_set --use_negative_samples --train_mode mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_linear_classification_new.py --mode no_test_set --use_negative_samples --train_mode moco_v2_imgnet_mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_linear_classification_new.py --mode no_test_set --use_negative_samples --train_mode byol_imgnet_mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_linear_classification_new.py --mode no_test_set --use_negative_samples --train_mode imgnet_mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_linear_classification_new.py --mode no_test_set --use_negative_samples --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# Negative with test set
# python analyze_linear_classification_new.py --mode default --use_negative_samples --train_mode mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_linear_classification_new.py --mode default --use_negative_samples --train_mode moco_v2_imgnet_mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_linear_classification_new.py --mode default --use_negative_samples --train_mode byol_imgnet_mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_linear_classification_new.py --mode default --use_negative_samples --train_mode imgnet_mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_linear_classification_new.py --mode default --use_negative_samples --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
# python analyze_linear_classification_new.py --mode default --use_negative_samples --train_mode raw_feature --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
from analyze_feature_variation_negative import NEGATIVE_LABEL, NEGATIVE_LABEL_SETS, get_negative_dataset_folder_paths
import training_utils
from training_utils import CLIPDataset, TensorDataset, make_numpy_loader, make_image_loader, make_tensor_loader, make_optimizer, make_scheduler
from utils import divide, normalize, load_pickle, save_obj_as_pickle
from prepare_clip_dataset import QUERY_TITLE_DICT, LABEL_SETS
import importlib
import random
import argparse
import large_scale_feature_chunks
from analyze_feature_variation import argparser, get_dataset_folder_paths
from datetime import datetime
from tqdm import tqdm
import copy
import time
import numpy as np
from faiss_utils import KNearestFaissFeatureChunks
import faiss_utils
import torch
import clip
import os

import sys
sys.path.append("./CLIP")

# from large_scale_feature import argparser, get_clip_loader, get_clip_features, get_feature_name, FlickrAccessor, FlickrFolder, get_flickr_accessor


device = "cuda" if torch.cuda.is_available() else "cpu"

MODE_DICT = {
    'default': {
        'VAL_SET_RATIO': 0.1,
        'TEST_SET_RATIO': 0.1,
        'TRAIN_SET_RATIO': 0.8,
    },
    'no_test_set': {
        'TEST_SET_RATIO': 0.3,
        'TRAIN_SET_RATIO': 0.7,
    },
}


class HyperParameter():
    def __init__(self, network_name, epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0.):
        self.network_name = network_name
        self.epochs = epochs
        self.step = step
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

    def get_detail_str(self):
        if self.network_name in TRAIN_MODES_CATEGORY['nearest_mean']:
            return self.network_name
        else:
            return "_".join([self.network_name, 'ep', self.epochs, 'step', self.step, 'b', self.batch_size, 'lr', self.lr, 'wd', self.weight_decay])


HYPER_DICT = {
    'mlp': HyperParameter('mlp', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0.),
    'linear': HyperParameter('linear', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0.),
    'cnn': HyperParameter('cnn', epochs=150, step=60, batch_size=64, lr=0.1, weight_decay=1e-5),
}

ALL_PRETRAINED_WEIGHTS = ['moco_yfcc_feb18_gpu_8_bucket_0', 'imgnet', 'moco_imgnet', 'byol_imgnet', None]
ALL_FEATURE_TYPES = ['image', 'clip', 'cnn_feature']
ALL_NETWORK_TYPES = HYPER_DICT.keys()
class TrainMode():
    def __init__(self, feature_type, pretrained_weight=None, network_type='cnn'):
        assert pretrained_weight in ALL_PRETRAINED_WEIGHTS
        assert feature_type in ALL_FEATURE_TYPES
        assert network_type in ALL_NETWORK_TYPES
        if network_type == 'cnn':
            assert feature_type == 'image'
        else:
            assert feature_type != 'image'
        self.feature_type = feature_type
        self.pretrained_weight = pretrained_weight
        self.network_type = network_type

TRAIN_MODES_CATEGORY = {
    'cnn_scratch': TrainMode('image', pretrained_weight=None, network_type='cnn'),
    'cnn_imgnet': TrainMode('image', pretrained_weight='imgnet', network_type='cnn'),
    'cnn_moco': TrainMode('image', pretrained_weight='moco_imgnet', network_type='cnn'),
    'cnn_byol': TrainMode('image', pretrained_weight='byol_imgnet', network_type='cnn'),
    'cnn_moco_yfcc_feb18_gpu_8_bucket_0': TrainMode('image', pretrained_weight='moco_yfcc_feb18_gpu_8_bucket_0', network_type='cnn'),
    'moco_v2_imgnet_linear': TrainMode('cnn_feature', pretrained_weight='moco_imgnet', network_type='linear'),
    'byol_imgnet_linear': TrainMode('cnn_feature', pretrained_weight='byol_imgnet', network_type='linear'),
    'imgnet_linear': TrainMode('cnn_feature', pretrained_weight='imgnet', network_type='linear'),
    'raw_feature': TrainMode('cnn_feature', pretrained_weight=None, network_type='linear'),
    'moco_v2_yfcc_feb18_bucket_0_gpu_8_linear': TrainMode('cnn_feature', pretrained_weight='moco_yfcc_feb18_gpu_8_bucket_0', network_type='linear'),
    'moco_v2_imgnet_mlp': TrainMode('cnn_feature', pretrained_weight='moco_imgnet', network_type='mlp'),
    'byol_imgnet_mlp': TrainMode('cnn_feature', pretrained_weight='byol_imgnet', network_type='mlp'),
    'imgnet_mlp': TrainMode('cnn_feature', pretrained_weight='imgnet', network_type='mlp'),
    'moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp': TrainMode('cnn_feature', pretrained_weight='moco_yfcc_feb18_gpu_8_bucket_0', network_type='mlp'),
    'linear' : TrainMode('clip', pretrained_weight=None, network_type='linear'),
    'mlp': TrainMode('clip', pretrained_weight=None, network_type='mlp'),
}
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
#                        help="(Must be true for linear classification) Only keep the max scoring images if set True")
# argparser.add_argument("--nn_size",
#                        default=2048, type=int,
#                        help="If avoid_multiple_class set to True, then first retrieve this number of top score images, and filter out duplicate")
argparser.add_argument("--excluded_label_set",
                       default=['tech_7', 'imagenet1K'],
                       help="Do not evaluate on these listed label set")
argparser.add_argument("--only_label_set",
                       default=None,
                       help="If set to a [labet_set], only evaluate on this label set")
argparser.add_argument("--use_negative_samples",
                       action='store_true',
                       help="If set True, then the dataset contains a negative class (run analyze_feature_variation_negative.py first)")
argparser.add_argument('--train_mode',
                       default='linear', choices=TRAIN_MODES_CATEGORY.keys(),
                       help='Train mode')
argparser.add_argument('--mode',
                       default='default', choices=MODE_DICT.keys(),
                       help='Mode for dataset split')
argparser.add_argument('--seed',
                       default=None, type=int,
                       help='Seed for experiment')

def get_seed_str(seed):
    if seed == None:
        return ""
    else:
        return f"_seed_{seed}"

def use_val_set(mode):
    return 'VAL_SET_RATIO' in MODE_DICT[mode]

def dataset_str(mode):
    if use_val_set(mode):
        return "_".join(['train', str(MODE_DICT[mode]['TRAIN_SET_RATIO']),
                         'val', str(MODE_DICT[mode]['VAL_SET_RATIO']),
                         'test', str(MODE_DICT[mode]['TEST_SET_RATIO'])])
    else:
        return "_".join(['train', str(MODE_DICT[mode]['TRAIN_SET_RATIO']),
                         'test', str(MODE_DICT[mode]['TEST_SET_RATIO'])])

def get_all_query(query_dict):
    all_query = sorted(list(query_dict.keys()))
    # For negative
    if NEGATIVE_LABEL in all_query:
        print(f"Removed {NEGATIVE_LABEL}")
        all_query.remove(NEGATIVE_LABEL)
    #
    return all_query

# def remove_random_crop_from_loader(loader):
#     _, test_transform = training_utils.get_imgnet_transforms()
#     loader.__dict__['dataset'].__dict__['transform'] = test_transform
#     return loader

def make_dataset_dict(query_dict, mode):
    dataset_dict = {}
    for b_idx in query_dict:
        print(f"<<<<<<<<<<<First create split the dataset for bucket {b_idx}")
        dataset_dict[b_idx] = split_dataset(query_dict[b_idx], mode)
    return dataset_dict

def split_dataset(query_dict, mode):
    dataset_dict = {}

    def gather_data(query, indices):
        return {
            'clip_features': [query_dict[query]['clip_features'][i] for i in indices],
            'metadata': [query_dict[query]['metadata'][i] for i in indices],
            'D': [query_dict[query]['D'][i] for i in indices],
        }

    for query in query_dict:
        num_of_data = len(query_dict[query]['metadata'])
        # for query in all_query:
        #     assert num_of_data == len(query_dict[query]['metadata'])
        data_indices = list(range(num_of_data))
        random.shuffle(data_indices)
        if use_val_set(mode):
            val_set_size = int(MODE_DICT[mode]['VAL_SET_RATIO'] * num_of_data)
        else:
            val_set_size = 0
        val_set_indices = data_indices[:val_set_size]

        test_set_size = int(MODE_DICT[mode]['TEST_SET_RATIO'] * num_of_data)
        test_set_indices = data_indices[val_set_size:val_set_size+test_set_size]
        train_set_size = int(MODE_DICT[mode]['TRAIN_SET_RATIO'] * num_of_data)
        train_set_indices = data_indices[val_set_size+test_set_size:]
        total_size = len(train_set_indices) + len(val_set_indices) + len(test_set_indices)
        if not total_size == num_of_data:
            import pdb; pdb.set_trace()
        dataset_dict[query] = {}
        dataset_dict[query]['train'] = gather_data(query, train_set_indices)
        if use_val_set(mode):
            dataset_dict[query]['val'] = gather_data(query, val_set_indices)
        dataset_dict[query]['test'] = gather_data(query, test_set_indices)
        # TODO: Handle when dataset_dict has empty val set
        dataset_dict[query]['all'] = gather_data(query, data_indices)

    return dataset_dict

def make_features_dict(dataset_dict, train_mode):
    features_dict = {}  # Saved the features of splitted dataset
    feature_name, feature_extractor = make_feature_extractor(train_mode)
    for b_idx in dataset_dict:
        print(f"<<<<<<<<<<<First store features for bucket {b_idx}")
        features_dict[b_idx] = extract_features(dataset_dict[b_idx], feature_name, feature_extractor)
    return features_dict

def extract_features(dataset_dict_i, feature_name, feature_extractor, batch_size=64):
    # batch size here is simply used to extract the features, not for final training purposes
    all_query = sorted(list(dataset_dict_i.keys()))
    features_dict_i = {}
    if feature_extractor == None:
        for k_name in dataset_dict_i[all_query[0]]:
            items = []
            for q_idx, query in enumerate(all_query):
                items += [(f, q_idx)
                          for f in dataset_dict_i[query][k_name][feature_name]]
            features_dict_i[k_name] = items
    else:
        feature_extractor = feature_extractor.cuda()
        for k_name in dataset_dict_i[all_query[0]]:
            items = []
            for q_idx, query in enumerate(all_query):
                items += [(f, q_idx)
                          for f in dataset_dict_i[query][k_name]['metadata']]
            loader = make_image_loader(items, batch_size, shuffle=False, fixed_crop=True)
            extracted_items = []
            for inputs, labels in tqdm(loader):
                inputs = inputs.cuda()
                outputs = feature_extractor(inputs)
                for output, label in zip(outputs, labels):
                    extracted_items.append((output.cpu(), int(label)))
                    assert int(label) == items[len(extracted_items)-1][1]
            features_dict_i[k_name] = extracted_items
    return features_dict_i

def get_loader_func(train_mode, batch_size):
    assert train_mode in TRAIN_MODES_CATEGORY.keys()
    feature_type = TRAIN_MODES_CATEGORY[train_mode].feature_type
    if feature_type == 'clip':
        return lambda items, is_train_mode: make_numpy_loader(items, batch_size, shuffle=is_train_mode)
    elif feature_type == 'image':
        # always do center cropping
        return lambda items, is_train_mode: make_image_loader(items, batch_size, shuffle=is_train_mode, fixed_crop=True)
    elif feature_type == 'cnn_feature':
        return lambda items, is_train_mode: make_tensor_loader(items, batch_size, shuffle=is_train_mode)
    else:
        raise NotImplementedError()

def get_all_loaders_from_features_dict(all_features_dict, train_mode, hyperparameter, excluded_bucket_idx=0):
    all_bucket = sorted(list(all_features_dict.keys()))
    print(f"Excluding {excluded_bucket_idx} from the loaders")
    features_dict = {k: all_features_dict[k]
                     for k in all_bucket if k != excluded_bucket_idx}
    loaders_dict = {}

    loader_func = get_loader_func(train_mode, hyperparameter.batch_size)

    for k_name in features_dict[list(features_dict.keys())[0]]:
        items = []
        for b_idx in list(features_dict.keys()):
            items += features_dict[b_idx][k_name]
        if k_name == 'train':
            is_train_mode = True
        else:
            is_train_mode = False
        loader = loader_func(items, is_train_mode)
        loaders_dict[k_name] = loader
    return loaders_dict

def get_excluded_bucket_idx(moco_model):
    moco_paths = moco_model.split(os.sep)
    model_configs = moco_paths[-2].split("_")
    excluded_bucket_idx = model_configs[model_configs.index('idx')+1]
    return int(excluded_bucket_idx)

def get_loaders_from_features_dict(features_dict, train_mode, hyperparameter):
    loaders_dict = {}  # Saved the splitted loader for each bucket

    loader_func = get_loader_func(train_mode, hyperparameter.batch_size)

    for b_idx in features_dict:
        loaders_dict[b_idx] = {}
        for k_name in features_dict[b_idx]:
            items = features_dict[b_idx][k_name]
            if k_name == 'train':
                is_train_mode = True
            else:
                is_train_mode = False
            loader = loader_func(items, is_train_mode)
            loaders_dict[b_idx][k_name] = loader
    return loaders_dict

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output

def make_feature_extractor(train_mode):
    assert train_mode in TRAIN_MODES_CATEGORY.keys()
    feature_type = TRAIN_MODES_CATEGORY[train_mode].feature_type
    if feature_type == 'clip':
        feature_extractor = None
        feature_name = 'clip_features'
    elif feature_type == 'image':
        feature_extractor = None
        feature_name = 'metadata'
    elif feature_type == 'cnn_feature':
        feature_extractor = make_cnn_model(TRAIN_MODES_CATEGORY[train_mode].pretrained_weight, output_size=None, train_mode='freeze')
        feature_name = 'metadata'
    return feature_name, feature_extractor

def make_cnn_model(pretrained_weight, output_size=1000, train_mode='finetune'):
    print(f"Using ResNet 50 (frozen feature extractor)")
    pretrained = False
    selfsupervised = False
    if pretrained_weight == 'imgnet':
        pretrained = True
    elif pretrained_weight == 'moco_imgnet':
        selfsupervised = 'moco_v2'
    elif pretrained_weight == 'byol_imgnet':
        selfsupervised = 'byol'
    elif pretrained_weight == "moco_yfcc_feb18_gpu_8_bucket_0":
        selfsupervised = "moco_v2_yfcc_feb18_bucket_0_gpu_8"
    elif pretrained_weight == None:
        pass
    else:
        raise NotImplementedError()
    model = training_utils.make_model(
        'resnet50',
        pretrained,
        selfsupervised,
        train_mode=train_mode,
        output_size=output_size
    )
    if train_mode == 'freeze':
        model.eval()
    return model

def get_input_size(train_mode):
    feature_type = TRAIN_MODES_CATEGORY[train_mode].feature_type
    if feature_type == 'clip':
        input_size = 1024
    elif feature_type == 'cnn_feature':
        input_size = 2048
    elif feature_type == 'image':
        input_size = None
    else:
        raise NotImplementedError()
    return input_size

def make_model(train_mode, input_size=1024, output_size=1000):
    network_type = TRAIN_MODES_CATEGORY[train_mode].network_type
    input_size = get_input_size(train_mode)
    if network_type == 'mlp':
        print(f"Using a mlp network with input size {input_size}")
        mlp = MLP(input_size, 2048, output_size)
        return mlp
    elif network_type == 'linear':
        print(f"Using a single linear layer")
        fc = torch.nn.Linear(input_size, output_size)
        # import pdb; pdb.set_trace()
        # fc.weight.data.normal_(mean=0.0, std=0.01)
        # fc.bias.data.zero_()
        return fc
    elif network_type == 'cnn':
        return make_cnn_model(TRAIN_MODES_CATEGORY[train_mode].pretrained_weight,
                              output_size=output_size,
                              train_mode='finetune')

def train(loaders,
          train_mode, output_size,
          epochs=150, lr=0.1, weight_decay=1e-5, step_size=60,
          finetuned_model=None):
    if finetuned_model == None:
        network = make_model(train_mode, output_size).cuda()
        print("Retraining..")
    else:
        network = finetuned_model
        print("Finetuning..")
    
    optimizer = make_optimizer(network, lr, weight_decay)
    scheduler = make_scheduler(optimizer, step_size=step_size)
    criterion = torch.nn.NLLLoss(reduction='mean')

    avg_results = {'train': {'loss_per_epoch': [], 'acc_per_epoch': []},
                   'test':  {'loss_per_epoch': [], 'acc_per_epoch': []}}
    if 'val' in loaders:
        avg_results['val'] = {'loss_per_epoch': [], 'acc_per_epoch': []}
        model_selection_criterion = 'val'
        phases = ['train', 'val', 'test']
    else:
        model_selection_criterion = 'test'
        phases = ['train', 'test']

    best_result = {'best_acc': 0, 'best_epoch': None, 'best_network': None}

    for epoch in range(0, epochs):
        print(f"Epoch {epoch}")
        for phase in phases:
            if phase == 'train':
                network.train()
            else:
                network.eval()

            running_loss = 0.0
            running_corrects = 0.
            count = 0

            pbar = loaders[phase]

            for batch, data in enumerate(pbar):
                inputs, labels = data
                count += inputs.size(0)

                inputs = inputs.cuda()
                labels = labels.cuda()

                if phase == 'train':
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = network(inputs)
                    _, preds = torch.max(outputs, 1)

                    log_probability = torch.nn.functional.log_softmax(
                        outputs, dim=1)
                    loss = criterion(log_probability, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            avg_loss = float(running_loss)/count
            avg_acc = float(running_corrects)/count
            avg_results[phase]['loss_per_epoch'].append(avg_loss)
            avg_results[phase]['acc_per_epoch'].append(avg_acc)
            if phase == 'train':
                scheduler.step()

            if phase == model_selection_criterion:
                if avg_acc > best_result['best_acc']:
                    print(
                        f"Best {model_selection_criterion} accuracy at epoch {epoch} being {avg_acc}")
                    best_result['best_epoch'] = epoch
                    best_result['best_acc'] = avg_acc
                    best_val_epoch_train_acc = avg_results['train']['acc_per_epoch'][-1]
                    print(
                        f"Train accuracy at epoch {epoch} being {best_val_epoch_train_acc}")
                    best_result['best_network'] = copy.deepcopy(
                        network.state_dict())

            print(f"Epoch {epoch}: Average {phase} Loss {avg_loss}, Accuracy {avg_acc:.2%}")
        print()
    print(f"Best Test Accuracy (for best {model_selection_criterion} model): {avg_results['test']['acc_per_epoch'][best_result['best_epoch']]:.2%}")
    print(f"Best Test Accuracy overall: {max(avg_results['test']['acc_per_epoch']):.2%}")
    network.load_state_dict(best_result['best_network'])
    test_acc = test(loaders['test'], network, train_mode, save_loc=None, class_names=None)
    print(f"Verify the best test accuracy for best {model_selection_criterion} is indeed {test_acc:.2%}")
    acc_result = {set_name: avg_results[set_name]['acc_per_epoch'][best_result['best_epoch']] for set_name in phases}
    return network, acc_result, best_result, avg_results
    # acc_result is {'train': best_val_epoch_train_acc, 'val': best_val_acc, 'test': test_acc}

def test(test_loader, network, train_mode, save_loc=None, class_names=None):
    # class_names should be sorted!!
    # If class_names != None, then return avg_acc, per_class_acc_dict
    if type(class_names) != type(None):
        assert sorted(class_names) == class_names
        idx_to_class = {idx: class_name for idx,
                        class_name in enumerate(class_names)}
        per_class_acc_dict = {idx: {'corrects': 0., 'counts': 0.}
                              for idx in idx_to_class.keys()}
    else:
        per_class_acc_dict = None

    network = network.cuda().eval()
    running_corrects = 0.
    count = 0

    pbar = test_loader

    for batch, data in enumerate(pbar):
        inputs, labels = data
        count += inputs.size(0)

        inputs = inputs.cuda()
        labels = labels.cuda()

        with torch.set_grad_enabled(False):
            outputs = network(inputs)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)
        if per_class_acc_dict != None:
            for label_i, pred_i in zip(labels.data, preds):
                per_class_acc_dict[int(label_i)]['corrects'] += int(pred_i == label_i)
                per_class_acc_dict[int(label_i)]['counts'] += 1
        # pbar.set_postfix(acc=float(running_corrects)/count)

    avg_acc = float(running_corrects)/count
    print(f"Best Test Accuracy on test set: {avg_acc}")
    if save_loc:
        torch.save(network.state_dict(), save_loc)
    if per_class_acc_dict != None:
        per_class_acc_dict_copy = {}
        for idx in idx_to_class:
            per_class_acc_dict_copy[idx_to_class[idx]] = per_class_acc_dict[idx]
        return avg_acc, per_class_acc_dict_copy
    else:
        return avg_acc

def avg_per_class_accuracy(per_class_accuracy_dict):
    total_count = 0.
    total_per_class_acc = 0.
    for class_name in per_class_accuracy_dict:
        total_per_class_acc += per_class_accuracy_dict[class_name]['corrects'] / \
            per_class_accuracy_dict[class_name]['counts']
        total_count += 1.
    return total_per_class_acc/total_count

def only_positive_accuracy(per_class_accuracy_dict):
    total_count = 0.
    total_correct = 0.
    for class_name in per_class_accuracy_dict:
        if class_name != NEGATIVE_LABEL:
            total_count += per_class_accuracy_dict[class_name]['counts']
            total_correct += per_class_accuracy_dict[class_name]['corrects']
    return total_correct/total_count

def run_baseline(all_loaders_dict, all_query, train_mode):
    result_baseline_dict = {'models': None, 
                            'accuracy_matrix': None,
                            'per_class_accuracy_dict': None,
                            'only_positive_accuracy_test': None,
                            'avg_per_class_accuracy_test': None,
                            'best_result' : None,
                            'avg_results' : None}
    all_model, all_accuracy, best_result, avg_results = train(all_loaders_dict,
                                                              args.train_mode,
                                                              len(all_query),
                                                              epochs=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].epochs,
                                                              lr=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].lr,
                                                              weight_decay=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].weight_decay,
                                                              step_size=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].step)
    print(all_accuracy)
    result_baseline_dict['accuracy_matrix'] = all_accuracy
    result_baseline_dict['models'] = all_model
    test_accuracy_all, per_class_accuracy_all = test(all_loaders_dict['test'], all_model, train_mode, class_names=all_query)
    result_baseline_dict['best_result'] = best_result
    result_baseline_dict['avg_results'] = avg_results
    result_baseline_dict['per_class_accuracy_dict'] = per_class_accuracy_all
    result_baseline_dict['only_positive_accuracy_test'] = only_positive_accuracy(per_class_accuracy_all)
    result_baseline_dict['avg_per_class_accuracy_test'] = avg_per_class_accuracy(per_class_accuracy_all)
    print(f"Baseline: {test_accuracy_all:.4%} (per sample), {result_baseline_dict['only_positive_accuracy_test']:.4%} (pos only), {result_baseline_dict['avg_per_class_accuracy_test']:.4%} (per class avg)")
    return result_baseline_dict

def run_single(loaders_dict, all_query, train_mode):
    result_single_dict = {'models': {}, # key is bucket index
                          'b1_b2_accuracy_matrix': None,
                          'accuracy': {},  # key is bucket index
                          'b1_b2_per_class_accuracy_dict': {},  # key is bucket index
                          'only_positive_accuracy_test': None,
                          'avg_per_class_accuracy_test': None,
                          'best_result_single': {},  # key is bucket index
                          'avg_results_single': {}}  # key is bucket index
    all_bucket = len(list(loaders_dict.keys()))
    single_accuracy_test = np.zeros((all_bucket, all_bucket))
    only_positive_accuracy_test = np.zeros((all_bucket, all_bucket))
    avg_per_class_accuracy_test = np.zeros((all_bucket, all_bucket))
    b1_b2_per_class_accuracy_dict = {}
    for b1 in range(all_bucket):
        b1_b2_per_class_accuracy_dict[b1] = {}
        single_model, single_accuracy_b1, best_result, avg_results = train(loaders_dict[b1],
                                                                           train_mode,
                                                                           len(all_query),
                                                                           epochs=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].epochs,
                                                                           lr=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].lr,
                                                                           weight_decay=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].weight_decay,
                                                                           step_size=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].step)
        result_single_dict['models'][b1] = single_model
        result_single_dict['accuracy'][b1] = single_accuracy_b1
        result_single_dict['best_result_single'][b1] = best_result
        result_single_dict['avg_results_single'][b1] = avg_results
        for b2 in range(all_bucket):
            test_loader_b2 = loaders_dict[b2]['test']
            single_accuracy_b1_b2, per_class_accuracy_b1_b2 = test(test_loader_b2, single_model, train_mode, class_names=all_query)
            b1_b2_per_class_accuracy_dict[b1][b2] = per_class_accuracy_b1_b2
            only_positive_accuracy_test[b1][b2] = only_positive_accuracy(per_class_accuracy_b1_b2)
            avg_per_class_accuracy_test[b1][b2] = avg_per_class_accuracy(per_class_accuracy_b1_b2)
            single_accuracy_test[b1][b2] = single_accuracy_b1_b2
            print(f"Train {b1}, test on {b2}: {single_accuracy_b1_b2:.4%} (per sample), {only_positive_accuracy_test[b1][b2]:.4%} (pos only), {avg_per_class_accuracy_test[b1][b2]:.4%} (per class avg)")
    result_single_dict['b1_b2_accuracy_matrix'] = single_accuracy_test
    result_single_dict['b1_b2_per_class_accuracy_dict'] = b1_b2_per_class_accuracy_dict
    result_single_dict['only_positive_accuracy_test'] = only_positive_accuracy_test
    result_single_dict['avg_per_class_accuracy_test'] = avg_per_class_accuracy_test
    return result_single_dict

def run_single_finetune(loaders_dict, all_query, train_mode):
    result_single_finetune_dict = {'models': {}, # key is bucket index
                                   'b1_b2_accuracy_matrix': None,
                                   'accuracy': {},  # key is bucket index
                                   'b1_b2_per_class_accuracy_dict': {},  # key is bucket index
                                   'only_positive_accuracy_test': None,
                                   'avg_per_class_accuracy_test': None,
                                   'best_result_single': {},  # key is bucket index
                                   'avg_results_single': {}}  # key is bucket index
    all_bucket = len(list(loaders_dict.keys()))
    single_accuracy_test = np.zeros((all_bucket, all_bucket))
    only_positive_accuracy_test = np.zeros((all_bucket, all_bucket))
    avg_per_class_accuracy_test = np.zeros((all_bucket, all_bucket))
    b1_b2_per_class_accuracy_dict = {}
    single_model = None
    for b1 in range(all_bucket):
        b1_b2_per_class_accuracy_dict[b1] = {}
        single_model, single_accuracy_b1, best_result, avg_results = train(loaders_dict[b1],
                                                                           train_mode,
                                                                           len(all_query),
                                                                           epochs=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].epochs,
                                                                           lr=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].lr,
                                                                           weight_decay=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].weight_decay,
                                                                           step_size=HYPER_DICT[TRAIN_MODES_CATEGORY[train_mode].network_type].step,
                                                                           finetuned_model=single_model)
        result_single_finetune_dict['models'][b1] = single_model
        result_single_finetune_dict['accuracy'][b1] = single_accuracy_b1
        result_single_finetune_dict['best_result_single'][b1] = best_result
        result_single_finetune_dict['avg_results_single'][b1] = avg_results
        for b2 in range(all_bucket):
            test_loader_b2 = loaders_dict[b2]['test']
            single_accuracy_b1_b2, per_class_accuracy_b1_b2 = test(test_loader_b2, single_model, train_mode, class_names=all_query)
            b1_b2_per_class_accuracy_dict[b1][b2] = per_class_accuracy_b1_b2
            only_positive_accuracy_test[b1][b2] = only_positive_accuracy(per_class_accuracy_b1_b2)
            avg_per_class_accuracy_test[b1][b2] = avg_per_class_accuracy(per_class_accuracy_b1_b2)
            single_accuracy_test[b1][b2] = single_accuracy_b1_b2
            print(f"Train {b1}, test on {b2}: {single_accuracy_b1_b2:.4%} (per sample), {only_positive_accuracy_test[b1][b2]:.4%} (pos only), {avg_per_class_accuracy_test[b1][b2]:.4%} (per class avg)")
    result_single_finetune_dict['b1_b2_accuracy_matrix'] = single_accuracy_test
    result_single_finetune_dict['b1_b2_per_class_accuracy_dict'] = b1_b2_per_class_accuracy_dict
    result_single_finetune_dict['only_positive_accuracy_test'] = only_positive_accuracy_test
    result_single_finetune_dict['avg_per_class_accuracy_test'] = avg_per_class_accuracy_test
    return result_single_finetune_dict

if __name__ == '__main__':
    args = argparser.parse_args()
    if not args.avoid_multiple_class and not args.use_negative_samples:
        import pdb; pdb.set_trace()

    start = time.time()

    if args.seed == None:
        print("Not using a random seed")
    else:
        random.seed(args.seed)
    seed_str = get_seed_str(args.seed)

    if args.use_negative_samples:
        print("Use Negative samples")
        dataset_paths_dict = get_negative_dataset_folder_paths(
            args.folder_path, args.num_of_bucket)
    else:
        dataset_paths_dict = get_dataset_folder_paths(
            args.folder_path, args.num_of_bucket, args.query_title,
            args.class_size, args.avoid_multiple_class, reverse_order=args.reverse_order, nn_size=args.nn_size)
    bucket_dict = {}

    excluded_bucket_idx = get_excluded_bucket_idx(args.moco_model)

    for label_set in dataset_paths_dict.keys():
        if label_set in args.excluded_label_set:
            print(f"<<<<<<<<<<<<<<<<<<<<<<<<<Skipping label set {label_set}")
            print()
            continue
        if args.only_label_set:
            if label_set == args.only_label_set:
                print(f"Only do label set {args.only_label_set}")
            else:
                print(
                    f"<<<<<<<<<<<<<<<<<<<<<<<<<Skipping label set {label_set}")
                print()
                continue

        main_label_set_path = dataset_paths_dict[label_set]['main_label_set_path']
        sub_folder_paths = dataset_paths_dict[label_set]['sub_folder_paths']
        if args.use_negative_samples:
            print("Not checking the info dict..")
        else:
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
                'num_of_bucket': args.num_of_bucket,
                'moco_model': args.moco_model,
                'arch': args.arch,
            }
            info_dict_path = os.path.join(
                main_label_set_path, "info_dict.pickle")
            if os.path.exists(info_dict_path):
                saved_info_dict = load_pickle(info_dict_path)
                if not saved_info_dict == info_dict:
                    print("Info dict does not align")
                    import pdb; pdb.set_trace()
            else:
                print("No info dict was saved")
                import pdb; pdb.set_trace()

        print(f"Processing {label_set}.. ")

        query_dict = {}  # Saved the query results for each data bucket

        query_dict_path = os.path.join(
            main_label_set_path, "query_dict.pickle")
        if not os.path.exists(query_dict_path):
            print(f"Query dict does not exist for {label_set}")
            import pdb
            pdb.set_trace()
            continue
        query_dict = load_pickle(query_dict_path)
        queries = list(query_dict[0].keys())
        print(queries)
        print(f"We have {len(queries)} queries.")

        ############### Create Datasets
        dataset_dict_path = os.path.join(main_label_set_path,
                                         f"dataset_features_dict_{dataset_str(args.mode)}{seed_str}.pickle")
        if os.path.exists(dataset_dict_path):
            print(f"{dataset_dict_path} already exists.")
            dataset_dict = load_pickle(dataset_dict_path)
        else:
            dataset_dict = make_dataset_dict(query_dict, args.mode) # Will save dataset_dict in file loc
            save_obj_as_pickle(dataset_dict_path, dataset_dict)
        
        ############### Create Features
        features_dict_path = os.path.join(main_label_set_path,
                                          f"features_dict_{dataset_str(args.mode)}_{args.train_mode}{seed_str}.pickle")
        if os.path.exists(features_dict_path):
            print(f"{features_dict_path} already exists.")
            features_dict = load_pickle(features_dict_path)
        else:
            features_dict = make_features_dict(dataset_dict, args.train_mode)
            save_obj_as_pickle(features_dict_path, features_dict)

        ############### Create DataLoaders
        all_loaders_dict_path = os.path.join(main_label_set_path,
                                             f"all_loaders_from_features_dict_{dataset_str(args.mode)}_{args.train_mode}{seed_str}_ex_{excluded_bucket_idx}.pickle")
        if os.path.exists(all_loaders_dict_path):
            print(f"{all_loaders_dict_path} already exists.")
            all_loaders_dict = load_pickle(all_loaders_dict_path)
        else:
            all_loaders_dict = get_all_loaders_from_features_dict(features_dict, args.train_mode, HYPER_DICT[TRAIN_MODES_CATEGORY[args.train_mode].network_type], excluded_bucket_idx=excluded_bucket_idx)
            save_obj_as_pickle(all_loaders_dict_path, all_loaders_dict)

        loaders_dict_path = os.path.join(main_label_set_path,
                                         f"loaders_from_features_dict_{dataset_str(args.mode)}_{args.train_mode}{seed_str}.pickle")

        if os.path.exists(loaders_dict_path):
            print(f"{loaders_dict_path} already exists.")
            loaders_dict = load_pickle(loaders_dict_path)
        else:
            loaders_dict = get_loaders_from_features_dict(features_dict, args.train_mode, HYPER_DICT[TRAIN_MODES_CATEGORY[args.train_mode].network_type])
            save_obj_as_pickle(loaders_dict_path, loaders_dict)

        all_query = sorted(list(dataset_dict[0].keys()))
        print(all_query)
        ############### Run Baseline Experiment
        results_dict_all_path = os.path.join(main_label_set_path,
                                             f"results_dict_all_{dataset_str(args.mode)}_{args.train_mode}{seed_str}_ex_{excluded_bucket_idx}.pickle")
        if not os.path.exists(results_dict_all_path):
            result_baseline_dict = run_baseline(all_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_all_path, result_baseline_dict)
            print(f"Saved at {results_dict_all_path}")
        else:
            print(f"Baseline result saved at {results_dict_all_path}")

        ############### Run Single Bucket Experiment
        results_dict_single_path = os.path.join(main_label_set_path,
                                                f"results_dict_single_{dataset_str(args.mode)}_{args.train_mode}{seed_str}.pickle")

        if not os.path.exists(results_dict_single_path):
            result_single_dict = run_single(loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_single_path, result_single_dict)
            print(f"Saved at {results_dict_single_path}")
        else:
            print(results_dict_single_path + " already exists")

        ############### Run Single Bucket (Finetune) Experiment
        results_dict_single_finetune_path = os.path.join(main_label_set_path,
                                                         f"results_dict_single_finetune_{dataset_str(args.mode)}_{args.train_mode}{seed_str}.pickle")

        if not os.path.exists(results_dict_single_finetune_path):
            result_single_finetune_dict = run_single_finetune(loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_single_finetune_path, result_single_finetune_dict)
            print(f"Saved at {results_dict_single_finetune_path}")
        else:
            print(results_dict_single_finetune_path + " already exists")
        
