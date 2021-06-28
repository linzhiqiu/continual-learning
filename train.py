import training_utils
from utils import load_pickle, save_obj_as_pickle
# from prepare_clip_dataset import QUERY_TITLE_DICT, LABEL_SETS
import random
import argparse
# from analyze_feature_variation import argparser, get_dataset_folder_paths
from tqdm import tqdm
import copy
import time
import numpy as np
import torch
import os

NEGATIVE_LABEL = "NEGATIVE"

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
    'mlp_tuned': HyperParameter('mlp', epochs=100, step=60, batch_size=256, lr=0.5, weight_decay=0.),
    'linear_tuned_2': HyperParameter('linear', epochs=100, step=45, batch_size=256, lr=0.5, weight_decay=0.),
    'linear_tuned': HyperParameter('linear', epochs=100, step=60, batch_size=256, lr=1., weight_decay=0.),
    'linear': HyperParameter('linear', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0.),
    'cnn': HyperParameter('cnn', epochs=100, step=60, batch_size=64, lr=0.1, weight_decay=1e-5),
    'cnn_lower_lr': HyperParameter('cnn', epochs=100, step=30, batch_size=64, lr=0.01, weight_decay=1e-5),
}

ALL_PRETRAINED_WEIGHTS = ['moco_yfcc_feb18_gpu_8_bucket_0', 'imgnet', 'moco_imgnet', 'byol_imgnet', None]
ALL_FEATURE_TYPES = ['image', 'clip', 'cnn_feature']
ALL_NETWORK_TYPES = HYPER_DICT.keys()
class TrainMode():
    def __init__(self, feature_type, pretrained_weight=None, network_type='cnn'):
        assert pretrained_weight in ALL_PRETRAINED_WEIGHTS
        assert feature_type in ALL_FEATURE_TYPES
        assert network_type in ALL_NETWORK_TYPES
        if network_type in ['cnn', 'cnn_lower_lr']:
            assert feature_type == 'image'
        else:
            assert feature_type != 'image'
        self.feature_type = feature_type
        self.pretrained_weight = pretrained_weight
        self.network_type = network_type

TRAIN_MODES_CATEGORY = {
    'cnn_scratch': TrainMode('image', pretrained_weight=None, network_type='cnn'),
    'cnn_scratch_lower_lr': TrainMode('image', pretrained_weight=None, network_type='cnn_lower_lr'),
    'cnn_imgnet': TrainMode('image', pretrained_weight='imgnet', network_type='cnn'),
    'cnn_moco': TrainMode('image', pretrained_weight='moco_imgnet', network_type='cnn'),
    'cnn_byol': TrainMode('image', pretrained_weight='byol_imgnet', network_type='cnn'),
    'cnn_moco_yfcc_feb18_gpu_8_bucket_0': TrainMode('image', pretrained_weight='moco_yfcc_feb18_gpu_8_bucket_0', network_type='cnn'),
    'cnn_moco_yfcc_feb18_gpu_8_bucket_0_lower_lr': TrainMode('image', pretrained_weight='moco_yfcc_feb18_gpu_8_bucket_0', network_type='cnn_lower_lr'),
    'moco_v2_imgnet_linear': TrainMode('cnn_feature', pretrained_weight='moco_imgnet', network_type='linear'),
    'moco_v2_imgnet_linear_tuned': TrainMode('cnn_feature', pretrained_weight='moco_imgnet', network_type='linear_tuned'),
    'byol_imgnet_linear': TrainMode('cnn_feature', pretrained_weight='byol_imgnet', network_type='linear'),
    'byol_imgnet_linear_tuned': TrainMode('cnn_feature', pretrained_weight='byol_imgnet', network_type='linear_tuned'),
    'imgnet_linear': TrainMode('cnn_feature', pretrained_weight='imgnet', network_type='linear'),
    'imgnet_linear_tuned': TrainMode('cnn_feature', pretrained_weight='imgnet', network_type='linear_tuned'),
    'raw_feature': TrainMode('cnn_feature', pretrained_weight=None, network_type='linear'),
    'moco_v2_yfcc_feb18_bucket_0_gpu_8_linear': TrainMode('cnn_feature', pretrained_weight='moco_yfcc_feb18_gpu_8_bucket_0', network_type='linear'),
    'moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned': TrainMode('cnn_feature', pretrained_weight='moco_yfcc_feb18_gpu_8_bucket_0', network_type='linear_tuned'),
    'moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned_2': TrainMode('cnn_feature', pretrained_weight='moco_yfcc_feb18_gpu_8_bucket_0', network_type='linear_tuned_2'),
    'moco_v2_imgnet_mlp': TrainMode('cnn_feature', pretrained_weight='moco_imgnet', network_type='mlp'),
    'moco_v2_imgnet_mlp_tuned': TrainMode('cnn_feature', pretrained_weight='moco_imgnet', network_type='mlp_tuned'),
    'byol_imgnet_mlp': TrainMode('cnn_feature', pretrained_weight='byol_imgnet', network_type='mlp'),
    'byol_imgnet_mlp_tuned': TrainMode('cnn_feature', pretrained_weight='byol_imgnet', network_type='mlp_tuned'),
    'imgnet_mlp': TrainMode('cnn_feature', pretrained_weight='imgnet', network_type='mlp'),
    'imgnet_mlp_tuned': TrainMode('cnn_feature', pretrained_weight='imgnet', network_type='mlp_tuned'),
    'moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp': TrainMode('cnn_feature', pretrained_weight='moco_yfcc_feb18_gpu_8_bucket_0', network_type='mlp'),
    'moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp_tuned': TrainMode('cnn_feature', pretrained_weight='moco_yfcc_feb18_gpu_8_bucket_0', network_type='mlp_tuned'),
    'linear' : TrainMode('clip', pretrained_weight=None, network_type='linear'),
    'mlp': TrainMode('clip', pretrained_weight=None, network_type='mlp'),
    'linear_tuned': TrainMode('clip', pretrained_weight=None, network_type='linear_tuned'),
    'mlp_tuned': TrainMode('clip', pretrained_weight=None, network_type='mlp_tuned'),
}

argparser = argparse.ArgumentParser()
argparser.add_argument("--folder_path",
                       default='/compute/autobot-1-1/zhiqiu/yfcc_dynamic_10',
                       help="The folder with the images and query_dict.pickle")
argparser.add_argument("--exp_result_path",
                       default='/project_data/ramanan/zhiqiu/yfcc_dynamic_10',
                       help="The folder with the images and query_dict.pickle")
# argparser.add_argument('--num_of_bucket', default=11, type=int,
#                        help='number of bucket')
argparser.add_argument("--dataset_name",
                    #    default='dynamic_300_cleaned', # best test model
                       default='dynamic_300', # best training loss
                    #    default='dynamic_300_second_run',  # best training loss
                    #    default='dynamic_negative_300_cleaned_bucket_1_only',
                       help="nly evaluate on this label set")
argparser.add_argument('--train_mode',
                       default='linear', choices=TRAIN_MODES_CATEGORY.keys(),
                       help='Train mode')
argparser.add_argument('--mode',
                       default='default', choices=MODE_DICT.keys(),
                       help='Mode for dataset split')
argparser.add_argument('--seed',
                       default=None, type=int,
                       help='Seed for experiment')
argparser.add_argument('--excluded_bucket_idx',
                       default=0, type=int,
                       help='Excluding this bucket from all experiments')

def get_seed_str(seed):
    if seed == None:
        return "seed_None"
    else:
        return f"seed_{seed}"

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
            loader = training_utils.make_image_loader(items, batch_size, shuffle=False, fixed_crop=True)
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
        return lambda items, is_train_mode: training_utils.make_numpy_loader(items, batch_size, shuffle=is_train_mode)
    elif feature_type == 'image':
        # always do center cropping
        return lambda items, is_train_mode: training_utils.make_image_loader(items, batch_size, shuffle=is_train_mode, fixed_crop=True)
    elif feature_type == 'cnn_feature':
        return lambda items, is_train_mode: training_utils.make_tensor_loader(items, batch_size, shuffle=is_train_mode)
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

# def get_excluded_bucket_idx(moco_model):
#     moco_paths = moco_model.split(os.sep)
#     model_configs = moco_paths[-2].split("_")
#     excluded_bucket_idx = model_configs[model_configs.index('idx')+1]
#     return int(excluded_bucket_idx)

def get_loaders_from_features_dict(features_dict, train_mode, hyperparameter, excluded_bucket_idx=0):
    loaders_dict = {}  # Saved the splitted loader for each bucket

    loader_func = get_loader_func(train_mode, hyperparameter.batch_size)

    for b_idx in features_dict:
        if type(excluded_bucket_idx) == int and b_idx == excluded_bucket_idx:
            continue
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

def get_curr_and_prev_loaders_from_features_dict(features_dict, train_mode, hyperparameter, excluded_bucket_idx=0):
    all_bucket = sorted(list(features_dict.keys()))
    print(f"Excluding {excluded_bucket_idx} from the sequential loaders")
    if type(excluded_bucket_idx) == int:
        features_dict = {k: features_dict[k]
                        for k in all_bucket if k != excluded_bucket_idx}
    curr_and_prev_loaders_dict = {}
    loader_func = get_loader_func(train_mode, hyperparameter.batch_size)

    for b_idx in all_bucket:
        curr_and_prev_loaders_dict[b_idx] = {}
        assert 'val' not in features_dict[b_idx]
        train_items = []
        prev_b_idx = b_idx - 1
        if prev_b_idx in features_dict:
            print(f"Add {prev_b_idx} to {b_idx} train loader")
            train_items += features_dict[prev_b_idx]['train']
        train_items += features_dict[b_idx]['train']
        train_loader = loader_func(train_items, True)
        curr_and_prev_loaders_dict[b_idx]['train'] = train_loader
        
        test_items = []
        test_items += features_dict[b_idx]['test']
        test_loader = loader_func(test_items, False)
        curr_and_prev_loaders_dict[b_idx]['test'] = test_loader
    return curr_and_prev_loaders_dict

def get_curr_and_random_prev_loaders_from_features_dict(features_dict, train_mode, hyperparameter, excluded_bucket_idx=0):
    all_bucket = sorted(list(features_dict.keys()))
    print(f"Excluding {excluded_bucket_idx} from the sequential loaders")
    if type(excluded_bucket_idx) == int:
        features_dict = {k: features_dict[k]
                         for k in all_bucket if k != excluded_bucket_idx}
    curr_and_random_prev_loaders_dict = {}
    loader_func = get_loader_func(train_mode, hyperparameter.batch_size)

    train_buffer = [] # buffer is cap at size 2 bucket
    n = 0. # number of seen examples in the stream
    k = None # number of examples in two buckets. will be calculated in in first iteration. Assume each bucket has same size!
    # import pdb; pdb.set_trace()
    for idx, b_idx in enumerate(all_bucket):
            
        curr_and_random_prev_loaders_dict[b_idx] = {}
        assert 'val' not in features_dict[b_idx]
        n += len(features_dict[b_idx]['train'])
        if idx == 0:
            train_buffer = features_dict[b_idx]['train']
            k = 2 * n # This is currently 2 * bucket size
        elif idx == 1:
            train_buffer += features_dict[b_idx]['train']
        else:
            print(f'select new examples with probability {k/n:.2f} = {k} / {n}')
            prob = k/n
            new_items_to_add_to_buffer = []
            for item in features_dict[b_idx]['train']:
                if random.random() <= prob:
                    new_items_to_add_to_buffer.append(item)
            print(f"Selected {len(new_items_to_add_to_buffer)} samples out of {k} incoming samples")
            random.shuffle(train_buffer)
            train_buffer = train_buffer[:len(train_buffer) - len(new_items_to_add_to_buffer)]
            train_buffer += new_items_to_add_to_buffer
        
        train_loader = loader_func(train_buffer.copy(), True) # Important! TO use copy() otherwise it will be changed
        curr_and_random_prev_loaders_dict[b_idx]['train'] = train_loader
        
        test_items = []
        test_items += features_dict[b_idx]['test']
        test_loader = loader_func(test_items.copy(), False)
        curr_and_random_prev_loaders_dict[b_idx]['test'] = test_loader
    # import pdb; pdb.set_trace()
    return curr_and_random_prev_loaders_dict

def get_sequential_loaders_from_features_dict(features_dict, train_mode, hyperparameter, excluded_bucket_idx=0):
    all_bucket = sorted(list(features_dict.keys()))
    print(f"Excluding {excluded_bucket_idx} from the sequential loaders")
    if type(excluded_bucket_idx) == int:
        features_dict = {k: features_dict[k]
                        for k in all_bucket if k != excluded_bucket_idx}
    sequential_loaders_dict = {}

    loader_func = get_loader_func(train_mode, hyperparameter.batch_size)

    for b_idx in all_bucket:
        sequential_loaders_dict[b_idx] = {}
        assert 'val' not in features_dict[b_idx]

        train_items = []
        for curr_b_idx in all_bucket:
            if curr_b_idx <= b_idx:
                train_items += features_dict[curr_b_idx]['train']
        train_loader = loader_func(train_items, True)
        sequential_loaders_dict[b_idx]['train'] = train_loader
            
        test_items = []
        test_items += features_dict[b_idx]['test']
        test_loader = loader_func(test_items, False)
        sequential_loaders_dict[b_idx]['test'] = test_loader
    return sequential_loaders_dict

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
    if network_type == 'mlp' or network_type == 'mlp_tuned':
        print(f"Using a mlp network with input size {input_size}")
        mlp = MLP(input_size, 2048, output_size)
        return mlp
    elif network_type in ['linear' , 'linear_tuned', 'linear_tuned_2']:
        print(f"Using a single linear layer")
        fc = torch.nn.Linear(input_size, output_size)
        # import pdb; pdb.set_trace()
        # fc.weight.data.normal_(mean=0.0, std=0.01)
        # fc.bias.data.zero_()
        return fc
    elif network_type in ['cnn', 'cnn_lower_lr']:
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
    optimizer = training_utils.make_optimizer(network, lr, weight_decay)
    scheduler = training_utils.make_scheduler(optimizer, step_size=step_size)
    criterion = torch.nn.NLLLoss(reduction='mean')

    avg_results = {'train': {'loss_per_epoch': [], 'acc_per_epoch': []},
                   'test':  {'loss_per_epoch': [], 'acc_per_epoch': []}}
    if 'val' in loaders:
        avg_results['val'] = {'loss_per_epoch': [], 'acc_per_epoch': []}
        phases = ['train', 'val', 'test']
    else:
        phases = ['train', 'test']

    # Save best training loss model
    best_result = {'best_loss': None, 'best_acc': 0, 'best_epoch': None, 'best_network': None}

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

                if best_result['best_loss'] == None or avg_loss < best_result['best_loss']:
                    print(
                        f"Best training loss at epoch {epoch} being {avg_loss} with accuracy {avg_acc}")
                    best_result['best_epoch'] = epoch
                    best_result['best_acc'] = avg_acc
                    best_result['best_loss'] = avg_loss
                    best_result['best_network'] = copy.deepcopy(network.state_dict())

            print(
                f"Epoch {epoch}: Average {phase} Loss {avg_loss}, Accuracy {avg_acc:.2%}")
        print()
    print(
        f"Test Accuracy (for best training loss model): {avg_results['test']['acc_per_epoch'][best_result['best_epoch']]:.2%}")
    print(
        f"Best Test Accuracy overall: {max(avg_results['test']['acc_per_epoch']):.2%}")
    network.load_state_dict(best_result['best_network'])
    test_acc = test(loaders['test'], network, train_mode,
                    save_loc=None, class_names=None)
    print(f"Verify the best test accuracy for best training loss is indeed {test_acc:.2%}")
    acc_result = {set_name: avg_results[set_name]['acc_per_epoch']
                  [best_result['best_epoch']] for set_name in phases}
    return network, acc_result, best_result, avg_results
    # acc_result is {'train': best_val_epoch_train_acc, 'val': best_val_acc, 'test': test_acc}

# def train_best_test(loaders,
#           train_mode, output_size,
#           epochs=150, lr=0.1, weight_decay=1e-5, step_size=60,
#           finetuned_model=None):
#     if finetuned_model == None:
#         network = make_model(train_mode, output_size).cuda()
#         print("Retraining..")
#     else:
#         network = finetuned_model
#         print("Finetuning..")
    
#     optimizer = training_utils.make_optimizer(network, lr, weight_decay)
#     scheduler = training_utils.make_scheduler(optimizer, step_size=step_size)
#     criterion = torch.nn.NLLLoss(reduction='mean')

#     avg_results = {'train': {'loss_per_epoch': [], 'acc_per_epoch': []},
#                    'test':  {'loss_per_epoch': [], 'acc_per_epoch': []}}
#     if 'val' in loaders:
#         avg_results['val'] = {'loss_per_epoch': [], 'acc_per_epoch': []}
#         model_selection_criterion = 'val'
#         phases = ['train', 'val', 'test']
#     else:
#         model_selection_criterion = 'test'
#         phases = ['train', 'test']

#     best_result = {'best_acc': 0, 'best_epoch': None, 'best_network': None}

#     for epoch in range(0, epochs):
#         print(f"Epoch {epoch}")
#         for phase in phases:
#             if phase == 'train':
#                 network.train()
#             else:
#                 network.eval()

#             running_loss = 0.0
#             running_corrects = 0.
#             count = 0

#             pbar = loaders[phase]

#             for batch, data in enumerate(pbar):
#                 inputs, labels = data
#                 count += inputs.size(0)

#                 inputs = inputs.cuda()
#                 labels = labels.cuda()

#                 if phase == 'train':
#                     optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = network(inputs)
#                     _, preds = torch.max(outputs, 1)

#                     log_probability = torch.nn.functional.log_softmax(
#                         outputs, dim=1)
#                     loss = criterion(log_probability, labels)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             avg_loss = float(running_loss)/count
#             avg_acc = float(running_corrects)/count
#             avg_results[phase]['loss_per_epoch'].append(avg_loss)
#             avg_results[phase]['acc_per_epoch'].append(avg_acc)
#             if phase == 'train':
#                 scheduler.step()

#             if phase == model_selection_criterion:
#                 if avg_acc > best_result['best_acc']:
#                     print(
#                         f"Best {model_selection_criterion} accuracy at epoch {epoch} being {avg_acc}")
#                     best_result['best_epoch'] = epoch
#                     best_result['best_acc'] = avg_acc
#                     best_val_epoch_train_acc = avg_results['train']['acc_per_epoch'][-1]
#                     print(
#                         f"Train accuracy at epoch {epoch} being {best_val_epoch_train_acc}")
#                     best_result['best_network'] = copy.deepcopy(
#                         network.state_dict())

#             print(f"Epoch {epoch}: Average {phase} Loss {avg_loss}, Accuracy {avg_acc:.2%}")
#         print()
#     print(f"Best Test Accuracy (for best {model_selection_criterion} model): {avg_results['test']['acc_per_epoch'][best_result['best_epoch']]:.2%}")
#     print(f"Best Test Accuracy overall: {max(avg_results['test']['acc_per_epoch']):.2%}")
#     network.load_state_dict(best_result['best_network'])
#     test_acc = test(loaders['test'], network, train_mode, save_loc=None, class_names=None)
#     print(f"Verify the best test accuracy for best {model_selection_criterion} is indeed {test_acc:.2%}")
#     acc_result = {set_name: avg_results[set_name]['acc_per_epoch'][best_result['best_epoch']] for set_name in phases}
#     return network, acc_result, best_result, avg_results
#     # acc_result is {'train': best_val_epoch_train_acc, 'val': best_val_acc, 'test': test_acc}

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
                                                              train_mode,
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
    sorted_buckets = sorted(list(loaders_dict.keys()))
    bucket_index_to_index = {sorted_buckets[i]: i for i in range(all_bucket)}
    print("bucket_index_to_index:")
    print(bucket_index_to_index)
    result_single_dict['bucket_index_to_index'] = bucket_index_to_index
    single_accuracy_test = np.zeros((all_bucket, all_bucket))
    only_positive_accuracy_test = np.zeros((all_bucket, all_bucket))
    avg_per_class_accuracy_test = np.zeros((all_bucket, all_bucket))
    b1_b2_per_class_accuracy_dict = {}
    for b1 in sorted_buckets:
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
        for b2 in sorted_buckets:
            # if b1 == b2:
            #     import pdb; pdb.set_trace() # TODO
            test_loader_b2 = loaders_dict[b2]['test']
            single_accuracy_b1_b2, per_class_accuracy_b1_b2 = test(test_loader_b2, single_model, train_mode, class_names=all_query)
            b1_b2_per_class_accuracy_dict[b1][b2] = per_class_accuracy_b1_b2
            b1_idx = bucket_index_to_index[b1]
            b2_idx = bucket_index_to_index[b2]
            only_positive_accuracy_test[b1_idx][b2_idx] = only_positive_accuracy(per_class_accuracy_b1_b2)
            avg_per_class_accuracy_test[b1_idx][b2_idx] = avg_per_class_accuracy(per_class_accuracy_b1_b2)
            single_accuracy_test[b1_idx][b2_idx] = single_accuracy_b1_b2
            print(f"Train {b1}, test on {b2}: {single_accuracy_b1_b2:.4%} (per sample), {only_positive_accuracy_test[b1_idx][b2_idx]:.4%} (pos only), {avg_per_class_accuracy_test[b1_idx][b2_idx]:.4%} (per class avg)")
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
    sorted_buckets = sorted(list(loaders_dict.keys()))
    bucket_index_to_index = {sorted_buckets[i]: i for i in range(all_bucket)}
    print("bucket_index_to_index:")
    print(bucket_index_to_index)
    result_single_finetune_dict['bucket_index_to_index'] = bucket_index_to_index
    single_accuracy_test = np.zeros((all_bucket, all_bucket))
    only_positive_accuracy_test = np.zeros((all_bucket, all_bucket))
    avg_per_class_accuracy_test = np.zeros((all_bucket, all_bucket))
    b1_b2_per_class_accuracy_dict = {}
    single_model = None
    for b1 in sorted_buckets:
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
        for b2 in sorted_buckets:
            test_loader_b2 = loaders_dict[b2]['test']
            single_accuracy_b1_b2, per_class_accuracy_b1_b2 = test(test_loader_b2, single_model, train_mode, class_names=all_query)
            b1_b2_per_class_accuracy_dict[b1][b2] = per_class_accuracy_b1_b2
            b1_idx = bucket_index_to_index[b1]
            b2_idx = bucket_index_to_index[b2]
            only_positive_accuracy_test[b1_idx][b2_idx] = only_positive_accuracy(per_class_accuracy_b1_b2)
            avg_per_class_accuracy_test[b1_idx][b2_idx] = avg_per_class_accuracy(per_class_accuracy_b1_b2)
            single_accuracy_test[b1_idx][b2_idx] = single_accuracy_b1_b2
            print(f"Train {b1}, test on {b2}: {single_accuracy_b1_b2:.4%} (per sample), {only_positive_accuracy_test[b1_idx][b2_idx]:.4%} (pos only), {avg_per_class_accuracy_test[b1_idx][b2_idx]:.4%} (per class avg)")
    result_single_finetune_dict['b1_b2_accuracy_matrix'] = single_accuracy_test
    result_single_finetune_dict['b1_b2_per_class_accuracy_dict'] = b1_b2_per_class_accuracy_dict
    result_single_finetune_dict['only_positive_accuracy_test'] = only_positive_accuracy_test
    result_single_finetune_dict['avg_per_class_accuracy_test'] = avg_per_class_accuracy_test
    return result_single_finetune_dict

if __name__ == '__main__':
    args = argparser.parse_args()

    if args.seed == None:
        print("Not using a random seed")
    else:
        print(f"Using random seed {args.seed}")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    seed_str = get_seed_str(args.seed)

    bucket_dict = {}

    excluded_bucket_idx = args.excluded_bucket_idx
    folder_path = args.folder_path
    dataset_name = args.dataset_name
    exp_result_path = args.exp_result_path
    exp_result_save_path = os.path.join(exp_result_path, dataset_name)
    if not os.path.exists(exp_result_save_path):
        os.makedirs(exp_result_save_path)
    print(f"Working on dataset {dataset_name}")
    print(f"Dataset and result will be saved at {exp_result_save_path}")

    query_dict_path = os.path.join(folder_path, dataset_name, "query_dict.pickle")
    if not os.path.exists(query_dict_path):
        print(f"Query dict does not exist for {dataset_name}")
        exit(0)
    query_dict = load_pickle(query_dict_path)
    
    all_query = sorted(list(query_dict[list(query_dict.keys())[0]].keys()))
    print(f"We have {len(all_query)} classes.")
    print(all_query)

    ############### Create Datasets
    dataset_dict_path = os.path.join(exp_result_save_path,
                                     f"dataset_dict_{dataset_str(args.mode)}_{seed_str}.pickle")
    if os.path.exists(dataset_dict_path):
        print(f"{dataset_dict_path} already exists.")
        dataset_dict = load_pickle(dataset_dict_path)
    else:
        dataset_dict = make_dataset_dict(query_dict, args.mode) # Will save dataset_dict in file loc
        save_obj_as_pickle(dataset_dict_path, dataset_dict)
    
    ############### Create Features
    features_dict_path = os.path.join(exp_result_save_path,
                                      f"features_dict_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}.pickle")
    if os.path.exists(features_dict_path):
        print(f"{features_dict_path} already exists.")
        features_dict = load_pickle(features_dict_path)
    else:
        features_dict = make_features_dict(dataset_dict, args.train_mode)
        save_obj_as_pickle(features_dict_path, features_dict)

    ############### Create DataLoaders
    all_loaders_dict_path = os.path.join(exp_result_save_path,
                                         f"all_loaders_dict_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
    if os.path.exists(all_loaders_dict_path):
        print(f"{all_loaders_dict_path} already exists.")
        all_loaders_dict = load_pickle(all_loaders_dict_path)
    else:
        all_loaders_dict = get_all_loaders_from_features_dict(
                               features_dict,
                               args.train_mode,
                               HYPER_DICT[TRAIN_MODES_CATEGORY[args.train_mode].network_type],
                               excluded_bucket_idx=excluded_bucket_idx
                           )
        save_obj_as_pickle(all_loaders_dict_path, all_loaders_dict)

    loaders_dict_path = os.path.join(exp_result_save_path,
                                     f"loaders_dict_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

    if os.path.exists(loaders_dict_path):
        print(f"{loaders_dict_path} already exists.")
        loaders_dict = load_pickle(loaders_dict_path)
    else:
        loaders_dict = get_loaders_from_features_dict(
                           features_dict,
                           args.train_mode,
                           HYPER_DICT[TRAIN_MODES_CATEGORY[args.train_mode].network_type],
                           excluded_bucket_idx=excluded_bucket_idx
                       )
        save_obj_as_pickle(loaders_dict_path, loaders_dict)

    if not use_val_set(args.mode):
        # get_curr_and_random_prev_loaders_from_features_dict
        print("Since not using a validation set, we can perform sequential learning experiment")
        sequential_loaders_dict_path = os.path.join(exp_result_save_path,
                                                    f"sequential_loaders_dict_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

        if os.path.exists(sequential_loaders_dict_path):
            print(f"{sequential_loaders_dict_path} already exists.")
            sequential_loaders_dict = load_pickle(sequential_loaders_dict_path)
        else:
            sequential_loaders_dict = get_sequential_loaders_from_features_dict(
                                        features_dict,
                                        args.train_mode,
                                        HYPER_DICT[TRAIN_MODES_CATEGORY[args.train_mode].network_type],
                                        excluded_bucket_idx=excluded_bucket_idx
                                    )
            save_obj_as_pickle(sequential_loaders_dict_path, sequential_loaders_dict)
        
        curr_and_prev_loaders_dict_path = os.path.join(exp_result_save_path,
                                                    f"curr_and_prev_loaders_dict_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

        if os.path.exists(curr_and_prev_loaders_dict_path):
            print(f"{curr_and_prev_loaders_dict_path} already exists.")
            curr_and_prev_loaders_dict = load_pickle(curr_and_prev_loaders_dict_path)
        else:
            curr_and_prev_loaders_dict = get_curr_and_prev_loaders_from_features_dict(
                                            features_dict,
                                            args.train_mode,
                                            HYPER_DICT[TRAIN_MODES_CATEGORY[args.train_mode].network_type],
                                            excluded_bucket_idx=excluded_bucket_idx
                                        )
            save_obj_as_pickle(curr_and_prev_loaders_dict_path, curr_and_prev_loaders_dict)
        
        curr_and_random_prev_loaders_dict_path = os.path.join(exp_result_save_path,
                                                    f"curr_and_random_prev_loaders_dict_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

        if os.path.exists(curr_and_random_prev_loaders_dict_path):
            print(f"{curr_and_random_prev_loaders_dict_path} already exists.")
            curr_and_random_prev_loaders_dict = load_pickle(curr_and_random_prev_loaders_dict_path)
        else:
            curr_and_random_prev_loaders_dict = get_curr_and_random_prev_loaders_from_features_dict(
                                                    features_dict,
                                                    args.train_mode,
                                                    HYPER_DICT[TRAIN_MODES_CATEGORY[args.train_mode].network_type],
                                                    excluded_bucket_idx=excluded_bucket_idx
                                                )
            save_obj_as_pickle(curr_and_random_prev_loaders_dict_path, curr_and_random_prev_loaders_dict)
        
        ############### Run Curr and Random Prev (Retrain) Experiment
        results_dict_curr_and_random_prev_retrain_path = os.path.join(exp_result_save_path,
                                                            f"results_dict_curr_and_random_prev_retrain_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
        
        if not os.path.exists(results_dict_curr_and_random_prev_retrain_path):
            results_dict_curr_and_random_prev_retrain = run_single(curr_and_random_prev_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_curr_and_random_prev_retrain_path, results_dict_curr_and_random_prev_retrain)
            print(f"Saved at {results_dict_curr_and_random_prev_retrain_path}")
        else:
            print(results_dict_curr_and_random_prev_retrain_path + " already exists")
        
        ############### Run Curr and Random Prev (Finetune) Experiment
        results_dict_curr_and_random_prev_finetune_path = os.path.join(exp_result_save_path,
                                                            f"results_dict_curr_and_random_prev_finetune_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

        if not os.path.exists(results_dict_curr_and_random_prev_finetune_path):
            results_dict_curr_and_random_prev_finetune = run_single_finetune(curr_and_random_prev_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_curr_and_random_prev_finetune_path, results_dict_curr_and_random_prev_finetune)
            print(f"Saved at {results_dict_curr_and_random_prev_finetune_path}")
        else:
            print(results_dict_curr_and_random_prev_finetune_path + " already exists")

        ############### Run Curr and Prev (Retrain) Experiment
        results_dict_curr_and_prev_retrain_path = os.path.join(exp_result_save_path,
                                                            f"results_dict_curr_and_prev_retrain_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
        
        if not os.path.exists(results_dict_curr_and_prev_retrain_path):
            results_dict_curr_and_prev_retrain = run_single(curr_and_prev_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_curr_and_prev_retrain_path, results_dict_curr_and_prev_retrain)
            print(f"Saved at {results_dict_curr_and_prev_retrain_path}")
        else:
            print(results_dict_curr_and_prev_retrain_path + " already exists")
        
        ############### Run Curr and Prev (Finetune) Experiment
        results_dict_curr_and_prev_finetune_path = os.path.join(exp_result_save_path,
                                                            f"results_dict_curr_and_prev_finetune_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

        if not os.path.exists(results_dict_curr_and_prev_finetune_path):
            results_dict_curr_and_prev_finetune = run_single_finetune(curr_and_prev_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_curr_and_prev_finetune_path, results_dict_curr_and_prev_finetune)
            print(f"Saved at {results_dict_curr_and_prev_finetune_path}")
        else:
            print(results_dict_curr_and_prev_finetune_path + " already exists")

        ############### Run Sequential (Retrain) Experiment
        results_dict_sequential_retrain_path = os.path.join(exp_result_save_path,
                                                            f"results_dict_sequential_retrain_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

        if not os.path.exists(results_dict_sequential_retrain_path):
            results_dict_sequential_retrain = run_single(sequential_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_sequential_retrain_path, results_dict_sequential_retrain)
            print(f"Saved at {results_dict_sequential_retrain_path}")
        else:
            print(results_dict_sequential_retrain_path + " already exists")
        
        ############### Run Sequential (Finetune) Experiment
        results_dict_sequential_finetune_path = os.path.join(exp_result_save_path,
                                                            f"results_dict_sequential_finetune_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

        if not os.path.exists(results_dict_sequential_finetune_path):
            results_dict_sequential_finetune = run_single_finetune(sequential_loaders_dict, all_query, args.train_mode)
            save_obj_as_pickle(results_dict_sequential_finetune_path, results_dict_sequential_finetune)
            print(f"Saved at {results_dict_sequential_finetune_path}")
        else:
            print(results_dict_sequential_finetune_path + " already exists")

    # curr_and_randomprev_loaders_dict_path = os.path.join(exp_result_save_path,
    #                                                f"curr_and_randomprev_loaders_dict_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

    # if os.path.exists(curr_and_randomprev_loaders_dict_path):
    #     print(f"{curr_and_randomprev_loaders_dict_path} already exists.")
    #     curr_and_randomprev_loaders_dict = load_pickle(curr_and_randomprev_loaders_dict_path)
    # else:
    #     curr_and_randomprev_loaders_dict = get_curr_and_randomprev_loaders_from_features_dict(
    #                                            features_dict,
    #                                            args.train_mode,
    #                                            HYPER_DICT[TRAIN_MODES_CATEGORY[args.train_mode].network_type],
    #                                            excluded_bucket_idx=excluded_bucket_idx
    #                                        )
    #     save_obj_as_pickle(curr_and_randomprev_loaders_dict_path, curr_and_randomprev_loaders_dict)
        
    ############### Run Baseline Experiment
    results_dict_all_path = os.path.join(exp_result_save_path,
                                         f"results_dict_all_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")
    if not os.path.exists(results_dict_all_path):
        result_baseline_dict = run_baseline(all_loaders_dict, all_query, args.train_mode)
        save_obj_as_pickle(results_dict_all_path, result_baseline_dict)
        print(f"Saved at {results_dict_all_path}")
    else:
        print(f"Baseline result saved at {results_dict_all_path}")

    ############### Run Single Bucket Experiment
    results_dict_single_path = os.path.join(exp_result_save_path,
                                            f"results_dict_single_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

    if not os.path.exists(results_dict_single_path):
        result_single_dict = run_single(loaders_dict, all_query, args.train_mode)
        save_obj_as_pickle(results_dict_single_path, result_single_dict)
        print(f"Saved at {results_dict_single_path}")
    else:
        print(results_dict_single_path + " already exists")

    ############### Run Single Bucket (Finetune) Experiment
    results_dict_single_finetune_path = os.path.join(exp_result_save_path,
                                                     f"results_dict_single_finetune_{dataset_str(args.mode)}_{args.train_mode}_{seed_str}_ex_{excluded_bucket_idx}.pickle")

    if not os.path.exists(results_dict_single_finetune_path):
        result_single_finetune_dict = run_single_finetune(loaders_dict, all_query, args.train_mode)
        save_obj_as_pickle(results_dict_single_finetune_path, result_single_finetune_dict)
        print(f"Saved at {results_dict_single_finetune_path}")
    else:
        print(results_dict_single_finetune_path + " already exists")



    # import pdb; pdb.set_trace()

    
