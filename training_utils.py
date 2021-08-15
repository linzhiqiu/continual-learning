import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from models import self_supervised
from tqdm import tqdm

import copy

from torch.utils.data import Dataset

class CLIPDataset(Dataset):
    def __init__(self, samples, class_names=None):
        self.samples = samples
        self.class_names = class_names
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        sample, label = self.samples[index]
        sample = torch.from_numpy(sample)
        return sample, label


class TensorDataset(Dataset):
    def __init__(self, samples, class_names=None):
        self.samples = samples
        self.class_names = class_names

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample, label = self.samples[index]
        return sample, label

def make_numpy_loader(items, batch_size, shuffle=False, num_workers=4):
    return torch.utils.data.DataLoader(
        CLIPDataset(items),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

def make_tensor_loader(items, batch_size, shuffle=False, num_workers=4):
    return torch.utils.data.DataLoader(
        TensorDataset(items),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

def make_image_loader(items, batch_size, shuffle=False, fixed_crop=False, num_workers=4):
    items = [(m.get_path(), l) for m, l in items]
    train_transform, test_transform = get_imgnet_transforms()
    if shuffle and not fixed_crop:
        transform = train_transform
    else:
        transform = test_transform
    return make_loader(
        items, transform, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
    )

def get_unnormalize_func():
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    return inv_normalize

def get_imgnet_transforms():
    # Note that this is not exactly imagenet transform/moco transform for val set
    # Because we resize to 224 instead of 256
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, test_transform

def make_optimizer(network, lr, weight_decay=1e-5, momentum=0.9):
    optimizer = torch.optim.SGD(
        list(filter(lambda x: x.requires_grad, network.parameters())),
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum
    )
    return optimizer

def make_lbfgs_optimizer(network, lr):
    # lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None
    optimizer = torch.optim.LBFGS(
        list(filter(lambda x: x.requires_grad, network.parameters())),
        lr=lr,
        max_iter=20,
        tolerance_grad=1e-07, tolerance_change=1e-09,
        history_size=100, line_search_fn=None
    )
    return optimizer

def make_scheduler(optimizer, step_size=50, gamma=0.1):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
    return scheduler

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output

def make_model(arch, pretrained, selfsupervised, train_mode='finetune', output_size=1000):
    # if output_size is None, then remove last layer
    if arch == 'mlp':
        print(f"Using a mlp network with input size 1024")
        return MLP(1024, 2048, output_size)
    elif arch == 'linear':
        print(f"Using a single linear layer")
        return torch.nn.Linear(1024, output_size)

    if pretrained or selfsupervised:
        print("=> using pre-trained model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=pretrained)
        if arch == 'resnet50' and selfsupervised:
            if selfsupervised == "moco_v2":
                model = self_supervised.moco_v2(model)
            elif selfsupervised == "byol":
                model = self_supervised.byol(model)
            elif selfsupervised == "rot":
                model = self_supervised.rot(model)
            elif selfsupervised == "moco_v2_yfcc_feb18_bucket_0_gpu_8":
                model = self_supervised.moco_v2_yfcc_feb18_bucket_0_gpu_8(model)
            else:
                raise NotImplementedError()
    else:
        print("=> creating model '{}'".format(arch))
        model = models.__dict__[arch]()
        print("No model checkpoint is supplied")
    
    if arch == 'resnet50':
        # Change output size
        if output_size == None:
            print(f"removing the last layer")
            model.fc = torch.nn.Identity()
        elif model.fc.weight.shape[0] != output_size:
            print(f"changing the size of last layer to {output_size}")
            model.fc = torch.nn.Linear(model.fc.weight.shape[1], output_size)
        
        
        # Change requires_grad flag if train_mode == 'freeze'
        if train_mode == 'freeze':
            print("Freezing the model!!!!!!!!!!!!!!!!!!!!!!!")
            for p in model.parameters():
                p.requires_grad = False
            for p in model.fc.parameters():
                p.requires_grad = True
    else:
        raise NotImplementedError()
    return model

def train(train_loader, val_loader, test_loader, network, epochs=150, lr=0.1, step_size=60):
    network = network.cuda()
    optimizer = make_optimizer(network, lr)
    scheduler = make_scheduler(optimizer, step_size=step_size)
    
    criterion = torch.nn.NLLLoss(reduction='mean')

    avg_loss_per_epoch = []
    avg_acc_per_epoch = []
    avg_val_loss_per_epoch = []
    avg_val_acc_per_epoch = []

    avg_test_acc_per_epoch = []

    best_val_epoch_train_acc = 0
    best_val_acc = 0
    best_val_epoch = None

    best_val_network = None

    loaders = {'train' : train_loader, 'val' : val_loader, 'test' : test_loader}
    for epoch in range(0, epochs):
        print(f"Epoch {epoch}")
        for phase in loaders.keys():
            if phase == 'train':
                network.train()
            else:
                network.eval()

            running_loss = 0.0
            running_corrects = 0.
            count = 0

            pbar = tqdm(loaders[phase])

            for batch, data in enumerate(pbar):
                inputs, labels = data
                count += inputs.size(0)
                    
                inputs = inputs.cuda()
                labels = labels.cuda()
                # import pdb; pdb.set_trace()

                if phase == 'train': optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # import pdb; pdb.set_trace()
                    outputs = network(inputs)
                    _, preds = torch.max(outputs, 1)

                    log_probability = torch.nn.functional.log_softmax(outputs, dim=1)
                    # import pdb; pdb.set_trace()
                    loss = criterion(log_probability, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_postfix(loss=float(running_loss)/count, 
                                    acc=float(running_corrects)/count,
                                    epoch=epoch,
                                    phase=phase)
                
            avg_loss = float(running_loss)/count
            avg_acc = float(running_corrects)/count
            if phase == 'train': 
                avg_loss_per_epoch.append(avg_loss)
                avg_acc_per_epoch.append(avg_acc)
                scheduler.step()
            elif phase == 'val':
                avg_val_loss_per_epoch.append(avg_loss)
                avg_val_acc_per_epoch.append(avg_acc)
                if avg_acc > best_val_acc:
                    print(f"Best val accuracy at epoch {epoch} being {avg_acc}")
                    best_val_epoch = epoch
                    best_val_acc = avg_acc
                    best_val_epoch_train_acc = avg_acc_per_epoch[-1]
                    print(f"Train accuracy at epoch {epoch} being {best_val_epoch_train_acc}")
                    best_val_network = copy.deepcopy(network.state_dict())
                    # print("Test additional:")
                    # test(test_loader, network, save_loc=None)
            else:
                avg_test_acc_per_epoch.append(avg_acc)
            print(f"Average {phase} Loss {avg_loss}, Accuracy {avg_acc}")
        print()
    print(f"Best Test Accuracy (for best val model): {avg_test_acc_per_epoch[best_val_epoch]}")
    print(f"Best Test Accuracy overall: {max(avg_test_acc_per_epoch)}")
    network.load_state_dict(best_val_network)
    test(test_loader, network, save_loc=None)
    return network

def test(test_loader, network, save_loc=None, class_names=None):
    network = network.cuda().eval()
    running_corrects = 0.
    count = 0

    pbar = tqdm(test_loader)

    # if class_names:
    #     class_accs = {}
    #     for i, class_name in enumerate(class_names):
    #         class_accs[i] = {'corrects' : 0., 'counts' : 0.}

    for batch, data in enumerate(pbar):
        inputs, labels = data
        count += inputs.size(0)
            
        inputs = inputs.cuda()
        labels = labels.cuda()
        # import pdb; pdb.set_trace()

        with torch.set_grad_enabled(False):
            outputs = network(inputs)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)
        pbar.set_postfix(acc=float(running_corrects)/count)
        
    avg_acc = float(running_corrects)/count
    print(f"Best Test Accuracy on test set: {avg_acc}")
    if save_loc:
        torch.save(network.state_dict(), save_loc)
    return avg_acc
