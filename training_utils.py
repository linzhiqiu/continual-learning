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
from torchvision.datasets.folder import default_loader
class SimpleDataset(Dataset):
    def __init__(self, samples, transform, class_names=None):
        self.samples = samples
        self.transform = transform
        self.class_names = class_names
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        path, label = self.samples[index]
        sample = default_loader(path)
        sample = self.transform(sample)
        return sample, label

def get_imgnet_transforms():
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

def make_optimizer(network, lr):
    optimizer = torch.optim.SGD(network.parameters(), 
                                lr=lr,
                                weight_decay=1e-5,
                                momentum=0.9)
    return optimizer

def make_scheduler(optimizer, step_size=50):
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=step_size,
                    gamma=0.1
                )
    return scheduler

def make_loader(dataset, transform, shuffle=False, batch_size=256, num_workers=0):
    return torch.utils.data.DataLoader(
        SimpleDataset(dataset, transform), 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
    )

def make_model(arch, pretrained, selfsupervised):
    if pretrained or selfsupervised:
        print("=> using pre-trained model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=True)
        if selfsupervised:
            if selfsupervised == "moco_v2":
                model = self_supervised.moco_v2(model)
            elif selfsupervised == "byol":
                model = self_supervised.byol(model)
            elif selfsupervised == "rot":
                model = self_supervised.rot(model)
            elif selfsupervised == "deepcluster":
                model = self_supervised.deepcluster(model)
            elif selfsupervised == "relativeloc":
                model = self_supervised.relativeloc(model)
    else:
        print("=> creating model '{}'".format(arch))
        model = models.__dict__[arch]()
        print("No model checkpoint is supplied")
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
                    best_val_network = copy.deepcopy(network.state_dict())
                    print("Test additional:")
                    test(test_loader, network, save_loc=None)
            else:
                avg_test_acc_per_epoch.append(avg_acc)
            print(f"Average {phase} Loss {avg_loss}, Accuracy {avg_acc}")
        print()
    print(f"Best Test Accuracy (for best val model): {avg_test_acc_per_epoch[best_val_epoch]}")
    print(f"Best Test Accuracy overall: {max(avg_test_acc_per_epoch)}")
    network.load_state_dict(best_val_network)
    test(test_loader, network, save_loc=None)
    return network

def test(test_loader, network, save_loc='temp.model', class_names=None):
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
