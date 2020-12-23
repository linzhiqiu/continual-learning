import torch

MOCO_PATH = "/project_data/ramanan/zhiqiu/Self-Improving/models/moco_r50_v2-e3b0c442.pth"
INSTANCE_PATH = "/project_data/ramanan/zhiqiu/Self-Improving/models/lemniscate_resnet50_update.pth"
BYOL_PATH = "/project_data/ramanan/zhiqiu/Self-Improving/models/byol_r50-e3b0c442.pth"
ROT_PATH = "/project_data/ramanan/zhiqiu/Self-Improving/models/rotation_r50-cfab8ebb.pth"
DEEPCLUSTER_PATH = "/project_data/ramanan/zhiqiu/Self-Improving/models/byol_r50-e3b0c442.pth"
RELATIVELOC_PATH = "/project_data/ramanan/zhiqiu/Self-Improving/models/byol_r50-e3b0c442.pth"

def moco_v2(model, path=MOCO_PATH):
    checkpoint = torch.load(path)['state_dict']
    model.load_state_dict(checkpoint, strict=False)
    return model

def byol(model, path=BYOL_PATH):
    checkpoint = torch.load(path)['state_dict']
    model.load_state_dict(checkpoint, strict=False)
    return model


def rot(model, path=ROT_PATH):
    checkpoint = torch.load(path)['state_dict']
    model.load_state_dict(checkpoint, strict=False)
    return model

def deepcluster(model, path=DEEPCLUSTER_PATH):
    checkpoint = torch.load(path)['state_dict']
    model.load_state_dict(checkpoint, strict=False)
    return model

def relativeloc(model, path=RELATIVELOC_PATH):
    checkpoint = torch.load(path)['state_dict']
    model.load_state_dict(checkpoint, strict=False)
    return model