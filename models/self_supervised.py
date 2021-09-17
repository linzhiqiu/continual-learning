import torch

MOCO_PATH = "/data3/zhiqiul/self_supervised_models/moco_r50_v2-e3b0c442.pth"
INSTANCE_PATH = "/data3/zhiqiul/self_supervised_models/lemniscate_resnet50_update.pth"
BYOL_PATH = "/data3/zhiqiul/self_supervised_models/byol_r50-e3b0c442.pth"
ROT_PATH = "/data3/zhiqiul/self_supervised_models/rotation_r50-cfab8ebb.pth"
MOCO_YFCC_GPU_8_PATH = "/data3/zhiqiul/self_supervised_models/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar"
MOCO_YFCC_GPU_4_RESNET18_PATH = "/data3/zhiqiul/self_supervised_models/yfcc_moco_models/sep_16_bucket_11_idx_0_gpu_4_resnet18/checkpoint_0199.pth.tar"

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


def load_moco_ckpt(model, path):
    checkpoint = torch.load(path)
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    return model

def moco_v2_yfcc_feb18_bucket_0_gpu_8(model, path=MOCO_YFCC_GPU_8_PATH):
    return load_moco_ckpt(model, path=path)

def moco_v2_yfcc_sep16_bucket_0_gpu_4_resnet18(model, path=MOCO_YFCC_GPU_4_RESNET18_PATH):
    return load_moco_ckpt(model, path=path)
