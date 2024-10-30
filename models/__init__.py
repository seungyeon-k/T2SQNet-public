import torch

from models.modules import FC_vec
from models.resnet import ResNet
from models.detr3d import DERT3D
from models.param_head import ParamHead
from models.voxel_head import VoxelHead
from models.resnet3d import generate_model
from utils.yaml_utils import dictconfig_to_dict

model_type_dict={
    'fc_vec': FC_vec,
    'resnet': ResNet,
    'detr3d': DERT3D,
    'voxel_head' : VoxelHead,
	'param_head': ParamHead,
    'resnet3d': generate_model,
}

def get_model(model_cfg):
    """Recursively create model.

    Args:
        model_cfg (dict): model configuration.
            key "arch" shoud be contained and corresponding value should be exist among keys of "model_type_dict".
            Remain keys are means other arguments for initialize model class.
    Returns:
        torch.nn.Module: pytorch model class
    """
    model_cfg = dictconfig_to_dict(model_cfg)
    model_type = model_cfg.pop("arch")
    for k, v in model_cfg.items():
        if isinstance(v, dict) and v.get("arch", 'not_model') in model_type_dict.keys():
            model_cfg[k] = get_model(v)
    model = model_type_dict[model_type](**model_cfg)

    if model_cfg.get('pretrained', None):
        pretrained_model_path = model_cfg['pretrained']
        ckpt = torch.load(pretrained_model_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])

    return model.float()

