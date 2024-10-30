import random
import numpy as np
import torch
import os
import yaml
import omegaconf
from omegaconf import OmegaConf
import ast

def save_yaml(filename, text):
	'''parse string as yaml then dump as a file'''
	with open(filename, 'w') as f:
		yaml.dump(yaml.safe_load(text), f, default_flow_style=False)
		
def parse_arg_type(val):
    if val[0] == '[' and val[-1] == ']':
        return ast.literal_eval(val)
    if val.isnumeric():
        return int(val)
    if (val == 'True') or (val == 'true'):
        return True
    if (val == 'False') or (val == 'false'):
        return False
    if (val == 'None'):
        return None
    try:
        return float(val)
    except:
        return str(val)

def parse_unknown_args(l_args):
    """convert the list of unknown args into dict
    this does similar stuff to OmegaConf.from_cli()
    I may have invented the wheel again..."""
    n_args = len(l_args) // 2
    kwargs = {}
    for i_args in range(n_args):
        key = l_args[i_args*2]
        val = l_args[i_args*2 + 1]
        assert '=' not in key, 'optional arguments should be separated by space'
        kwargs[key.strip('-')] = parse_arg_type(val)
    return kwargs

def parse_nested_args(d_cmd_cfg):
    """produce a nested dictionary by parsing dot-separated keys
    e.g. {key1.key2 : 1}  --> {key1: {key2: 1}}"""
    d_new_cfg = {}
    for key, val in d_cmd_cfg.items():
        l_key = key.split('.')
        d = d_new_cfg
        for i_key, each_key in enumerate(l_key):
            if i_key == len(l_key) - 1:
                d[each_key] = val
            else:
                if each_key not in d:
                    d[each_key] = {}
                d = d[each_key]
    return d_new_cfg

def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True
	
def dictconfig_to_dict(cfg: omegaconf.dictconfig.DictConfig) -> dict:
	new_dict = {}
	for k, v in cfg.items():
		if isinstance(v, omegaconf.dictconfig.DictConfig):
			new_dict[k] = dictconfig_to_dict(v)
		elif isinstance(v, omegaconf.listconfig.ListConfig):
			new_dict[k] = listconfig_to_list(v)
		else:
			new_dict[k] = v
	return new_dict

def listconfig_to_list(cfg: omegaconf.listconfig.ListConfig) -> list:
	new_list = []
	for v in cfg:
		if isinstance(v, omegaconf.dictconfig.DictConfig):
			new_list.append(dictconfig_to_dict(v))
		elif isinstance(v, omegaconf.listconfig.ListConfig):
			new_list.append(listconfig_to_list(v))
		else:
			new_list.append(v)
	return new_list

def load_config(cfg_model):
	for key, cfg_module in cfg_model.items():
		if 'module' in key:
			cfg_model[key] = OmegaConf.load(
				os.path.join(
					cfg_module.path, cfg_module.path.split('/')[-2]+".yml")
				).model
			cfg_model[key]['pretrained'] = os.path.join(
				cfg_module.path, cfg_module.checkpoint
			)

	return cfg_model

def update_dict(base_dict, dict_for_update):
    for k, v in dict_for_update.items():
        if k in base_dict.keys() and isinstance(base_dict[k], dict) and isinstance(v, dict):
            base_dict[k] = update_dict(base_dict[k], v)
        else:
            base_dict[k] = v
    return base_dict

def load_cfg_with_base(config_dir):
    cfg = OmegaConf.load(config_dir)
    cfg = dictconfig_to_dict(cfg)
    if 'base_yml' in cfg.keys():
        base_yml_dir = cfg.pop('base_yml')
        base_cfg = load_cfg_with_base(base_yml_dir)
        base_cfg = dictconfig_to_dict(base_cfg)
        cfg = update_dict(base_cfg, cfg)
    cfg = OmegaConf.create(cfg)
    return cfg

def gallery(array, ncols=3):
	nindex, height, width, intensity = array.shape
	nrows = nindex//ncols
	assert nindex == nrows*ncols
	result = (array.reshape(nrows, ncols, height, width, intensity)
			  .swapaxes(1,2)
			  .reshape(height*nrows, width*ncols, intensity))
	return result

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'
def clear_line(n=1):
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR)