import torch
from copy import deepcopy

def get_scheduler(cfg_scheduler, optimizer):
    cfg_scheduler_copy = deepcopy(cfg_scheduler)
    scheduler_type = cfg_scheduler_copy.pop('type')
    if scheduler_type == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **cfg_scheduler_copy)