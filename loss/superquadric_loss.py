import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch

class SuperquadricLoss(nn.Module):
    def __init__(self):
        super(SuperquadricLoss, self).__init__()

    def forward(self, pred_poses, pred_params, gt_pc):

        return 