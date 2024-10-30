import torch
import torch.nn as nn
from tablewarenet.tableware import name_to_class

class ParamHead(nn.Module):
    def __init__(
        self,
        class_list,
        **predictors
        ):
        super().__init__()
        self.class_name_list = class_list
        self.class_list = [name_to_class[cls] for cls in class_list]
        self.sq_predictors = nn.ModuleList([])
        for _, val in predictors.items():
            self.sq_predictors.append(val)

    def forward(
        self,
        queries
        ):
        params = []
        for q, sq_predictor in zip(queries.split(int(queries.shape[1]/len(self.sq_predictors)), dim=1), self.sq_predictors):
            params.append(sq_predictor(q))
        return params
    
    def param2pc(
        self,
        class_idx,
        params,
        *args,
        **kwargs
        ):
        B = params.shape[0]
        pos = params[...,0:3]
        angle = torch.atan2(params[...,4], params[...,3])
        T = torch.eye(4).repeat(B, 1, 1).to(params.device).float()
        T[..., 0, 0] = torch.cos(angle)
        T[..., 0, 1] = -torch.sin(angle)
        T[..., 1, 0] = torch.sin(angle)
        T[..., 1, 1] = torch.cos(angle)
        T[..., 0:3, 3] = pos
        objects = self.class_list[class_idx](
            SE3=T,
            params=params[...,5:],
            device=params.device
        )
        objects.params_ranger()
        objects.construct()
        pc_pred = objects.get_differentiable_point_cloud(dtype='torch', use_mask=True)
        return pc_pred, T, objects.params, objects.nonsymmetric_idx