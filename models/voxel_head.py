import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.chamfer_loss import ChamferLoss
from tablewarenet.tableware import *

class VoxelHead(nn.Module):
    def __init__(
        self,
        backbone,
        pos_head,
        ori_head,
        param_head,
        loss_config,
        cls_name,
        common_feature_layer=None,
        pos_from_global=False,
        ori_from_global=True,
        param_from_global=True,
        *args,
        **kwargs
        ):
        super().__init__()
        self.backbone = backbone
        self.pos_head = pos_head
        self.ori_head = ori_head
        self.param_head = param_head
        self.cls_idx = name_to_idx[cls_name]
        self.l1loss = nn.L1Loss()
        self.mseloss = nn.MSELoss()
        self.cham_loss = ChamferLoss(use_mask=True, reduce=True)
        self.cham_met = ChamferLoss(use_mask=False, reduce=True)
        self.dummy_object = name_to_class[cls_name](SE3=torch.eye(4), params='random', device='cpu')
        self.loss_config = loss_config
        self.common_feature_layer = common_feature_layer
        self.pos_from_global = pos_from_global
        self.ori_from_global = ori_from_global
        self.param_from_global = param_from_global
        
    def forward(self, voxel, voxel_scale):
        _, _, W, H, D = voxel.size()
        equi_feature, inv_feature = self.backbone(voxel)
        if self.common_feature_layer is not None:
            inv_feature = self.common_feature_layer(inv_feature)
        all_feature = torch.cat([inv_feature, equi_feature], dim=-1)
        # get position
        if self.pos_from_global:
            pos = self.pos_head(inv_feature)
        else:
            pos = self.pos_head(all_feature)
        pos = torch.cat([
            (pos[...,[0]] - 0.5) * voxel_scale.unsqueeze(-1) * W,
            (pos[...,[1]] - 0.5) * voxel_scale.unsqueeze(-1) * H,
            pos[...,[2]] * voxel_scale.unsqueeze(-1) * D
        ], dim=-1)
        # get orientation
        if self.ori_from_global:
            ori = self.ori_head(inv_feature)
        else:
            ori = self.ori_head(all_feature)
        ori = F.normalize(ori, dim=-1)
        # get parameter
        if self.param_from_global:
            param = self.param_head(inv_feature)
        else:
            param = self.param_head(all_feature)
        return torch.cat([pos, ori, param], dim=-1)
        
    def chamfer_loss(self, pred_pos, pred_ori, pred_param, gt_diff_pc):
        device = pred_param.device
        B = pred_param.shape[0]
        angle = torch.atan2(pred_ori[...,1], pred_ori[...,0])
        T = torch.eye(4).repeat(B, 1, 1).to(device).float()
        T[..., 0, 0] = torch.cos(angle)
        T[..., 0, 1] = -torch.sin(angle)
        T[..., 1, 0] = torch.sin(angle)
        T[..., 1, 1] = torch.cos(angle)
        T[..., 0:3, 3] = pred_pos
        obj = idx_to_class[self.cls_idx](SE3=T, params=pred_param, device=device)
        obj.params_ranger()
        obj.construct()
        pred_diff_pc = obj.get_differentiable_point_cloud(dtype='torch', use_mask=True)
        return self.cham_loss(pred_diff_pc, gt_diff_pc)
        
    def position_loss(self, pred_pos, gt_pos):
        return self.mseloss(pred_pos, gt_pos)
    
    def ori_loss(self, pred_ori, gt_ori):
        return self.l1loss(pred_ori, gt_ori)
    
    def param_loss(self, pred_param, gt_param, use_only_nonsymmetric=False):
        if use_only_nonsymmetric:
            idx = self.dummy_object.nonsymmetric_idx
            return self.l1loss(pred_param[...,idx], gt_param[...,idx])
        else:
            return self.l1loss(pred_param, gt_param)
    
    def chamfer_metric(self, pred_pos, pred_ori, pred_param, gt_pos, gt_ori, gt_param, downsample=1024):
        device = pred_param.device
        B = pred_param.shape[0]
        # get pred pose
        pred_angle = torch.atan2(pred_ori[...,1], pred_ori[...,0])
        pred_pose = torch.eye(4).repeat(B, 1, 1).to(device)
        pred_pose[..., 0, 0] = torch.cos(pred_angle)
        pred_pose[..., 0, 1] = -torch.sin(pred_angle)
        pred_pose[..., 1, 0] = torch.sin(pred_angle)
        pred_pose[..., 1, 1] = torch.cos(pred_angle)
        pred_pose[..., 0:3, 3] = pred_pos
        # get gt pose
        gt_angle = torch.atan2(gt_ori[...,1], gt_ori[...,0])
        gt_pose = torch.eye(4).repeat(B, 1, 1).to(device)
        gt_pose[..., 0, 0] = torch.cos(gt_angle)
        gt_pose[..., 0, 1] = -torch.sin(gt_angle)
        gt_pose[..., 1, 0] = torch.sin(gt_angle)
        gt_pose[..., 1, 1] = torch.cos(gt_angle)
        gt_pose[..., 0:3, 3] = gt_pos
        # construct objects
        pred_pcs = []
        gt_pcs = []
        for i in range(B):
            pred_obj = idx_to_class[self.cls_idx](SE3=pred_pose[i], params=pred_param[i], device=device)
            gt_obj = idx_to_class[self.cls_idx](SE3=gt_pose[i], params=gt_param[i], device=device)
            # get pred pc
            pred_obj.params_ranger()
            pred_obj.construct()
            pred_pc = torch.from_numpy(pred_obj.get_point_cloud())
            pred_pcs.append(pred_pc)
            # get gt pc
            gt_obj.params_ranger()
            gt_obj.construct()
            gt_pc = torch.from_numpy(gt_obj.get_point_cloud())
            gt_pcs.append(gt_pc)
        # get chamfer loss
        pred_pcs = torch.stack(pred_pcs)
        gt_pcs = torch.stack(gt_pcs)
        return self.cham_met(pred_pcs, gt_pcs)
    
    def get_loss(self, pred_pos, pred_ori, pred_param, gt_pos, gt_ori, gt_param, gt_diff_pc):
        if self.loss_config["position"]["weight"] > 0:
            pos_loss = self.position_loss(pred_pos, gt_pos)
        else:
            pos_loss = torch.tensor(0.)
            
        if self.loss_config["orientation"]["weight"] > 0:
            ori_loss = self.ori_loss(pred_ori, gt_ori)
        else:
            ori_loss = torch.tensor(0.)
            
        if self.loss_config["param"]["weight"] > 0:
            param_loss = self.param_loss(pred_param, gt_param, self.loss_config["param"]["use_only_nonsymmetric"])
        else:
            param_loss = torch.tensor(0.)
            
        if self.loss_config["chamfer"]["weight"] > 0:
            chamfer_loss = self.chamfer_loss(pred_pos, pred_ori, pred_param, gt_diff_pc)
        else:
            chamfer_loss = torch.tensor(0.)
            
        loss = pos_loss * self.loss_config["position"]["weight"] +\
               ori_loss * self.loss_config["orientation"]["weight"] +\
               param_loss * self.loss_config["param"]["weight"] +\
               chamfer_loss * self.loss_config["chamfer"]["weight"]
                
        return loss, pos_loss, ori_loss, param_loss, chamfer_loss
    
    def train_step(self, data, optimizer, device, *args, **kwargs):
        voxel = data["voxel"].to(device).float()
        voxel_scale = data["voxel_scale"].to(device).float()
        gt_pos = data["pos"].to(device).float()
        gt_ori = data["ori"].to(device).float()
        gt_param = data["param"].to(device).float()
        gt_diff_pc = data["diff_pc"].to(device).float()
        
        # model forward
        pred_pose_param = self(voxel, voxel_scale)
        
        # separate 
        pred_pos = pred_pose_param[...,:3]
        pred_ori = pred_pose_param[...,3:5]
        pred_param = pred_pose_param[...,5:]

        loss, pos_loss, ori_loss, param_loss, chamfer_loss = self.get_loss(pred_pos, pred_ori, pred_param, gt_pos, gt_ori, gt_param, gt_diff_pc)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {
            'loss' : loss.item(),
            'pos_loss' : pos_loss.item(),
            'ori_loss' : ori_loss.item(),
            'param_loss' : param_loss.item(),
            'chamfer_loss' : chamfer_loss.item(),
            'voxel': voxel,
            'gt_pc': {
                'cls' : self.cls_idx,
                'pos': gt_pos,
                'ori': gt_ori, 
                'param': gt_param,
                },
            'pred_pc': {
                'cls' : self.cls_idx,
                'pos': pred_pos,
                'ori': pred_ori,
                'param': pred_param,
                },
            }

    def validation_step(self, data, device, *args, **kwargs):
        with torch.no_grad():
            voxel = data["voxel"].to(device).float()
            voxel_scale = data["voxel_scale"].to(device).float()
            gt_pos = data["pos"].to(device).float()
            gt_ori = data["ori"].to(device).float()
            gt_param = data["param"].to(device).float()
            gt_diff_pc = data["diff_pc"].to(device).float()
            
            # model forward
            pred_pose_param = self(voxel, voxel_scale)
            
            # separate 
            pred_pos = pred_pose_param[...,:3]
            pred_ori = pred_pose_param[...,3:5]
            pred_param = pred_pose_param[...,5:]

            loss, pos_loss, ori_loss, param_loss, chamfer_loss = self.get_loss(pred_pos, pred_ori, pred_param, gt_pos, gt_ori, gt_param, gt_diff_pc)
            chamfer_metric = self.chamfer_metric(pred_pos, pred_ori, pred_param, gt_pos, gt_ori, gt_param)
            
            return {
                'loss' : loss.item(),
                'pos_loss' : pos_loss.item(),
                'ori_loss' : ori_loss.item(),
                'param_loss' : param_loss.item(),
                'chamfer_loss' : chamfer_loss.item(),
                'chamfer_metric' : chamfer_metric.item(),
                'voxel': voxel,
                'gt_pc': {
                    'cls' : self.cls_idx,
                    'pos': gt_pos,
                    'ori': gt_ori, 
                    'param': gt_param,
                    },
                'pred_pc': {
                    'cls' : self.cls_idx,
                    'pos': pred_pos,
                    'ori': pred_ori,
                    'param': pred_param,
                    },
                }