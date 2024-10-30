import numpy as np
import torchvision
import torch
import wandb
from omegaconf import OmegaConf
from torchvision.utils import make_grid

from tablewarenet.tableware import  idx_to_class
from loss.chamfer_loss import ChamferLoss
from loss.metrics import averageMeter

resize = torchvision.transforms.Resize((32, 32))
chamfer_loss = ChamferLoss()

class VoxelLogger:
    def __init__(self, prefix, items, best_model_metric=None, *args, **kwargs):
        self.prefix = prefix
        self.items = OmegaConf.to_container(items)
        self.best_model_metric = best_model_metric
        self.best_model_metric_val = None
        self.reset()

    def reset(self):
        for key, val in self.items.items():
            if val["type"] == 'average_meter':
                self.items[key]["value"] = averageMeter()
            else:
                self.items[key]["value"] = None
           
    def process_iter(self, data):
        for key, val in data.items():
            if key in self.items:
                if self.items[key]["type"] == 'average_meter':
                    self.items[key]["value"].update(val)
                else:
                    self.items[key]["value"] = val
    
    def get_scalars(self):
        scalars = {}
        for key, val in self.items.items():
            if val["type"] == 'scalar':
                scalars[key] = val["value"]
            elif val["type"] == 'average_meter' :
                scalars[key] = val["value"].avg
        return scalars
    
    def log_by_interval(self, i):
        for key, _ in self.items.items():
            if i % self.items[key]["interval"] == 0 and i > 0:
                self.log_wandb_instance(key, i)
        
    def log_all(self, i):
        for key, _ in self.items.items():
            self.log_wandb_instance(key, i)
                
    def log_wandb_instance(self, key, i):
        type = self.items[key]["type"]
        val = self.items[key]["value"]
        dict = self.items[key]
        key = self.prefix + '_' + key
        
        if type == "scalar":
            wandb_dict = {key : val}
            wandb.log(wandb_dict, step=i)
            
        elif type == "average_meter":
            wandb_dict = {key : val.avg}
            wandb.log(wandb_dict, step=i)
            
        elif type == "image":
            wandb_dict = {key : [wandb.Image(val)]}
            wandb.log(wandb_dict, step=i)
            
        elif type == "grid_images":
            B, V, C, H, W = val.size()
            grid = make_grid(val.reshape(B*V, C, H, W), nrow=V).permute(1, 2, 0).cpu().detach().numpy()
            wandb_dict = {key : [wandb.Image(grid)]}
            wandb.log(wandb_dict, step=i)
        
        elif type == "param_to_pc":
            cls, pos, ori, param = val["cls"], val["pos"], val["ori"], val["param"]
            color = dict["color"]
            obj_cls = idx_to_class[cls]
            B = len(pos)
            pose = torch.eye(4).repeat(B, 1, 1).to(pos.device)
            angle = torch.atan2(ori[...,1], ori[...,0])
            pose[..., 0, 0] = torch.cos(angle)
            pose[..., 0, 1] = -torch.sin(angle)
            pose[..., 1, 0] = torch.sin(angle)
            pose[..., 1, 1] = torch.cos(angle)
            pose[:, 0:3, 3] = pos
            wandb_pc_list = []
            for batch_idx in range(B):
                obj = obj_cls(
                    SE3=pose[batch_idx],
                    params=param[batch_idx],
                    device=param.device
                )
                obj.params_ranger()
                obj.construct()
                pc_numpy = obj.get_point_cloud()
                rgb = np.zeros_like(pc_numpy)
                rgb[...,:] = np.array(color)
                pc = np.concatenate([pc_numpy, rgb], axis=1)
                wandb_pc_list.append(wandb.Object3D(pc))
            wandb_dict = {key : wandb_pc_list}
            wandb.log(wandb_dict, step=i)
            
        elif type == "param_to_pc_with_gt":
            cls, pos, ori, param, color = val["cls"], val["pos"], val["ori"], val["param"], dict["color"]
            gt_val = self.items[dict["with_pc"]]["value"]
            gt_cls, gt_pos, gt_ori, gt_param, gt_color = gt_val["cls"], gt_val["pos"], gt_val["ori"], gt_val["param"], self.items[dict["with_pc"]]["color"]
            obj_cls = idx_to_class[cls]
            gt_obj_cls = idx_to_class[gt_cls]
            B = len(pos)
            pose = torch.eye(4).repeat(B, 1, 1).to(pos.device)
            angle = torch.atan2(ori[...,1], ori[...,0])
            pose[..., 0, 0] = torch.cos(angle)
            pose[..., 0, 1] = -torch.sin(angle)
            pose[..., 1, 0] = torch.sin(angle)
            pose[..., 1, 1] = torch.cos(angle)
            pose[:, 0:3, 3] = pos
            gt_pose = torch.eye(4).repeat(B, 1, 1).to(pos.device)
            gt_angle = torch.atan2(gt_ori[...,1], gt_ori[...,0])
            gt_pose[..., 0, 0] = torch.cos(gt_angle)
            gt_pose[..., 0, 1] = -torch.sin(gt_angle)
            gt_pose[..., 1, 0] = torch.sin(gt_angle)
            gt_pose[..., 1, 1] = torch.cos(gt_angle)
            gt_pose[:, 0:3, 3] = gt_pos
            wandb_pc_list = []
            for batch_idx in range(B):
                obj = obj_cls(
                    SE3=pose[batch_idx],
                    params=param[batch_idx],
                    device=param.device
                )
                obj.params_ranger()
                obj.construct()
                pc_numpy = obj.get_point_cloud()
                rgb = np.zeros_like(pc_numpy)
                rgb[...,:] = np.array(color)
                pc = np.concatenate([pc_numpy, rgb], axis=1)
                gt_obj = gt_obj_cls(
                    SE3=gt_pose[batch_idx],
                    params=gt_param[batch_idx],
                    device=gt_param.device
                )
                gt_obj.params_ranger()
                gt_obj.construct()
                gt_pc_numpy = gt_obj.get_point_cloud()
                gt_rgb = np.zeros_like(gt_pc_numpy)
                gt_rgb[...,:] = np.array(gt_color)
                gt_pc = np.concatenate([gt_pc_numpy, gt_rgb], axis=1)
                pc_all = np.concatenate([pc, gt_pc], axis=0)
                wandb_pc_list.append(wandb.Object3D(pc_all))
            wandb_dict = {key : wandb_pc_list}
            wandb.log(wandb_dict, step=i)
        
        elif type == "voxel":
            voxel = val
            voxel_size = dict["voxel_size"]
            downsample = dict["downsample"]
            B, _, w, h, d = voxel.size()
            X = torch.arange(w) * voxel_size
            Y = torch.arange(h) * voxel_size
            Z = torch.arange(d) * voxel_size
            grid_x, grid_y, grid_z = torch.meshgrid(X, Y, Z, indexing='ij')
            grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).to(voxel.device)
            wandb_pc_list = []
            for batch_idx in range(B):
                vox = voxel[batch_idx,...]
                pc_out_box = grid[(vox[0] == 1.) & (vox[1]==0.)].detach().cpu().numpy()
                rgb_out_box = np.zeros_like(pc_out_box)
                rgb_out_box[...,:] = np.array([255, 0, 0])
                pc_out_box = np.concatenate([pc_out_box, rgb_out_box], axis=-1)
                pc_in_box = grid[(vox[0] == 1.) & (vox[1]==1.)].detach().cpu().numpy()
                rgb_in_box = np.zeros_like(pc_in_box)
                rgb_in_box[...,:] = np.array([0, 255, 0])
                pc_in_box = np.concatenate([pc_in_box, rgb_in_box], axis=-1)
                pc = np.concatenate([pc_in_box, pc_out_box], axis=0)
                pc = pc[torch.randint(len(pc), [downsample])]
                wandb_pc_list.append(wandb.Object3D(pc))
            wandb_dict = {key : wandb_pc_list}
            wandb.log(wandb_dict, step=i)
    
    def get_best_model_booleans(self):
        key = self.best_model_metric
        type = self.items[key]["type"]
        val = self.items[key]["value"]
        dict = self.items[key]    
        
        if type == "scalar":
            if self.best_model_metric_val is None:
                bool = True
            elif dict['criterion'] == '↓':
                bool = self.best_model_metric_val > val
            elif dict['criterion'] == '↑':
                bool = self.best_model_metric_val < val
            if bool:
                self.best_model_metric_val = val
            return {key: bool}
            
        elif type == "average_meter":
            if self.best_model_metric_val is None:
                bool = True
            elif dict['criterion'] == '↓':
                bool = self.best_model_metric_val > val.avg
            elif dict['criterion'] == '↑':
                bool = self.best_model_metric_val < val.avg
            if bool:
                self.best_model_metric_val = val.avg
            return {key: bool}
        