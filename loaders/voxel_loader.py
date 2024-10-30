import torch
import os
import os.path as osp
import random
import pickle
from copy import deepcopy
from torch.utils.data import Dataset
from tqdm import tqdm
from omegaconf import OmegaConf

from tablewarenet.tableware import *
class_num = len(name_to_idx)

class VoxelDataset(Dataset):
    def __init__(self, split, roots, class_name, voxel_trans_noise_max, voxel_size_noise_max, max_data_num=99999, preload=True, *args, **kwargs):
        self.file_list = []
        for root in roots:
            self.file_list += [osp.join(osp.join(root, class_name, split), file)
                               for file
                               in os.listdir(osp.join(root, class_name, split))]
        random.shuffle(self.file_list)
        self.class_name = class_name
        self.preload = preload
        self.max_data_num = max_data_num
        self.load_voxel_size(root)
        if self.preload:
            self.load()
        else:
            self.get_valid_file_list()
        self.voxel_trans_noise_max = voxel_trans_noise_max
        self.voxel_size_noise_max = voxel_size_noise_max
        print(f'total data num is {self.__len__()}')
    
    def load_voxel_size(self, root):
        config_dir = osp.join(root, 'dataset_config.yml')
        dataset_args = OmegaConf.load(config_dir)
        self.voxel_size = dataset_args["voxel_size"][self.class_name]
    
    def load_item(self, file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        f.close()
        # load position, orientation
        pose = data["object_pose"]
        pos = pose[0:3, 3]
        ori = torch.tensor([pose[1, 1], pose[1, 0]])
        # load parameter, differentiable point cloud
        param = data["object_param"]
        obj = name_to_class[self.class_name](
                SE3=pose,
                params=param,
                device='cpu'
            )
        diff_pc = obj.get_differentiable_point_cloud(dtype='torch', use_mask=True).squeeze()
        obj.params_deranger()
        param_deranged = obj.params
        # alignment for Bowl symmetry
        if self.class_name == "Bowl" and param_deranged[0] < param_deranged[1]:
            ori = torch.tensor([-pose[1, 0], pose[1, 1]])
            param_deranged[[0, 1]] = param_deranged[[1, 0]]
        param_deranged = param_deranged.detach()
        
        return {
            "raw_voxel": deepcopy(data["vox"]),
            "bound1": deepcopy(data['bound1']),
            "bound2": deepcopy(data['bound2']),
            "pos": deepcopy(pos),
            "ori": deepcopy(ori),
            "param": deepcopy(param_deranged),
            "diff_pc": deepcopy(diff_pc)
        }

    def load(self):
        self.raw_voxels = []
        self.bound1 = []
        self.bound2 = []
        self.pos = []
        self.ori = []
        self.param = []
        self.diff_pc = []
        data_num = 0
        for file in tqdm(self.file_list, desc='loading data...'):
            try:
                data = self.load_item(file)
                self.raw_voxels.append(data["raw_voxel"])
                self.bound1.append(data["bound1"])
                self.bound2.append(data["bound2"])
                self.pos.append(data["pos"])
                self.ori.append(data["ori"])
                self.param.append(data["param"])
                self.diff_pc.append(data["diff_pc"])
                data_num += 1
            except:
                print(f"file {file} is not valid")
            if data_num >= self.max_data_num:
                break
            
    def get_valid_file_list(self):
        new_file_list = []
        data_num = 0
        for file in tqdm(self.file_list, desc='checking data...'):
            try:
                with open(file, 'rb') as f:
                    _ = pickle.load(f)
                f.close()
                new_file_list.append(file)
                data_num += 1
                if data_num >= self.max_data_num:
                    break
            except:
                print(f"file {file} is not valid")
        self.file_list = new_file_list
            
    def __len__(self):
        if self.preload:
            return min(len(self.raw_voxels), self.max_data_num)
        else:
            return min(len(self.file_list), self.max_data_num)
    
    def __getitem__(self, idx):
        if self.preload:
            raw_voxel = deepcopy(self.raw_voxels[idx])
            bound1 = deepcopy(self.bound1[idx])
            bound2 = deepcopy(self.bound2[idx])
            pos = deepcopy(self.pos[idx])
            ori = deepcopy(self.ori[idx])
            param = deepcopy(self.param[idx])
            diff_pc = deepcopy(self.diff_pc[idx])
        else:
            data = self.load_item(self.file_list[idx])
            raw_voxel = data["raw_voxel"]
            bound1 = data["bound1"]
            bound2 = data["bound2"]
            pos = data["pos"]
            ori = data["ori"]
            param = data["param"]
            diff_pc = data["diff_pc"]
            
            
        if len(raw_voxel.shape) == 4:
            raw_voxel = raw_voxel.float().mean(dim=0)
        raw_voxel = raw_voxel.float()
        
        w_min1, w_max1, h_min1, h_max1, d_min1, d_max1 = bound1
        w_min2, w_max2, h_min2, h_max2, d_min2, d_max2 = bound2

        trans_noise_x = torch.randint(low=-self.voxel_trans_noise_max, high=self.voxel_trans_noise_max+1, size=[])
        trans_noise_y = torch.randint(low=-self.voxel_trans_noise_max, high=self.voxel_trans_noise_max+1, size=[])
        size_noise_x = torch.randint(low=-self.voxel_size_noise_max[0], high=self.voxel_size_noise_max[0]+1, size=[])
        size_noise_y = torch.randint(low=-self.voxel_size_noise_max[1], high=self.voxel_size_noise_max[1]+1, size=[])
        size_noise_z = torch.randint(low=-self.voxel_size_noise_max[2], high=self.voxel_size_noise_max[2]+1, size=[])
        bbox_pos_trans = self.voxel_size * torch.tensor([trans_noise_x, trans_noise_y, size_noise_z/2])

        w_min2 += trans_noise_x - size_noise_x
        w_max2 += trans_noise_x + size_noise_x
        h_min2 += trans_noise_y - size_noise_y
        h_max2 += trans_noise_y + size_noise_y
        d_max2 += trans_noise_y + size_noise_z

        w_min1 += trans_noise_x
        w_max1 += trans_noise_x
        h_min1 += trans_noise_y
        h_max1 += trans_noise_y

        inside_voxel = torch.zeros_like(raw_voxel).fill_(0.)
        inside_voxel[w_min2:w_max2, h_min2:h_max2, d_min2:d_max2] = 1.

        voxel = torch.stack([raw_voxel, inside_voxel])
        voxel = voxel[:, w_min1:w_max1, h_min1:h_max1, d_min1:d_max1]
        
        pos[0:2] -= bbox_pos_trans[0:2]
        diff_pc[...,0:2] -= bbox_pos_trans[0:2]
        
        return {
            "voxel" : voxel.detach(),
            "voxel_scale" : torch.tensor(self.voxel_size),
            "pos" : pos.detach(),
            "ori" : ori.detach(),
            "param" : param.detach(),
            "diff_pc" : diff_pc.detach()
            }
