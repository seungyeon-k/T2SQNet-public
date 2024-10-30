import torch
import numpy as np
import os
import os.path as osp
import random
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm

from tablewarenet.tableware import name_to_idx

class DETR3DDataset(Dataset):
	def __init__(
		self,
		split,
		roots,
		input_type='mask',
		max_data_num=9999,
		reduce_ratio=1,
		max_obj_num=5,
		select_view=None,
		preload=True,
		relative_to_workspace=True,
		cutout=None,
		shuffle_dirs=True,
		distribute_data_eqaul_num=None,
		*args,
		**kwargs
		):
		if distribute_data_eqaul_num is not None:
			self.file_list = []
			for root in roots:
				self.file_list += [osp.join(osp.join(root, split), file) for file in os.listdir(osp.join(root, split))][:distribute_data_eqaul_num]
		else:
			self.file_list = []
			for root in roots:
				self.file_list += [osp.join(osp.join(root, split), file) for file in os.listdir(osp.join(root, split))]
			if shuffle_dirs:
				random.shuffle(self.file_list)
		self.maxpooler = torch.nn.MaxPool2d(reduce_ratio, stride=reduce_ratio)
		self.input_type = input_type
		self.max_data_num = max_data_num
		self.reduce_ratio = reduce_ratio
		self.max_obj_num = max_obj_num
		self.select_view = select_view
		self.preload = preload
		self.relative_to_workspace = relative_to_workspace
		if cutout is not None:
			self.cutout = RandomCutout(**cutout)
		else:
			self.cutout = None
		if preload:
			self.load()
			print(f'total data num is {self.__len__()}')
	
	def __getitem__(self, idx):
		if self.preload:
			mask_imgs = self.mask_imgs[idx]
			if self.cutout is not None:
				mask_imgs = self.cutout(mask_imgs)
			return {
				'mask_imgs': mask_imgs,
				'camera_projection_matrices': self.proj_matrices[idx],
				'img_size': self.img_size[idx],
				'gt_bbox': self.bbox[idx],
				'gt_cls': self.cls[idx],
				'gt_pose': self.pose[idx],
				'gt_param': self.param[idx],
				}
		else:
			file = self.file_list[idx]
			output_dict = self.load_item(file)
			if self.cutout is not None:
				output_dict['mask_imgs'] = self.cutout(output_dict['mask_imgs'])
			return output_dict

	def __len__(self):   
		return min(len(self.file_list), self.max_data_num)
	
	def load_item(self, file):
		
		with open(file, 'rb') as f:
			data = pickle.load(f)
		f.close()
		
		object_num = len(data['objects_pose'])
		if self.relative_to_workspace:
			workspace_origin = data["workspace_origin"]
		# add mask images
		
		if self.input_type == 'mask':
			if torch.is_tensor(data["mask_imgs"]):
				mask_imgs = data["mask_imgs"].repeat(3, 1, 1, 1).permute(1, 0, 2, 3).float()
			else:
				mask_imgs = torch.from_numpy(data["mask_imgs"]).repeat(3, 1, 1, 1).permute(1, 0, 2, 3).float()
			if self.select_view is not None:
				mask_imgs = mask_imgs[self.select_view]
			mask_imgs = self.maxpooler(mask_imgs).bool()
		elif self.input_type == 'rgb':
			if torch.is_tensor(data["mask_imgs"]):
				mask_imgs = data["rgb_imgs"].float().permute(0, 3, 1, 2)/255.
			else:
				mask_imgs = torch.from_numpy(data["rgb_imgs"]).float().permute(0, 3, 1, 2)/255.
			if self.select_view is not None:
				mask_imgs = mask_imgs[self.select_view]
			mask_imgs = self.maxpooler(mask_imgs)
		
		# add class
		cls = []
		for name in data['objects_class']:
			cls.append(name_to_idx[name])
		cls = torch.tensor(cls)
		
		# add bbox
		bbox = torch.from_numpy(data["objects_bbox"])
		if self.relative_to_workspace:
			bbox[..., 0:3] -= workspace_origin
		
		# add pose and params
		pose = torch.stack(data['objects_pose'])
		if self.relative_to_workspace:
			pose[..., 0:3, 3] -= workspace_origin
		param = data['objects_param']

		# padding objects
		if len(cls) < self.max_obj_num:
			padding_cls = torch.zeros(self.max_obj_num - object_num) - 1
			cls = torch.cat([cls, padding_cls], dim=0)
			padding_bbox = torch.zeros(self.max_obj_num - object_num, bbox.shape[-1])
			bbox = torch.cat([bbox, padding_bbox], dim=0)
			padding_pose = torch.eye(4).repeat(self.max_obj_num - object_num, 1, 1)
			pose = torch.cat([pose, padding_pose], dim=0)
			param += [torch.zeros(0) for _ in range(self.max_obj_num - object_num)]
		
		# add camera parameters
		proj_matrices = []
		if self.select_view is None:
			for cam in data["camera"]:
				camera_intr = cam['camera_intr'] / self.reduce_ratio
				camera_intr[2,2] = 1
				camera_pose = cam['camera_pose']
				if self.relative_to_workspace:
					camera_pose[0:3, 3] -= workspace_origin
				proj_matrix = torch.from_numpy(camera_intr @ np.linalg.inv(camera_pose)[:3])
				proj_matrices.append(proj_matrix)
		else:
			for i in self.select_view:
				cam = data["camera"][i]
				camera_intr = cam['camera_intr'] / self.reduce_ratio
				camera_intr[2,2] = 1
				camera_pose = cam['camera_pose']
				if self.relative_to_workspace:
					camera_pose[0:3, 3] -= workspace_origin
				proj_matrix = torch.from_numpy(camera_intr @ np.linalg.inv(camera_pose)[:3])
				proj_matrices.append(proj_matrix)
				
		proj_matrices = torch.stack(proj_matrices)
		img_size = torch.tensor(mask_imgs.size()[-2:])
		
		return {
			'mask_imgs': mask_imgs,
			'camera_projection_matrices': proj_matrices,
			'img_size': img_size,
			'gt_bbox': bbox,
			'gt_cls': cls,
			'gt_pose': pose,
			'gt_param': param,
			}
	
	def load(self):
		self.mask_imgs = []
		self.bbox = []
		self.cls = []
		self.pose = []
		self.param = []
		self.proj_matrices = []
		self.img_size = []

		data_num = 0
		for file in tqdm(self.file_list, desc='loading data...'):
			data = self.load_item(file) 
			self.mask_imgs.append(data["mask_imgs"])
			self.bbox.append(data["gt_bbox"])
			self.cls.append(data["gt_cls"])
			self.pose.append(data["gt_pose"])
			self.param.append(data["gt_param"])
			self.proj_matrices.append(data["camera_projection_matrices"])
			self.img_size.append(data["img_size"])
			data_num += 1
			if data_num >= self.max_data_num:
				break

class RandomCutout(object):
	def __init__(self, min_num_hole, max_num_hole, min_cutout_len, max_cutout_len):
		"""
		Randomly mask out one or several patches from each image in a batch.
		Args:
			min_num_hole (int): Minimum number of patches to cut out of each image.
			max_num_hole (int): Maximum number of patches to cut out of each image.
			min_cutout_len (int): Minimum length of the square cut.
			max_cutout_len (int): Maximum length of the square cut.
		"""
		self.min_num_hole = min_num_hole
		self.max_num_hole = max_num_hole
		self.min_cutout_len = min_cutout_len
		self.max_cutout_len = max_cutout_len

	def __call__(self, imgs):
		"""
		Args:
			imgs (Tensor): Tensor images of size (V, C, H, W).
		Returns:
			Tensor: Images with cutout of random rectangles applied.
		"""
		v, c, h, w = imgs.size()
		masks = torch.ones((v, 1, h, w), dtype=imgs.dtype, device=imgs.device)

		for i in range(v):
			num_holes = np.random.randint(self.min_num_hole, self.max_num_hole + 1)
			for _ in range(num_holes):
				y = np.random.randint(h)
				x = np.random.randint(w)
				length = np.random.randint(self.min_cutout_len, self.max_cutout_len + 1)

				y1 = np.clip(y - length // 2, 0, h)
				y2 = np.clip(y + length // 2, 0, h)
				x1 = np.clip(x - length // 2, 0, w)
				x2 = np.clip(x + length // 2, 0, w)

				masks[i, 0, y1: y2, x1: x2] = 0

		imgs = imgs * masks
		return imgs