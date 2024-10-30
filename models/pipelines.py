import numpy as np
import torch
import time
import os
import math
import pickle
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms as T
from copy import deepcopy
import platform
if platform.system() == "Linux":
	from utils.suppress_logging import suppress_output
	with suppress_output():
		from lang_sam import LangSAM
elif platform.system() == "Windows":
	from lang_sam import LangSAM

from models import get_model
from tablewarenet.tableware import name_to_class, idx_to_name, name_to_idx
from tablewarenet.utils import sq2depth, sq2occ_for_eval
from functions.voxel_carving import voxel_carving

class TSQPipeline():
	def __init__(
		self,
		bbox_model_path,
		bbox_config_path,
		param_model_paths,
		param_config_paths,
		voxel_data_config_path,
		device,
		dummy_data_paths=None,
		num_augs=5,
		debug_mode=False,
		):
		self.device = device
		if platform.system() == "Linux":
			from utils.suppress_logging import suppress_output
			with suppress_output():
				self.mask_predictor = LangSAM()
		elif platform.system() == "Windows":
			self.mask_predictor = LangSAM()

		# self.mask_predictor = LangSAM()
		self.mask_predictor.device = self.device
		self.load_models(bbox_model_path, bbox_config_path, param_model_paths, param_config_paths)
		self.load_voxel_infos(voxel_data_config_path)
		if dummy_data_paths is not None:
			self.load_dummy_data(dummy_data_paths)
			self.use_dummy_data = True
		else:
			self.use_dummy_data = False
		self.debug_mode = debug_mode
		self.num_augs = num_augs
		
	def load_models(self, bbox_model_path, bbox_config_path, param_model_paths, param_config_paths):
		
		# load bbox model
		bbox_cfg = OmegaConf.load(bbox_config_path)
		self.bbox_predictor = get_model(bbox_cfg['model'])
		bbox_model_state = torch.load(bbox_model_path)["model_state"]
		self.bbox_predictor.load_state_dict(bbox_model_state)
		self.bbox_predictor.eval().to(self.device)
		self.class_name_list = bbox_cfg['model']['class_name_list']
		self.query_num = bbox_cfg['model']['num_query']
		
		# load param model
		self.param_predictors = []
		for param_model_path, param_config_path in zip(param_model_paths, param_config_paths):
			param_cfg = OmegaConf.load(param_config_path)
			param_predictor = get_model(param_cfg['model'])
			param_model_state = torch.load(param_model_path, map_location=self.device)["model_state"]
			param_predictor.load_state_dict(param_model_state)
			param_predictor.eval().to(self.device)
			self.param_predictors.append(param_predictor)
		
	def load_voxel_infos(self, voxel_data_config_path):
		voxel_data_config = OmegaConf.load(voxel_data_config_path)
		self.voxel_size = voxel_data_config['voxel_size']
		self.max_bbox_size = voxel_data_config['max_bbox_size']
		self.marginal_bbox_size = voxel_data_config['marginal_bbox_size']

	def load_dummy_data(self, dummy_data_paths):
		mask_imgs_list = []
		projection_matrices_list = []
		img_size_list = []
		for dummy_data_path in dummy_data_paths:
			file_list = os.listdir(dummy_data_path)
			for file in tqdm(file_list, disable=True):
				with open(os.path.join(dummy_data_path, file), 'rb') as f:
					data = pickle.load(f)
				f.close()
				workspace_origin = data["workspace_origin"]
				try:
					mask_imgs = torch.from_numpy(data['mask_imgs']).float()[15:22]
				except:
					mask_imgs = data['mask_imgs'].float()[15:22]
				cls = []
				for name in data['objects_class']:
					cls.append(name)
				proj_matrices = []
				camera_intr_list = []
				camera_pose_list = []
				for cam in data["camera"][15:22]:
					camera_intr = cam['camera_intr']
					camera_pose = cam['camera_pose']
					camera_pose[0:3, 3] -= workspace_origin
					proj_matrix = torch.from_numpy(camera_intr @ np.linalg.inv(camera_pose)[:3])
					proj_matrices.append(proj_matrix)
					camera_intr_list.append(camera_intr)
					camera_pose_list.append(camera_pose)
				proj_matrices = torch.stack(proj_matrices)
				img_size = torch.tensor(mask_imgs.shape[-2:])
				mask_imgs_list.append(mask_imgs)
				projection_matrices_list.append(proj_matrices)
				img_size_list.append(img_size)
		self.dummy_mask_imgs = torch.stack(mask_imgs_list)
		self.dummy_projection_matrices = torch.stack(projection_matrices_list)
		self.dummy_img_size = torch.stack(img_size_list)
		self.dummy_len = len(self.dummy_mask_imgs)
  
	def get_random_dummy(self):
		rand_id = torch.randint(self.dummy_len, [7])
		return self.dummy_mask_imgs[rand_id], self.dummy_projection_matrices[rand_id], self.dummy_img_size[rand_id]
 
	def forward(
			self, imgs, camera_params, 
			text_prompt="tableware", conf_thld=0.75,
			output_all=False, from_mask_imgs=False):
		"""_summary_

		Args:
			imgs (V x C x H x W): 0 ~ 255 scale RGB image
			camera_params (_type_): _description_
			text_prompt (str, optional): _description_. Defaults to "tableware".
			conf_thld (float, optional): _description_. Defaults to 0.75.

		Returns:
			_type_: _description_
		"""
		with torch.no_grad():
			# get camera params
			camera_projection_matrices = camera_params["projection_matrices"].float().to(self.device)
			img_size = camera_params["camera_image_size"].float().to(self.device)

			# get mask images
			if not from_mask_imgs:
				t = time.time()
				imgs = self.rgb2mask(imgs, text_prompt, num_aug=self.num_augs)
				# print(f'ellapsed time from rgb to mask: {time.time() - t}')

			# get bounding boxes
			t = time.time()
			bboxes, cls = self.mask2bbox(imgs, camera_projection_matrices, img_size, conf_thld)
			# print(f'ellapsed time from mask to bbox: {time.time() - t}')

			# infer objects
			t = time.time()
			obj_list = self.infer_obj(bboxes, cls, imgs, camera_params)
			# print(f'ellapsed time from bbox to object: {time.time() - t}')

			if output_all and not from_mask_imgs:
				return imgs, bboxes, cls, obj_list
			elif output_all and from_mask_imgs:
				return bboxes, cls, obj_list
			else:
				return obj_list

	def evaluation(self, data, text_prompt="tableware", conf_thld=0.75):
		# dataset process
		workspace_origin = data["workspace_origin"]
		rgb_imgs = data['rgb_imgs']
		
		# numpy to torch
		if torch.is_tensor(rgb_imgs):
			rgb_imgs = rgb_imgs.permute(0, 3, 1, 2)
		else:
			rgb_imgs = torch.tensor(rgb_imgs).float().permute(0, 3, 1, 2)

		# camera process
		proj_matrices = []
		camera_intr_list = []
		camera_pose_list = []
		for cam in data["camera"]:
			camera_intr = cam['camera_intr']
			camera_pose = deepcopy(cam['camera_pose'])
			camera_pose[0:3, 3] -= workspace_origin
			proj_matrix = torch.from_numpy(
				camera_intr @ np.linalg.inv(camera_pose)[:3])
			proj_matrices.append(proj_matrix)
			camera_intr_list.append(camera_intr)
			camera_pose_list.append(camera_pose)
		proj_matrices = torch.stack(proj_matrices)
		img_size = torch.tensor(rgb_imgs.size()[-2:])
		camera_params = {
			"camera_image_size" : img_size,
			"projection_matrices" : proj_matrices,
			"camera_intr" : camera_intr_list,
			"camera_pose" : camera_pose_list,
		}        
		
		obj_list, _ = self.forward(
			rgb_imgs, camera_params, text_prompt=text_prompt, 
			conf_thld=conf_thld
		)
		
		# object process
		objects_class = []
		objects_pose = []
		objects_param = []
		for obj in obj_list:
			objects_class.append(obj.name)
			SE3 = obj.SE3
			SE3[:3, 3] += torch.tensor(workspace_origin).to(SE3)
			objects_pose.append(SE3)
			objects_param.append(obj.params)
		objects_pose = torch.stack(objects_pose)

		# get depth, mask, and tsdf
		depth_list = sq2depth(
			objects_class, objects_pose, objects_param, 
			data["camera"], sim_type=data['sim_type'])	
		occupancy = sq2occ_for_eval(
			objects_class, objects_pose, objects_param, 
			data["vol_bnds"], data["voxel_size"]+1e-8)
		# instance_occ_list = []
		# for c, po, pa in zip(objects_class, objects_pose, objects_param):
		# 	single_occ = sq2occ_for_eval(
		# 	[c], po.unsqueeze(0), pa.unsqueeze(0), 
		# 	data["vol_bnds"], data["voxel_size"])
		# 	instance_occ_list.append(single_occ)

		# info
		info = dict()
		info['vol_bnds'] = data["vol_bnds"]
		info['voxel_size'] = data["voxel_size"]

		# info new
		info['objects_class'] = objects_class
		info['objects_pose'] = objects_pose
		info['objects_param'] = objects_param

		return depth_list, occupancy, info

	def rgb2mask(self, imgs, text_prompt, num_aug=5):
		# get mask tensors from RGB 255 scale imgs
		
		# color jittering
		hue_trans = torch.linspace(-0.25, 0.25, num_aug-1)
		mask_jitt = []
		for _idx in range(num_aug):
			if _idx > 0:
				imgs_aug = T.functional.adjust_hue(imgs/255., hue_trans[_idx-1])
			else:
				imgs_aug = imgs
			mask_imgs = []
			for img in imgs_aug:
				image_pil = Image.fromarray(
					(img*255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
				)
				masks, _, _, _ = self.mask_predictor.predict(image_pil, text_prompt)
				masks = (masks.sum(dim=0) > 0).float()
				mask_imgs.append(masks)
			mask_imgs = torch.stack(mask_imgs).to(self.device)
			mask_jitt.append(mask_imgs)
		mask_jitt = torch.stack(mask_jitt) #  J x V x H x W
		mask_jitt = (mask_jitt.sum(dim=0) >= 0.99).float() # V x H x W
		return mask_jitt
	
	def mask2bbox(self, mask_imgs, camera_projection_matrices, img_size, conf_thld):
		if self.use_dummy_data:
			dummy_mask_imgs, dummy_projection_matrices, dummy_img_size = self.get_random_dummy()
			mask_imgs = torch.cat([mask_imgs.unsqueeze(0), dummy_mask_imgs.to(self.device)], dim=0).float()
			camera_projection_matrices = torch.cat([camera_projection_matrices.unsqueeze(0), dummy_projection_matrices.to(self.device)], dim=0).float()
			img_size = torch.cat([img_size.unsqueeze(0), dummy_img_size.to(self.device)], dim=0).float()
			bounding_box_list, conf_list, _, _ = self.bbox_predictor(
	   			mask_imgs.unsqueeze(2).repeat(1, 1, 3, 1, 1),
		  		camera_projection_matrices,
		  		img_size)
			bboxes = bounding_box_list[-1][0].squeeze()
			confs = conf_list[-1][0].squeeze()
		else:
			# get bboxes
			bounding_box_list, conf_list, _, _ = self.bbox_predictor(
				mask_imgs.unsqueeze(0).unsqueeze(2).repeat(1, 1, 3, 1, 1),
				camera_projection_matrices.unsqueeze(0),
				img_size.unsqueeze(0)
			)
			bboxes = bounding_box_list[-1].squeeze()
			confs = conf_list[-1].squeeze()
		valid_idxs = torch.arange(self.query_num).to(self.device)[confs > conf_thld]
		bbox_list = []
		cls_list = []
		for idx in valid_idxs:
			object_idx = int(idx // (self.query_num / len(self.class_name_list)))
			bbox_list.append(bboxes[idx])
			cls_list.append(idx_to_name[object_idx])
		return bbox_list, cls_list
	
	def bbox2voxel(self, bbox, object_class, mask_imgs, camera_params):
		
		# get parameters
		voxel_size = self.voxel_size[object_class]
		max_bbox_size = self.max_bbox_size[object_class]
		marginal_bbox_size = self.marginal_bbox_size[object_class]
		bbox = bbox.detach().cpu().numpy()
			
		# bounding box
		max_bbox = np.concatenate(
			(
				bbox[0:2], 
				np.array([bbox[2] - bbox[5] + max_bbox_size[2]]),
				max_bbox_size), 
			axis=0
		)
		marginal_bbox = np.concatenate(
			(
				bbox[0:2], 
				np.array([bbox[2] - bbox[5] + marginal_bbox_size[2]]),
				marginal_bbox_size), 
			axis=0
		)

		# voxel carving
		raw_voxel = voxel_carving(
			mask_imgs, camera_params, marginal_bbox, voxel_size,
			device='cuda:0', smoothed=True)
		w, h, d = raw_voxel.shape
		
		# get bounding box inside voxels
		w_min1 = math.floor(w * (marginal_bbox[3] - max_bbox[3]) / (2 * marginal_bbox[3]))
		w_max1 = math.ceil(w * (marginal_bbox[3] + max_bbox[3]) / (2 * marginal_bbox[3]))
		h_min1 = math.floor(h * (marginal_bbox[4] - max_bbox[4]) / (2 * marginal_bbox[4]))
		h_max1 = math.ceil(h * (marginal_bbox[4] + max_bbox[4]) / (2 * marginal_bbox[4]))
		d_min1 = 0
		d_max1 = math.ceil(d * (2 * max_bbox[5]) / (2 * marginal_bbox[5])) - 1

		# get bounding box inside voxels
		w_min2 = math.floor(w * (marginal_bbox[3] - bbox[3]) / (2 * marginal_bbox[3]))
		w_max2 = math.ceil(w * (marginal_bbox[3] + bbox[3]) / (2 * marginal_bbox[3]))
		h_min2 = math.floor(h * (marginal_bbox[4] - bbox[4]) / (2 * marginal_bbox[4]))
		h_max2 = math.ceil(h * (marginal_bbox[4] + bbox[4]) / (2 * marginal_bbox[4]))
		d_min2 = 0
		d_max2 = math.ceil(d * (2 * bbox[5]) / (2 * marginal_bbox[5])) - 1

		# get input voxel value
		inside_voxel = torch.zeros_like(raw_voxel).fill_(0.)
		inside_voxel[w_min2:w_max2, h_min2:h_max2, d_min2:d_max2] = 1.
		voxel = torch.stack([raw_voxel, inside_voxel])
		voxel = voxel[:, w_min1:w_max1, h_min1:h_max1, d_min1:d_max1]
		
		return voxel.to(self.device)

	def infer_obj(self, bboxes, cls, mask_imgs, camera_params):
		obj_list = []
		voxel_info = []
		for bbox, c in zip(bboxes, cls):
				
			# bbox to voxel
			object_idx = name_to_idx[c]
			object_class = c
			t = time.time()
			voxel = self.bbox2voxel(bbox, object_class, mask_imgs, camera_params)
			# print(f'bbox2voxel elapsed time: {time.time() - t}')

			# parameter prediction
			voxel_scale = torch.tensor(self.voxel_size[object_class]).unsqueeze(0).to(self.device)
			t = time.time()
			obj_info = self.param_predictors[object_idx](voxel.unsqueeze(0), voxel_scale).squeeze()
			# print(f'param_predictors elapsed time: {time.time() - t}')
			pose = torch.eye(4).to(self.device)
			pose[0:3, 3] = obj_info[0:3] + bbox[0:3]
			pose[2, 3] -= bbox[5]
			angle = torch.atan2(obj_info[4], obj_info[3])
			pose[0, 0] = torch.cos(angle)
			pose[0, 1] = -torch.sin(angle)
			pose[1, 0] = torch.sin(angle)
			pose[1, 1] = torch.cos(angle)
			
			# reconstruct object
			obj = name_to_class[object_class](
				SE3=pose.detach(),
				params=obj_info[5:].detach(),
				device=self.device
			)
			obj.params_ranger()
			obj.construct()
			obj_list.append(obj)
			
			# get voxel info
			voxel_info.append({"voxel" : voxel, "voxel_scale" : voxel_scale})

		return obj_list, voxel_info

		