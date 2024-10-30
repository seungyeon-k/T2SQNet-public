import os
import torch
import numpy as np
import pickle
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from signal import signal, SIGINT
try:
	import open3d.cuda.pybind as o3d
except:
	import open3d as o3d
from copy import deepcopy
import math
import multiprocessing as mp
from multiprocessing import Pool, Manager, set_start_method
set_start_method('spawn', force=True)
from functools import partial

from utils.debugging_windows import debugging_windows_voxelize

def handler(signalnum, frame):
	raise TypeError

def extract_and_append_bb(
		file,
		bbox_size_cls_dict_list,
		):
	
	# initialize for stop mp
	signal(SIGINT, handler)

	# name
	filename = file
	
	# read file
	with open(filename, 'rb') as f:
		data = pickle.load(f)
		f.close()			

	# load data
	objects_pose = data["objects_pose"]
	objects_param = data["objects_param"]
	objects_bbox = data["objects_bbox"]
	objects_class = data["objects_class"]

	# iterate for objects
	for obj_idx in range(len(objects_pose)):
			
		# get object info
		object_bbox = objects_bbox[obj_idx]		
		object_class = objects_class[obj_idx]

		bbox_size = torch.from_numpy(object_bbox[3:6])
		bbox_size_cls_dict_list[object_class].append(bbox_size)		

def process_file(
		file, 
		original_split_data_path,
		new_data_path,
		split,
		min_view_idx,
		max_view_idx,
		voxel_size_dict,
		voxel_size_for_debug,
		marginal_bbox_size_dict,
		max_bbox_size_dict,
		debug,
		save_data
		):

	# initialize for stop mp
	signal(SIGINT, handler)

	# read file
	file_idx = file.split('.')[0]
	filename = os.path.join(original_split_data_path, file)
	with open(filename, 'rb') as f:
		data = pickle.load(f)
		f.close()	

	# if debugging
	if debug:
		data_debug = dict()

	# load data
	mask_imgs = data["mask_imgs"][min_view_idx:max_view_idx+1]
	camera = data["camera"][min_view_idx:max_view_idx+1]
	objects_pose = data["objects_pose"]
	objects_param = data["objects_param"]
	objects_bbox = data["objects_bbox"]
	objects_class = data["objects_class"]
	camera_param_list = []
	img_list = []
	for cam, img in zip(camera, mask_imgs):
		camera_params = o3d.camera.PinholeCameraParameters()	
		camera_params.extrinsic = np.linalg.inv(
			cam['camera_pose'])
		camera_params.intrinsic.set_intrinsics(
			cam['camera_image_size'][1],
			cam['camera_image_size'][0],
			cam['camera_intr'][0,0],
			cam['camera_intr'][1,1],
			cam['camera_intr'][0,2],
			cam['camera_intr'][1,2])
		camera_param_list.append(camera_params)
		img_list.append(o3d.geometry.Image(
			np.asarray(img, dtype='float32'))
		)
	workspace_origin = data["workspace_origin"]

	# save object info in debug
	if debug:
		data_debug['objects_pose'] = deepcopy(objects_pose)
		data_debug["objects_param"] = deepcopy(objects_param)
		data_debug["objects_bbox"] = deepcopy(objects_bbox)
		data_debug["objects_class"] = deepcopy(objects_class)

	# total voxel carving data
	if debug:
		length = 0.767
		voxel_grid_total = o3d.geometry.VoxelGrid.create_dense(
			origin=[
				workspace_origin[0]-length/2, 
				workspace_origin[1]-length/2, 
				workspace_origin[2]], 
			color=[1,0,1], 
			voxel_size=voxel_size_for_debug, 
			width=length, 
			height =length, 
			depth=length)

		# voxel carving of total scene
		for img, params in zip(img_list, camera_param_list):
			voxel_grid_total.carve_silhouette(img, params)

		data_debug['voxel_grid_total'] = voxel_grid_total

	# iterate for objects
	for obj_idx in range(len(objects_pose)):
	
		# get object info
		object_pose = objects_pose[obj_idx]
		object_param = objects_param[obj_idx]
		object_bbox = objects_bbox[obj_idx]		
		object_class = objects_class[obj_idx]
		voxel_size = voxel_size_dict[object_class]
		max_bbox_size = torch.tensor(max_bbox_size_dict[object_class])
		marginal_bbox_size = torch.tensor(marginal_bbox_size_dict[object_class])

		# file name
		new_folder = os.path.join(
			new_data_path, object_class, split)
		new_filename = os.path.join(
			new_folder, f'{file_idx}_{obj_idx}.pkl')
		
		# bounding box
		max_bbox = np.concatenate(
			(
				object_bbox[0:2], 
				np.array([object_bbox[2] - object_bbox[5] + max_bbox_size[2]]),
				max_bbox_size), 
			axis=0
		)
		marginal_bbox = np.concatenate(
			(
				object_bbox[0:2], 
				np.array([object_bbox[2] - object_bbox[5] + marginal_bbox_size[2]]),
				marginal_bbox_size), 
			axis=0
		)
		if debug:
			data_debug["voxel_size"] = deepcopy(voxel_size)
			data_debug["object_bbox"] = deepcopy(object_bbox)
			data_debug["max_bbox"] = deepcopy(max_bbox)
			data_debug["marginal_bbox"] = deepcopy(marginal_bbox)

		# # initialize voxel grid
		voxel_grid_original = o3d.geometry.VoxelGrid.create_dense(
			origin=marginal_bbox[0:3] - marginal_bbox[3:6],
			color=[0.7,0.7,0.7],
			voxel_size=voxel_size,
			width=marginal_bbox[3] * 2,
			height=marginal_bbox[4] * 2,
			depth=marginal_bbox[5] * 2,
		)

		# voxel carving
		w = round(marginal_bbox[3] * 2 / voxel_size)
		h = round(marginal_bbox[4] * 2 / voxel_size)
		d = round(marginal_bbox[5] * 2 / voxel_size)
		vox_stacked = []
		for img, params in zip(img_list, camera_param_list):
			voxel_grid = deepcopy(voxel_grid_original)
			voxel_grid.carve_silhouette(img, params, keep_voxels_outside_image=True)
			voxels = voxel_grid.get_voxels()  # returns list of voxels
			try:
				list_indices = list(vx.grid_index for vx in voxels)
			except:
				print(filename)
			indices = np.stack(list_indices)
			indices_tensor = torch.from_numpy(indices).long()
			vox = torch.zeros(w, h, d)
			vox[
				indices_tensor[:, 0], 
				indices_tensor[:, 1], 
				indices_tensor[:, 2]
			] = 1
			vox = vox.to(torch.bool)
			vox_stacked.append(vox)
		vox_stacked = torch.stack(vox_stacked)
		vox = vox_stacked

		if debug:
			data_debug["voxel_grid"] = deepcopy(voxel_grid)

		# max bounding box
		w_min1 = math.floor(w * (marginal_bbox[3] - max_bbox[3]) / (2 * marginal_bbox[3]))
		w_max1 = math.ceil(w * (marginal_bbox[3] + max_bbox[3]) / (2 * marginal_bbox[3]))
		h_min1 = math.floor(h * (marginal_bbox[4] - max_bbox[4]) / (2 * marginal_bbox[4]))
		h_max1 = math.ceil(h * (marginal_bbox[4] + max_bbox[4]) / (2 * marginal_bbox[4]))
		d_min1 = 0
		d_max1 = math.ceil(d * (2 * max_bbox[5]) / (2 * marginal_bbox[5])) - 1
		bound1 = [w_min1, w_max1, h_min1, h_max1, d_min1, d_max1]

		# get bounding box inside voxels
		w_min2 = math.floor(w * (marginal_bbox[3] - object_bbox[3]) / (2 * marginal_bbox[3]))
		w_max2 = math.ceil(w * (marginal_bbox[3] + object_bbox[3]) / (2 * marginal_bbox[3]))
		h_min2 = math.floor(h * (marginal_bbox[4] - object_bbox[4]) / (2 * marginal_bbox[4]))
		h_max2 = math.ceil(h * (marginal_bbox[4] + object_bbox[4]) / (2 * marginal_bbox[4]))
		d_min2 = 0
		d_max2 = math.ceil(d * (2 * object_bbox[5]) / (2 * marginal_bbox[5])) - 1
		bound2 = [w_min2, w_max2, h_min2, h_max2, d_min2, d_max2]
		if debug:
			data_debug["vox"] = deepcopy(vox)
			data_debug["bound1"] = deepcopy(bound1)
			data_debug["bound2"] = deepcopy(bound2)

		# object info
		object_pose[:3, 3] -= object_bbox[:3]
		object_pose[2, 3] += object_bbox[5]

		# processed data			
		data_processed = {
			"vox" : vox,
			"bound1": bound1,
			"bound2": bound2,
			"object_pose" : object_pose, 
			"object_class" : object_class, 
			"object_param": object_param,
		}

		# debug
		if debug:
			debugging_windows_voxelize(data_debug)
		
		# save
		if save_data:
			# save
			with open(new_filename, 'wb') as f:
				pickle.dump(
					data_processed, f, pickle.HIGHEST_PROTOCOL)
				f.close()

if __name__ == '__main__':

	# argparser
	parser = argparse.ArgumentParser()
	parser.add_argument('--debug', action='store_true', help="debugging dataset")
	parser.add_argument('--folder_name', default=None, help="dataset folder name")
	args = parser.parse_args()

	################################################
	################## INITIALIZE ##################
	################################################

	# save
	save_data = True
	use_mp_for_bb = False

	# parameters
	voxel_size_for_debug = 0.005
	min_view_idx = 15
	max_view_idx = 21
	marginal_bbox_scale = 1.5
	max_bbox_scale = 1.2
	max_voxel_num = 100
	debug = args.debug

	# dataset path
	original_data_path = args.folder_name
	new_data_path = f'{args.folder_name}_voxelized'
	if not os.path.exists(new_data_path):
		os.makedirs(new_data_path)	

	# load dataset config
	yml_path = os.path.join(original_data_path, 'dataset_config.yml')
	dataset_args = OmegaConf.load(yml_path)

	# get class info
	class_list = dataset_args.object_types
	for cls in class_list:
		new_class_data_path = os.path.join(new_data_path, cls)
		if not os.path.exists(new_class_data_path):
			os.makedirs(new_class_data_path)
		for split in ['training', 'validation', 'test']:
			new_class_split_data_path = os.path.join(new_data_path, cls, split)
			if not os.path.exists(new_class_split_data_path):
				os.makedirs(new_class_split_data_path)

	################################################
	########### BOUNDING BOX CALCULATION ###########
	################################################

	# config name
	new_yaml_path = os.path.join('configs', 'voxelize_config.yml')
	
	# obtain bounding box
	dataset_args = OmegaConf.load(new_yaml_path)
	marginal_bbox_size_dict = dataset_args.marginal_bbox_size
	max_bbox_size_dict = dataset_args.max_bbox_size
	voxel_size_dict = dataset_args.voxel_size

	################################################
	############# DATASET VOXELIZATION #############
	################################################

	# split
	for split in ['training', 'validation', 'test']:

		# folder name
		original_split_data_path = os.path.join(
			original_data_path, split)
		original_split_data_list = os.listdir(original_split_data_path)
		dataset_len = len(original_split_data_list)

		# multiprocessing
		if debug:
			P = Pool(1)
			dataset_len = 1
			original_split_data_list = original_split_data_list[:1]
		else:
			num_cores = mp.cpu_count()
			P = Pool()
		with P as pool:
			list(
				tqdm(
					pool.imap(
						partial(
							process_file, 
							original_split_data_path=original_split_data_path,
							new_data_path=new_data_path,
							split=split,
							min_view_idx=min_view_idx,
							max_view_idx=max_view_idx,
							voxel_size_dict=voxel_size_dict,
							voxel_size_for_debug=voxel_size_for_debug,
							marginal_bbox_size_dict=marginal_bbox_size_dict, 
							max_bbox_size_dict=max_bbox_size_dict,
							debug=debug,
							save_data=save_data
						), 
						original_split_data_list
					),
					desc=f"{split} dataset processing ... ", 
					total=dataset_len
				)
			)

			pool.close()
			pool.join()

		if debug:
			break