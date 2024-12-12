import torch
import argparse
import numpy as np
import open3d as o3d
import os
from PIL import Image
from omegaconf import OmegaConf

from control.control_env import ControlSimulationEnv
from models.pipelines import TSQPipeline
from utils.yaml_utils import parse_unknown_args, parse_nested_args

def load_png_images_in_order(directory_path):
    	
	# image list
	images = []

	# file names
	file_names = sorted(
		[file_name for file_name in os.listdir(directory_path) if file_name.startswith('rgb_') and file_name.endswith('.png')],
		key=lambda x: int(x.split('_')[1].split('.')[0])
	)

	# load images
	for file_name in file_names:
		file_path = os.path.join(directory_path, file_name)
		img = Image.open(file_path)
		img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
		images.append(img)

	# stack
	images = torch.stack(images)
	return images

def get_camera_parameters(env, num_cameras, reduce_ratio):
	
	# get view poses
	camera_view_poses = env.get_view_poses(
		num_cameras=num_cameras)

	# get camera parameters
	camera_param_list = []
	for view_pose in camera_view_poses:
			
		# set camera parameters
		camera_param = env.sim._get_camera_param(
			camera_pose=view_pose,
			camera_intrinsic=(
				env.sim.realsense_intrinsic / reduce_ratio
			),
			camera_image_size=[
				int(env.sim.camera_image_size[0]/reduce_ratio), 
				int(env.sim.camera_image_size[1]/reduce_ratio)
			]
		)
		camera_param_list.append(camera_param)
	
	# set rendering frames
	render_frame_list = [15, 16, 17, 18, 19, 20, 21]
	camera_param_list = [camera_param_list[i] for i in render_frame_list]

	return camera_param_list

def process_camera_parameters(env, camera_param_list):

	proj_matrices = []
	camera_intr_list = []
	camera_pose_list = []

	for cam in camera_param_list:
		camera_intr = cam['camera_intr']
		camera_pose = cam['camera_pose']
		camera_image_size = cam['camera_image_size']
		camera_pose[0:3, 3] -= env.workspace_center
		proj_matrix = torch.from_numpy(
			camera_intr @ np.linalg.inv(camera_pose)[:3])
		proj_matrices.append(proj_matrix)
		camera_intr_list.append(camera_intr)
		camera_pose_list.append(camera_pose)	
	proj_matrices = torch.stack(proj_matrices)
	img_size = torch.tensor(camera_image_size)

	return {
		"camera_image_size" : img_size,
		"projection_matrices" : proj_matrices,
		"camera_intr" : camera_intr_list,
		"camera_pose" : camera_pose_list,
	}   

if __name__ == "__main__":
	
	# argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str)
	parser.add_argument('--device', default=0)
	parser.add_argument('--sim_type', type=str, default='table')

	# process cfg
	args, unknown = parser.parse_known_args()
	d_cmd_cfg = parse_unknown_args(unknown)
	d_cmd_cfg = parse_nested_args(d_cmd_cfg)
	cfg = OmegaConf.load(args.config)
	cfg = OmegaConf.merge(cfg, d_cmd_cfg)

	# set device
	if args.device == 'cpu':
		device = 'cpu'
	elif args.device == 'any':
		device = 'cuda'
	else:
		device = f'cuda:{args.device}'

	# set environment
	if args.sim_type in ['table', 'shelf']:
		env = ControlSimulationEnv(
			enable_gui=False, 
			object_types=[],
			num_objects=0,
			sim_type=args.sim_type,
			control_mode=True
		)
	else:
		raise ValueError(f'sim_type {args.sim_type} not in ["table", "shelf"]')

	# camera setting
	num_cameras = cfg.num_cameras
	reduce_ratio = cfg.reduce_ratio	

	# t2sqnet setting
	text_prompt = cfg.text_prompt
	conf_thld = cfg.conf_thld
	t2sqnet_cfg = cfg.get('t2sqnet_config', False)

	# load data
	img_list = load_png_images_in_order(f'examples/{args.sim_type}')

	# load camera
	camera_param_list = get_camera_parameters(env, num_cameras, reduce_ratio)
	camera_params = process_camera_parameters(env, camera_param_list)

	# load model
	tsqnet = TSQPipeline(
		t2sqnet_cfg.bbox_model_path,
		t2sqnet_cfg.bbox_config_path,
		t2sqnet_cfg.param_model_paths,
		t2sqnet_cfg.param_config_paths,
		t2sqnet_cfg.voxel_data_config_path,
		device=device,
		dummy_data_paths=t2sqnet_cfg.dummy_data_paths,
		num_augs=t2sqnet_cfg.num_augs
	)

	# tsqnet inference
	results = tsqnet.forward(
		img_list, camera_params, 
		text_prompt=text_prompt, 
		conf_thld=conf_thld,
		output_all=True
	)

	# object info
	obj_list = results[-1][0]
	for obj in obj_list:
		obj.SE3[:3, 3] += torch.tensor(env.workspace_center).to(obj.SE3)
		obj.construct()
		obj.send_to_device('cpu')

	# draw objects
	mesh_list = [obj.get_mesh() for obj in obj_list]
	o3d.visualization.draw_geometries(mesh_list)


