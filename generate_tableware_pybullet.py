import os
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from functools import partial
from multiprocessing import Pool, set_start_method, current_process
set_start_method('spawn', force=True)

from control.control_env import ControlSimulationEnv
from utils.yaml_utils import save_yaml
from utils.debugging_windows import debugging_windows_pybullet

def generate_data(
		data_num,
		split_data_path,
		enable_gui,
		object_types,
		num_objects,
		num_cameras,
		reduce_ratio,
		debug,
		sim_type='shelf'
		):
		
	# file name
	filename = os.path.join(split_data_path, f"{data_num}.pkl")

	# get current process index
	process_cur = current_process()
	process_idx = process_cur._identity[0]

	# reset environment
	env = ControlSimulationEnv(
		enable_gui=enable_gui, 
		object_types=object_types,
		num_objects=num_objects,
  		sim_type=sim_type,
		process_idx=process_idx
	)
	env.reset()
	
	# initialize lists
	mask_img_list = []
	camera_param_list = []
	view_poses = env.get_view_poses(num_cameras=num_cameras)

	# get images
	for view_pose in view_poses:
		
		# set camera parameters
		camera_param = env.sim._get_camera_param(
			camera_pose=view_pose,
			camera_intrinsic=(
				env.sim.realsense_intrinsic / reduce_ratio
			),
			camera_image_size=[
				int(env.sim.camera_image_size[0]/reduce_ratio), int(env.sim.camera_image_size[1]/reduce_ratio)
			]
		)
		
		# get camera images
		_, _, mask_image = env.sim.get_camera_data(
			camera_param
		)
		mask_image[mask_image < 3] = 0
		mask_image[mask_image >= 3] = 1
		mask_image = mask_image.astype(bool)

		# append images
		mask_img_list.append(mask_image)
		camera_param_list.append(camera_param)
	
	# stack
	mask_img_list = np.stack(mask_img_list)

	# get labels
	pc = []
	diff_pc = []
	bbox = []
	object_poses = []
	shape_classes = []
	shape_parameters = []
	for obj in env.object_infos:

		# object pc
		pc.append(obj.get_point_cloud())

		# object differentiable pc
		diff_pc.append(
			obj.get_differentiable_point_cloud(dtype='numpy').squeeze()
		)
		
		# bounding box
		bbox_min, bbox_max = obj.get_bounding_box()
		bbox_center = (bbox_min + bbox_max)/2
		bbox_size = bbox_max - bbox_center
		bbox.append(
			np.concatenate([bbox_center, bbox_size], axis=0)
		)

		# object poses
		object_poses.append(obj.SE3)

		# shape class
		shape_classes.append(obj.name)

		# shape parameters
		shape_parameters.append(obj.params)

	# stack
	pc = np.stack(pc)
	bbox = np.stack(bbox)

	# save data
	data = {
		"mask_imgs" : mask_img_list,
		"camera" : camera_param_list,
		"objects_pose": object_poses,
		"objects_class": shape_classes,
		"objects_param": shape_parameters,
		# "objects_pc": pc,
		"objects_diff_pc": diff_pc,
		"objects_bbox": bbox,
		"workspace_origin" : env.workspace_center
	}
	if debug: # debug
		debugging_windows_pybullet(data)
	with open(filename, 'wb') as f: # save
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
		f.close()

	env.sim.close()

if __name__ == '__main__':
	
	# argparser
	parser = argparse.ArgumentParser()
	parser.add_argument('--enable_gui', action='store_true', help='show PyBullet simulation window')
	parser.add_argument('--debug', action='store_true', help="debugging dataset")
	parser.add_argument('--data_path', type=str, default='datasets', help="path to data")
	parser.add_argument('--sim_type', type=str, default='table', help="environment setting")
	parser.add_argument('--folder_name', default=None, help="dataset folder name")
	parser.add_argument('--num_cameras', type=int, default=8, help="number of cameras")
	parser.add_argument('--num_objects', type=int, default=4, help="number of objects")
	parser.add_argument('--full_view', action='store_true', help="whether full view or not")
	parser.add_argument('--reduce_ratio', type=float, default=2, help="reduce resolution of the mask image")
	parser.add_argument('--training_num', type=int, default=1, help="number of training scenes")
	parser.add_argument('--validation_num', type=int, default=1, help="number of validation scenes")
	parser.add_argument('--test_num', type=int, default=1, help="number of testing scenes")
	parser.add_argument('--object_types', nargs='+', 
		default=['WineGlass', 'Bowl', 'Bottle', 'BeerBottle', 'HandlessCup', 'Mug', 'Dish'],
		help="choose object types"
	)
	args = parser.parse_args()

	# print args
	for key in vars(args):
		print(f"[{key}] = {getattr(args, key)}")

	if args.folder_name is None:
		folder_name = datetime.now().strftime('%Y%m%d-%H%M')
	else:
		folder_name = args.folder_name	

	# dataset path
	data_path = os.path.join(args.data_path, folder_name)
	if not os.path.exists(data_path):
		os.makedirs(data_path)

	# save dataset config
	yml_path = os.path.join(data_path, 'dataset_config.yml')
	save_yaml(yml_path, OmegaConf.to_yaml(vars(args)))

	# parameters
	num_cameras = args.num_cameras
	num_objects = args.num_objects
	enable_gui = args.enable_gui
	object_types = args.object_types
	debug = args.debug
	reduce_ratio = args.reduce_ratio
	sim_type= args.sim_type

	# split
	for split in ['training', 'validation', 'test']:
		if getattr(args, split+"_num") == 0:
			continue

		split_data_path = os.path.join(data_path, split)
		if not os.path.exists(split_data_path):
			os.makedirs(split_data_path)
		dataset_len = getattr(args, split+"_num")
		
		# get pool
		if debug:
			P = Pool(1)
			dataset_len = 1
		else:
			P = Pool()

		# multiprocessing
		with P as pool:
			list(
				tqdm(
						pool.imap(
							partial(
								generate_data, 
								split_data_path=split_data_path,
								enable_gui=enable_gui,
								object_types=object_types,
								num_objects=num_objects,
								num_cameras=num_cameras,
								reduce_ratio=reduce_ratio,
								debug=debug,
								sim_type=sim_type
							), 
						list(range(dataset_len))
					),
					desc=f"{split} dataset ... ", 
					total=dataset_len
				)
			)		

			pool.close()
			pool.join()

		# break if debug
		if debug:
			break

