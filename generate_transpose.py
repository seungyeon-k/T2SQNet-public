import numpy as np
import os
import pickle
import sys
from tqdm import tqdm
from PIL import Image
sys.path.append(os.getcwd())
from datetime import datetime
from omegaconf import OmegaConf

from control.control_env import ControlSimulationEnv
from utils.yaml_utils import save_yaml
from utils.argparse_blender import ArgumentParserForBlender
from utils.debugging_windows import debugging_windows_transpose

def run(args):

	# print args
	for key in vars(args):
		print(f"[{key}] = {getattr(args, key)}")

	# additional parameters
	enable_gui = args.enable_gui
	object_types = args.object_types
	num_objects = args.num_objects
	sim_type = args.sim_type
	num_cameras = args.num_cameras
	data_num = args.data_num
	reduce_ratio = args.reduce_ratio
	use_blender = args.use_blender
	device_id = args.device
	debug = args.debug

	# initialize for blender
	if use_blender:
		renderer_root_dir = './assets/materials'
		obj_texture_image_root_path = './data/assets/imagenet'  
		use_background = True  		
		output_modality_dict = {'RGB': 1,
								'IR': 0,
								'NOCS': 0,
								'Mask': 0,
								'Normal': 0}

	# folder name
	if args.folder_name is None:
		folder_name = datetime.now().strftime('%Y%m%d-%H%M')
	else:
		folder_name = args.folder_name	

	# dataset path
	data_path = os.path.join(args.data_path, folder_name)
	if not os.path.exists(data_path):
		os.makedirs(data_path)

	# temp path
	temp_path = 'temp_objects'
	if not os.path.exists(temp_path):
		os.makedirs(temp_path)
	temp_image_path = 'temp_images'
	if not os.path.exists(temp_image_path):
		os.makedirs(temp_image_path)

	# additional import
	if use_blender:
		from blender.render import blender_init_scene, blender_render

	############################################################
	######################## CONTROL ENV #######################
	############################################################

	# reset environment
	env = ControlSimulationEnv(
		enable_gui=enable_gui, 
		object_types=object_types,
		num_objects=num_objects,
		sim_type=sim_type,
	)

	# get view poses
	camera_view_poses = env.get_view_poses(num_cameras=num_cameras)

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
	
	# additional parameters
	camera_image_size = camera_param_list[0]['camera_image_size']
	camera_focal = camera_param_list[0]['camera_intr'][0, 0] 

	# set rendering frames
	if debug:
		render_frame_list = [0, 1, 2]
		camera = [camera_param_list[i] for i in render_frame_list]
		camera_view_poses = [c['camera_pose'] for c in camera]
	else:
		render_frame_list = list(range(len(camera_param_list)))

	# save dataset config
	yml_path = os.path.join(data_path, 'dataset_config.yml')
	save_yaml(yml_path, OmegaConf.to_yaml(vars(args)))

	############################################################
	##################### DATA GENERATION ######################
	############################################################

	pbar = tqdm(
		total=data_num, 
		desc=f"TRansPose object {num_objects} ... ", 
		leave=False
	)	

	# data generation
	for round_idx in range(data_num):
		
		# get urdfs
		urdfs_and_poses_dict = env.reset(return_urdf_and_poses=True)

		# get images
		mask_img_list = []
		depth_img_list = []
		camera_param_list = []
		for view_pose in camera_view_poses:
			
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
			_, depth_image, mask_image = env.sim.get_camera_data(
				camera_param
			)
			mask_image[mask_image < 3] = 0
			mask_image[mask_image >= 3] = 1
			mask_image = mask_image.astype(bool)

			# append images
			mask_img_list.append(mask_image)
			depth_img_list.append(depth_image)
			camera_param_list.append(camera_param)

		# stack
		mask_img_list = np.stack(mask_img_list)
		depth_img_list = np.stack(depth_img_list)

		# get labels
		SE3s = []
		mesh_paths = []
		for obj in env.object_infos:
			
			# object poses
			SE3s.append(obj['SE3'])

			# meshes
			mesh_paths.append(obj['mesh_path'])

		SE3s = np.stack(SE3s)

		# get blender image
		if use_blender:
    			
			# blender initialize
			renderer, quaternion_list, translation_list, path_scene = blender_init_scene(
				renderer_root_dir, 
				temp_image_path,
				obj_texture_image_root_path, 
				urdfs_and_poses_dict, 
				0, 
				device_id, 
				camera_image_size,
				camera_view_poses,
				sim_type,
				use_background,
				glass_idx=1)
			
			# remove temp urdf and meshes
			env.remove_urdf_and_mesh()

			# blender rendering
			save_name_list_dict = blender_render(
				renderer, 
				quaternion_list, 
				translation_list, 
				path_scene, 
				render_frame_list, 
				output_modality_dict, 
				camera_focal, 
				is_init=True)

			# load images and cat
			rgb_img_list = []
			for save_name in save_name_list_dict['RGB']:
				rgb_path = os.path.join(path_scene, save_name)
				rgb_img = np.array(Image.open(rgb_path).convert("RGB"))
				rgb_img_list.append(rgb_img)

			rgb_img_list = np.stack(rgb_img_list)

		# save data
		data = {
			"rgb_imgs" : rgb_img_list,
			"mask_imgs" : mask_img_list,
			"depth_imgs": depth_img_list,
			"camera" : camera_param_list,
			"SE3s" : SE3s,
			"mesh_paths": mesh_paths,
			"workspace_origin" : env.workspace_center
		}

		# debug
		if debug:
			debugging_windows_transpose(data)

		# save data
		if not debug:
			filename = os.path.join(data_path, f"{round_idx:04d}.pkl")
			with open(filename, 'wb') as f: # save
				pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
				f.close()

		# pbar update
		pbar.update(1)

		# if debug
		if debug:
			break

	# close environment
	pbar.close()
	env.sim.close()

if __name__ == "__main__":

	# argparser
	parser = ArgumentParserForBlender()
	parser.add_argument('---debug', action='store_true', help="debugging dataset")
	parser.add_argument('---enable_gui', action='store_true', help='show PyBullet simulation window')
	parser.add_argument('---data_path', type=str, default='datasets', help="path to data")
	parser.add_argument('---sim_type', type=str, default='table', help="environment setting")
	parser.add_argument('---folder_name', default=None, help="dataset folder name")
	parser.add_argument('---num_cameras', type=int, default=36, help="number of cameras")
	parser.add_argument('---num_objects', type=int, default=4, help="number of objects")
	parser.add_argument('---reduce_ratio', type=float, default=2, help="reduce resolution of the mask image")
	parser.add_argument('---data_num', type=int, default=2, help="number of training scenes")
	parser.add_argument('---use_blender', action='store_true', help="make blender dataset")
	parser.add_argument('---device', default=0)
	parser.add_argument('---object_types', nargs='+', 
		default=['WineGlass', 'Bowl', 'Bottle', 'BeerBottle', 'HandlessCup', 'Mug', 'Dish'],
		help="choose object types"
	)
	args = parser.parse_args()
	
	# run
	run(args)