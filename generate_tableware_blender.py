import os
import torch
import numpy as np
import pickle
import random
from omegaconf import OmegaConf
from tqdm import tqdm
from signal import signal, SIGINT
from PIL import Image
import open3d as o3d
from functools import partial
from copy import deepcopy
import sys
sys.path.append(os.getcwd())

from control.control_env import ControlSimulationEnv
from functions.utils import matrices_to_quats
from tablewarenet.tableware import name_to_class
from tablewarenet.utils import sq2depth, sq2mask, sq2tsdf
from utils.yaml_utils import save_yaml
from utils.argparse_blender import ArgumentParserForBlender
from utils.debugging_windows import debugging_windows_blender

def get_urdfs_and_poses_dict(
		objects_class, 
		objects_pose, 
		objects_param, 
		process_idx=None,
		temp_path='temp_objects'):
	
	# initialize
	mesh_path_list = []
	urdf_path_list = []	
	urdfs_and_poses_dict = dict()
	n_objects = len(objects_class)

	# make urdfs and poses
	for i in range(n_objects):
		
		cls = objects_class[i]
		pose = objects_pose[i]
		param = objects_param[i]

		# create object
		tableware = name_to_class[cls](
			pose, params=param, device='cpu'
		)

		# pose
		position = tableware.SE3[0:3, 3].detach().cpu().numpy()
		orientation = matrices_to_quats(tableware.SE3[0:3, 0:3].detach().cpu().numpy())

		# mesh
		mesh = tableware.get_mesh(transform=False)

		# make temporary object file
		randnum = np.random.randint(low=0, high=1e9)		
		if process_idx is not None:
			mesh_name = f'temp_object_{process_idx}_{randnum}.obj'
			urdf_name = f'temp_object_{process_idx}_{randnum}.urdf'
		else:
			mesh_name = f'temp_object_{randnum}.obj'
			urdf_name = f'temp_object_{randnum}.urdf'
		mesh_path = os.path.join(temp_path, mesh_name)
		urdf_path = os.path.join(temp_path, urdf_name)		

		# save mesh			
		o3d.io.write_triangle_mesh(mesh_path, mesh)

		# urdf content
		urdf_contents = f"""
		<robot name="object_{randnum}">
			<link name="link_object_{randnum}">
				<visual>
					<geometry>
						<mesh filename="{mesh_name}" />
					</geometry>
				</visual>

				<collision>
					<geometry>
						<mesh filename="{mesh_name}" />
					</geometry>
				</collision>
				
				<inertial>
					<origin rpy="0 0 0" xyz="0 0 0"/>
					<mass value="0.3" />
					<inertia ixx="0.3" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.3" />
				</inertial>
			</link>
		</robot>
		"""		
		file = open(urdf_path, "w")
		file.write(urdf_contents)
		file.close()
		# time.sleep(0.2)

		# path append
		mesh_path_list.append(mesh_path)
		urdf_path_list.append(urdf_path)
		
		# get position and quaternion
		urdfs_and_poses_dict[f'{i}'] = []
		urdfs_and_poses_dict[f'{i}'].append(1.0)
		urdfs_and_poses_dict[f'{i}'].append(orientation)
		urdfs_and_poses_dict[f'{i}'].append(position)
		urdfs_and_poses_dict[f'{i}'].append(urdf_path)

	return urdfs_and_poses_dict, mesh_path_list, urdf_path_list

def remove_urdf_and_mesh(mesh_path_list, urdf_path_list):
	for mesh_path, urdf_path in zip(mesh_path_list, urdf_path_list):
		os.remove(mesh_path)
		os.remove(urdf_path)

def process_file(
		file, 
		original_split_data_path,
		new_data_path,
		temp_path,
		temp_image_path,
		split,
		device_id,
		debug,
		sim_type=None,
		use_tsdf=True,
		use_blender=False,
		camera_for_tsdf=None
		):

	# read file
	file_idx = file.split('.')[0]
	filename = os.path.join(original_split_data_path, file)
	with open(filename, 'rb') as f:
		data = pickle.load(f)
		f.close()	

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
		
		# material split
		background_split = [0, 15, 30, 43]
		wood_split = [0, 3, 6, 10]

		if split == 'training':
			background_idx = random.choice(
				list(range(background_split[0], background_split[1])))
			wood_idx = random.choice(
				list(range(wood_split[0], wood_split[1])))
		elif split == 'validation':
			background_idx = random.choice(
				list(range(background_split[1], background_split[2])))
			wood_idx = random.choice(
				list(range(wood_split[1], wood_split[2])))
		elif split == 'test':
			background_idx = random.choice(
				list(range(background_split[2], background_split[3])))
			wood_idx = random.choice(
				list(range(wood_split[2], wood_split[3])))
		else:
			raise ValueError('check split!')

	# values
	if use_tsdf:
		workspace_origin = data["workspace_origin"]
		vol_bnds = np.array([
			[workspace_origin[0] - 0.3, workspace_origin[0] + 0.3],
			[workspace_origin[1] - 0.4, workspace_origin[1] + 0.4],
			[workspace_origin[2], workspace_origin[2] + 0.3]
		])
		voxel_size = 0.005
		if camera_for_tsdf is None:
			camera_for_tsdf = data["camera"]

	# get current process index
	process_idx = None

	# set camera
	camera = deepcopy(data["camera"])
	camera_view_poses = [c['camera_pose'] for c in camera]
	camera_image_size = camera[0]['camera_image_size']
	camera_focal = camera[0]['camera_intr'][0, 0]

	# set rendering frames
	if debug:
		render_frame_list = [0, 1, 2]
		camera = [camera[i] for i in render_frame_list]
		camera_view_poses = [c['camera_pose'] for c in camera]
	else:
		render_frame_list = list(range(15, 22))
		camera = [camera[i] for i in render_frame_list]
		camera_view_poses = [c['camera_pose'] for c in camera]

	# load data
	objects_pose = torch.stack(data["objects_pose"])
	objects_param = data["objects_param"]
	objects_bbox = data["objects_bbox"]
	objects_class = data["objects_class"]
	if debug:
		data_debug = dict()
		data_debug['objects_pose'] = deepcopy(objects_pose)
		data_debug["objects_param"] = deepcopy(objects_param)
		data_debug["objects_bbox"] = deepcopy(objects_bbox)
		data_debug["objects_class"] = deepcopy(objects_class)

	# get depth image and mask image
	depth_list = sq2depth(
		objects_class, objects_pose, objects_param, camera, sim_type=sim_type)	
	mask_list = sq2mask(
		objects_class, objects_pose, objects_param, camera)

	# get tsdf
	if use_tsdf:
		tsdf = sq2tsdf(
			objects_class, objects_pose, objects_param, 
			camera_for_tsdf, vol_bnds, voxel_size)

	# get blender image
	if use_blender:

		# make urdfs
		urdfs_and_poses_dict, mesh_path_list, urdf_path_list = get_urdfs_and_poses_dict(
			objects_class, objects_pose, objects_param, 
			process_idx=process_idx,
			temp_path=temp_path
		)

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
			glass_idx=1,
			background_idx=background_idx,
			table_idx=wood_idx,
			)

		# remove temp urdf and meshes
		remove_urdf_and_mesh(mesh_path_list, urdf_path_list)

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
		rgb_list = []
		for save_name in save_name_list_dict['RGB']:
			rgb_path = os.path.join(path_scene, save_name)
			rgb_img= np.array(Image.open(rgb_path).convert("RGB"))
			rgb_list.append(rgb_img)

		rgb_list = torch.from_numpy(np.stack(rgb_list))

	# new data
	new_data = deepcopy(data)
	new_data["camera"] = camera
	new_data["mask_imgs"] = mask_list
	new_data["depth_imgs"] = depth_list
	if use_blender:
		new_data["rgb_imgs"] = rgb_list
		new_data['background_idx'] = background_idx
		new_data['wood_idx'] = wood_idx
	if use_tsdf:
		new_data["tsdf"] = tsdf
		new_data["tsdf_vol_bnds"] = vol_bnds
		new_data["tsdf_voxel_size"] = voxel_size

	# debug
	if debug: 
		debugging_windows_blender(new_data)

	# save data
	if not debug:
		filename = os.path.join(
			new_data_path, split, f'{file_idx}.pkl')
		with open(filename, 'wb') as f:
			pickle.dump(new_data, f, pickle.HIGHEST_PROTOCOL)
			f.close()

if __name__ == '__main__':

	# argparser
	parser = ArgumentParserForBlender()
	parser.add_argument('---debug', action='store_true', help="debugging dataset")
	parser.add_argument('---use_blender', action='store_true', help="make blender dataset")
	parser.add_argument('---use_tsdf', action='store_true', help="make blender dataset")
	parser.add_argument('---max_data_num', nargs='+', type=int,
		default=[9999999, 9999999, 9999999],
		help="split list"
	)	
	parser.add_argument('---device', default=0)
	parser.add_argument('---folder_name', default=None, help="dataset folder name")
	parser.add_argument('---split_list', nargs='+',
		default=['training', 'validation', 'test'],
		help="split list"
	)
	args = parser.parse_args()

	################################################
	################## INITIALIZE ##################
	################################################

	# parameters
	debug = args.debug
	use_blender = args.use_blender
	use_tsdf = args.use_tsdf
	max_data_num_list = args.max_data_num
	device_id = args.device
	original_data_path = args.folder_name
	split_list = args.split_list

	# check split list
	if len(max_data_num_list) != len(split_list):
		raise ValueError('max data num does not match to split list')

	# load dataset config
	yml_path = os.path.join(original_data_path, 'dataset_config.yml')
	dataset_args = OmegaConf.load(yml_path)
	sim_type = dataset_args.sim_type
	num_cameras = dataset_args.num_cameras
	reduce_ratio = dataset_args.reduce_ratio
	reduce_ratio_for_tsdf = 1

	# new dataset path
	new_data_path = f'{args.folder_name}_processed'
	for split in split_list:
		if not os.path.exists(os.path.join(new_data_path, split)):
			os.makedirs(os.path.join(new_data_path, split))	
	new_yaml_path = os.path.join(new_data_path, 'dataset_config.yml')
	save_yaml(new_yaml_path, OmegaConf.to_yaml(dataset_args))
	
	# additional import
	if use_blender:
		from blender.render import blender_init_scene, blender_render

	# temp path
	temp_path = 'temp_objects'
	if not os.path.exists(temp_path):
		os.makedirs(temp_path)
	temp_image_path = 'temp_images'
	if not os.path.exists(temp_image_path):
		os.makedirs(temp_image_path)

	################################################
	############# NEW CAMERA PARAMETERS ############
	################################################
	
	# get env
	env = ControlSimulationEnv(
		enable_gui=False, 
		object_types=[''],
		num_objects=1,
  		sim_type=sim_type,
	)
	view_poses = env.get_view_poses(num_cameras=num_cameras)

	# get view poses for tsdf
	camera_for_tsdf = []
	view_poses_for_tsdf = env.get_view_poses_for_tsdf(
		num_phi=6, num_theta=10)
	for view_pose in view_poses_for_tsdf:
				
		# set camera parameters
		camera_param = env.sim._get_camera_param(
			camera_pose=view_pose,
			camera_intrinsic=(
				env.sim.realsense_intrinsic / reduce_ratio_for_tsdf
			),
			camera_image_size=[
				int(env.sim.camera_image_size[0]/reduce_ratio_for_tsdf), 
				int(env.sim.camera_image_size[1]/reduce_ratio_for_tsdf)
			]
		)
		camera_for_tsdf.append(camera_param)

	################################################
	############# DATASET VOXELIZATION #############
	################################################

	# split
	for split, max_data_num in zip(split_list, max_data_num_list):

		# folder name
		original_split_data_path = os.path.join(
			original_data_path, split)
		original_split_data_list = os.listdir(original_split_data_path)
		dataset_len = len(original_split_data_list)

		# pbar
		if debug:
			dataset_len = 1
			original_split_data_list = original_split_data_list[:1]
		else:
			dataset_len = min(max_data_num, dataset_len)
			original_split_data_list = original_split_data_list[:dataset_len]
		pbar = tqdm(
			total=dataset_len, 
			desc=f"{split} dataset ... ", 
			leave=False
		)			
		
		# process
		for file in original_split_data_list:
			process_file(
				file,
				original_split_data_path=original_split_data_path,
				new_data_path=new_data_path,
				temp_path=temp_path,
				temp_image_path=temp_image_path,
				split=split,
				device_id=device_id,
				debug=debug,
				sim_type=sim_type,
				use_blender=use_blender,
				use_tsdf=use_tsdf,
				camera_for_tsdf=camera_for_tsdf
			)
			pbar.update(1)

		# close
		pbar.close()

		# break if debug
		if debug:
			break