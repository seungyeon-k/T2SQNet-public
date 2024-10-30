import os
import numpy as np
import torch
import open3d as o3d
import pybullet as p
import math
from math import floor
from copy import deepcopy

from tablewarenet.tableware import *
from functions.fusion import TSDFVolume
from functions.utils import exp_se3

def depth_map_error(pred_depth, gt_depth, thld=0.05):
	"""
	Args:
		predicted_depth (B x W x H): 
		gt_depth (B x W x H):
		thld (float)
	"""
	return (((pred_depth - gt_depth).abs() / gt_depth) < thld).mean(dim=[1,2]).mean(dim=0)

def mask_iou(pred_mask, gt_mask):
	"""
	Args:
		predicted_depth (B x W x H): boolean tensor
		gt_depth (B x W x H): boolean tensor
	"""
	return ((pred_mask & gt_mask).sum(dim=[-1, -2]) / (pred_mask | gt_mask).sum(dim=[-1, -2])).mean(dim=0)

def sq2depth(classes, poses, params, camera_params, data_type='torch', sim_type=None):
	"""superquadrics to depth images

	Args:
		classes (list of str, len=N_obj): object classes
		poses (N_obj x 4 x 4 tensor): object poses
		params (list of tensor, len=N_obj): object param list
		camera_params (list of camera param dictionary, len=N_camera)

	Returns:
		N_camera x H x W tensor
	"""
	
	# load camera data
	N_obj = poses.shape[0]
	N_view = len(camera_params)
	camera_intr = camera_params[0]['camera_intr']
	camera_z_near = camera_params[0]['camera_z_near']
	camera_z_far = camera_params[0]['camera_z_far']
	h, w = camera_params[0]['camera_image_size']
	
	# initialize
	vis = o3d.visualization.Visualizer()
	vis.create_window(visible=False, width=w, height=h)
	ctr = vis.get_view_control()
	ctr.set_constant_z_near(camera_z_near)
	ctr.set_constant_z_far(camera_z_far)
	intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, camera_intr[0, 0], camera_intr[1, 1], camera_intr[0, 2], camera_intr[1, 2])
	
	# add objects
	for i in range(N_obj):
		obj = name_to_class[classes[i]](
			SE3=poses[i],
			params=params[i],
			device='cpu'
		)
		vis.add_geometry(obj.get_mesh())

	# add environment
	if sim_type == 'table':

		# info
		high_table_path = os.path.join(
			'assets', 'table', 'mesh', 'high_table.obj'
		)
		high_table_position = [0.61, -0.0375, 0.243-0.05/2]
		high_table_orientation = [1.0, 0.0, 0.0, 0.0]

		# add table
		mesh = o3d.io.read_triangle_mesh(high_table_path, True)
		T = np.eye(4)
		T[:3, :3] = mesh.get_rotation_matrix_from_quaternion(high_table_orientation)
		T[:3, 3] = high_table_position
		mesh.transform(T)
		vis.add_geometry(mesh)

	elif sim_type == 'shelf':
		
		# info
		shelf_path = os.path.join(
			'assets', 'shelf', 'mesh', 'shelf.obj'
		)
		shelf_position = [0.61, 0., 0.003]
		shelf_orientation = [0.0, 0.0, 0.0, 1.0]

		# add table
		mesh = o3d.io.read_triangle_mesh(shelf_path, True)
		T = np.eye(4)
		T[:3, :3] = mesh.get_rotation_matrix_from_quaternion(shelf_orientation)
		T[:3, 3] = shelf_position
		mesh.transform(T)
		vis.add_geometry(mesh)

	# get depth maps
	depth_list = []
	for i in range(N_view):
		
		# camera parameter
		camera_pose = camera_params[i]['camera_pose']
		new_cam_param = o3d.camera.PinholeCameraParameters()
		new_cam_param.intrinsic = intrinsic
		new_cam_param.extrinsic = np.linalg.inv(camera_pose)
		ctr.convert_from_pinhole_camera_parameters(
			new_cam_param, allow_arbitrary=True)
		
		# depth image render
		vis.poll_events()
		vis.update_renderer()
		depth_rn = np.asarray(vis.capture_depth_float_buffer(True))
		depth_rn = depth_rn.astype(np.float32)
		
		# append
		depth_list.append(depth_rn)

	# depth image process
	depth_list = np.stack(depth_list)
	depth_list[np.where(depth_list == 0)] = camera_z_far

	# data type
	if data_type == 'torch':
		depth_list = torch.from_numpy(depth_list)
	return depth_list

def sq2mask(classes, poses, params, camera_params, data_type='torch'):
	"""superquadrics to mask images

	Args:
		classes (list of str, len=N_obj): object classes
		poses (N_obj x 4 x 4 tensor): object poses
		params (list of tensor, len=N_obj): object param list
		camera_params (list of camera param dictionary, len=N_camera)

	Returns:
		N_camera x H x W boolean tensor
	"""

	# load camera data
	camera_z_near = camera_params[0]['camera_z_near']
	camera_z_far = camera_params[0]['camera_z_far']

	# make mask
	depth_list = sq2depth(classes, poses, params, camera_params, sim_type=None)
	mask = (depth_list > camera_z_near) & (depth_list < camera_z_far)
	
	# data type
	if data_type == 'numpy':
		mask = mask.numpy()
	return mask

def depth2mask(depth_imgs, near=0, far=1.5, data_type='torch'):

	# make mask
	depth_list = depth_imgs
	mask = (depth_list > near) & (depth_list < far)
	
	# data type
	if data_type == 'numpy':
		mask = mask.numpy()
	return mask

def sq2tsdf(classes, poses, params, camera_params, vol_bnds, voxel_size, depth_imgs=None, sim_type=None, trunc_ratio=5):
	"""superquadrics to depth images

	Args:
		classes (list of str, len=N_obj): object classes
		poses (N_obj x 4 x 4 tensor): object poses
		params (list of tensor, len=N_obj): object param list
		camera_params (list of camera param dictionary, len=N_camera)

	Returns:
		N_camera x H x W tensor
	"""

	# get camera parameters
	N_view = len(camera_params)
	camera_intr = camera_params[0]['camera_intr']
	camera_z_near = camera_params[0]['camera_z_near']
	camera_z_far = camera_params[0]['camera_z_far']
	h, w = camera_params[0]['camera_image_size']    
	camera_poses = []
	for i in range(N_view):
		camera_pose = camera_params[i]['camera_pose']    
		camera_poses.append(camera_pose)

	# tsdf initialize
	tsdf = TSDFVolume(vol_bnds, voxel_size=voxel_size, use_gpu=True, trunc_ratio=trunc_ratio)

	# depth image
	if depth_imgs is None:
		depth_imgs = sq2depth(
			classes, poses, params, camera_params, data_type='numpy', sim_type=sim_type)

	for i in range(N_view):

		# load data
		depth_img = depth_imgs[i]
		color_img = np.ones((depth_img.shape[0], depth_img.shape[1], 3)) * 0.7
		camera_pose = camera_poses[i]

		# integration
		tsdf.integrate(color_img, depth_img, camera_intr, camera_pose, obs_weight=1.)

	# get volume
	volume = np.asarray(tsdf.get_volume()[0])
	# volume = np.transpose(volume, [1, 0, 2])

	return volume

def depth2tsdf(depth_imgs, camera_params, vol_bnds, voxel_size):
	"""superquadrics to depth images

	Args:
		classes (list of str, len=N_obj): object classes
		poses (N_obj x 4 x 4 tensor): object poses
		params (list of tensor, len=N_obj): object param list
		camera_params (list of camera param dictionary, len=N_camera)

	Returns:
		N_camera x H x W tensor
	"""

	# get camera parameters
	N_view = len(camera_params)
	camera_intr = camera_params[0]['camera_intr']
	camera_z_near = camera_params[0]['camera_z_near']
	camera_z_far = camera_params[0]['camera_z_far']
	h, w = camera_params[0]['camera_image_size']    
	camera_poses = []
	for i in range(N_view):
		camera_pose = camera_params[i]['camera_pose']    
		camera_poses.append(camera_pose)

	# tsdf initialize
	tsdf = TSDFVolume(vol_bnds, voxel_size=voxel_size, use_gpu=True)

	# fusion
	for i in range(N_view):

		# load data
		depth_img = depth_imgs[i]
		color_img = np.ones((depth_img.shape[0], depth_img.shape[1], 3)) * 0.7
		camera_pose = camera_poses[i]

		# integration
		tsdf.integrate(color_img, depth_img, camera_intr, camera_pose, obs_weight=1.)

	# get volume
	volume = np.asarray(tsdf.get_volume()[0])
	# volume = np.transpose(volume, [1, 0, 2])

	return volume

def occ2depth(occ_map, camera_params, vol_bnds, voxel_size, data_type='torch', sim_type=None):
	###
	occ_map = deepcopy(occ_map)
	occ_map[0:20, 0:10, :] = 0
	occ_map[0:20, 70:80, :] = 0
	###
	X = torch.linspace(vol_bnds[0, 0], vol_bnds[0, 1], occ_map.shape[0])
	Y = torch.linspace(vol_bnds[1, 0], vol_bnds[1, 1], occ_map.shape[1])
	Z = torch.linspace(vol_bnds[2, 0], vol_bnds[2, 1], occ_map.shape[2])
	XYZ = torch.stack(torch.meshgrid(X, Y, Z), dim=-1)
	pred_pc = XYZ[occ_map].reshape(-1, 3).cpu().detach().numpy()
	pred_pc_o3d = o3d.geometry.PointCloud()
	pred_pc_o3d.points = o3d.utility.Vector3dVector(pred_pc)
	voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pred_pc_o3d, voxel_size=voxel_size)
 
	# load camera data
	N_view = len(camera_params)
	camera_intr = camera_params[0]['camera_intr']
	camera_z_near = camera_params[0]['camera_z_near']
	camera_z_far = camera_params[0]['camera_z_far']
	h, w = camera_params[0]['camera_image_size']

	# initialize
	vis = o3d.visualization.Visualizer()
	vis.create_window(visible=False, width=w, height=h)
	ctr = vis.get_view_control()
	ctr.set_constant_z_near(camera_z_near)
	ctr.set_constant_z_far(camera_z_far)
	intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, camera_intr[0, 0], camera_intr[1, 1], camera_intr[0, 2], camera_intr[1, 2])

	# add objects
	vis.add_geometry(voxel_grid)

	# add environment
	if sim_type == 'table':
		# info
		high_table_path = os.path.join(
			'assets', 'table', 'mesh', 'high_table.obj'
		)
		high_table_position = [0.61, -0.0375, 0.243-0.05/2]
		high_table_orientation = [1.0, 0.0, 0.0, 0.0]
		# add table
		mesh = o3d.io.read_triangle_mesh(high_table_path, True)
		T = np.eye(4)
		T[:3, :3] = mesh.get_rotation_matrix_from_quaternion(high_table_orientation)
		T[:3, 3] = high_table_position
		mesh.transform(T)
		vis.add_geometry(mesh)

	elif sim_type == 'shelf':
		# info
		shelf_path = os.path.join(
			'assets', 'shelf', 'mesh', 'shelf.obj'
		)
		shelf_position = [0.61, 0., 0.003]
		shelf_orientation = [0.0, 0.0, 0.0, 1.0]
		# add table
		mesh = o3d.io.read_triangle_mesh(shelf_path, True)
		T = np.eye(4)
		T[:3, :3] = mesh.get_rotation_matrix_from_quaternion(shelf_orientation)
		T[:3, 3] = shelf_position
		mesh.transform(T)
		vis.add_geometry(mesh)
  
	# get depth maps
	depth_list = []
	for i in range(N_view):
		# camera parameter
		camera_pose = camera_params[i]['camera_pose']
		new_cam_param = o3d.camera.PinholeCameraParameters()
		new_cam_param.intrinsic = intrinsic
		new_cam_param.extrinsic = np.linalg.inv(camera_pose)
		ctr.convert_from_pinhole_camera_parameters(
			new_cam_param, allow_arbitrary=True)
		# depth image render
		vis.poll_events()
		vis.update_renderer()
		depth_rn = np.asarray(vis.capture_depth_float_buffer(True))
		depth_rn = depth_rn.astype(np.float32)
		# append
		depth_list.append(depth_rn)

	# depth image process
	depth_list = np.stack(depth_list)
	depth_list[np.where(depth_list == 0)] = camera_z_far

	# data type
	if data_type == 'torch':
		depth_list = torch.from_numpy(depth_list)
	return depth_list
    

def sq2occ(classes, poses, params, vol_bnds, voxel_size):
	"""superquadrics to depth images

	Args:
		classes (list of str, len=N_obj): object classes
		poses (N_obj x 4 x 4 tensor): object poses
		params (list of tensor, len=N_obj): object param list
		camera_params (list of camera param dictionary, len=N_camera)

	"""

	# load camera data
	N_obj = len(poses)
	max_bbox_size = vol_bnds[:, 1] - vol_bnds[:, 0]
	
	# add objects
	for i in range(N_obj):
		obj = name_to_class[classes[i]](
			SE3=poses[i],
			params=params[i],
			device='cpu'
		)
		if i == 0:
			mesh = obj.get_mesh()
		else:
			mesh += obj.get_mesh()

	# voxelization
	voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
		mesh, voxel_size=voxel_size, 
		min_bound=vol_bnds[:, 0], max_bound=vol_bnds[:, 1])
	
	# process voxelization results
	voxels = voxel_grid.get_voxels()
	list_indices = list(vx.grid_index for vx in voxels)
	indices = np.stack(list_indices)
	w = floor(max_bbox_size[0] / voxel_size + 0.5)
	h = floor(max_bbox_size[1] / voxel_size + 0.5)
	d = floor(max_bbox_size[2] / voxel_size + 0.5)
	occupancy = np.zeros((w, h, d))
	occupancy[
		indices[:, 0], 
		indices[:, 1], 
		indices[:, 2]
	] = 1
	occupancy = (occupancy != 0)

	return occupancy

def transpose2detph(SE3s, mesh_paths, camera_params, data_type='torch', sim_type=None):
	"""superquadrics to depth images

	Args:
		classes (list of str, len=N_obj): object classes
		poses (N_obj x 4 x 4 tensor): object poses
		params (list of tensor, len=N_obj): object param list
		camera_params (list of camera param dictionary, len=N_camera)

	Returns:
		N_camera x H x W tensor
	"""

	# load camera data
	N_obj = len(mesh_paths)
	N_view = len(camera_params)
	camera_intr = camera_params[0]['camera_intr']
	camera_z_near = camera_params[0]['camera_z_near']
	camera_z_far = camera_params[0]['camera_z_far']
	h, w = camera_params[0]['camera_image_size']
 	
	# initialize
	vis = o3d.visualization.Visualizer()
	vis.create_window(visible=False, width=w, height=h)
	ctr = vis.get_view_control()
	ctr.set_constant_z_near(camera_z_near)
	ctr.set_constant_z_far(camera_z_far)
	intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, camera_intr[0, 0], camera_intr[1, 1], camera_intr[0, 2], camera_intr[1, 2])
 
	# add objects
	for i in range(N_obj):
		obj_path = mesh_paths[i]
		obj_mesh = o3d.io.read_triangle_mesh(obj_path)
		R = obj_mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
		obj_mesh.rotate(R, center=(0, 0, 0))
		obj_mesh.transform(SE3s[i])
		vis.add_geometry(obj_mesh)

	# add environment
	if sim_type == 'table':

		# info
		high_table_path = os.path.join(
			'assets', 'table', 'mesh', 'high_table.obj'
		)
		high_table_position = [0.61, -0.0375, 0.243-0.05/2]
		high_table_orientation = [1.0, 0.0, 0.0, 0.0]

		# add table
		mesh = o3d.io.read_triangle_mesh(high_table_path, True)
		T = np.eye(4)
		T[:3, :3] = mesh.get_rotation_matrix_from_quaternion(high_table_orientation)
		T[:3, 3] = high_table_position
		mesh.transform(T)
		vis.add_geometry(mesh)

	elif sim_type == 'shelf':
		
		# info
		shelf_path = os.path.join(
			'assets', 'shelf', 'mesh', 'shelf.obj'
		)
		shelf_position = [0.61, 0., 0.003]
		shelf_orientation = [0.0, 0.0, 0.0, 1.0]

		# add table
		mesh = o3d.io.read_triangle_mesh(shelf_path, True)
		T = np.eye(4)
		T[:3, :3] = mesh.get_rotation_matrix_from_quaternion(shelf_orientation)
		T[:3, 3] = shelf_position
		mesh.transform(T)
		vis.add_geometry(mesh)

	# get depth maps
	depth_list = []
	for i in range(N_view):
		
		# camera parameter
		camera_pose = camera_params[i]['camera_pose']
		new_cam_param = o3d.camera.PinholeCameraParameters()
		new_cam_param.intrinsic = intrinsic
		new_cam_param.extrinsic = np.linalg.inv(camera_pose)
		ctr.convert_from_pinhole_camera_parameters(
			new_cam_param, allow_arbitrary=True)
		
		# depth image render
		vis.poll_events()
		vis.update_renderer()
		depth_rn = np.asarray(vis.capture_depth_float_buffer(True))
		depth_rn = depth_rn.astype(np.float32)
		
		# append
		depth_list.append(depth_rn)

	# depth image process
	depth_list = np.stack(depth_list)
	depth_list[np.where(depth_list == 0)] = camera_z_far

	# data type
	if data_type == 'torch':
		depth_list = torch.from_numpy(depth_list)
	return depth_list

def transpose2tsdf(SE3s, mesh_paths, camera_params, vol_bnds, voxel_size):
	"""superquadrics to depth images

	Args:
		classes (list of str, len=N_obj): object classes
		poses (N_obj x 4 x 4 tensor): object poses
		params (list of tensor, len=N_obj): object param list
		camera_params (list of camera param dictionary, len=N_camera)

	Returns:
		N_camera x H x W tensor
	"""

	# get camera parameters
	N_view = len(camera_params)
	camera_intr = camera_params[0]['camera_intr']
	camera_poses = []
	for i in range(N_view):
		camera_pose = camera_params[i]['camera_pose']    
		camera_poses.append(camera_pose)

	# tsdf initialize
	tsdf = TSDFVolume(vol_bnds, voxel_size=voxel_size, use_gpu=True)

	# depth image
	depth_imgs = transpose2detph(
		SE3s, mesh_paths, camera_params, data_type='torch', sim_type=None)

	for i in range(N_view):

		# load data
		depth_img = depth_imgs[i]
		color_img = np.ones((depth_img.shape[0], depth_img.shape[1], 3)) * 0.7
		camera_pose = camera_poses[i]

		# integration
		tsdf.integrate(color_img, depth_img, camera_intr, camera_pose, obs_weight=1.)

	# get volume
	volume = np.asarray(tsdf.get_volume()[0])
	# volume = np.transpose(volume, [1, 0, 2])

	return volume

def transpose2occ(SE3s, mesh_paths, vol_bnds, voxel_size):
	"""superquadrics to depth images

	Args:
		classes (list of str, len=N_obj): object classes
		poses (N_obj x 4 x 4 tensor): object poses
		params (list of tensor, len=N_obj): object param list
		camera_params (list of camera param dictionary, len=N_camera)

	Returns:
		N_camera x H x W tensor
	"""

	# load camera data
	N_obj = len(SE3s)
	max_bbox_size = vol_bnds[:, 1] - vol_bnds[:, 0]
	
	# add objects
	for i in range(N_obj):

		obj_path = mesh_paths[i]
		obj_mesh = o3d.io.read_triangle_mesh(obj_path)
		R = obj_mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
		obj_mesh.rotate(R, center=(0, 0, 0))
		obj_mesh.transform(SE3s[i])

		if i == 0:
			mesh = obj_mesh
		else:
			mesh += obj_mesh

	# voxelization
	voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
		mesh, voxel_size=voxel_size, 
		min_bound=vol_bnds[:, 0], max_bound=vol_bnds[:, 1])
	
	# process voxelization results
	voxels = voxel_grid.get_voxels()
	list_indices = list(vx.grid_index for vx in voxels)
	indices = np.stack(list_indices)
	w = round(max_bbox_size[0] / voxel_size)
	h = round(max_bbox_size[1] / voxel_size)
	d = round(max_bbox_size[2] / voxel_size)
	occupancy = np.zeros((w, h, d))
	occupancy[
		indices[:, 0], 
		indices[:, 1], 
		indices[:, 2]
	] = 1
	occupancy = (occupancy != 0)

	return occupancy


############################################################
####################### for eval ###########################
############################################################

def get_view_poses_for_tsdf(num_phi=6, num_theta=10, radius=0.8, workspace_center=np.array([0., 0., 0.])):
	# camera pose initialize
	view_poses = []
	rotating_phis = np.linspace(
		start = 20/180*np.pi, stop = 80/180*np.pi, num=num_phi)
	rotating_angles = np.linspace(
		start = -np.pi, stop = np.pi, num=num_theta+1)[:-1]
	# shelf params
	z_axis = np.array([0, 0, 1])
	v = - np.cross(z_axis, workspace_center)
	screw = np.concatenate((z_axis, v))
	# camera poses
	for phi in rotating_phis:
		# reference ee pose
		view_pose_init = np.eye(4)
		view_pose_init[:3, :3] = np.array([
			[0, -np.sin(phi), np.cos(phi)], 
			[-1, 0			 , 0],
			[0, -np.cos(phi), -np.sin(phi)]]
		)
		view_pose_init[:3, 3] = np.array(
			[-radius*np.cos(phi), 0, radius*np.sin(phi)])
		view_pose_init[:3, 3] += workspace_center
		# rotating theta
		for theta in rotating_angles:
			rotating_SE3 = exp_se3(theta * screw)
			view_pose = rotating_SE3.dot(view_pose_init)
			view_poses.append(view_pose)
	return view_poses

def get_camera_param(
		camera_pose,
		camera_intrinsic,
		camera_image_size
	):

	# modified camera intrinsic
	fx = (camera_intrinsic[0] + camera_intrinsic[1]) / 2
	fy = (camera_intrinsic[0] + camera_intrinsic[1]) / 2
	px = float(camera_image_size[1]) / 2
	py = float(camera_image_size[0]) / 2

	# camera view matrix
	camera_view_matrix = deepcopy(camera_pose)
	camera_view_matrix[:, 1:3] = -camera_view_matrix[:, 1:3]
	camera_view_matrix = np.linalg.inv(camera_view_matrix).T.reshape(-1)
	
	# camera z near/far values (arbitrary value)
	camera_z_near = 0.01
	camera_z_far = 20

	# camera intrinsic matrix
	camera_intrinsic_matrix = np.array(
		[[fx, 0, px],
			[0, fy, py],
			[0, 0, 1]]
	)

	# camera projection matrix
	camera_fov_h = (math.atan(py / fy) * 2 / np.pi) * 180
	camera_projection_matrix = p.computeProjectionMatrixFOV(
		fov=camera_fov_h,
		aspect=float(camera_image_size[1]) / float(camera_image_size[0]),
		nearVal=camera_z_near,
		farVal=camera_z_far
	)  

	camera_param = {
		'camera_image_size': camera_image_size,
		'camera_intr': camera_intrinsic_matrix,
		'camera_pose': camera_pose,
		'camera_view_matrix': camera_view_matrix,
		'camera_projection_matrix': camera_projection_matrix,
		'camera_z_near': camera_z_near,
		'camera_z_far': camera_z_far
	}
	return camera_param

def camera_params_for_tsdf(realsense_intrinsic, camera_image_size, workspace_center=np.array([0., 0., 0.])):
	view_poses_for_tsdf =  get_view_poses_for_tsdf(workspace_center=workspace_center)
	camera_for_tsdf = []
	for view_pose in view_poses_for_tsdf:
		# set camera parameters
		camera_param = get_camera_param(
			camera_pose=view_pose,
			camera_intrinsic=realsense_intrinsic,
			camera_image_size=[int(camera_image_size[0]), int(camera_image_size[1])]
		)
		camera_for_tsdf.append(camera_param)
	return(camera_for_tsdf)

def sq2occ_for_eval(
	classes, poses, params,
    vol_bnds,
	voxel_size,
	realsense_intrinsic = np.array([606.1148681640625, 605.2857055664062, 325.19329833984375, 246.53085327148438]),
	camera_image_size = [480, 640],
	workspace_center = np.array([0.61, -0.0375, 0.243])
 	):
	camera_params = camera_params_for_tsdf(realsense_intrinsic, camera_image_size, workspace_center)
	tsdf = sq2tsdf(classes, poses, params, camera_params, vol_bnds, voxel_size)
	return tsdf < 0

def transpose2tsdf_for_eval(
	SE3s, mesh_paths,
    vol_bnds,
	voxel_size,
	realsense_intrinsic = np.array([606.1148681640625, 605.2857055664062, 325.19329833984375, 246.53085327148438]),
	camera_image_size = [480, 640],
	workspace_center = np.array([0.61, -0.0375, 0.243])
 	):
	camera_params = camera_params_for_tsdf(realsense_intrinsic, camera_image_size, workspace_center)
	tsdf = transpose2tsdf(SE3s, mesh_paths, camera_params, vol_bnds, voxel_size)
	return tsdf

def transpose2occ_for_eval(
	SE3s, mesh_paths,
    vol_bnds,
	voxel_size,
	realsense_intrinsic = np.array([606.1148681640625, 605.2857055664062, 325.19329833984375, 246.53085327148438]),
	camera_image_size = [480, 640],
	workspace_center = np.array([0.61, -0.0375, 0.243])
 	):
	tsdf = transpose2tsdf_for_eval(SE3s, mesh_paths, vol_bnds, voxel_size, realsense_intrinsic, camera_image_size, workspace_center)
	return tsdf < 0
