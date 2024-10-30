import numpy as np
import torch
import open3d as o3d

from functions.utils import quats_to_matrices, get_SE3s
from control.gripper import Gripper

############################################################
######################## RENDERING #########################
############################################################

def render_pc(pc, furniture=None, coordinate=False):
	
	# list
	mesh_list = []
 
	# add pc
	mesh_list += [add_pc(pc)]

	# add coordinate
	if coordinate:
		coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
		mesh_list += [coord]

	# add furniture
	if furniture is not None:
		mesh_list += add_furniture(furniture)

	o3d.visualization.draw_geometries(mesh_list)

def render_objects(objs, furniture=None):

	# list
	mesh_list = []
 
	# add geometries and lighting
	mesh_list += add_objects(objs)

	# add furniture
	if furniture is not None:
		mesh_list += add_furniture(furniture)

	o3d.visualization.draw_geometries(mesh_list)

def render_bboxes_with_objects(objs, bboxes, furniture=None):
	
	# list
	mesh_list = []
 
	# add geometries and lighting
	mesh_list += add_objects(objs)

	# add bboxes
	mesh_list += add_bboxes(bboxes)

	# add furniture
	if furniture is not None:
		mesh_list += add_furniture(furniture)

	o3d.visualization.draw_geometries(mesh_list)

def render_pc_with_objects(objs, pc, furniture=None):
	
	# list
	mesh_list = []
 
	# add geometries and lighting
	mesh_list += add_objects(objs)

	# add pc
	mesh_list += [add_pc(pc)]

	# add furniture
	if furniture is not None:
		mesh_list += add_furniture(furniture)

	o3d.visualization.draw_geometries(mesh_list)

def render_grasp_poses(
		objs, grasp_poses_list, 
		collision_scores_list, furniture=None):
	
	# list
	mesh_list = []
 
	# add geometries and lighting
	mesh_list += add_objects(objs)

	# add grasp poses
	for grasp_poses, collision_scores in zip(grasp_poses_list, collision_scores_list):
		
		# print(f'grasp_poses: {grasp_poses}')
		# print(f'collision_scores: {collision_scores}')

		# empty grasp pose
		if len(grasp_poses) == 0:
			continue
		
		# add grasp poses
		for i, grasp_pose in enumerate(grasp_poses):
			gripper = add_gripper(grasp_pose)
			if collision_scores is not None:
				if collision_scores[i] >= 0:
					gripper.paint_uniform_color([0, 1, 0])
				elif collision_scores[i] < 0:
					gripper.paint_uniform_color([1, 0, 0])
			mesh_list += [gripper]

	# add furniture
	if furniture is not None:
		mesh_list += add_furniture(furniture)

	o3d.visualization.draw_geometries(mesh_list)

############################################################
###################### ADD FUNCTIONS #######################
############################################################

def add_pc(pc, colors=None):
		
	# point cloud dtype
	if torch.is_tensor(pc):
		pc = pc.cpu().float().numpy()

	# point cloud mesh
	pc_o3d = o3d.geometry.PointCloud()
	pc_o3d.points = o3d.utility.Vector3dVector(pc)

	return pc_o3d

def add_bboxes(bboxes, colors=None):
	
	# list
	mesh_list = []
	
	# add bboxes
	for bbox in bboxes:

		# bbox dtype
		if torch.is_tensor(bbox):
			bbox = bbox.detach().cpu().float().numpy()

		# get bbox
		bbox_min = bbox[:3] - bbox[3:]
		bbox_max = bbox[:3] + bbox[3:]
		obj_bbox = o3d.geometry.AxisAlignedBoundingBox()
		obj_bbox.min_bound = bbox_min
		obj_bbox.max_bound = bbox_max
		
		# append
		mesh_list.append(obj_bbox)

	return mesh_list

def add_objects(objs, colors=None):
	
	# list
	mesh_list = []	

	# add geometries and lighting
	for obj in objs:
		
		# get mesh
		mesh = obj.get_mesh()
		mesh.compute_vertex_normals()
		mesh.paint_uniform_color(1 * np.random.rand(3))

		# mesh 
		mesh_list.append(mesh)

	return mesh_list

def add_gripper(grasp_pose, colors=None):
	
	# gripper mesh
	gripper = Gripper(grasp_pose.numpy(), 0.08).mesh

	return gripper

def add_furniture(furniture, exclude_list=[], real_shelf=False):
		
	# list
	mesh_list = []

	# get table parts
	part_keys = [key for key in furniture.keys() 
		if (key.startswith('table') or key.startswith('shelf'))
	]

	# get position and orientation
	global_position = furniture['global_position']
	global_orientation = furniture['global_orientation']
	global_SO3 = quats_to_matrices(global_orientation)
	global_SE3 = get_SE3s(global_SO3, global_position)

	for part_key in part_keys:
			
		# exclude
		if part_key.split('_')[-1] in exclude_list:
			continue

		# load info	
		size = furniture[part_key]['size']
		position = furniture[part_key]['position']

		# for figure drawing
		if real_shelf and part_key == 'shelf_left':
			size[1] = 0.018
			position[1] = 0.3745 + 0.018
		if real_shelf and part_key == 'shelf_right':
			size[1] = 0.018
			position[1] = -0.3745 - 0.018
		if real_shelf and part_key == 'shelf_upper':
			size[2] = 0.022
			position[2] = 0.759 + 0.022

		# get open3d mesh
		mesh = o3d.geometry.TriangleMesh.create_box(
			width = size[0], 
			height = size[1], 
			depth = size[2]
		)
		mesh.translate([-size[0]/2, -size[1]/2, -size[2]/2])
		mesh.compute_vertex_normals()

		# transform
		mesh.translate(position)
		mesh.transform(global_SE3)
		if part_key.startswith('table'):
			mesh.paint_uniform_color([222/255, 184/255, 135/255])
		elif part_key.startswith('shelf'):
			mesh.paint_uniform_color([101/255/2, 67/255/2, 33/255/2])

		# append
		mesh_list.append(mesh)	

	return mesh_list