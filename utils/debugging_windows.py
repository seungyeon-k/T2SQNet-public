import random
import torch
import math
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tablewarenet.tableware import name_to_class, name_to_idx

# color template (2023 pantone top 10 colors)
colors = np.zeros((10, 3))
colors[0, :] = [208, 28, 31] # fiery red
colors[1, :] = [207, 45, 113] # beetroot purple
colors[2, :] = [249, 77, 0] # tangelo
colors[3, :] = [250, 154, 133] # peach pink
colors[4, :] = [247, 208, 0] # empire yellow
colors[5, :] = [253, 195, 198] # crystal rose
colors[6, :] = [57, 168, 69] # classic green
colors[7, :] = [193, 219, 60] # love bird 
colors[8, :] = [75, 129, 191] # blue perennial 
colors[9, :] = [161, 195, 218] # summer song
colors = colors / 255  

#########################################################
################### DEBUG FOR PYBULLET ##################
#########################################################

def debugging_windows_pybullet(data):

	object_poses = data['objects_pose'] 
	shape_classes = data['objects_class']
	shape_parameters = data['objects_param']
	# pc = data['objects_pc']
	diff_pc = data['objects_diff_pc']
	bbox = data['objects_bbox']
	workspace_origin = data['workspace_origin']

	# number of objects
	assert ( 
		len(object_poses) == len(shape_classes)
	), f"len of object_poses is {len(object_poses)} and len of shape_classes is {len(shape_classes)}."
	assert ( 
		len(object_poses) == len(shape_parameters)
	), f"len of object_poses is {len(object_poses)} and len of shape_parameters is {len(shape_parameters)}."
	# assert ( 
	# 	len(object_poses) == len(pc)
	# ), f"len of object_poses is {len(object_poses)} and len of pc is {len(pc)}."
	assert ( 
		len(object_poses) == len(bbox)
	), f"len of object_poses is {len(object_poses)} and len of bbox is {len(bbox)}."    
	num_objects = len(object_poses)
	
	# object processing
	obj_mesh_list = []
	# obj_pc_list = []
	obj_diff_pc_list = []
	obj_bbox_list = []

	# table info
	table_size = [0.5, 0.895, 0.05]
	table = o3d.geometry.TriangleMesh.create_box(
		width=table_size[0], 
		height=table_size[1], 
		depth=table_size[2]
	)
	table.translate([-table_size[0]/2, -table_size[1]/2, -table_size[2]])
	table.translate(workspace_origin)
	table.paint_uniform_color([222/255,184/255,135/255])

	for idx in range(num_objects):
		
		# color
		color = colors[name_to_idx[shape_classes[idx]]]

		# mesh
		obj = name_to_class[shape_classes[idx]](
			object_poses[idx], shape_parameters[idx], 'cpu'
		)
		obj_mesh = obj.get_mesh()
		obj_mesh.paint_uniform_color(color)
		obj_mesh_list.append(obj_mesh)        
		
		# # pc
		# obj_pc = o3d.geometry.PointCloud()
		# obj_pc.points = o3d.utility.Vector3dVector(pc[idx])
		# obj_pc.paint_uniform_color(color)
		# obj_pc_list.append(obj_pc)  	

		# diff pc
		obj_diff_pc = o3d.geometry.PointCloud()
		obj_diff_pc.points = o3d.utility.Vector3dVector(diff_pc[idx])
		obj_diff_pc.paint_uniform_color(color)
		obj_diff_pc_list.append(obj_diff_pc)  	

		# bbox
		bbox_min = bbox[idx, :3] - bbox[idx, 3:]
		bbox_max = bbox[idx, :3] + bbox[idx, 3:]
		obj_bbox = o3d.geometry.AxisAlignedBoundingBox()
		obj_bbox.min_bound = bbox_min
		obj_bbox.max_bound = bbox_max
		obj_bbox.color = color
		obj_bbox_list.append(obj_bbox)  	

	# visualizer
	vis1 = o3d.visualization.Visualizer()
	vis1.create_window(window_name="Object Mesh", width=960-2, height=520-32, left=960*0+1, top=31)
	vis2 = o3d.visualization.Visualizer()
	vis2.create_window(window_name="Object Point Cloud", width=960-2, height=520-32, left=960*1+1, top=31)
	vis3 = o3d.visualization.Visualizer()
	vis3.create_window(window_name="Object Differentiable Point Cloud", width=960-2, height=520-32, left=960*0+1, top=520+31)
	vis4 = o3d.visualization.Visualizer()
	vis4.create_window(window_name="Object Bounding Box", width=960-2, height=520-32, left=960*1+1, top=520+31)

	# coordinate frame
	frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

	vis1.add_geometry(frame)
	vis1.add_geometry(table)
	for idx in range(num_objects):
		vis1.add_geometry(obj_mesh_list[idx])
	vis2.add_geometry(frame)
	vis2.add_geometry(table)
	# for idx in range(num_objects):
	# 	vis2.add_geometry(obj_pc_list[idx])
	vis3.add_geometry(frame)
	vis3.add_geometry(table)
	for idx in range(num_objects):
		vis3.add_geometry(obj_diff_pc_list[idx])
	vis4.add_geometry(frame)
	vis4.add_geometry(table)
	for idx in range(num_objects):
		vis4.add_geometry(obj_bbox_list[idx])

	while True:
		vis1.update_geometry(frame)
		vis1.update_geometry(table)
		for idx in range(num_objects):
			vis1.update_geometry(obj_mesh_list[idx])
		if not vis1.poll_events():
			break
		vis1.update_renderer()

		vis2.update_geometry(frame)
		vis2.update_geometry(table)
		# for idx in range(num_objects):
		# 	vis2.update_geometry(obj_pc_list[idx])
		if not vis2.poll_events():
			break
		vis2.update_renderer()

		vis3.update_geometry(frame)
		vis3.update_geometry(table)
		for idx in range(num_objects):
			vis3.update_geometry(obj_diff_pc_list[idx])
		if not vis3.poll_events():
			break
		vis3.update_renderer()

		vis4.update_geometry(frame)
		vis4.update_geometry(table)
		for idx in range(num_objects):
			vis4.update_geometry(obj_bbox_list[idx])
		if not vis4.poll_events():
			break
		vis4.update_renderer()

	vis1.destroy_window()
	vis2.destroy_window()
	vis3.destroy_window()
	vis4.destroy_window()

#########################################################
################### DEBUG FOR BLENDER ###################
#########################################################

def debugging_windows_blender(data):

	# load data
	objects_pose = data['objects_pose'] 
	objects_class = data['objects_class']
	objects_param = data['objects_param']
	mask_imgs = data['mask_imgs']
	depth_imgs = data['depth_imgs']
	if 'rgb_imgs' in data:
		rgb_imgs = data['rgb_imgs']
	if 'tsdf' in data:
		tsdf = data['tsdf']
		tsdf = np.transpose(tsdf, [1, 0, 2])
		voxel_size = data['tsdf_voxel_size']
		vol_bnds = data['tsdf_vol_bnds']
	num_objects = len(objects_pose)

	# depth image scaling
	closest_depth = 0.5
	farthest_depth = 1.0
	a = 1 / (closest_depth - farthest_depth)
	b = farthest_depth / (farthest_depth - closest_depth)
	depth_imgs = np.clip(a * depth_imgs + b, 0, 1).squeeze()	

	# object processing
	obj_mesh_list = []

	# table info
	table_size = [0.5, 0.895, 0.05]
	table_position = [0.405, -0.0375, 0.243-0.05/2]
	table = o3d.geometry.TriangleMesh.create_box(
		width=table_size[0], 
		height=table_size[1], 
		depth=table_size[2]
	)
	table.translate([-table_size[0]/2, -table_size[1]/2, -table_size[2]/2])
	table.translate(table_position)
	table.paint_uniform_color([222/255,184/255,135/255])

	# load object	
	for idx in range(num_objects):
			
		# color
		color = colors[name_to_idx[objects_class[idx]]]

		# mesh
		obj = name_to_class[objects_class[idx]](
			objects_pose[idx], objects_param[idx], 'cpu'
		)
		obj_mesh = obj.get_mesh()
		obj_mesh.paint_uniform_color(color)
		obj_mesh_list.append(obj_mesh)        	 	

	# load tsdf
	mask_voxels = np.where(tsdf < 0)
	y_list = mask_voxels[0]
	x_list = mask_voxels[1]
	z_list = mask_voxels[2]
	for i, (x, y, z) in enumerate(zip(x_list, y_list, z_list)):
		mesh_box = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
		x_translate = voxel_size * x + vol_bnds[0, 0]
		y_translate = voxel_size * y + vol_bnds[1, 0]
		z_translate = voxel_size * z + vol_bnds[2, 0]
		mesh_box.translate([x_translate, y_translate, z_translate])
		if i == 0:
			mesh_total = mesh_box
		else:
			mesh_total += mesh_box	

	# visualizer
	vis1 = o3d.visualization.Visualizer()
	vis1.create_window(window_name="Object Mesh", width=960-2, height=520-32, left=960*0+1, top=31)
	vis2 = o3d.visualization.Visualizer()
	vis2.create_window(window_name="TSDF", width=960-2, height=520-32, left=960*1+1, top=31)

	# coordinate frame
	frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

	# add geometry
	vis1.add_geometry(frame)
	vis1.add_geometry(table)
	for idx in range(num_objects):
		vis1.add_geometry(obj_mesh_list[idx])
	vis2.add_geometry(frame)
	vis2.add_geometry(table)
	vis2.add_geometry(mesh_total)

	# update geometry
	while True:
		vis1.update_geometry(frame)
		vis1.update_geometry(table)
		for idx in range(num_objects):
			vis1.update_geometry(obj_mesh_list[idx])
		if not vis1.poll_events():
			break
		vis1.update_renderer()
		vis2.update_geometry(frame)
		vis2.update_geometry(table)
		vis2.update_geometry(mesh_total)
		if not vis2.poll_events():
			break
		vis2.update_renderer()

	vis1.destroy_window()
	vis2.destroy_window()

	# Visualize image datas
	for i in random.sample(list(range(len(mask_imgs))), 3):
		plt.figure(1)
		plt.suptitle(f"images (camera view {i})")
		plt.subplot(1, 3, 1)
		plt.imshow(mask_imgs[i], cmap='gray')
		plt.gca().set_title('mask image')
		plt.subplot(1, 3, 2)
		plt.imshow(depth_imgs[i], cmap='gray')
		plt.gca().set_title('depth image')
		if 'rgb_imgs' in data:
			plt.subplot(1, 3, 3)
			plt.imshow(rgb_imgs[i])
			plt.gca().set_title('rgb image')

		plt.tight_layout()
		plt.show()

#########################################################
################ DEBUG FOR VOXELIZATION #################
#########################################################

def debugging_windows_voxelize(data):
	
	# load data
	objects_pose = data['objects_pose'] 
	objects_class = data['objects_class']
	objects_param = data['objects_param']
	objects_bbox = data['objects_bbox']
	voxel_grid_total = data['voxel_grid_total']
	object_bbox = data['object_bbox']
	max_bbox = data['max_bbox']
	if 'marginal_bbox' in data:
		marginal_bbox = data['marginal_bbox']
	voxel_grid = data['voxel_grid']
	vox = data['vox']
	bound1 = data['bound1']
	bound2 = data['bound2']
	voxel_size = data['voxel_size']
	num_objects = len(objects_pose)

	# object processing
	obj_mesh_list = []
	obj_bbox_list = []

	# table info
	table_size = [0.5, 0.895, 0.05]
	table_position = [0.405, -0.0375, 0.243-0.05/2]
	table = o3d.geometry.TriangleMesh.create_box(
		width=table_size[0], 
		height=table_size[1], 
		depth=table_size[2]
	)
	table.translate([-table_size[0]/2, -table_size[1]/2, -table_size[2]/2])
	table.translate(table_position)
	table.paint_uniform_color([222/255,184/255,135/255])
		
	for idx in range(num_objects):
			
		# color
		color = colors[name_to_idx[objects_class[idx]]]

		# mesh
		obj = name_to_class[objects_class[idx]](
			objects_pose[idx], objects_param[idx], 'cpu'
		)
		obj_mesh = obj.get_mesh()
		obj_mesh.paint_uniform_color(color)
		obj_mesh_list.append(obj_mesh)        	

		# bbox
		bbox_min = objects_bbox[idx, :3] - objects_bbox[idx, 3:]
		bbox_max = objects_bbox[idx, :3] + objects_bbox[idx, 3:]
		obj_bbox = o3d.geometry.AxisAlignedBoundingBox()
		obj_bbox.min_bound = bbox_min
		obj_bbox.max_bound = bbox_max
		obj_bbox.color = color
		obj_bbox_list.append(obj_bbox)  	

	# bbox
	target_bbox_min = object_bbox[:3] - object_bbox[3:]
	target_bbox_max = object_bbox[:3] + object_bbox[3:]
	target_obj_bbox = o3d.geometry.AxisAlignedBoundingBox()
	target_obj_bbox.min_bound = target_bbox_min
	target_obj_bbox.max_bound = target_bbox_max
	target_obj_bbox.color = color
	max_bbox_min = max_bbox[:3] - max_bbox[3:]
	max_bbox_max = max_bbox[:3] + max_bbox[3:]
	max_obj_bbox = o3d.geometry.AxisAlignedBoundingBox()
	max_obj_bbox.min_bound = max_bbox_min
	max_obj_bbox.max_bound = max_bbox_max
	max_obj_bbox.color = color
	if 'marginal_bbox' in data:
		marginal_bbox_min = marginal_bbox[:3] - marginal_bbox[3:]
		marginal_bbox_max = marginal_bbox[:3] + marginal_bbox[3:]
		marginal_obj_bbox = o3d.geometry.AxisAlignedBoundingBox()
		marginal_obj_bbox.min_bound = marginal_bbox_min
		marginal_obj_bbox.max_bound = marginal_bbox_max
		marginal_obj_bbox.color = color    		

	# processed voxel
	X, Y, Z = torch.where(vox[-1] == True)
	indices_tensor = torch.cat(
		(X.unsqueeze(-1), Y.unsqueeze(-1), Z.unsqueeze(-1)),
		dim=-1
	)
	indices_tensor1 = indices_tensor[
		(indices_tensor[:, 0] >= bound1[0])
		* (indices_tensor[:, 0] <= bound1[1])
		* (indices_tensor[:, 1] >= bound1[2])
		* (indices_tensor[:, 1] <= bound1[3])
		* (indices_tensor[:, 2] >= bound1[4])
		* (indices_tensor[:, 2] <= bound1[5])
		* ~((indices_tensor[:, 0] >= bound2[0])
		* (indices_tensor[:, 0] <= bound2[1])
		* (indices_tensor[:, 1] >= bound2[2])
		* (indices_tensor[:, 1] <= bound2[3])
		* (indices_tensor[:, 2] >= bound2[4])
		* (indices_tensor[:, 2] <= bound2[5]))
	]

	indices_tensor2 = indices_tensor[
		(indices_tensor[:, 0] >= bound2[0])
		* (indices_tensor[:, 0] <= bound2[1])
		* (indices_tensor[:, 1] >= bound2[2])
		* (indices_tensor[:, 1] <= bound2[3])
		* (indices_tensor[:, 2] >= bound2[4])
		* (indices_tensor[:, 2] <= bound2[5])
	]

	pc_surr = indices_tensor1.detach().cpu().numpy() * voxel_size
	pc = indices_tensor2.detach().cpu().numpy() * voxel_size
	pcd_surr = o3d.geometry.PointCloud()
	pcd_surr.points = o3d.utility.Vector3dVector(pc_surr)
	pcd_surr.paint_uniform_color(colors[0])
	pcd_surr.translate(marginal_bbox[:3] - marginal_bbox[3:])
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pc)
	pcd.paint_uniform_color(colors[6])
	pcd.translate(marginal_bbox[:3] - marginal_bbox[3:])

	# visualizer
	vis1 = o3d.visualization.Visualizer()
	vis1.create_window(window_name="Objects & bounding boxes", width=960-2, height=520-32, left=960*0+1, top=31)
	vis2 = o3d.visualization.Visualizer()
	vis2.create_window(window_name="Voxel carving", width=960-2, height=520-32, left=960*1+1, top=31)
	vis3 = o3d.visualization.Visualizer()
	vis3.create_window(window_name="Object voxel carving", width=960-2, height=520-32, left=960*0+1, top=520+31)
	vis4 = o3d.visualization.Visualizer()
	vis4.create_window(window_name="Object Bounding Box", width=960-2, height=520-32, left=960*1+1, top=520+31)

	# coordinate frame
	frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

	# add geometry
	vis1.add_geometry(frame)
	vis1.add_geometry(table)
	for idx in range(num_objects):
		vis1.add_geometry(obj_mesh_list[idx])
		vis1.add_geometry(obj_bbox_list[idx])

	vis2.add_geometry(frame)
	vis2.add_geometry(table)
	vis2.add_geometry(voxel_grid_total)
	vis2.add_geometry(target_obj_bbox)
	vis2.add_geometry(max_obj_bbox)
	if 'marginal_bbox' in data:
		vis2.add_geometry(marginal_obj_bbox)

	vis3.add_geometry(frame)
	vis3.add_geometry(table)
	vis3.add_geometry(voxel_grid)
	vis3.add_geometry(target_obj_bbox)
	vis3.add_geometry(max_obj_bbox)
	if 'marginal_bbox' in data:
		vis3.add_geometry(marginal_obj_bbox)

	vis4.add_geometry(pcd)
	vis4.add_geometry(pcd_surr)
	vis4.add_geometry(target_obj_bbox)
	vis4.add_geometry(max_obj_bbox)
	if 'marginal_bbox' in data:
		vis4.add_geometry(marginal_obj_bbox)

	# update geometry
	while True:
		vis1.update_geometry(frame)
		vis1.update_geometry(table)
		for idx in range(num_objects):
			vis1.update_geometry(obj_mesh_list[idx])
			vis1.update_geometry(obj_bbox_list[idx])
		if not vis1.poll_events():
			break
		vis1.update_renderer()

		vis2.update_geometry(frame)
		vis2.update_geometry(table)
		vis2.update_geometry(voxel_grid_total)
		vis2.update_geometry(target_obj_bbox)
		vis2.update_geometry(max_obj_bbox)
		if 'marginal_bbox' in data:
			vis2.update_geometry(marginal_obj_bbox)
		if not vis2.poll_events():
			break
		vis2.update_renderer()

		vis3.update_geometry(frame)
		vis3.update_geometry(table)
		vis3.update_geometry(voxel_grid)
		vis3.update_geometry(target_obj_bbox)
		vis3.update_geometry(max_obj_bbox)
		if 'marginal_bbox' in data:
			vis3.update_geometry(marginal_obj_bbox)
		if not vis3.poll_events():
			break
		vis3.update_renderer()

		vis4.update_geometry(pcd)
		vis4.update_geometry(pcd_surr)
		vis4.update_geometry(target_obj_bbox)
		vis4.update_geometry(max_obj_bbox)
		if 'marginal_bbox' in data:
			vis4.update_geometry(marginal_obj_bbox)
		if not vis4.poll_events():
			break
		vis4.update_renderer()

	vis1.destroy_window()
	vis2.destroy_window()
	vis3.destroy_window()
	vis4.destroy_window()

#########################################################
############# DEBUG FOR TRANSPOSE DATASET ###############
#########################################################

def debugging_windows_transpose(data):

	# load data
	SE3s = data['SE3s'] 
	mesh_paths = data['mesh_paths']
	mask_imgs = data['mask_imgs']
	depth_imgs = data['depth_imgs']
	if 'rgb_imgs' in data:
		rgb_imgs = data['rgb_imgs']
	num_objects = len(SE3s)

	# depth image scaling
	closest_depth = 0.5
	farthest_depth = 1.0
	a = 1 / (closest_depth - farthest_depth)
	b = farthest_depth / (farthest_depth - closest_depth)
	depth_imgs = np.clip(a * depth_imgs + b, 0, 1).squeeze()	

	# object processing
	obj_mesh_list = []

	# table info
	table_size = [0.5, 0.895, 0.05]
	table_position = [0.405, -0.0375, 0.243-0.05/2]
	table = o3d.geometry.TriangleMesh.create_box(
		width=table_size[0], 
		height=table_size[1], 
		depth=table_size[2]
	)
	table.translate([-table_size[0]/2, -table_size[1]/2, -table_size[2]/2])
	table.translate(table_position)
	table.paint_uniform_color([222/255,184/255,135/255])

	# load object	
	for idx in range(num_objects):
			
		# color
		color = random.choice(colors)

		# mesh
		obj_path = mesh_paths[idx]
		obj_mesh = o3d.io.read_triangle_mesh(obj_path)
		R = obj_mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
		obj_mesh.rotate(R, center=(0, 0, 0))
		obj_mesh.paint_uniform_color(color)
		obj_mesh.transform(SE3s[idx])
		obj_mesh_list.append(obj_mesh)        	 	

	# visualizer
	vis1 = o3d.visualization.Visualizer()
	vis1.create_window(window_name="Object Mesh", width=960-2, height=520-32, left=960*0+1, top=31)

	# coordinate frame
	frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

	# add geometry
	vis1.add_geometry(frame)
	vis1.add_geometry(table)
	for idx in range(num_objects):
		vis1.add_geometry(obj_mesh_list[idx])

	# update geometry
	while True:
		vis1.update_geometry(frame)
		vis1.update_geometry(table)
		for idx in range(num_objects):
			vis1.update_geometry(obj_mesh_list[idx])
		if not vis1.poll_events():
			break
		vis1.update_renderer()

	vis1.destroy_window()

	# Visualize image datas
	for i in random.sample(list(range(len(mask_imgs))), 3):
		plt.figure(1)
		plt.suptitle(f"images (camera view {i})")
		plt.subplot(1, 3, 1)
		plt.imshow(mask_imgs[i], cmap='gray')
		plt.gca().set_title('mask image')
		plt.subplot(1, 3, 2)
		plt.imshow(depth_imgs[i], cmap='gray')
		plt.gca().set_title('depth image')
		if 'rgb_imgs' in data:
			plt.subplot(1, 3, 3)
			plt.imshow(rgb_imgs[i])
			plt.gca().set_title('rgb image')

		plt.tight_layout()
		plt.show()