import numpy as np
import torch
import time
import random
import torch
import open3d as o3d
import os
from tqdm import tqdm
from copy import deepcopy
from matplotlib import colormaps
import platform
if platform.system() == "Linux":
	from utils.suppress_logging import suppress_output
	with suppress_output():
		import pybullet as p
elif platform.system() == "Windows":
	import pybullet as p
else:
	print('OS is not Linux or Windows!')

from control.sim_with_table import PybulletSim
from control.sim_with_table_robot import PybulletRobotSim
from control.sim_with_shelf import PybulletShelfSim
from control.sim_with_shelf_robot import PybulletShelfRobotSim
from functions.utils import exp_se3, matrices_to_quats, quats_to_matrices, exp_so3
from tablewarenet.tableware import name_to_class

class ControlSimulationEnv:
	def __init__(
		self, 
		enable_gui=True,
		object_types=['WineGlass', 'Bowl', 'Bottle', 'BeerBottle', 'HandlessCup', 'Mug', 'Dish'], 
		num_objects=4,
		sim_type='shelf',
		open_shelf=True,
		process_idx=None,
		control_mode=False
		):
			
		# pybullet settings
		if sim_type == "table":
			self.sim = PybulletSim(enable_gui=enable_gui)
		elif sim_type == "table_with_robot":
			self.sim = PybulletRobotSim(enable_gui=enable_gui)
		elif sim_type == "shelf":
			self.sim = PybulletShelfSim(enable_gui=enable_gui, open_shelf=open_shelf)
		elif sim_type == "shelf_with_robot":
			self.sim = PybulletShelfRobotSim(enable_gui=enable_gui, open_shelf=open_shelf)

		# environment settings
		self.num_pts_down_wo_plane = 2048
		self.object_ids = []
		self.object_infos = []
		self.object_types_list = []
		self.num_objects = num_objects
		self.object_types = object_types	
		self.workspace_center = np.array([
			(self.sim.workspace_bounds[0][0] + self.sim.workspace_bounds[0][1])/2,
			(self.sim.workspace_bounds[1][0] + self.sim.workspace_bounds[1][1])/2,
			self.sim.workspace_bounds[2][0],
		])
		self.ws2cam = deepcopy(self.sim.ws2cam)
		self.ee2cam = np.array([[ 0.68838548, -0.72529384, -0.008618  ,  0.02603617],
								[ 0.72530726,  0.6884223 , -0.00202673, -0.0718539 ],
								[ 0.0074028 , -0.00485553,  0.99996081,  0.0739693 ],
								[ 0.        ,  0.        ,  0.        ,  1.        ]])

		# control setting
		self.control_mode = control_mode

		# temp object path
		self.temp_path = 'temp_objects'
		if not os.path.exists(self.temp_path):
			os.makedirs(self.temp_path)
		self.process_idx = process_idx

		# transpose object path
		if 'TRansPose' in self.object_types:
			self.transpose_path = 'assets/TRansPose'
			self.transpose_obj_list = os.listdir(self.transpose_path)

	#############################################################
	######################### FOR RESET #########################
	#############################################################

	def reset(
			self, 
			return_urdf_and_poses=False, 
			use_convex_decomp=False,
			target_object=None):

		while True:

			# remove objects
			for obj_id in self.object_ids:
				p.removeBody(obj_id)

			# initialize
			self.return_urdf_and_poses = return_urdf_and_poses
			self.use_convex_decomp = use_convex_decomp
			self.object_ids = []
			self.object_infos = []
			self.object_types_list = []
			if self.return_urdf_and_poses or self.use_convex_decomp:
				self.mesh_path_list = []
			if self.return_urdf_and_poses:
				self.urdf_path_list = []

			# initialize for reset
			self.reset_positions = []
			self.reset_orientations = []

			# load objects
			self._random_drop(target_object=target_object)
			time.sleep(1)

			# wait until objects stop moving
			flag = False
			old_po = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
			for _ in range(150):
				time.sleep(0.1)
				new_ps = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
				if np.sum((new_ps - old_po) ** 2) < 1e-6:
					flag = True
					break
				old_po = new_ps
			if not flag:
				# print('objects are still moving')
				continue
			
			# save 
			for i, obj_id in enumerate(self.object_ids):
				current_position = np.array(p.getBasePositionAndOrientation(obj_id)[0])
				current_orientation = np.array(p.getBasePositionAndOrientation(obj_id)[1])
				rotmat = quats_to_matrices(current_orientation)
				SE3 = np.concatenate([rotmat, current_position.reshape([3, 1])], axis=1)
				SE3 = np.concatenate([SE3, np.array([[0., 0., 0., 1.]])], axis=0)
				if self.object_types_list[i] == 'TRansPose':
					self.object_infos[i]['SE3'] = torch.from_numpy(SE3).float()
				else:
					self.object_infos[i].SE3 = torch.from_numpy(SE3).float()
					self.object_infos[i].construct()
				self.reset_positions.append(current_position)
				self.reset_orientations.append(current_orientation)

			# exceptions
			if not self.is_in_workspace():
				# print('objects are out of workspace')
				continue
			if not self.is_not_tilted():
				# print('objects are tilted')
				continue
			if not self.is_not_float():
				# print('objects are floating')
				continue

			# convex decomposition
			if self.use_convex_decomp:
				self.mesh_convexification()

			break

		# return urdf and poses
		if self.return_urdf_and_poses:
			
			urdfs_and_poses_dict = dict()
			for i, urdf_path in enumerate(self.urdf_path_list):
				
				# get position and quaternion
				obj_id = self.object_ids[i]
				current_position = np.array(p.getBasePositionAndOrientation(obj_id)[0])
				current_orientation = np.array(p.getBasePositionAndOrientation(obj_id)[1])

				urdfs_and_poses_dict[f'{i}'] = []
				urdfs_and_poses_dict[f'{i}'].append(1.0)
				urdfs_and_poses_dict[f'{i}'].append(current_orientation)
				urdfs_and_poses_dict[f'{i}'].append(current_position)
				urdfs_and_poses_dict[f'{i}'].append(urdf_path)
			
			return urdfs_and_poses_dict

	def is_in_workspace(self):
		flag = True
		for obj, obj_type in zip(self.object_infos, self.object_types_list):
			if obj_type == 'TRansPose':
				position = obj['SE3'][0:3,3]
			else:
				position = obj.SE3[0:3,3]
			if position[0] < self.sim.workspace_bounds[0, 0] or position[0] > self.sim.workspace_bounds[0, 1] or position[1] < self.sim.workspace_bounds[1, 0] or position[1] > self.sim.workspace_bounds[1, 1] or position[2] < self.sim.workspace_bounds[2, 0] or position[2] > self.sim.workspace_bounds[2, 1]:
				flag = False
		return flag

	def is_not_tilted(self):
		flag = True
		for obj, obj_type in zip(self.object_infos, self.object_types_list):
			if obj_type == 'TRansPose':
				z_axis = obj['SE3'][0:3,2]
			else:
				z_axis = obj.SE3[0:3,2]
			theta = torch.acos(torch.sum(z_axis * torch.tensor([0, 0, 1])))
			if theta >= np.pi / 30:
				flag = False
		return flag

	def is_not_float(self):
		flag = True
		for obj, obj_type in zip(self.object_infos, self.object_types_list):
			if obj_type == 'TRansPose':
				z_pos = obj['SE3'][2,3]
			else:
				z_pos = obj.SE3[2,3]	
			if obj_type == 'TRansPose':
				pass
			else:
				if z_pos >= self.sim.workspace_bounds[2][0] + 0.03:
					flag = False
		return flag

	#############################################################
	############### FOR ENVIRONMENT INITIALIZE ##################
	#############################################################

	def _random_drop(
			self,
			distance_threshold=0.03,
			target_object=None
		):

		# sample object positions
		while True:
			xy_pos = np.random.rand(self.num_objects, 2)
			xy_pos = (
				np.expand_dims(self.sim.spawn_bounds[:2, 0], 0)
				+ np.expand_dims(
					self.sim.spawn_bounds[:2, 1] - self.sim.spawn_bounds[:2, 0], 0
				) * xy_pos
			)
   
			if self.num_objects == 1:
				break

			distance_list = []
			for i in range(self.num_objects - 1):
				for j in range(i + 1, self.num_objects):
					distance = np.sqrt(np.sum((xy_pos[i] - xy_pos[j])**2))
					distance_list += [distance]

			if min(distance_list) > distance_threshold:
				break

		# except target object
		if target_object is not None:
			object_types_wo_target = deepcopy(self.object_types)
			if target_object in object_types_wo_target:
				object_types_wo_target.remove(target_object)
			if len(object_types_wo_target) == 0:
				raise ValueError('the set of object types without target is empty')

		# make all objects locate around the center of workspce
		for i in range(self.num_objects):
			
			# object pose
			position = np.append(
					xy_pos[i], 
					self.sim.spawn_bounds[2, 0]
				)
			orientation = [0, 0, np.random.rand()*2*np.pi]
			rotmat = exp_so3(np.array(orientation))
			pose = torch.eye(4)
			pose[0:3, 0:3] = torch.from_numpy(rotmat)
			pose[0:3, 3] = torch.from_numpy(position)
			
			# get random object type
			if target_object is not None:
				if i == 0:
					obj_type = target_object
				else:
					obj_type = random.choice(object_types_wo_target)
			else:
				obj_type = random.choice(self.object_types)

			# create object
			if obj_type == 'TRansPose': # create transpose
				obj_path = os.path.join(
					self.transpose_path,
					random.choice(self.transpose_obj_list)
				)
				self._create_transpose(pose, obj_path)

			else: # create tableware
				if self.control_mode:
					obj = name_to_class[obj_type](
						pose, params='random', device='cpu', t=0.05, process_mesh=False
					)    				
				else:
					obj = name_to_class[obj_type](
						pose, params='random', device='cpu'
					)
				self._create_tableware(obj)
			
			# time sleep
			time.sleep(0.2)

	def _create_transpose(
			self,
			pose, obj_path,
			object_color=None
		):

		# pose
		position = pose[0:3, 3].detach().cpu().numpy()
		orientation = matrices_to_quats(pose[0:3, 0:3].detach().cpu().numpy())
		
		# mesh
		mesh = o3d.io.read_triangle_mesh(obj_path)
		R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
		mesh.rotate(R, center=(0, 0, 0))
		
		# make temporary object file
		randnum = np.random.randint(low=0, high=1e9)		
		if self.process_idx is not None:
			mesh_name = f'temp_object_{self.process_idx}_{randnum}.obj'
			urdf_name = f'temp_object_{self.process_idx}_{randnum}.urdf'
		else:
			mesh_name = f'temp_object_{randnum}.obj'
			urdf_name = f'temp_object_{randnum}.urdf'
		mesh_path = os.path.join(self.temp_path, mesh_name)
		urdf_path = os.path.join(self.temp_path, urdf_name)

		# save mesh			
		o3d.io.write_triangle_mesh(mesh_path, mesh)

		# mesh path append
		if self.return_urdf_and_poses or self.use_convex_decomp:
			self.mesh_path_list.append(mesh_path)

		# urdf content
		if self.return_urdf_and_poses:
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
			time.sleep(0.2)

			# path append
			self.urdf_path_list.append(urdf_path)

		# color info
		if object_color is None:
			object_color = colormaps['Blues'](np.random.rand() * 0.7 + 0.3)
		
		# declare collision
		collision_id = p.createCollisionShape(
			p.GEOM_MESH, 
			fileName=mesh_path
		)
		visualshape_id = p.createVisualShape(
			p.GEOM_MESH, 
			fileName=mesh_path
		)

		# remove mesh
		if (not self.return_urdf_and_poses) and (not self.use_convex_decomp):
			os.remove(mesh_path)
  
		# create object
		body_id = p.createMultiBody(
			0.05, 
			collision_id, 
			visualshape_id, 
			position, 
			orientation
		)
		p.changeDynamics(
			body_id, 
			-1,
			rollingFriction=0.02,
			spinningFriction=0.02,
			lateralFriction=0.4,
			mass=0.1
		)
		p.changeVisualShape(
			body_id, 
			-1, 
			rgbaColor=object_color
		)
		self.object_ids.append(body_id)

		# keep object information
		transpose_obj = {
			'SE3': pose,
			'mesh': mesh,
			'mesh_path': obj_path
		}
		self.object_infos.append(transpose_obj)
		self.object_types_list.append('TRansPose')

		return body_id

	def _create_tableware(
			self,
			tableware,
			object_color=None
		):

		# pose
		position = tableware.SE3[0:3, 3].detach().cpu().numpy()
		orientation = matrices_to_quats(tableware.SE3[0:3, 0:3].detach().cpu().numpy())
		
		# mesh
		mesh = tableware.get_mesh(transform=False)

		# make temporary object file
		randnum = np.random.randint(low=0, high=1e9)		
		if self.process_idx is not None:
			mesh_name = f'temp_object_{self.process_idx}_{randnum}.obj'
			urdf_name = f'temp_object_{self.process_idx}_{randnum}.urdf'
		else:
			mesh_name = f'temp_object_{randnum}.obj'
			urdf_name = f'temp_object_{randnum}.urdf'
		mesh_path = os.path.join(self.temp_path, mesh_name)
		urdf_path = os.path.join(self.temp_path, urdf_name)			

		# save mesh
		o3d.io.write_triangle_mesh(mesh_path, mesh)

		# mesh path append
		if self.return_urdf_and_poses or self.use_convex_decomp:
			self.mesh_path_list.append(mesh_path)

		# urdf content
		if self.return_urdf_and_poses:
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
			time.sleep(0.2)

			# path append
			self.urdf_path_list.append(urdf_path)

		# color info
		if object_color is None:
			object_color = colormaps['Blues'](np.random.rand() * 0.7 + 0.3)
		
		# declare collision
		collision_id = p.createCollisionShape(
			p.GEOM_MESH, 
			fileName=mesh_path
		)
		visualshape_id = p.createVisualShape(
			p.GEOM_MESH, 
			fileName=mesh_path
		)

		# remove mesh
		if (not self.return_urdf_and_poses) and (not self.use_convex_decomp):
			os.remove(mesh_path)
  
		# create object
		body_id = p.createMultiBody(
			0.05, 
			collision_id, 
			visualshape_id, 
			position, 
			orientation
		)
		p.changeDynamics(
			body_id, 
			-1,
			rollingFriction=0.02,
			spinningFriction=0.02,
			lateralFriction=0.4,
			mass=0.1
		)
		p.changeVisualShape(
			body_id, 
			-1, 
			rgbaColor=object_color
		)
		self.object_ids.append(body_id)

		# keep object information
		self.object_infos.append(tableware)
		self.object_types_list.append('Tableware')

		return body_id

	def mesh_convexification(self, object_color=None, enable_pbar=True):
		
		# import libraries
		import coacd
		import trimesh	

		# pbar
		if enable_pbar:
			pbar = tqdm(
				total=len(self.mesh_path_list), 
				desc=f"doing mesh convex decomposition ... ", 
				leave=False
			)				
		
		for i, mesh_path in enumerate(self.mesh_path_list):
				
			# mesh = coacd.Mesh(
			# 	np.asarray(mesh.vertices), 
			# 	np.asarray(mesh.triangles))

			# load triangle mesh
			mesh = trimesh.load(mesh_path, force='mesh')
			new_mesh_path = f'{mesh_path.split(".")[0]}_cdp.obj'
			
			# convex decomposition
			mesh = coacd.Mesh(mesh.vertices, mesh.faces)	
			if platform.system() == "Linux":
				with suppress_output():
					parts = coacd.run_coacd(mesh)
			elif platform.system() == "Windows":
				parts = coacd.run_coacd(mesh)

			# assemble parts
			mesh_parts = []
			for vs, fs in parts:
				mesh_parts.append(trimesh.Trimesh(vs, fs))
			scene = trimesh.Scene()
			np.random.seed(0)
			for part in mesh_parts:
				part.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
				scene.add_geometry(part)
			scene.export(new_mesh_path)
			
			# get initialize
			obj_id = self.object_ids[i]
			position = p.getBasePositionAndOrientation(obj_id)[0]
			orientation = p.getBasePositionAndOrientation(obj_id)[1]
			p.removeBody(obj_id)
			
			# color info
			if object_color is None:
				object_color = colormaps['Blues'](np.random.rand() * 0.7 + 0.3)

			# declare collision
			collision_id = p.createCollisionShape(
				p.GEOM_MESH, 
				fileName=new_mesh_path
			)
			visualshape_id = p.createVisualShape(
				p.GEOM_MESH, 
				fileName=new_mesh_path
			)

			# remove mesh
			os.remove(mesh_path)
			os.remove(new_mesh_path)

			# create object
			new_body_id = p.createMultiBody(
				0.05, 
				collision_id, 
				visualshape_id, 
				position, 
				orientation
			)
			p.changeDynamics(
				new_body_id, 
				-1,
				rollingFriction=0.02,
				spinningFriction=0.02,
				lateralFriction=0.4,
				mass=0.1
			)
			p.changeVisualShape(
				new_body_id, 
				-1, 
				rgbaColor=object_color
			)

			# replace object
			self.object_ids[i] = new_body_id

			# pbar update
			if enable_pbar:
				pbar.update(1)

		# close tqdm
		if enable_pbar:
			pbar.close()

	def remove_urdf_and_mesh(self):
		for mesh_path, urdf_path in zip(self.mesh_path_list, self.urdf_path_list):
			os.remove(mesh_path)
			os.remove(urdf_path)

	#############################################################
	##################### RESET FUNCTIONS #######################
	#############################################################

	def reset_object_poses(self):
		for idx, object_id in enumerate(self.object_ids):
			p.resetBasePositionAndOrientation(
				object_id, 
				self.reset_positions[idx], 
				self.reset_orientations[idx])
	
	def remove_dropped_objects(self):
		
		# check removal indices
		remove_indices = [idx for idx in range(len(self.object_ids)) 
			if p.getBasePositionAndOrientation(self.object_ids[idx])[0][2] < self.sim.workspace_bounds[2][0] - 0.05]
		remove_indices.reverse()

		# remove indices
		for idx in remove_indices:
			p.removeBody(self.object_ids[idx])
			self.object_ids.pop(idx)
			self.object_infos.pop(idx)
			self.object_types_list.pop(idx)

	def update_object_poses(self):
		for idx, obj_id in enumerate(self.object_ids):
			current_position = np.array(p.getBasePositionAndOrientation(obj_id)[0])
			current_orientation = np.array(p.getBasePositionAndOrientation(obj_id)[1])
			rotmat = quats_to_matrices(current_orientation)
			SE3 = np.concatenate([rotmat, current_position.reshape([3, 1])], axis=1)
			SE3 = np.concatenate([SE3, np.array([[0., 0., 0., 1.]])], axis=0)
			if self.object_types_list[idx] == 'TRansPose':
				self.object_infos[idx]['SE3'] = torch.from_numpy(SE3).float()
			else:
				self.object_infos[idx].SE3 = torch.from_numpy(SE3).float()
				self.object_infos[idx].construct()

	def get_view_poses(self, num_cameras=36):
		
		# camera pose initialize
		view_poses = []
		
		# shelf params
		z_axis = np.array([0, 0, 1])
		v = - np.cross(z_axis, self.workspace_center)
		screw = np.concatenate((z_axis, v))
		rotating_angles = np.linspace(start = -np.pi, stop = np.pi, num=num_cameras+1)[:-1]
		cam_pose_pybullet_init = deepcopy(self.ws2cam)
		cam_pose_pybullet_init[0:3, 3] += self.workspace_center

		# camera poses
		for rotating_angle in rotating_angles:
			rotating_SE3 = exp_se3(rotating_angle * screw)
			cam_pose_pybullet = rotating_SE3.dot(cam_pose_pybullet_init)
			view_poses.append(cam_pose_pybullet)

		return view_poses

	def get_view_poses_for_tsdf(self, num_phi=6, num_theta=36, radius=0.8):
			
		# camera pose initialize
		view_poses = []
		
		rotating_phis = np.linspace(
			start = 20/180*np.pi, stop = 80/180*np.pi, num=num_phi)
		rotating_angles = np.linspace(
			start = -np.pi, stop = np.pi, num=num_theta+1)[:-1]

		# shelf params
		z_axis = np.array([0, 0, 1])
		v = - np.cross(z_axis, self.workspace_center)
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
			view_pose_init[:3, 3] += self.workspace_center

			# rotating theta
			for theta in rotating_angles:
				rotating_SE3 = exp_se3(theta * screw)
				view_pose = rotating_SE3.dot(view_pose_init)
				view_poses.append(view_pose)

		return view_poses