import math
import threading
import time
import torch
import copy
import numpy as np
import open3d as o3d
import platform
if platform.system() == "Linux":
	from utils.suppress_logging import suppress_output
	with suppress_output():
		import pybullet as p
		import pybullet_data
elif platform.system() == "Windows":
	import pybullet as p
	import pybullet_data
else:
	print('OS is not Linux or Windows!')
from copy import deepcopy
from functions.utils import get_SE3s
from control.gripper import Gripper
from functions.utils import exp_se3, matrices_to_quats, log_SO3, exp_so3
from functions.utils import quats_to_matrices, get_SE3s
from tablewarenet.primitives import SuperQuadric

class PybulletShelfSim:
	def __init__(self, enable_gui, open_shelf):
		
		# environment settings
		self.plane_z = -0.8
		self.low_table_position = [0.26, 0, -0.075/2-0.012]
		self.high_shelf_position = [0.61, 0., 0.003]
		self.high_shelf_orientation =  [0.0, 0.0, 1., 0.]

		# spawn offset
		spawn_offset_x = 0.15
		spawn_offset_y = 0.15

		# set workspace bounds
		self.workspace_bounds = np.array(
			[
				[self.high_shelf_position[0] - 0.374/2, self.high_shelf_position[0] + 0.374/2],
				[self.high_shelf_position[1] - 0.767/2, self.high_shelf_position[1] + 0.767/2],
				[self.high_shelf_position[2] + 0.3905 + 0.022/2, self.high_shelf_position[2] + 0.3905 + 0.022/2 + 0.300]
			]
		) 
		self.spawn_bounds = np.array(
			[
				[self.high_shelf_position[0] - 0.180, self.high_shelf_position[0] + 0.180 - spawn_offset_x],
				[self.high_shelf_position[1] - 0.360 + spawn_offset_y, self.high_shelf_position[1] + 0.360 - spawn_offset_y],
				[self.high_shelf_position[2] + 0.3905 + 0.022/2, self.high_shelf_position[2] + 0.3905 + 0.022/2 + 0.200]
			]
		)
  
		# Start PyBullet simulation
		if enable_gui:
			if platform.system() == "Linux":
				with suppress_output():
					self._physics_client = p.connect(p.GUI)
			else:
				self._physics_client = p.connect(p.GUI)
		else:
			self._physics_client = p.connect(p.DIRECT)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, -9.8)
		self.close_event = threading.Event()
		self.step_sim_thread = threading.Thread(target=self.step_simulation)
		self.step_sim_thread.daemon = True
		self.step_sim_thread.start()

		# Add ground plane
		self._plane_id = p.loadURDF("plane.urdf", [0, 0, self.plane_z])

		# # Add table
		table_path = 'assets/table/'
		self._low_table_id = p.loadURDF(table_path + 'low_table.urdf', self.low_table_position, useFixedBase=True)

		# Add shelf
		if open_shelf:
			shelf_path = 'assets/shelf/shelf.urdf'
		else:
			shelf_path = 'assets/shelf/shelf_old.urdf'
		self._high_shelf_id = p.loadURDF(shelf_path, self.high_shelf_position, self.high_shelf_orientation, useFixedBase=True)

		self.realsense_intrinsic = np.array(
			[606.1148681640625, 605.2857055664062, 325.19329833984375, 246.53085327148438]
		) # fx, fy, px, py
		self.camera_image_size=[480, 640]
		self.kinect_pose = np.array([
			[-0.43471579, -0.10620786,  0.894283,   -0.04457319],
 			[-0.9005385 ,  0.05926315, -0.43071834,  0.44833383],
 			[-0.00725236, -0.99257633, -0.12140689,  0.65361773],
 			[ 0.        ,  0.        ,  0.        ,  1.        ]
		])
		# camera list
		self.camera_params = {
   			# realsense
			0: self._get_camera_param(
				camera_pose = self.kinect_pose,
				camera_intrinsic = self.realsense_intrinsic,
				camera_image_size=[480, 640]
			),
		}
		self.ws2cam = np.array([[ 0.        , -0.25881905,  0.96592583, -0.73646976],
								[-1.        ,  0.        ,  0.        ,  0.        ],
								[ 0.        , -0.96592583, -0.25881905,  0.33230666],
								[ 0.        ,  0.        ,  0.        ,  1.        ]])

	# Step through simulation time
	def step_simulation(self):
		while True:
			p.stepSimulation()
			time.sleep(0.0001)
			if self.close_event.is_set():
				break

	def _get_camera_param(
			self, 
			camera_pose,
			camera_intrinsic,
			camera_image_size,
			camera_z_near=0.01,
			camera_z_far=20
		):

		# modified camera intrinsic
		fx = (camera_intrinsic[0] + camera_intrinsic[1]) / 2
		fy = (camera_intrinsic[0] + camera_intrinsic[1]) / 2
		px = float(camera_image_size[1]) / 2
		py = float(camera_image_size[0]) / 2

		# camera view matrix
		camera_view_matrix = copy.deepcopy(camera_pose)
		camera_view_matrix[:, 1:3] = -camera_view_matrix[:, 1:3]
		camera_view_matrix = np.linalg.inv(camera_view_matrix).T.reshape(-1)

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

	# Get latest RGB-D image
	def get_camera_data(self, cam_param):
		camera_data = p.getCameraImage(cam_param['camera_image_size'][1], cam_param['camera_image_size'][0],
									   cam_param['camera_view_matrix'], cam_param['camera_projection_matrix'],
									   shadow=1, renderer=p.ER_TINY_RENDERER)
		color_image = np.asarray(camera_data[2]).reshape(
			[cam_param['camera_image_size'][0], cam_param['camera_image_size'][1], 4]
		)[:, :, :3]  # remove alpha channel
		z_buffer = np.asarray(camera_data[3]).reshape(cam_param['camera_image_size'])
		camera_z_near = cam_param['camera_z_near']
		camera_z_far = cam_param['camera_z_far']
		depth_image = (2.0 * camera_z_near * camera_z_far) / (
			camera_z_far + camera_z_near - (2.0 * z_buffer - 1.0) * (
				camera_z_far - camera_z_near
			)
		)
		mask_image = np.asarray(camera_data[4]).reshape(cam_param['camera_image_size'][0:2])
		return color_image, depth_image, mask_image

	def end_thread(self, thread):
		self.close_event.set()
		while thread.is_alive():
			time.sleep(0.01)
		self.close_event.clear()

	def close(self):

		if platform.system() == "Linux":
			with suppress_output():
				if self.step_sim_thread.is_alive():
					self.end_thread(self.step_sim_thread)
				p.disconnect(self._physics_client)
		elif platform.system() == "Windows":
			if self.step_sim_thread.is_alive():
				self.end_thread(self.step_sim_thread)
			p.disconnect(self._physics_client)

	#############################################################
	##################### GET SHELF INFO ########################
	#############################################################

	def _get_shelf_info(self):
		
		# thickness
		thickness = 0.2 # side : 0.018, upper : 0.022
  
		# Make open3d box
		shelf_bottom_size = [0.392, 0.803, 0.022]
		shelf_left_size = [0.392, thickness, 0.759]
		shelf_right_size = [0.392, thickness, 0.759]
		shelf_middle_size = [0.374, 0.767, 0.022]
		shelf_back_size = [0.018, 0.767, 0.759]
		shelf_upper_size = [0.392, 0.803, thickness]

		# translates
		shelf_bottom_pos = [0, 0, 0]
		shelf_left_pos = [0, 0.3745 + thickness/2, 0.3905]
		shelf_right_pos = [0, -0.3745 - thickness/2, 0.3905]
		shelf_middle_pos = [0.009, 0, 0.3905]
		shelf_back_pos = [-0.187, 0, 0.3905]
		shelf_upper_pos = [0, 0, 0.759 + thickness/2]

		shelf_position = np.array(self.high_shelf_position)
		shelf_orientation = np.array(self.high_shelf_orientation)

		shelf_info = {
			'shelf_bottom': {
				'size': shelf_bottom_size,
				'position': shelf_bottom_pos
			},
			'shelf_left': {
				'size': shelf_left_size,
				'position': shelf_left_pos
			},
			'shelf_right': {
				'size': shelf_right_size,
				'position': shelf_right_pos
			},
			'shelf_middle': {
				'size': shelf_middle_size,
				'position': shelf_middle_pos
			},
			'shelf_back': {
				'size': shelf_back_size,
				'position': shelf_back_pos
			},
			'shelf_upper': {
				'size': shelf_upper_size,
				'position': shelf_upper_pos
			},
			'global_position': shelf_position,
			'global_orientation': shelf_orientation
		}

		return shelf_info

	def _get_shelf_sq_values(self, exclude_list=[]):
	  
		# get table info
		shelf = self._get_shelf_info()

		# get table parts
		part_keys = [key for key in shelf.keys() if key.startswith('shelf')]

		# get position and orientation
		global_position = shelf['global_position']
		global_orientation = shelf['global_orientation']
		global_SO3 = quats_to_matrices(global_orientation)

		# initialize
		sq_list = []

		for part_key in part_keys:
				
			# exclude
			if part_key.split('_')[-1] in exclude_list:
				continue

			# load info	
			size = shelf[part_key]['size']
			position = shelf[part_key]['position']
			
			# transform
			part_SO3 = global_SO3
			part_position = global_position + global_SO3.dot(position)
			T = get_SE3s(part_SO3, part_position)
			T = torch.tensor(T).float()

			# parameters
			parameter = torch.tensor(
				[size[0]/2, size[1]/2, size[2]/2, 0.2, 0.2, 0.0])

			# append
			sq = SuperQuadric(T, parameter, type="superellipsoid")
			sq_list.append(sq)

		return sq_list