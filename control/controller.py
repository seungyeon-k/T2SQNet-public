import os
import random
import platform
import time
import cv2
import threading
from datetime import datetime
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
import numpy as np
import torch
import platform
from copy import deepcopy

from control.control_env import ControlSimulationEnv
from control.gripper import Gripper
from control.debug_utils import (
	render_pc,
	render_objects,
	render_grasp_poses,
	render_pc_with_objects,
)

class Controller:
	def __init__(self, cfg):

		# debug
		self.debug = cfg.get("debug", False)

		# device
		self.device = cfg.device
		self.device_for_ik = 'cpu'

		# experiment type
		self.exp_type = cfg.exp_type
		self.recog_type = cfg.recog_type
		self.sim_type = cfg.sim_type
		self.task_type = cfg.task_type
		if self.exp_type not in ['sim', 'real']:
			raise ValueError(f'exp_type {self.exp_type} not in ["sim", "real"]')
		if self.recog_type not in ['t2sqnet_gt', 't2sqnet_rgb']:
			raise ValueError(f'recog_type {self.recog_type} not in ["t2sqnet_gt", "t2sqnet_rgb"]')
		if self.sim_type not in ['table', 'shelf']:
			raise ValueError(f'sim_type {self.sim_type} not in ["table", "shelf"]')

		# task setting
		if self.task_type == 'clear_clutter':
			self.target_object = None
		elif self.task_type == 'target_retrieval':
			self.target_object = cfg.get("target_object", None)
			if self.target_object is None:
				raise ValueError('target object is not specified for target retrieval task')

		# simulation setting
		if self.exp_type == 'sim':
			self.enable_gui = cfg.get("enable_gui", True)
			self.data_type = cfg.data_type
			if self.data_type == 'tableware':
				self.object_types = cfg.object_types
			elif self.data_type == 'transpose':
				self.object_types = ['TRansPose']
				if self.task_type == 'target_retrieval':
					raise NotImplementedError
			self.sim_type = f'{cfg.sim_type}_with_robot'
			self.num_objects = cfg.num_objects
			self.use_convex_decomp = cfg.use_convex_decomp
			self.env = ControlSimulationEnv(
				enable_gui=self.enable_gui, 
				object_types=self.object_types,
				num_objects=self.num_objects,
				sim_type=self.sim_type,
				control_mode=True
			)
			if self.sim_type == 'table_with_robot':
				p.resetDebugVisualizerCamera(1.18, -124.96, -20.38, [0.69, 0.06, 0.61])
			elif self.sim_type == 'shelf_with_robot':
				p.resetDebugVisualizerCamera(1.00, -131.76, -6.78, [0.39, 0.04, 0.60])
			self.env.reset(
				use_convex_decomp=self.use_convex_decomp,
				target_object=self.target_object)
			self.env.sim.get_franka_for_ik(device=self.device_for_ik)

			# robot setting
			if platform.system() == "Windows":
				self.speed = 0.05
				self.staright_speed = 0.05
				self.speed_gripper = 0.01
				self.grasp_force = 100
			elif platform.system() == "Linux":
				self.speed = 0.002
				self.staright_speed = 0.002
				self.speed_gripper = 0.005
				self.grasp_force = 100

		# real world setting
		elif self.exp_type == 'real':
			self.ip = cfg.ip
			self.port = cfg.port
			self.listener = Listener(self.ip, self.port)

			self.sim_type = f'{cfg.sim_type}_with_robot'
			self.env = ControlSimulationEnv(
				enable_gui=False, 
				object_types=[],
				num_objects=0,
				sim_type=self.sim_type,
				control_mode=True
			)
			self.env.sim.get_franka_for_ik(device=self.device_for_ik)

		# segmentation setting
		self.save_depth_imgs = cfg.get("save_depth_imgs", False) if self.exp_type == 'real' else False
		self.background_sam = cfg.get("background_sam", True) if self.exp_type == 'real' else False

		# environment setting
		if 'table' in self.sim_type:
			self.furniture_info = self.env.sim._get_table_info()
			self.furniture_sq_values = self.env.sim._get_table_sq_values()
		elif 'shelf' in self.sim_type:
			self.furniture_info = self.env.sim._get_shelf_info()
			self.furniture_sq_values = self.env.sim._get_shelf_sq_values()

		# robot control setting
		self.approach_distance = cfg.approach_distance
		self.angle_bound = cfg.angle_bound * torch.pi
		self.contain_camera = cfg.contain_camera
		if self.recog_type in ['t2sqnet_gt', 't2sqnet_rgb']:
			self.locked_joint_7 = cfg.get("locked_joint_7", None)
			if self.locked_joint_7 is not None:
				self.locked_joint_7 = cfg.locked_joint_7 * np.pi
		else:
			self.locked_joint_7 = None
		self.n_afterimage = 5
		gripper_open = Gripper(
			np.eye(4), 0.08, 
			contain_camera=self.contain_camera,
			locked_joint_7=self.locked_joint_7)
		self.gripper_pc_open = gripper_open.get_gripper_afterimage_pc(
			pc_dtype='torch', number_of_points=2048, 
			distance=self.approach_distance,
			n_gripper=self.n_afterimage
		)
		gripper_closed = Gripper(
			np.eye(4), 0.0, 
			contain_camera=self.contain_camera,
			locked_joint_7=self.locked_joint_7)
		self.gripper_pc_closed = gripper_closed.get_gripper_afterimage_pc(
			pc_dtype='torch', number_of_points=2048, 
			distance=self.approach_distance,
			n_gripper=self.n_afterimage
		)
		self.max_iter = cfg.ik_max_iter
		self.step_size1 = cfg.ik_step_size1
		self.step_size2 = cfg.ik_step_size2

		if self.contain_camera:
			self.bracket_offset = 0.008
		else:
			self.bracket_offset = 0.0

		# task setting for target retrieval
		self.placing_grid = self.get_2D_grid(
			resolution=0.03, dtype='torch'
		)
		self.place_offset = cfg.place_offset
		self.action_types = cfg.action_types

		# camera setting
		self.num_cameras = cfg.num_cameras
		self.reduce_ratio = cfg.reduce_ratio

		# data save folder
		folder_name = datetime.now().strftime('%Y%m%d-%H%M')
		self.save_folder = os.path.join('control_results', folder_name)
		if not os.path.exists(self.save_folder):
			os.makedirs(self.save_folder)

		# save workspace center
		np.save(
			os.path.join(self.save_folder, 'furniture_info'),
			self.furniture_info)
		np.save(
			os.path.join(self.save_folder, 'workspace_center'),
			self.env.workspace_center)

		#############################################################
		###################### SETUP MODEL ##########################
		#############################################################

		# setup t2sqnet model
		t2sqnet_cfg = cfg.get('t2sqnet_config', False)
		if t2sqnet_cfg and self.recog_type == 't2sqnet_rgb':
			
			# import model
			from models.pipelines import TSQPipeline
			
			# get setting
			self.text_prompt = cfg.text_prompt
			self.conf_thld = cfg.conf_thld
			self.sequential = cfg.sequential

			# load model
			self.tsqnet = TSQPipeline(
				t2sqnet_cfg.bbox_model_path,
				t2sqnet_cfg.bbox_config_path,
				t2sqnet_cfg.param_model_paths,
				t2sqnet_cfg.param_config_paths,
				t2sqnet_cfg.voxel_data_config_path,
				device=self.device,
				dummy_data_paths=t2sqnet_cfg.dummy_data_paths,
				num_augs=t2sqnet_cfg.num_augs
			)

			# for real-time segmentation inference
			if self.background_sam:
				self.initialize_img_list()
				self.finish_segmentation = False
				self.event = threading.Event()
				self.thread_sam = threading.Thread(
					target=self.realtime_segmentation, daemon=True)
				self.thread_sam.start()

	#############################################################
	###################### CLEAR CLUTTER ########################
	#############################################################
 
	def clear_clutter(self):

		# declutter methods
		if self.recog_type in ['t2sqnet_gt', 't2sqnet_rgb']:
			self.clear_clutter_tableware()

		# end sam thrad
		if self.background_sam:
			if self.thread_sam.is_alive():
				self.end_thread(self.thread_sam)

	def clear_clutter_tableware(self):

		# initialize data
		self.iter = 1

		while True:

			# iteration
			print(
			f'''
			*************************************************
			***************** CLEAR CLUTTER *****************
			****************** ITERATION {self.iter} ******************
			*************************************************
			'''
			)

			# initialize
			folder_name = os.path.join(self.save_folder, str(self.iter))
			if not os.path.exists(folder_name):
				os.makedirs(folder_name)
			self.initialize_img_list()
		
			# recognition
			recog_results = self.recognition()
			if recog_results is None:
				print(f'[1] observation failure')
				continue
			print(f'[1] recognized objects: {len(recog_results)}')
			if self.debug:
				render_objects(
					recog_results, 
					furniture=self.furniture_info)

			# check clutter is cleared
			if len(recog_results) == 0:
				print('[2] the scene is decluttered')
				break

			# sequential grasping
			if self.sequential:
				
				for _ in range(len(recog_results)):
					
					# grasp pose generation
					grasp_poses_list, collision_scores_list = self.grasp_pose_generation(
						recog_results)
					for i, collision_scores in enumerate(collision_scores_list):
						print(f'[2] {i}th object collision-free grasp poses: {(collision_scores > 0).sum().item()}')

					# sort grasp poses
					sorted_grasp_poses, sorted_object_indices = self.sort_grasp_poses(
						grasp_poses_list, collision_scores_list, return_object_indices=True
					)
					print(f'[3] collision-free grasp poses: {len(sorted_grasp_poses)}')
					if len(sorted_grasp_poses) == 0:
						break

					# solve batchwise ik
					sorted_joint_angles, successes = self.env.sim.solve_ik_grasping(
						sorted_grasp_poses.to(self.device_for_ik),
						approach_distance=self.approach_distance,
						bracket_offset=self.bracket_offset,
						locked_joint_7=self.locked_joint_7,
						max_iter=self.max_iter,
						step_size1=self.step_size1, step_size2=self.step_size2,
					)
					sorted_grasp_poses = sorted_grasp_poses[successes]
					sorted_object_indices = sorted_object_indices[successes]
					sorted_joint_angles = sorted_joint_angles[successes]
					print(f'[4] ik solved grasp poses: {len(sorted_joint_angles)}')
					if len(sorted_joint_angles) == 0:
						if self.exp_type == 'real':
							self.listener.send_grasp('failed')
							self.listener.close_connection()
						break

					# execute controller
					if self.exp_type == 'sim':
						success, object_indice = self.execute_grasping_sim(
							sorted_joint_angles, sorted_grasp_poses, 
							recog_results, sorted_object_indices=sorted_object_indices)
					elif self.exp_type == 'real':
						success, object_indice = self.execute_grasping_real(
							sorted_joint_angles, sorted_grasp_poses, 
							recog_results, sorted_object_indices=sorted_object_indices,
							receive_success=False)
					print(f'[5] control success: {success}')

					# delete grasped object
					if success:
						recog_results.pop(object_indice.item())
					else:
						break
					
					# update iteration
					self.iter += 1

					# make folder
					folder_name = os.path.join(self.save_folder, str(self.iter))
					if not os.path.exists(folder_name):
						os.makedirs(folder_name)

				# finished
				print('finished!')
				return True

			# non-sequential grasping (re-recognize)
			else:

				# grasp pose generation
				grasp_poses_list, collision_scores_list = self.grasp_pose_generation(
					recog_results)
				for i, collision_scores in enumerate(collision_scores_list):
					print(f'[2] {i}th object collision-free grasp poses: {(collision_scores > 0).sum().item()}')
				if self.debug:
					render_grasp_poses(
						recog_results, 
						grasp_poses_list,
						collision_scores_list,
						furniture=self.furniture_info)

				# sort grasp poses
				sorted_grasp_poses = self.sort_grasp_poses(
					grasp_poses_list, collision_scores_list
				)
				print(f'[3] collision-free grasp poses: {len(sorted_grasp_poses)}')
				if len(sorted_grasp_poses) == 0:
					break

				# solve batchwise ik
				sorted_joint_angles, successes = self.env.sim.solve_ik_grasping(
					sorted_grasp_poses.to(self.device_for_ik),
					approach_distance=self.approach_distance,
					bracket_offset=self.bracket_offset,
					locked_joint_7=self.locked_joint_7,
					max_iter=self.max_iter,
					step_size1=self.step_size1, step_size2=self.step_size2,
				)
				sorted_grasp_poses = sorted_grasp_poses[successes]
				sorted_joint_angles = sorted_joint_angles[successes]
				print(f'[4] ik solved grasp poses: {len(sorted_joint_angles)}')
				if len(sorted_joint_angles) == 0:
					if self.exp_type == 'real':
						self.listener.send_grasp('failed')
						self.listener.close_connection()
					break

				# execute controller
				if self.exp_type == 'sim':
					success = self.execute_grasping_sim(
						sorted_joint_angles, sorted_grasp_poses, recog_results)
				elif self.exp_type == 'real':
					success = self.execute_grasping_real(
						sorted_joint_angles, sorted_grasp_poses, recog_results)
				print(f'[5] control success: {success}')

				# update iteration
				self.iter += 1

		# finished
		return True

	#############################################################
	#################### TARGET RETRIEVAL #######################
	#############################################################
 
	def target_retrieval(self):

		# initialize data
		self.iter = 1

		while True:
	
			# iteration
			print(
			f'''
			*************************************************
			**************** TARGET RETRIEVAL ***************
			****************** ITERATION {self.iter} ******************
			*************************************************
			'''
			)

			# initialize
			folder_name = os.path.join(self.save_folder, str(self.iter))
			if not os.path.exists(folder_name):
				os.makedirs(folder_name)
			self.initialize_img_list()

			# recognition
			recog_results = self.recognition()
			if recog_results is None:
				print(f'[1] observation failure')
				continue
			print(f'[1] recognized objects: {len(recog_results)}')

			# target object is in recognized objects
			recognized_types = [r.name for r in recog_results]
			count = recognized_types.count(self.target_object)
			if count != 1:
				raise ValueError('the number of target objects is not one')
			else:
				print('[2] the number of target objects: 1')
				self.target_idx = recognized_types.index(self.target_object)

			# sequential rearranging
			if self.sequential:
					
				while True:

					# check target object is graspable
					graspablility_score, solely_graspable, target_grasp_poses, target_collision_scores = self.get_target_object_graspability(
						recog_results
					)
					if (graspablility_score == 0) and solely_graspable:
						
						print('[3] target object is graspable')

						# select one grasp pose
						sorted_grasp_poses = self.sort_grasp_poses(
							target_grasp_poses, target_collision_scores
						)
						print(f'[4] collision-free target grasp poses: {len(sorted_grasp_poses)}')

						# solve batchwise ik
						sorted_joint_angles, successes = self.env.sim.solve_ik_grasping(
							sorted_grasp_poses.to(self.device_for_ik),
							approach_distance=self.approach_distance,
							bracket_offset=self.bracket_offset,
							locked_joint_7=self.locked_joint_7,
							max_iter=self.max_iter,
							step_size1=self.step_size1, step_size2=self.step_size2,
						)
						sorted_grasp_poses = sorted_grasp_poses[successes]
						sorted_joint_angles = sorted_joint_angles[successes]
						print(f'[5] ik solved grasp poses: {len(sorted_joint_angles)}')
						if len(sorted_joint_angles) == 0:
							if self.exp_type == 'real':
								self.listener.send_grasp('failed')
								self.listener.close_connection()
							break

						# execute controller
						if self.exp_type == 'sim':
							success = self.execute_grasping_sim(
								sorted_joint_angles, sorted_grasp_poses, recog_results)
						elif self.exp_type == 'real':
							success = self.execute_grasping_real(
								sorted_joint_angles, sorted_grasp_poses, recog_results)
						print(f'[6] control success: {success}')

						# finished
						return True

					else:
							
						# get score
						print(f'[3] target object is not graspable; score = {graspablility_score}')

					# sample actions
					actions = self.sample_rearranging_actions(recog_results)
					print(f'[4] sampled rearranging actions: {len(actions)}')
					if len(actions) == 0:
						break

					# sort rearranging actions
					sorted_actions, sorted_graspability_scores = self.sort_rearranging_actions(
						actions, recog_results)
					print(f'[5] best score after rearrangement: {sorted_graspability_scores[0]}')

					# solve batchwise ik
					sorted_joint_angles, action_types, successes = self.env.sim.solve_ik_rearranging(
						sorted_actions,
						device=self.device_for_ik,
						approach_distance=self.approach_distance,
						bracket_offset=self.bracket_offset,
						locked_joint_7=self.locked_joint_7,
						max_iter=self.max_iter,
						step_size1=self.step_size1, step_size2=self.step_size2,
					)
					sorted_actions = [
						sorted_actions[i] for i in range(len(sorted_actions)) if successes[i]]
					sorted_joint_angles = sorted_joint_angles[successes]
					action_types = [
						action_types[i] for i in range(len(action_types)) if successes[i]]
					print(f'[6] ik solved rearranging actions: {len(sorted_joint_angles)}')
					if len(sorted_joint_angles) == 0:
						if self.exp_type == 'real':
							self.listener.send_grasp('failed')
							self.listener.close_connection()
						break

					# execute controller
					if self.exp_type == 'sim':
						success, action = self.execute_rearranging_sim(
							sorted_joint_angles, sorted_actions, action_types, 
							recog_results, return_action=True)
					elif self.exp_type == 'real':
						success, action = self.execute_rearranging_real(
							sorted_joint_angles, sorted_actions, action_types, 
							recog_results, return_action=True, 
							receive_success=False)
					print(f'[7] control success: {success}')

					# delete grasped object
					if success:
						recog_results = self.update_recog_results(
							recog_results, action
						)
					else:
						break
					
					# update iteration
					self.iter += 1

					# make folder
					folder_name = os.path.join(self.save_folder, str(self.iter))
					if not os.path.exists(folder_name):
						os.makedirs(folder_name)

			else:
				
				# check target object is graspable
				graspablility_score, solely_graspable, target_grasp_poses, target_collision_scores = self.get_target_object_graspability(
					recog_results
				)
				if (graspablility_score == 0) and solely_graspable:
					
					print('[3] target object is graspable')

					# select one grasp pose
					sorted_grasp_poses = self.sort_grasp_poses(
						target_grasp_poses, target_collision_scores
					)
					print(f'[4] collision-free target grasp poses: {len(sorted_grasp_poses)}')

					# solve batchwise ik
					sorted_joint_angles, successes = self.env.sim.solve_ik_grasping(
						sorted_grasp_poses.to(self.device_for_ik),
						approach_distance=self.approach_distance,
						bracket_offset=self.bracket_offset,
						locked_joint_7=self.locked_joint_7,
						max_iter=self.max_iter,
						step_size1=self.step_size1, step_size2=self.step_size2,
					)
					sorted_grasp_poses = sorted_grasp_poses[successes]
					sorted_joint_angles = sorted_joint_angles[successes]
					print(f'[5] ik solved grasp poses: {len(sorted_joint_angles)}')
					if len(sorted_joint_angles) == 0:
						if self.exp_type == 'real':
							self.listener.send_grasp('failed')
							self.listener.close_connection()
						break

					# execute controller
					if self.exp_type == 'sim':
						success = self.execute_grasping_sim(
							sorted_joint_angles, sorted_grasp_poses, recog_results)
					elif self.exp_type == 'real':
						success = self.execute_grasping_real(
							sorted_joint_angles, sorted_grasp_poses, recog_results)
					print(f'[6] control success: {success}')

					# finished
					return True

				else:
						
					# get score
					print(f'[3] target object is not graspable; score = {graspablility_score}')

				# sample actions
				actions = self.sample_rearranging_actions(recog_results)
				print(f'[4] sampled rearranging actions: {len(actions)}')
				if len(actions) == 0:
					break

				# sort rearranging actions
				sorted_actions, sorted_graspability_scores = self.sort_rearranging_actions(
					actions, recog_results)
				print(f'[5] best score after rearrangement: {sorted_graspability_scores[0]}')

				# solve batchwise ik
				sorted_joint_angles, action_types, successes = self.env.sim.solve_ik_rearranging(
					sorted_actions,
					device=self.device_for_ik,
					approach_distance=self.approach_distance,
					bracket_offset=self.bracket_offset,
					locked_joint_7=self.locked_joint_7,
					max_iter=self.max_iter,
					step_size1=self.step_size1, step_size2=self.step_size2,
				)
				sorted_actions = [
					sorted_actions[i] for i in range(len(sorted_actions)) if successes[i]]
				sorted_joint_angles = sorted_joint_angles[successes]
				action_types = [
					action_types[i] for i in range(len(action_types)) if successes[i]]
				print(f'[6] ik solved rearranging actions: {len(sorted_joint_angles)}')
				if len(sorted_joint_angles) == 0:
					if self.exp_type == 'real':
						self.listener.send_grasp('failed')
						self.listener.close_connection()
					break

				# execute controller
				if self.exp_type == 'sim':
					success = self.execute_rearranging_sim(
						sorted_joint_angles, sorted_actions, action_types, recog_results)
				elif self.exp_type == 'real':
					success = self.execute_rearranging_real(
						sorted_joint_angles, sorted_actions, action_types, recog_results)
				print(f'[7] control success: {success}')

				# update iteration
				self.iter += 1

	#############################################################
	######################## FUNCTIONS ##########################
	#############################################################

	def initialize_img_list(self):
		joint_angles = self.get_joint_angles_for_observation()
		self.rgb_img_list = [None] * len(joint_angles)
		if self.save_depth_imgs:
			self.depth_img_list = [None] * len(joint_angles) 
		if self.background_sam:
			self.mask_img_list = [None] * len(joint_angles)	
		self.segmented_indices = [False] * len(joint_angles)

	def observation(self):

		# camera setting
		if self.recog_type == 't2sqnet_rgb':
			camera_param_list = self.get_camera_parameters()
	
		# gt observation
		if self.recog_type == 't2sqnet_gt':
			if self.exp_type == 'real':
				raise ValueError('ground-truth is not available in real-world setting.')

		# rgb observation
		elif self.recog_type == 't2sqnet_rgb':
			
			# simulation
			if self.exp_type == 'sim':
				
				# get images
				joint_angles = self.get_joint_angles_for_observation()
				self.env.sim.move_joints(joint_angles[0])
				time.sleep(1)
				for i, camera_param in enumerate(camera_param_list):

					# get camera images
					rgb_image, _, _ = self.env.sim.get_camera_data(
						camera_param
					)

					# append images
					rgb_image = torch.tensor(
						rgb_image).float().permute(2, 0, 1)
					self.rgb_img_list[i] = rgb_image

				# stack
				self.env.sim.robot_go_home()
				self.rgb_img_list = torch.stack(self.rgb_img_list)

			# realworld
			elif self.exp_type == 'real':

				# send joint angle
				print('getting observation data...')
				data = self.listener.recv_vision()
				if not data == b'request_joint_angle':
					raise ValueError('invalid receved data; must be "request_joint_angle"')
				joint_angles = self.get_joint_angles_for_observation()
				self.listener.send_grasp(joint_angles)

				# receive vision data
				for i in range(len(joint_angles)):
					data = self.listener.recv_vision()
					if data == b'failed':
						self.listener.send_grasp('failed')
						return None, None
					rgb_image = data[b'rgb_image']
					depth_image = data[b'depth_image']
					self.listener.send_grasp(f'{i+1}th_view_received')

					# resize images
					rgb_image = cv2.resize(
						rgb_image, 
						dsize=(
							int(rgb_image.shape[1]/self.reduce_ratio), 
							int(rgb_image.shape[0]/self.reduce_ratio)),
						interpolation=cv2.INTER_CUBIC
					)
					if self.save_depth_imgs:
						depth_image = cv2.resize(
							depth_image, 
							dsize=(
								int(depth_image.shape[1]/self.reduce_ratio), 
								int(depth_image.shape[0]/self.reduce_ratio)),
							interpolation=cv2.INTER_CUBIC
						) 

					# numpy to torch
					rgb_image = torch.tensor(
						rgb_image).float().permute(2, 0, 1)
					if self.save_depth_imgs:
						depth_image = torch.tensor(depth_image.astype(np.float32)).float()

					# append
					self.rgb_img_list[i] = rgb_image
					if self.save_depth_imgs:
						self.depth_img_list[i] = depth_image

				# stack
				if self.background_sam:
					self.finish_segmentation = False
					while not self.finish_segmentation:
						time.sleep(0.01)
					self.rgb_img_list = torch.stack(self.rgb_img_list)
					self.mask_img_list = torch.stack(self.mask_img_list)
					if self.save_depth_imgs:
						self.depth_img_list = torch.stack(self.depth_img_list)
				else:
					self.rgb_img_list = torch.stack(self.rgb_img_list)
					if self.save_depth_imgs:
						self.depth_img_list = torch.stack(self.depth_img_list)

			if self.background_sam:
				return self.mask_img_list, camera_param_list	
			else:
				return self.rgb_img_list, camera_param_list

	def recognition(self):

		# ground truth
		if self.recog_type == 't2sqnet_gt':
			if self.data_type == 'tableware':
				return self.env.object_infos
			elif self.data_type == 'transpose':
				raise NotImplementedError

		# t2sqnet from rgb
		elif self.recog_type == 't2sqnet_rgb':
			
			# observation
			img_list, camera_params = self.observation()
			camera_params = self.process_camera_parameters(camera_params)
			if img_list is None:
				return None
			
			# save images
			np.save(
				os.path.join(self.save_folder, str(self.iter), 'rgb_img_list'), 
				self.rgb_img_list.permute(0, 2, 3, 1).cpu().numpy() / 255.0)
			if self.background_sam:
				np.save(
					os.path.join(self.save_folder, str(self.iter), 'mask_img_list'), 
					self.mask_img_list.cpu().numpy())
			if self.save_depth_imgs:
				np.save(
					os.path.join(self.save_folder, str(self.iter), 'depth_img_list'), 
					self.depth_img_list.cpu().numpy())    			

			# tsqnet inference
			results = self.tsqnet.forward(
				img_list, camera_params, 
				text_prompt=self.text_prompt, 
				conf_thld=self.conf_thld,
				from_mask_imgs=self.background_sam,
				output_all=True
			)

			# object info
			obj_list = results[-1][0]
			for obj in obj_list:
				obj.SE3[:3, 3] += torch.tensor(self.env.workspace_center).to(obj.SE3)
				obj.construct()
				obj.send_to_device('cpu')
			obj_info = self.get_objects_info(obj_list)
			np.save(
				os.path.join(self.save_folder, str(self.iter), 'obj_info'),
				obj_info)

			# save inference results
			bboxes =  results[-3]
			for bbox in bboxes:
				bbox[:3] += torch.tensor(self.env.workspace_center).to(bbox)
				bbox = bbox.detach().cpu()
			bbox_info = {
				'objects_class': results[-2],
				'bboxes': bboxes
			}
			np.save(
				os.path.join(self.save_folder, str(self.iter), 'bbox_info'),
				bbox_info)

			# save mask images
			if not self.background_sam:
				mask_img_list = results[-4]
				np.save(
					os.path.join(self.save_folder, str(self.iter), 'mask_img_list'), 
					mask_img_list.cpu().numpy())				

			return obj_list

	def get_camera_parameters(self):
	
		# get view poses
		camera_view_poses = self.env.get_view_poses(
			num_cameras=self.num_cameras)

		# get camera parameters
		camera_param_list = []
		for view_pose in camera_view_poses:
				
			# set camera parameters
			camera_param = self.env.sim._get_camera_param(
				camera_pose=view_pose,
				camera_intrinsic=(
					self.env.sim.realsense_intrinsic / self.reduce_ratio
				),
				camera_image_size=[
					int(self.env.sim.camera_image_size[0]/self.reduce_ratio), 
					int(self.env.sim.camera_image_size[1]/self.reduce_ratio)
				]
			)
			camera_param_list.append(camera_param)
		
		# set rendering frames
		render_frame_list = [15, 16, 17, 18, 19, 20, 21]
		camera_param_list = [camera_param_list[i] for i in render_frame_list]
	
		return camera_param_list

	def process_camera_parameters(self, camera_param_list):

		proj_matrices = []
		camera_intr_list = []
		camera_pose_list = []

		for cam in camera_param_list:
			camera_intr = cam['camera_intr']
			camera_pose = cam['camera_pose']
			camera_image_size = cam['camera_image_size']
			if self.recog_type in ['t2sqnet_gt', 't2sqnet_rgb']:
				camera_pose[0:3, 3] -= self.env.workspace_center
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

	def realtime_segmentation(self):
		
		# run background
		while True:
			
			# initialize
			all_segmented = False
			
			# run until all rgbs are segmented
			for i, mask_img in enumerate(self.mask_img_list):

				# mask image is empty
				if mask_img is None:

					# segmentation	
					if self.rgb_img_list[i] is not None:
						start_time = time.time()
						result = self.tsqnet.rgb2mask(
							self.rgb_img_list[i].unsqueeze(0), self.text_prompt)
						end_time = time.time()
						self.mask_img_list[i] = result.squeeze(0)
						if not self.segmented_indices[i]:
							self.segmented_indices[i] = True
							print(f'{i+1}th rgb image is segmented using SAM, shape: {self.mask_img_list[i].shape}, ellapsed time: {end_time - start_time}(s)')
					else:
						break

				# all images are segmented
				if i == len(self.mask_img_list) - 1:
					all_segmented = True

			# segmentation finished
			if all_segmented:
				self.finish_segmentation = True

	def end_thread(self, thread:threading.Thread):
		self.event.set()
		while thread.is_alive():
			time.sleep(0.1)
		self.event.clear()

	def get_objects_info(self, obj_list):
		return {
			'objects_pose': [obj.SE3.cpu().numpy() for obj in obj_list],
			'objects_class': [obj.name for obj in obj_list],
			'objects_param': [obj.params.cpu().numpy() for obj in obj_list]
		}

	#############################################################
	######################## DECLUTTER ##########################
	#############################################################

	def sort_grasp_poses(
			self, grasp_poses_list, collision_scores_list,
			return_object_indices=False):

		# total grasp poses
		total_grasp_poses = torch.cat(grasp_poses_list, dim=0)
		total_collision_scores = torch.cat(collision_scores_list, dim=0)
		if return_object_indices:
			total_object_indices = []
			for i, grasp_poses in enumerate(grasp_poses_list):
				total_object_indices += [i] * len(grasp_poses)
			total_object_indices = torch.tensor(total_object_indices)

		# sort grasp poses
		sorted_collision_scores = total_collision_scores.argsort(descending=True)
		sorted_grasp_poses = total_grasp_poses[
			sorted_collision_scores]
		total_collision_scores = total_collision_scores[
			sorted_collision_scores]
		if return_object_indices:
			total_object_indices = total_object_indices[
				sorted_collision_scores]
		
		# collision free grasp poses
		sorted_grasp_poses = sorted_grasp_poses[total_collision_scores > 0]
		if return_object_indices:
			total_object_indices = total_object_indices[total_collision_scores > 0]

		# return
		if return_object_indices:
			return sorted_grasp_poses, total_object_indices	
		else:
			return sorted_grasp_poses

	def execute_grasping_sim(
			self, sorted_joint_angles, sorted_grasp_poses, recog_results, 
			sorted_object_indices=None, gripper_width=0.08):
			
		for i, (joint_angles, grasp_pose) in enumerate(zip(sorted_joint_angles, sorted_grasp_poses)):
			success_grasping = self.env.sim.object_grasping(
				joint_angles,
				speed=self.speed,
				speed_gripper=self.speed_gripper,
				gripper_width=gripper_width, 
				grasp_force=self.grasp_force)				

			# check if success
			if success_grasping:
				time.sleep(1)
				self.env.remove_dropped_objects()
				if self.recog_type == 't2sqnet_gt':
					self.env.update_object_poses()
				
				# save action
				if self.recog_type == 't2sqnet_rgb':
					data = {
						'action_type': 'grasping',
						'joint_angles': joint_angles.detach().cpu().numpy().astype(np.float64),
						'furniture_info': self.furniture_info,
						'objects_info': self.get_objects_info(recog_results),
						'grasp_pose': grasp_pose.detach().cpu().numpy()
					}
				np.save(
					os.path.join(self.save_folder, str(self.iter), 'action'), 
					data)	

				if sorted_object_indices is not None:
					return True, sorted_object_indices[i]
				else:
					return True
			else:
				self.env.sim.robot_go_home()
				self.env.reset_object_poses()
				print('failed! try another grasp pose.')

		if sorted_object_indices is not None:
			return False, sorted_object_indices[i]
		else:
			return False

	def execute_grasping_real(
			self, sorted_joint_angles, sorted_grasp_poses, recog_results,
			sorted_object_indices=None, receive_success=True):
		
		# currently pick one
		joint_angles = sorted_joint_angles[0]
		grasp_pose = sorted_grasp_poses[0]
		if sorted_object_indices is not None:
			object_indice = sorted_object_indices[0]

		# send data
		data = self.listener.recv_vision()
		if not data == b'request_action':
			raise ValueError('invalid receved data; must be "request_action"')
		if recog_results is not None:
			if self.recog_type == 't2sqnet_rgb':
				data = {
					'action_type': 'grasping',
					'joint_angles': joint_angles.detach().cpu().numpy().astype(np.float64),
					'furniture_info': self.furniture_info,
					'objects_info': self.get_objects_info(recog_results),
					'grasp_pose': grasp_pose.detach().cpu().numpy()
				}
			elif self.recog_type == 'dvgo_mask':
				data = {
					'action_type': 'grasping',
					'joint_angles': joint_angles.detach().cpu().numpy().astype(np.float64),
					'furniture_info': self.furniture_info,
					'recog_info': recog_results,
					'grasp_pose': grasp_pose.detach().cpu().numpy()
				}
			elif self.recog_type == 'graspnerf':
				data = {
					'action_type': 'grasping',
					'joint_angles': joint_angles.detach().cpu().numpy().astype(np.float64),
					'furniture_info': self.furniture_info,
					'recog_info': recog_results.detach().cpu().numpy(),
					'grasp_pose': grasp_pose.detach().cpu().numpy()
				} 				 				
		else:
			data = {
				'action_type': 'grasping',
				'joint_angles': joint_angles.detach().cpu().numpy().astype(np.float64),
				'furniture_info': None,
				'objects_info': None,
				'grasp_pose': grasp_pose.detach().cpu().numpy()
			}
		self.listener.send_grasp(data)

		# save action
		np.save(
			os.path.join(self.save_folder, str(self.iter), 'action'), 
			data)		

		# receive success
		if receive_success:
			data = self.listener.recv_vision()
			if data == b'success':
				self.listener.send_grasp('success')
				if sorted_object_indices is not None:
					return True, object_indice
				else:
					return True
			elif data == b'failed':
				self.listener.send_grasp('failed')
				if sorted_object_indices is not None:
					return False, object_indice
				else:
					return False
		else:
			return True, object_indice

	#############################################################
	#################### TARGET RETRIEVAL #######################
	#############################################################

	def get_target_object_graspability(self, recog_results):

		# grasp pose generation
		grasp_poses_list, collision_scores_list, modified_scores_list = self.grasp_pose_generation(
			recog_results, return_modified_score=True)
		
		# target object graspability
		target_grasp_poses = [grasp_poses_list[self.target_idx]]
		target_collision_scores = [collision_scores_list[self.target_idx]]
		target_modified_collision_scores = [modified_scores_list[self.target_idx]]

		# target graspability
		graspability_score = target_modified_collision_scores[0].min().item()

		return graspability_score, True, target_grasp_poses, target_collision_scores

	def sample_rearranging_actions(
			self, 
			recog_results,
			n_actions=40):
		
		# check actions types
		for action_type in self.action_types:
			if action_type not in ['pick_and_place']:
				raise ValueError('action type should be in ["pick_and_place"]')

		# initialize
		actions = []

		# get pick-and-place actions
		if 'pick_and_place' in self.action_types:
			
			# initialize
			pick_and_place_actions = []

			# available grasp poses
			grasp_poses_list, collision_scores_list = self.grasp_pose_generation(
				recog_results)

			# get grasp poses
			for i, (grasp_poses, collision_scores) in enumerate(zip(grasp_poses_list, collision_scores_list)):	
				
				# exclude target for pick and place
				if i == self.target_idx:
					continue

				# object pc
				obj = recog_results[i]
				SE3 = obj.SE3
				pc = obj.get_point_cloud(
					number_of_points=256, dtype='torch')
				
				grasp_poses = grasp_poses[collision_scores > 0]
				for grasp_pose in grasp_poses:
					action_dict = {
						'action_type': 'pick_and_place',
						'object_idx': i,
						'object_pose': SE3,
						'object_pc': pc,
						'grasp_pose': grasp_pose
					}
				
					pick_and_place_actions.append(action_dict)
			
			# append actions
			actions += pick_and_place_actions
			actions = random.sample(actions, min(len(actions), n_actions))

		# sample actions
		for action in actions:
			
			# action type
			action_type = action['action_type']

			# pick and place
			if action_type == 'pick_and_place':
				
				# get random place pose
				placeable_poses = self.get_placeable_poses(
					action, recog_results)

				# update
				if len(placeable_poses) > 0:
					place_pose = random.choice(placeable_poses)
					place_pose[:3, 3] += self.place_offset
					action['place_pose'] = place_pose
					action['success'] = True
				else:
					action['success'] = False

		# valid actions
		actions = [a for a in actions if a['success']]

		return actions

	def get_placeable_poses(self, sampled_action, recog_results):

		# get settings
		object_idx = sampled_action['object_idx']
		grasp_pose = sampled_action['grasp_pose']
		object_pose = sampled_action['object_pose'].to(grasp_pose)
		object_pc = sampled_action['object_pc'].to(grasp_pose)
		direction = - grasp_pose[:3, 2]
		RotZ_angle = torch.atan2(grasp_pose[1, 2], grasp_pose[0, 2])
		recog_results_wo_target = recog_results[:]
		recog_results_wo_target.pop(object_idx)
		
		# grid settings
		placing_grid = deepcopy(self.placing_grid)
		placing_grid = torch.cat(
			[placing_grid, 
			torch.tensor([[object_pose[2, 3]]]).repeat(placing_grid.shape[0], 1)
			], dim=-1)
		gripper_SE3s = torch.eye(4).repeat(len(placing_grid), 1, 1).to(grasp_pose)
		gripper_SE3s[:, :3, 3] = placing_grid

		# get afterimage gripper object pc and 
		pc_gripper = (
			grasp_pose[:3, :3] @ self.gripper_pc_open.T 
			+ grasp_pose[:3, 3].unsqueeze(-1)).T
		pc_object = (
			object_pc.unsqueeze(0)
			+ torch.linspace(0, self.approach_distance, self.n_afterimage).reshape(self.n_afterimage, 1, 1).to(grasp_pose)
			* direction.reshape(1, 1, 3)
		).reshape(-1, 3)
		pc_total = torch.cat([pc_object, pc_gripper], dim=0)	

		# transform pc to canonical coordinate
		new_pose = torch.zeros(4, 4).to(grasp_pose)
		new_pose[0:3, 0:3] = torch.tensor([
			[torch.cos(RotZ_angle), -torch.sin(RotZ_angle), 0.],
			[torch.sin(RotZ_angle),  torch.cos(RotZ_angle), 0.],
			[0., 0., 1.]]).T.to(grasp_pose)
		new_pose[0:3, 3] = (
			(torch.eye(3).to(grasp_pose) - new_pose[0:3, 0:3]) 
			@ object_pose[:3, 3]
		)
		pc_total = (
			new_pose[:3, :3] @ pc_total.T 
			+ new_pose[:3, 3].unsqueeze(-1)
			- object_pose[:3, 3].unsqueeze(-1)).T

		# check collision
		sdfs = self.get_sdfs_of_pc(
			gripper_SE3s, pc_total, recog_results_wo_target)

		# calculate score
		scores = sdfs - 1

		# reachable poses
		placing_grid = placing_grid[scores > 0]
		n_placeable = len(placing_grid)

		# update object poses
		placeable_poses = torch.eye(4).unsqueeze(0).repeat(n_placeable, 1, 1).to(grasp_pose)
		placeable_poses[:, :3, :3] = (new_pose[:3, :3] @ object_pose[:3, :3]).unsqueeze(0).repeat(n_placeable, 1, 1)
		placeable_poses[:, :3, 3] = placing_grid

		return placeable_poses

	def sort_rearranging_actions(self, actions, recog_results):

		# graspability scores
		graspability_scores = []

		for action in actions:
			
			# initialize
			updated_recog_results = deepcopy(recog_results)
			action_type = action['action_type']

			# pick and place action
			if action_type == 'pick_and_place':
					
				# load infos
				object_idx = action['object_idx']
				place_pose = action['place_pose']

				# update scene
				updated_recog_results[object_idx].SE3 = place_pose
				updated_recog_results[object_idx].construct()

				# get target object graspability
				graspablility_score, _, _, _ = self.get_target_object_graspability(
					updated_recog_results
				)
				graspability_scores.append(graspablility_score)

		# sort graspability score
		graspability_scores = torch.tensor(graspability_scores)
		sorted_graspability_scores = graspability_scores.argsort(descending=False)
		sorted_actions = [
			actions[i] for i in sorted_graspability_scores
		]
		graspability_scores = graspability_scores[
			sorted_graspability_scores]

		return sorted_actions, graspability_scores.tolist()

	def update_recog_results(self, recog_results, action):
		
		# action type
		action_type = action['action_type']

		# pick and place action
		if action_type == 'pick_and_place':
				
			# load infos
			object_idx = action['object_idx']
			place_pose = action['place_pose']

			# update scene
			recog_results[object_idx].SE3 = place_pose
			recog_results[object_idx].construct()

		return recog_results

	def execute_rearranging_sim(
			self, sorted_joint_angles, sorted_actions, action_types, 
			recog_results, return_action=False):
				
		for joint_angles, action_info, action_type in zip(sorted_joint_angles, sorted_actions, action_types):
			
			if action_type == 'pick_and_place':
				success_rearranging = self.env.sim.pick_and_place(
					joint_angles,
					speed=self.speed,
					speed_gripper=self.speed_gripper, 
					grasp_force=self.grasp_force)

			# check if success
			if success_rearranging:
				time.sleep(1)
				if self.recog_type == 't2sqnet_gt':
					self.env.update_object_poses()
				
				# save action
				if self.recog_type == 't2sqnet_rgb':
					
					# data
					data = {
						'action_type': action_type,
						'joint_angles': joint_angles.detach().cpu().numpy().astype(np.float64),
						'furniture_info': self.furniture_info,
						'objects_info': self.get_objects_info(recog_results),
						'action_info': action_info
					}
					
					np.save(
						os.path.join(self.save_folder, str(self.iter), 'action'), 
						data)		
				
				if return_action:
					return True, action_info
				else:
					return True
			else:
				self.env.sim.robot_go_home()
				self.env.reset_object_poses()
				print('failed! try another grasp pose.')

		return False, action_info

	def execute_rearranging_real(
			self, sorted_joint_angles, sorted_actions, 
			action_types, recog_results, return_action=False, 
			receive_success=True):

		# currently pick one
		joint_angles = sorted_joint_angles[0]
		action_info = sorted_actions[0]
		action_type = action_types[0]

		# send data
		data = self.listener.recv_vision()
		if not data == b'request_action':
			raise ValueError('invalid receved data; must be "request_action"')
		data = {
			'action_type': action_type,
			'joint_angles': joint_angles.detach().cpu().numpy().astype(np.float64),
			'furniture_info': self.furniture_info,
			'objects_info': self.get_objects_info(recog_results)
		}
		self.listener.send_grasp(data)

		# save action
		if self.recog_type == 't2sqnet_rgb':
			data['action_info'] = action_info
			np.save(
				os.path.join(self.save_folder, str(self.iter), 'action'), 
				data)		

		# receive success
		if receive_success:
			data = self.listener.recv_vision()
			if data == b'success':
				self.listener.send_grasp('success')
				if return_action:
					return True, action_info
				else:
					return True

			elif data == b'failed':
				self.listener.send_grasp('failed')
				if return_action:
					return False, action_info
				else:
					return False
		else:
			return True, action_info

	#############################################################
	########################## UTILS ############################
	#############################################################

	def grasp_pose_generation(
			self, recog_results, 
			return_modified_score=False):
			
		# superquadric
		if self.recog_type in ['t2sqnet_gt', 't2sqnet_rgb']:
			
			# list
			grasp_poses_list = []
			
			# get grasp poses
			for i, obj in enumerate(recog_results):	
				obj_grasp_poses = obj.get_grasp_poses(
					desired_dir_sq=torch.tensor([1, 0, 0]), 
					dir_angle_bound_sq=self.angle_bound,
					flip_sq=True,
					desired_dir_sp=torch.tensor([1, 0, 0]), 
					dir_angle_bound_sp=self.angle_bound,
					flip_sp=False)
				grasp_poses_list.append(obj_grasp_poses)

			# check collision
			results = self.check_collision_gripper_poses(
				grasp_poses_list, recog_results, 
				return_modified_score=return_modified_score,
				gripper_pc=self.gripper_pc_open
			)

			# return
			if return_modified_score:
				return grasp_poses_list, results[0], results[1]
			else:
				return grasp_poses_list, results
		else:
			raise NotImplementedError

	def check_collision_gripper_poses(
			self, gripper_poses_list, recog_results, gripper_pc=None,
			return_modified_score=False):
		'''
		input:	gripper_poses: list of (n x 4 x 4) 
				recog_results: list of (tableware object)
		output: scores: list of (n)
		'''

		# initialize
		scores_list = []
		if return_modified_score:
			modified_scores_list = []

		# object index
		for i, grasp_poses in enumerate(gripper_poses_list):

			# check collision without target
			recog_results_wo_target = recog_results[:]
			recog_results_wo_target.pop(i)

			# get sdf values
			sdfs = self.get_sdfs_of_pc(
				grasp_poses, gripper_pc, recog_results_wo_target)
			scores = sdfs - 1
			scores_list.append(scores)

			# get sdf values for modified collision score
			if return_modified_score:
				sdfs = self.get_sdfs_of_pc(
					grasp_poses, gripper_pc, recog_results_wo_target, 
					return_min=False)
				modified_scores = (sdfs - 1 <= 0).sum(dim=0)
				modified_scores_list.append(modified_scores)

		# return scores
		if return_modified_score:
			return scores_list, modified_scores_list
		else:
			return scores_list

	def get_sdfs_of_pc(
			self, gripper_poses, pc, recog_results,
			return_min=True, debug=False):

		# number of grippers
		ndim_gripper_poses = gripper_poses.ndim
		if ndim_gripper_poses == 2:
			gripper_poses = gripper_poses.unsqueeze(0)
		elif ndim_gripper_poses == 3:
			pass
		else:
			raise ValueError('check gripper_poses shape!')

		# gripper point cloud
		target_pc = deepcopy(pc).permute(1,0).unsqueeze(0)
		target_pc = (
			gripper_poses[:, :3, :3] @ target_pc 
			+ gripper_poses[:, :3, 3].unsqueeze(-1)
		).permute(0,2,1) # n_gr x n_pc x 3

		# for debugging
		if debug:
			for i in range(min(len(target_pc), 5)):
				render_pc_with_objects(
					recog_results,
					target_pc[i],
					furniture=self.furniture_info)

		# flatten
		if ndim_gripper_poses == 2:
			target_pc = target_pc.squeeze(0)
		if ndim_gripper_poses == 3:
			n_gripper = target_pc.shape[0]
			n_pc = target_pc.shape[1]
			target_pc = target_pc.reshape(-1, 3)

		# get sdf value
		sdfs = []
		for obj in self.furniture_sq_values:
			sdf = obj.get_sdf_value(target_pc).unsqueeze(0)
			sdfs.append(sdf)
		for obj in recog_results:
			sdf = obj.get_sdf_values(target_pc, mode='1')
			sdfs.append(sdf)
		sdfs = torch.cat(sdfs, dim=0) # n_env x (n_gripper x n_pc)
		n_env = len(sdfs)

		# reshape sdf
		if ndim_gripper_poses == 2:
			sdfs = sdfs.min(dim=-1)[0]
			if return_min:
				sdfs = sdfs.min()
		elif ndim_gripper_poses == 3:
			sdfs = sdfs.reshape(n_env, n_gripper, n_pc)
			sdfs = sdfs.min(dim=-1)[0]
			if return_min:
				sdfs = sdfs.min(dim=0)[0]

		return sdfs

	def get_2D_grid(self, resolution=0.015, dtype='torch'):
		
		# get workspace bounds
		workspace_bounds = torch.from_numpy(self.env.sim.workspace_bounds).float()

		# get the number of grids
		size_x = torch.round(
			(workspace_bounds[0, 1] - workspace_bounds[0, 0]) / resolution
		).type(torch.int)
		size_y = torch.round(
			(workspace_bounds[1, 1] - workspace_bounds[1, 0]) / resolution
		).type(torch.int)
		
		# meshgrid
		x = torch.linspace(workspace_bounds[0, 0], workspace_bounds[0, 1], size_x)
		y = torch.linspace(workspace_bounds[1, 0], workspace_bounds[1, 1], size_y)
		X, Y = torch.meshgrid(x, y, indexing='ij')
		
		# reshape grids
		X = X.reshape(-1, 1)
		Y = Y.reshape(-1, 1)
		grid = torch.cat([X, Y], dim=1)

		return grid
	
	def get_joint_angles_for_observation(self):
		
		if 'table' in self.sim_type:
			joint_angles = np.array([
				[ 0.3319, -1.6649,  1.0888, -2.4297, -0.0565,  1.5136,  1.9762],
				[ 0.8750, -1.6790,  1.0121, -2.6250, -0.2410,  1.3700,  2.0308],
				[ 1.4630, -1.6821,  0.9663, -2.6826, -0.4058,  1.0484,  2.0895],
				[ 1.9262, -1.6762,  1.0129, -2.5899, -0.4765,  0.6868,  2.0828],
				[-0.9368, -1.6801, -1.0108, -2.6279,  0.2049,  1.4774, -0.4347],
				[-0.4907, -1.4720, -1.1088, -2.4095,  0.1470,  1.5808, -0.5495],
				[-0.0394, -1.3357, -1.2348, -2.0935,  0.1119,  1.5601, -0.6202]
			])

		elif 'shelf' in self.sim_type:
			joint_angles = np.array([
				[-0.1705, -1.6906,  0.7590, -2.0262,  0.0104,  1.5099,  1.5680],
				[ 0.2919, -1.7100,  0.6964, -2.2598, -0.2128,  1.5327,  1.5964],
				[ 0.8359, -1.7127,  0.6326, -2.3710, -0.4939,  1.3856,  1.6585],
				[ 1.3745, -1.7127,  0.6407, -2.3503, -0.7275,  1.0935,  1.6679],
				[ 1.7727, -1.7033,  0.7184, -2.2227, -0.8926,  0.8034,  1.5665],
				[ 1.9988, -1.6807,  0.8043, -2.0108, -1.0613,  0.5941,  1.3526],
				[ 2.0736, -1.6615,  0.8405, -1.7295, -1.2782,  0.4904,  1.0246]
			])

		return joint_angles