import math
import threading
import time
import copy
import torch
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
from functions.utils import matrices_to_quats
from control.sim_with_shelf import PybulletShelfSim

# for custom inverse kinematics
from robot.openchains_lib import Franka
from robot.openchains_torch import inverse_kinematics

class PybulletShelfRobotSim(PybulletShelfSim):
	def __init__(self, enable_gui, open_shelf):
		super().__init__(enable_gui, open_shelf)
  
		# Add Franka Panda Emika robot
		robot_path = 'assets/panda/panda_with_gripper.urdf'
		self._robot_body_id = p.loadURDF(robot_path, [0.0, 0.0, 0.0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

		# Get revolute joint indices of robot (skip fixed joints)
		robot_joint_info = [p.getJointInfo(self._robot_body_id, i) for i in range(p.getNumJoints(self._robot_body_id))]
		self._robot_joint_indices = [x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
		self._robot_joint_lower_limit = [x[8] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
		self._robot_joint_upper_limit = [x[9] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
		self._finger_joint_indices = [8, 9]
		self._joint_epsilon = 0.01  # joint position threshold in radians for blocking calls (i.e. move until joint difference < epsilon)

		# Move robot to home joint configuration
		self._robot_home_joint_config = [
			0.0301173169862714, 
			-1.4702106391932968, 
			0.027855688427362513, 
			-2.437557753144649, 
			0.14663284881434122, 
			2.308719465520647, 
			0.7012385825324389]
		self.robot_go_home()

		# robot end-effector index
		self._robot_EE_joint_idx = 7
		self._robot_tool_joint_idx = 9
		self._robot_tool_tip_joint_idx = 9

		# Set friction coefficients for gripper fingers
		p.changeDynamics(
			self._robot_body_id, 7,
			lateralFriction=1, # 0.1
			spinningFriction=1, # 0.1
			rollingFriction=1,
			frictionAnchor=True
		)
		p.changeDynamics(
			self._robot_body_id, 8,
			lateralFriction=1, # 0.1
			spinningFriction=1, # 0.1
			rollingFriction=1,
			frictionAnchor=True
		)
		p.changeDynamics(
			self._robot_body_id, 9,
			lateralFriction=1, # 0.1
			spinningFriction=1, # 0.1
			rollingFriction=1,
			frictionAnchor=True
		)

	#############################################################
	###################### INVERSE KINEMATICS ###################
	#############################################################

	def get_franka_for_ik(self, device='cpu'):
		
		# declare franka for inverse kinematics
		self.device = device
		self.franka_for_ik = Franka(device=device)

	def check_ik(
			self, SE3, 
			initial_joint_state=None,
			thld=1e-3, use_pybullet_ik=False):
		
		# use pybullet inverse kinematics
		if use_pybullet_ik:
				
			# solve ik
			SE3 = SE3.cpu().numpy()
			position = deepcopy(SE3[:3, 3])
			orientation = matrices_to_quats(SE3[:3, :3])			
			target_joint_state = np.array(
				p.calculateInverseKinematics(
					self._robot_body_id,  
					self._robot_EE_joint_idx, 
					position, 
					orientation,
					maxNumIterations=100000, 
					residualThreshold=0.0001,
					lowerLimits=self._robot_joint_lower_limit,
					upperLimits=self._robot_joint_upper_limit
			))
			target_joint_state = target_joint_state[:len(self._robot_joint_indices)]

			# check if success
			for i in self._robot_joint_indices:
				p.resetJointState(self._robot_body_id, i, target_joint_state[i])
			ee_state = p.getLinkState(self._robot_body_id, 7)
			ee_quat = np.array(ee_state[5])
			ee_pos = np.array(ee_state[4])
			success = (np.linalg.norm(position - ee_pos) < thld) * (np.linalg.norm(orientation - ee_quat) < thld)

		# use custom inverse kinematics (non-batch)
		else:

			# initialize
			if initial_joint_state is None:
				target_joint_state = deepcopy(self._robot_home_joint_config)
			else:
				target_joint_state = initial_joint_state
			
			# solve ik
			SE3 = SE3.unsqueeze(0).float()
			target_joint_state = torch.tensor(
				target_joint_state).unsqueeze(0).to(SE3)
			target_joint_state, dict_infos = inverse_kinematics(
				target_joint_state, 
				SE3,
				self.franka_for_ik,
				max_iter=5000,
				step_size1=0.01,
				step_size2=0.001,
				tolerance=thld,
				device=self.device)
			target_joint_state = target_joint_state.squeeze().tolist()

			# check if success
			success = (dict_infos['final_error'] < thld) * dict_infos['joint limit check']
			success = bool(success.item())

		return target_joint_state, success

	def solve_batchwise_ik(
			self, SE3s, 
			initial_joint_state=None, thld=1e-4,
			max_iter=1000, step_size1=0.1, step_size2=0.001,
			bracket_offset=0.0,	locked_joint_7=None):
	
		# initialize
		if initial_joint_state is None:
			target_joint_state = deepcopy(self._robot_home_joint_config)
		else:
			target_joint_state = initial_joint_state
		
		# consider bracket offset
		SE3s[:, :3, 3] = SE3s[:, :3, 3] - bracket_offset * SE3s[:, :3, 2]

		# locked_joints
		locked_joints = [None] * 6 + [locked_joint_7]

		# solve ik
		target_joint_states = torch.tensor(
			target_joint_state).unsqueeze(0).repeat(len(SE3s), 1).to(SE3s)
		target_joint_states, dict_infos = inverse_kinematics(
			target_joint_states, 
			SE3s,
			self.franka_for_ik,
			max_iter=max_iter,
			step_size1=step_size1,
			step_size2=step_size2,
			tolerance=thld,
			locked_joints=locked_joints,
			device=self.device)

		# check if success
		successes = (dict_infos['final_error'] < thld) * dict_infos['joint limit check']
		successes = successes.to(torch.bool)

		return target_joint_states, successes

	def solve_ik_grasping(
			self, SE3s, 
			initial_joint_state=None, thld=1e-4,
			approach_distance=0.2, n_segment=3,
			max_iter=1000, step_size1=0.1, step_size2=0.001,
			bracket_offset=0.008, locked_joint_7=None):

		# get approach poses
		SE3s = SE3s.unsqueeze(0).repeat(n_segment+2, 1, 1, 1)
		approach_vector = approach_distance * torch.linspace(
			1, 0, n_segment+2).unsqueeze(-1).unsqueeze(-1).to(SE3s)
		SE3s[:, :, :3, 3] = (SE3s[:, :, :3, 3]
			- approach_vector * SE3s[:, :, :3, 2]
		)
		n_approach = SE3s.shape[0]
		n_grasp = SE3s.shape[1]
		SE3s = SE3s.reshape(-1, 4, 4)

		# solve ik
		target_joint_states, successes = self.solve_batchwise_ik(
			SE3s, initial_joint_state=initial_joint_state, thld=thld,
			max_iter=max_iter, step_size1=step_size1, step_size2=step_size2,
			bracket_offset=bracket_offset,
			locked_joint_7=locked_joint_7
		)

		# check if success
		target_joint_states = target_joint_states.reshape(n_approach, n_grasp, -1)
		target_joint_states = target_joint_states.permute(1, 0, 2)
		successes = successes.reshape(n_approach, n_grasp)
		successes = successes.prod(dim=0).to(torch.bool)

		return target_joint_states, successes

	def solve_ik_rearranging(
			self, actions, 
			device='cpu',
			initial_joint_state=None, thld=1e-4,
			approach_distance=0.2, n_segment=3,
			max_iter=1000, step_size1=0.1, step_size2=0.001,
			bracket_offset=0.008, locked_joint_7=None):

		# load action
		SE3s = []
		action_types = []
		for action in actions:
			
			# load info
			action_type = action['action_type']

			# pick and place action
			if action_type == 'pick_and_place':
				
				# load infos
				object_pose = action['object_pose']
				place_pose = action['place_pose']
				grasp_gripper_pose = action['grasp_pose']

				# place gripper pose
				place_gripper_pose = (
					place_pose @ torch.inverse(object_pose) @ grasp_gripper_pose)
				SE3 = torch.cat(
					[grasp_gripper_pose.unsqueeze(0), place_gripper_pose.unsqueeze(0)],
					dim=0
				)

				# append
				SE3s.append(SE3)
				action_types.append(action_type)

		# get total SE3s
		SE3s = torch.stack(SE3s)

		# get approach poses
		SE3s = SE3s.unsqueeze(0).repeat(n_segment+2, 1, 1, 1, 1)
		approach_vector = approach_distance * torch.linspace(
			1, 0, n_segment+2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(SE3s)
		SE3s[:, :, :, :3, 3] = (SE3s[:, :, :, :3, 3]
			- approach_vector * SE3s[:, :, :, :3, 2]
		)
		SE3s = torch.cat([
			SE3s[:, :, 0, :, :], SE3s[:, :, 1, :, :]
		], dim=0)
		n_approach = SE3s.shape[0]
		n_action = SE3s.shape[1]
		SE3s = SE3s.reshape(-1, 4, 4).to(device)

		# solve ik
		target_joint_states, successes = self.solve_batchwise_ik(
			SE3s, initial_joint_state=initial_joint_state, thld=thld,
			max_iter=max_iter, step_size1=step_size1, step_size2=step_size2,
			bracket_offset=bracket_offset, locked_joint_7=locked_joint_7
		)

		# check if success
		target_joint_states = target_joint_states.reshape(n_approach, n_action, -1)
		target_joint_states = target_joint_states.permute(1, 0, 2)
		successes = successes.reshape(n_approach, n_action)
		successes = successes.prod(dim=0).to(torch.bool)

		return target_joint_states, action_types, successes

	#############################################################
	#################### CONTROLLER - SIMULATION ################
	#############################################################

	def robot_go_home(self, blocking=True, speed=0.1):
		self.move_joints(self._robot_home_joint_config, blocking, speed)
			
	def reset_robot(self):
		for i in self._robot_joint_indices:
			p.resetJointState(self._robot_body_id, i, self._robot_home_joint_config[i])
   
	def move_robot_joint(self, jointangles):
		for i in self._robot_joint_indices:
			p.resetJointState(self._robot_body_id, i, jointangles[i])

	# object grasping with ik
	def object_grasping(
			self, joint_angles, 
			speed=0.001, speed_gripper=0.01, gripper_width=0.08,
			grasp_force=100, blocking=True
		):

		################
		## CONTROLLER ##
		################

		# open gripper
		self.move_gripper(gripper_width/2, speed=speed_gripper, blocking=blocking)
		# approach
		for joint_angle in joint_angles:
			joint_angle = joint_angle.cpu().tolist()
			if not self.move_joints(joint_angle, speed=speed, blocking=blocking):
				return False
		# grasp
		self.grasp_object(force=grasp_force, blocking=blocking)
		# retrieve
		for joint_angle in joint_angles.flip(dims=(0,)):
			joint_angle = joint_angle.cpu().tolist()
			if not self.move_joints(joint_angle, speed=speed, blocking=blocking):
				return False
		# go to home
		self.robot_go_home(speed=speed)
		# open gripper
		self.move_gripper(0.04, speed=speed_gripper, blocking=blocking)

		return True

	# pick and place with ik
	def pick_and_place(
			self, joint_angles, 
			speed=0.001, speed_gripper=0.01, 
			grasp_force=100, blocking=True
		):

		################
		## CONTROLLER ##
		################

		# number of segments
		n_segment = int(len(joint_angles) / 2)

		# open gripper
		self.move_gripper(0.04, speed=speed_gripper, blocking=blocking)
		# approach
		for joint_angle in joint_angles[:n_segment]:
			joint_angle = joint_angle.cpu().tolist()
			if not self.move_joints(joint_angle, speed=speed, blocking=blocking):
				return False
		# grasp
		self.grasp_object(force=grasp_force, blocking=blocking)
		# retrieve
		for joint_angle in joint_angles[:n_segment].flip(dims=(0,)):
			joint_angle = joint_angle.cpu().tolist()
			if not self.move_joints(joint_angle, speed=speed, blocking=blocking):
				return False
		# go to home
		self.robot_go_home(speed=speed)
		# go to place
		for joint_angle in joint_angles[n_segment:]:
			joint_angle = joint_angle.cpu().tolist()
			if not self.move_joints(joint_angle, speed=speed, blocking=blocking):
				return False
		# place
		self.move_gripper(0.04, speed=speed_gripper, blocking=blocking)
		# go back
		for joint_angle in joint_angles[n_segment:].flip(dims=(0,)):
			joint_angle = joint_angle.cpu().tolist()
			if not self.move_joints(joint_angle, speed=speed, blocking=blocking):
				return False
		# go to home
		self.robot_go_home(speed=speed)

		return True

	#############################################################
	#################### ACTION PRIMITIVES ######################
	#############################################################

	# move joints
	def move_joints(
			self, target_joint_state, 
			blocking=False, speed=0.03):
		
		# move joints
		p.setJointMotorControlArray(
			self._robot_body_id, 
			self._robot_joint_indices,
			p.POSITION_CONTROL, 
			target_joint_state,
			positionGains=speed * np.array([1, 1, 1, 1, 1, 2, 1])
   		)

		# Block call until joints move to target configuration
		if blocking:
			actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._robot_joint_indices]
			timeout_t0 = time.time()
			while not all([np.abs(actual_joint_state[i] - target_joint_state[i]) < self._joint_epsilon for i in
						   range(6)]):
				if time.time() - timeout_t0 > 3:
					return False
				actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._robot_joint_indices]
				time.sleep(0.001)
			return True
		else:
			return 1

	# move in cartesian coordinates
	def move_carts(
			self, SE3,
			blocking=False, speed=0.03, 
			initial_joint_state=None, return_joint_angle=False):
		
		# inverse kinematics
		target_joint_state, success = self.check_ik(
			SE3, initial_joint_state=initial_joint_state
		)

		# check ik is solved
		if not success:
			if return_joint_angle:
				return success, target_joint_state
			else:
				return success
	
		# move joints
		if return_joint_angle:
			return self.move_joints(
				target_joint_state, 
				blocking=blocking, 
				speed=speed
			), target_joint_state
		else:
			return self.move_joints(
				target_joint_state, 
				blocking=blocking, 
				speed=speed
			)

	# move in cartesian coordinates with "straight line"
	def move_carts_straight(
			self, SE3, 
			blocking=False, speed=0.03, return_joint_angle=False,
			n_segment=10):

		# define segments of straight line motion
		ee_state = p.getLinkState(
			self._robot_body_id, 
			self._robot_EE_joint_idx
		)
		position_initial = deepcopy(np.array(ee_state[4]))
		position_initial = torch.tensor(position_initial).to(SE3)
		
		# inverse kinematics of segments
		target_joint_state_list = []
		initial_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._robot_joint_indices]
		for i in range(n_segment):
			
			# get SE3 segment
			SE3_seg = deepcopy(SE3)
			SE3_seg[:3, 3] = (
				position_initial 
				+ (SE3_seg[:3, 3] - position_initial) 
				* (i + 1) / n_segment)

			# inverse kinematics
			results = self.move_carts(
				SE3_seg, 
				blocking=blocking, speed=speed, 
				initial_joint_state=initial_joint_state,
				return_joint_angle=return_joint_angle)
			
			# move joints
			if return_joint_angle:
				if results[0]:
					target_joint_state_list.append(results[1])
					initial_joint_state = results[1]
				else:
					return False, target_joint_state_list
			else:
				if not results:
					return False

		# results
		if return_joint_angle:
			return True, target_joint_state_list
		else:
			return True

	# move gripper
	def move_gripper(self, target_width, blocking=False, speed=0.03):
		
		# target joint state
		target_joint_state = np.array([target_width, target_width])
		
		# Move joints
		p.setJointMotorControlArray(
			self._robot_body_id, 
			self._finger_joint_indices,
			p.POSITION_CONTROL, 
			target_joint_state,
			positionGains=speed * np.ones(len(self._finger_joint_indices))
		)

		# Block call until joints move to target configuration
		if blocking:
			actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._finger_joint_indices]
			timeout_t0 = time.time()
			while not all([np.abs(actual_joint_state[i] - target_joint_state[i]) < self._joint_epsilon for i in range(len(actual_joint_state))]):
				if time.time() - timeout_t0 > 5:
					break
				actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._finger_joint_indices]
				time.sleep(0.001)

	# grasp the object with force
	def grasp_object(self, blocking=False, force=200):
		
		# target joint state
		target_joint_state = np.array([0.0, 0.0])
		forces = np.array([force, force])
		
		# Move joints
		p.setJointMotorControlArray(
			self._robot_body_id, 
			self._finger_joint_indices,
			p.POSITION_CONTROL, 
			target_joint_state,
			forces=forces
		)

		# Block call until joints move to target configuration
		if blocking:
			timeout_t0 = time.time()
			while True:
				if time.time() - timeout_t0 > 1:
					break
				time.sleep(0.001)

	#############################################################
	######################## TEMPORARY ##########################
	#############################################################

	# # object grasping with ik
	# def object_grasping(
	# 		self, SE3, 
	# 		approach_distance=0.2, 
	# 		speed=0.001, straight_speed=0.01, speed_gripper=0.01, 
	# 		grasp_force=100, blocking=True
	# 	):

	# 	# get poses
	# 	SE3_init = deepcopy(SE3)
	# 	SE3_init[:3, 3] = SE3_init[:3, 3] - approach_distance * SE3_init[:3, 2]

	# 	################
	# 	## CONTROLLER ##
	# 	################

	# 	# open gripper
	# 	self.move_gripper(0.04, speed=speed_gripper, blocking=blocking)
	# 	# before approach
	# 	if not self.move_carts(SE3_init, speed=speed, blocking=blocking):
	# 		return False
	# 	# approach
	# 	if not self.move_carts_straight(SE3, speed=straight_speed, blocking=blocking):
	# 		return False
	# 	# grasp
	# 	self.grasp_object(force=grasp_force, blocking=blocking)
	# 	# retrieve
	# 	if not self.move_carts_straight(SE3_init, speed=straight_speed,	blocking=blocking):
	# 		return False
	# 	# go to home
	# 	self.robot_go_home(speed=speed)
	# 	return True