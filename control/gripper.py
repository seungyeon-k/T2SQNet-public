import numpy as np
import torch
import open3d as o3d
from copy import deepcopy
from functions.utils import get_SE3s, exp_so3

class Gripper:
	def __init__(
			self, SE3, width=0, contain_camera=False, locked_joint_7=None,
			mesh_with_links=False):
		
		# initialize
		self.hand_SE3 = SE3
		self.gripper_width = width
		self.contain_camera = contain_camera
		self.locked_joint_7 = locked_joint_7
		if locked_joint_7 is None:
			self.locked_joint_7 = np.pi / 4
		if width < 0:
			print("gripper width exceeds minimum width. gripper width is set to 0")
			self.gripper_width = 0
		if width > 0.08:
			print("gripper width exceeds maximum width. gripper width is set to 0.08")
			self.gripper_width = 0.08

		# load meshes
		self.hand = o3d.io.read_triangle_mesh("assets/gripper/hand.ply")
		self.hand.compute_vertex_normals()
		self.hand.paint_uniform_color([0.9, 0.9, 0.9])
		self.finger1 = o3d.io.read_triangle_mesh("assets/gripper/finger.ply")
		self.finger1.compute_vertex_normals()
		self.finger1.paint_uniform_color([0.7, 0.7, 0.7])
		self.finger2 = o3d.io.read_triangle_mesh("assets/gripper/finger.ply")
		self.finger2.compute_vertex_normals()
		self.finger2.paint_uniform_color([0.7, 0.7, 0.7])

		# transform
		self.finger1_M = get_SE3s(np.identity(3), np.array([0, self.gripper_width/2, 0.1654/3]))
		self.finger2_M = get_SE3s(exp_so3(np.asarray([0, 0, 1]) * np.pi), np.array([0, -self.gripper_width/2, 0.1654/3]))
		self.finger1_SE3 = np.dot(self.hand_SE3, self.finger1_M)
		self.finger2_SE3 = np.dot(self.hand_SE3, self.finger2_M)			
		self.hand.transform(self.hand_SE3)
		self.finger1.transform(self.finger1_SE3)
		self.finger2.transform(self.finger2_SE3)
		self.mesh = self.hand + self.finger1 + self.finger2
		self.mesh.compute_vertex_normals()

		# for reachability
		if self.contain_camera:
			d = 0.008
		else:
			d = 0.0
		self.link6 = o3d.io.read_triangle_mesh("assets/gripper/link6.ply")
		self.link6.compute_vertex_normals()
		self.link6.paint_uniform_color([0.9, 0.9, 0.9])
		self.link6.rotate(
			self.link6.get_rotation_matrix_from_xyz(
				(- np.pi / 2, 0, 0)
			),
			center=(0, 0, 0)
		)
		self.link6.rotate(
			self.link6.get_rotation_matrix_from_xyz(
				(0, 0, - (self.locked_joint_7 - np.pi / 4))
			),
			center=[0.088, 0, 0]
		)
		self.link6.translate([-0.088, 0, - 0.107 - d])

		self.link7 = o3d.io.read_triangle_mesh("assets/gripper/link7.ply")
		self.link7.compute_vertex_normals()
		self.link7.paint_uniform_color([0.9, 0.9, 0.9])
		self.link7.rotate(
			self.link7.get_rotation_matrix_from_xyz(
				(0, 0, np.pi / 4)
			),
			center=(0, 0, 0)
		)
		self.link7.translate([0, 0, - 0.107 - d])

		# camera mesh
		if self.contain_camera:
			self.bracket = o3d.io.read_triangle_mesh("assets/panda/visual/bracket.obj")
			self.bracket.compute_vertex_normals()
			self.bracket.paint_uniform_color([0.3, 0.3, 0.3])
			self.bracket.translate([0, 0, - 0.002 - d])
			
			self.camera = o3d.io.read_triangle_mesh("assets/panda/visual/d435.obj")
			self.camera.compute_vertex_normals()
			self.camera.paint_uniform_color([0.6, 0.6, 0.6])
			roll = np.radians(0)
			pitch = np.radians(0)
			yaw = np.radians(-90)
			R = rpy_to_rotation_matrix(roll, pitch, yaw)
			self.camera = self.camera.rotate(R, center=np.array([0,0,0]))
			self.camera = self.camera.translate(np.array([0.069, 0, 0.01]))
			self.camera = self.camera.translate(np.array([0, 0, - 0.002 - d]))
		
		# mesh
		if mesh_with_links:
			link6_mesh = deepcopy(self.link6)
			link7_mesh = deepcopy(self.link7)
			bracket_mesh = deepcopy(self.bracket)
			camera_mesh = deepcopy(self.camera)
			link6_mesh.transform(self.hand_SE3)
			link7_mesh.transform(self.hand_SE3)
			bracket_mesh.transform(self.hand_SE3)
			camera_mesh.transform(self.hand_SE3)
			self.mesh = (
				self.mesh + link6_mesh + link7_mesh 
				+ bracket_mesh + camera_mesh
			)
			self.mesh.compute_vertex_normals()
		
	def get_gripper_afterimage_pc(
			self,
			pc_dtype='numpy',
			number_of_points=2048,
			distance=0.2,
			n_gripper=5,
			debug=False
		):

		# initialize
		gripper_mesh = deepcopy(self.mesh)
		link6_mesh = deepcopy(self.link6)
		link7_mesh = deepcopy(self.link7)

  		# camera (realsense)
		if self.contain_camera:
			gripper_mesh += self.bracket
			gripper_mesh += self.camera

		# approaching
		for i in range(n_gripper):
			gripper_afterimage_mesh = deepcopy(gripper_mesh)
			link6_afterimage_mesh = deepcopy(link6_mesh)
			link7_afterimage_mesh = deepcopy(link7_mesh)
			z_distance = - distance * i / (n_gripper - 1)
			gripper_afterimage_mesh.translate([0, 0, z_distance])
			link6_afterimage_mesh.translate([0, 0, z_distance])
			link7_afterimage_mesh.translate([0, 0, z_distance])
			if i == 0:
				gripper_mesh_total = gripper_afterimage_mesh
			else:
				gripper_mesh_total += gripper_afterimage_mesh
			gripper_mesh_total += link6_afterimage_mesh
			gripper_mesh_total += link7_afterimage_mesh

		# debug
		if debug:
			coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
			o3d.visualization.draw_geometries([gripper_mesh_total, coord])

		# sample gripper pc
		gripper_pc_total = gripper_mesh_total.sample_points_uniformly(
			number_of_points=number_of_points
		)
		gripper_pc_total.paint_uniform_color([0.5, 0.5, 0.5])

		# debug
		if debug:
			coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
			o3d.visualization.draw_geometries([gripper_pc_total, coord])

		# data type
		if pc_dtype == 'numpy':
			pc = np.asarray(gripper_pc_total.points)
		elif pc_dtype == 'torch':
			pc = torch.tensor(np.asarray(gripper_pc_total.points)).float()

		# return type
		return pc

def rpy_to_rotation_matrix(roll, pitch, yaw):
	# Calculate the cosine and sine of each angle
	cr = np.cos(roll)
	sr = np.sin(roll)
	cp = np.cos(pitch)
	sp = np.sin(pitch)
	cy = np.cos(yaw)
	sy = np.sin(yaw)

	# Define the rotation matrix components
	R = np.array([
		[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
		[sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
		[-sp, cp * sr, cp * cr]
	])
	
	return R