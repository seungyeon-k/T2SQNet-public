from functions.utils_torch import get_SE3s_torch
from .primitives import SuperQuadric
import torch
import numpy as np
import open3d as o3d

from tablewarenet.primitive_grasp_planner import (
	superellipsoid_grasp_planner,
	bent_superellipsoid_grasp_planner,
	superparaboloid_grasp_planner
)
	
class Tableware():
	def __init__(self, SE3, params, range, device='cpu', t=0.01, process_mesh=True):
		self.quadrics = []
		self.SE3 = SE3.float()
		self.range_torch = torch.tensor([range[key] for key in range]).to(device).float()
		self.device = device
		self.params = params
		if self.params  == 'random':
			self.params = torch.rand(len(self.range_torch)).to(self.device) * self.range_torch.diff(dim=1).squeeze() + self.range_torch[:,0]
		self.params = self.params.float()
		self.t = t
		self.process_mesh = process_mesh
		self.construct()
	
	def construct(self):
		pass
	
	def send_to_device(self, device):
		self.device = device
		self.SE3 = self.SE3.to(device)
		self.range_torch = self.range_torch.to(device)
		self.params = self.params.to(device)
		self.construct()

	def params_ranger(self):
		self.params = torch.sigmoid(self.params) * self.range_torch.diff(dim=1).squeeze() + self.range_torch[:,0]

	def params_deranger(self):
		self.params = torch.logit((self.params - self.range_torch[:,0]) / self.range_torch.diff(dim=1).squeeze(), eps=1e-6)
	
	def get_mesh(self, resolution=20, transform=True):
		for i, quadric in enumerate(self.quadrics):
			if i == 0:
				mesh = quadric.get_mesh(resolution=resolution)
			else:
				mesh += quadric.get_mesh(resolution=resolution)
		if not transform:
			mesh.transform(torch.inverse(self.SE3).detach().cpu().numpy())
			mesh.triangle_normals = o3d.utility.Vector3dVector([])
		return mesh
	
	def get_point_cloud(self, transform=True, number_of_points=1024, dtype='numpy'):
		mesh = self.get_mesh(transform=transform)
		pc = mesh.sample_points_uniformly(number_of_points=number_of_points)
		pc_numpy = np.asarray(pc.points)
		if dtype == 'torch':
			pc_numpy = torch.from_numpy(pc_numpy).float()
		return pc_numpy
	
	def get_bounding_box(self):
		pc_numpy = self.get_point_cloud()
		pc = o3d.geometry.PointCloud()
		pc.points = o3d.utility.Vector3dVector(pc_numpy)
		bbox = pc.get_axis_aligned_bounding_box()
		min_bound = bbox.get_min_bound()
		max_bound = bbox.get_max_bound()
		return min_bound, max_bound
	
	def get_oriented_bounding_box(self):
		pc_numpy = self.get_point_cloud(transform=False)
		pc = o3d.geometry.PointCloud()
		pc.points = o3d.utility.Vector3dVector(pc_numpy)
		bbox = pc.get_axis_aligned_bounding_box()
		bbox = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bbox)
		bbox.translate(self.SE3[0:3,[3]].cpu().detach().numpy())
		bbox.rotate(self.SE3[0:3,0:3].cpu().detach().numpy())
		return bbox
	
	def get_differentiable_point_cloud(self, dtype='numpy', use_mask=False):
		pc = []
		for idx, quadric in enumerate(self.quadrics):
			diff_pc = quadric.get_differentiable_point_cloud()
			if use_mask:
				mask = torch.ones_like(diff_pc[...,[0]]) * idx
				diff_pc = torch.cat([diff_pc, mask], dim=-1)
			pc.append(diff_pc)
		pc = torch.cat(pc, dim=-2)
		if dtype == 'numpy':
			pc = pc.cpu().detach().numpy()
		return pc

	def get_sdf_values(self, pc, mode='e1'):
		dist = []
		for quadric in self.quadrics:
				dist.append(
				quadric.get_sdf_value(pc, mode=mode)
			)
		dist = torch.stack(dist)
		return dist
  
class WineGlass(Tableware):
	def __init__(
		self,
		SE3,
		params,
		device,
		t=0.01,
		process_mesh=True
		):
		
		self.name = 'WineGlass'
		self.range = {
			"root_thickness" : [0.003, 0.006],
			"pillar_radius" : [0.003, 0.006],
			"cup_radius" : [0.03, 0.057],
			"total_height" : [0.155, 0.264],
			"height_pillar_cup_ratio":[0.4, 0.6],
			"radius_root_cup_ratio": [0.9, 1.1],
			"cup_e1" : [0.6, 2.0],
			"cup_k" : [-3.0, 0.0],
		}
		super().__init__(SE3, params, self.range, device, t, process_mesh)
		self.nonsymmetric_idx = [0,1,2,3,4,5,6,7]
		self.construct()

	def construct(self):
		root_thickness = self.params[...,[0]]
		pillar_radius = self.params[...,[1]]
		cup_radius = self.params[...,[2]]
		total_height = self.params[...,[3]]
		height_pillar_cup_ratio = self.params[...,[4]]
		radius_root_cup_ratio = self.params[...,[5]]
		cup_e1 = self.params[...,[6]]
		cup_k = self.params[...,[7]]
		ones = torch.ones_like(root_thickness)

		root_radius = cup_radius * radius_root_cup_ratio
		cup_height = total_height * height_pillar_cup_ratio
		pillar_height = total_height * (1-height_pillar_cup_ratio)
		
		SE3_root = self.SE3.clone()
		SE3_root[...,0:3,3] += self.SE3[...,0:3, 2] * root_thickness * 0.5
		params_root = torch.cat([root_radius, root_radius, root_thickness * 0.5, ones*0.2, ones*1.0, ones*0], dim=-1)

		SE3_pillar = SE3_root.clone()
		SE3_pillar[...,0:3,3] += self.SE3[...,0:3, 2] * (root_thickness + pillar_height) * 0.5
		params_pillar = torch.cat([pillar_radius, pillar_radius, pillar_height * 0.5, ones*0.2, ones*1.0, ones*0], dim=-1)
		
		SE3_cup = SE3_pillar.clone()
		SE3_cup[...,0:3,3] += self.SE3[...,0:3, 2] * (pillar_height * 0.5 + cup_height)
		params_cup = torch.cat([cup_radius, cup_radius, cup_height, cup_e1, ones * 1.0, cup_k], dim=-1)
		
		self.root = SuperQuadric(SE3_root, params_root, type="superellipsoid")
		self.pillar = SuperQuadric(SE3_pillar, params_pillar, type="superellipsoid")
		self.cup = SuperQuadric(SE3_cup, params_cup, type="superparaboloid", t=self.t, process_mesh=self.process_mesh)
		
		self.quadrics = [self.root, self.pillar, self.cup]

	def get_grasp_poses(
			self, 
			desired_dir_sq=None, dir_angle_bound_sq=np.pi/2, flip_sq=False,
			desired_dir_sp=None, dir_angle_bound_sp=np.pi/2, flip_sp=False):
		
		# pillar
		grasp_poses_pillar = superellipsoid_grasp_planner(
			self.pillar.SE3,
			self.pillar.parameters,
			n_gripper_cyl_r=24,
			n_gripper_cyl_h=6,
			d=0.09,
			desired_dir=desired_dir_sq,
			dir_angle_bound=dir_angle_bound_sq,
			flip=flip_sq
		)

		# cup
		grasp_poses_cup = superparaboloid_grasp_planner(
			self.cup.SE3,
			self.cup.parameters,
			n_gripper=24,
			d=0.09,
			desired_dir=desired_dir_sp,
			dir_angle_bound=dir_angle_bound_sp,
			flip=flip_sp
		)

		return torch.cat([grasp_poses_pillar, grasp_poses_cup], dim=0)

class Bowl(Tableware):
	def __init__(
		self,
		SE3,
		params,
		device,
		t=0.01,
		process_mesh=True
		):
		
		self.name = 'Bowl'
		self.range = {
			"bowl_a1" : [0.04, 0.15],
			"bowl_a2" : [0.04, 0.15],
			"bowl_height" : [0.02, 0.1],
			"bowl_e1" : [0.01, 0.3],
			"bowl_e2" : [0.1, 1.0],
			"bowl_k" : [-0.1, 0.3],
		}
		super().__init__(SE3, params, self.range, device, t, process_mesh)
		self.interperumtable_idx = [0, 1]
		self.nonsymmetric_idx = [2,3,4,5]
		self.construct()
		
	def construct(self):
		bowl_a1 = self.params[...,[0]]
		bowl_a2 = self.params[...,[1]]
		bowl_height = self.params[...,[2]]
		bowl_e1 = self.params[...,[3]]
		bowl_e2 = self.params[...,[4]]
		bowl_k = self.params[...,[5]]
		
		SE3_bowl = self.SE3.clone()
		SE3_bowl[...,0:3,3] += self.SE3[...,0:3,2] * bowl_height
		params_bowl = torch.cat([bowl_a1, bowl_a2, bowl_height, bowl_e1, bowl_e2, bowl_k], dim=-1)
		self.bowl = SuperQuadric(SE3_bowl, params_bowl, type="superparaboloid", t=self.t, process_mesh=self.process_mesh)
		self.quadrics = [self.bowl]

	def get_grasp_poses(
			self, 
			desired_dir_sq=None, dir_angle_bound_sq=np.pi/2, flip_sq=False,
			desired_dir_sp=None, dir_angle_bound_sp=np.pi/2, flip_sp=False):
			
		# bowl
		grasp_poses_bowl = superparaboloid_grasp_planner(
			self.bowl.SE3,
			self.bowl.parameters,
			n_gripper=36,
			d=0.10,
			desired_dir=desired_dir_sp,
			dir_angle_bound=dir_angle_bound_sp,
			flip=flip_sp
		)

		return grasp_poses_bowl

class Bottle(Tableware):
	def __init__(
		self,
		SE3,
		params,
		device,
		t=0.01,
		process_mesh=True
		):
		
		self.name = 'Bottle'
		self.range = {
			"bottom_radius" : [0.03, 0.055],
			"bottom_height" : [0.10, 0.23],
			"middle_height" : [0.03, 0.05],
			"top_radius" : [0.008, 0.012],
			"top_height" : [0.01, 0.02],
			"e2" : [0.2, 1.0],
		}
		super().__init__(SE3, params, self.range, device, t, process_mesh)
		self.nonsymmetric_idx = [0,1,2,3,4,5]
		self.construct()
		
	def construct(self):
		bottom_radius = self.params[...,[0]]
		bottom_height = self.params[...,[1]]
		middle_height = self.params[...,[2]]
		top_radius = self.params[...,[3]]
		top_height = self.params[...,[4]]
		e2 = self.params[...,[5]]
		ones = torch.ones_like(bottom_radius)
		
		SE3_bottom = self.SE3.clone()
		SE3_bottom[...,0:3,3] += self.SE3[...,0:3, 2] * bottom_height * 0.5
		params_bottom = torch.cat([bottom_radius, bottom_radius, bottom_height * 0.5, ones*0.2, e2, ones*0], dim=-1)
		self.bottom = SuperQuadric(SE3_bottom, params_bottom, type="superellipsoid")
		
		SE3_middle = SE3_bottom.clone()
		SE3_middle[...,0:3,3] += self.SE3[...,0:3, 2] * (bottom_height + middle_height) * 0.5
		middle_radius = (bottom_radius + top_radius) * 0.5
		middle_k = -(middle_radius - top_radius) / middle_radius
		params_middle = torch.cat([middle_radius, middle_radius, middle_height * 0.5, ones*0.2, e2, middle_k], dim=-1)
		self.middle = SuperQuadric(SE3_middle, params_middle, type="superellipsoid")
		
		SE3_top = SE3_middle.clone()
		SE3_top[...,0:3,3] += self.SE3[...,0:3, 2] * (middle_height + top_height) * 0.5
		params_top = torch.cat([top_radius, top_radius, top_height * 0.5, ones*0.2, ones*1.0, ones*0], dim=-1)
		self.top = SuperQuadric(SE3_top, params_top, type="superellipsoid")
		
		self.quadrics = [self.bottom, self.middle, self.top]

	def get_grasp_poses(
			self, 
			desired_dir_sq=None, dir_angle_bound_sq=np.pi/2, flip_sq=False,
			desired_dir_sp=None, dir_angle_bound_sp=np.pi/2, flip_sp=False):
			
		# bottom
		grasp_poses_bottom = superellipsoid_grasp_planner(
			self.bottom.SE3,
			self.bottom.parameters,
			n_gripper_cyl_r=24,
			n_gripper_cyl_h=6,
			d=0.1,
			desired_dir=desired_dir_sq,
			dir_angle_bound=dir_angle_bound_sq,
			flip=flip_sq
		)

		# top
		grasp_poses_top = superellipsoid_grasp_planner(
			self.top.SE3,
			self.top.parameters,
			n_gripper_cyl_r=24,
			n_gripper_cyl_h=3,
			d=0.1,
			desired_dir=desired_dir_sq,
			dir_angle_bound=dir_angle_bound_sq,
			flip=flip_sq
		)

		return torch.cat([grasp_poses_bottom, grasp_poses_top], dim=0)


class BeerBottle(Tableware):
	def __init__(
		self,
		SE3,
		params,
		device,
		t=0.01,
		process_mesh=True
		):
		
		self.name = 'BeerBottle'
		self.range = {
			"bottom_radius" : [0.025, 0.05],
			"bottom_height" : [0.12, 0.19],
			"middle_height" : [0.01, 0.07],
			"top_radius" : [0.014, 0.016],
			"top_height" : [0.07, 0.1],
			"top_k" : [-0.2, 0.0],
		}
		super().__init__(SE3, params, self.range, device, t, process_mesh)
		self.nonsymmetric_idx = [0,1,2,3,4,5]
		self.construct()
		
	def construct(self):
		bottom_radius = self.params[...,[0]]
		bottom_height = self.params[...,[1]]
		middle_height = self.params[...,[2]]
		top_radius = self.params[...,[3]]
		top_height = self.params[...,[4]]
		top_k = self.params[...,[5]]
		ones = torch.ones_like(bottom_radius)
		
		SE3_bottom = self.SE3.clone()
		SE3_bottom[...,0:3,3] += self.SE3[...,0:3, 2] * bottom_height * 0.5
		params_bottom = torch.cat([bottom_radius, bottom_radius, bottom_height * 0.5, ones*0.05, ones*1.0, ones*0], dim=-1)
		self.bottom = SuperQuadric(SE3_bottom, params_bottom, type="superellipsoid")
		
		SE3_middle = SE3_bottom.clone()
		SE3_middle[...,0:3,3] += self.SE3[...,0:3, 2] * (bottom_height + middle_height) * 0.5
		middle_radius = (bottom_radius + top_radius*(1-top_k)) * 0.5
		middle_k = -(middle_radius - top_radius*(1-top_k)) / middle_radius
		params_middle = torch.cat([middle_radius, middle_radius, middle_height * 0.5, ones*0.05, ones*1.0, middle_k], dim=-1)
		self.middle = SuperQuadric(SE3_middle, params_middle, type="superellipsoid")
		
		SE3_top = SE3_middle.clone()
		SE3_top[...,0:3,3] += self.SE3[...,0:3, 2] * (middle_height + top_height) * 0.5
		params_top = torch.cat([top_radius, top_radius, top_height * 0.5, ones*0.05, ones*1.0, top_k], dim=-1)
		self.top = SuperQuadric(SE3_top, params_top, type="superellipsoid")
		
		self.quadrics = [self.bottom, self.middle, self.top]

	def get_grasp_poses(
			self, 
			desired_dir_sq=None, dir_angle_bound_sq=np.pi/2, flip_sq=False,
			desired_dir_sp=None, dir_angle_bound_sp=np.pi/2, flip_sp=False):
			
		# bottom
		grasp_poses_bottom = superellipsoid_grasp_planner(
			self.bottom.SE3,
			self.bottom.parameters,
			n_gripper_cyl_r=24,
			n_gripper_cyl_h=6,
			d=0.1,
			desired_dir=desired_dir_sq,
			dir_angle_bound=dir_angle_bound_sq,
			flip=flip_sq
		)

		# top
		grasp_poses_top = superellipsoid_grasp_planner(
			self.top.SE3,
			self.top.parameters,
			n_gripper_cyl_r=24,
			n_gripper_cyl_h=3,
			d=0.1,
			desired_dir=desired_dir_sq,
			dir_angle_bound=dir_angle_bound_sq,
			flip=flip_sq
		)

		return torch.cat([grasp_poses_bottom, grasp_poses_top], dim=0)

class HandlessCup(Tableware):
	def __init__(
		self,
		SE3,
		params,
		device,
		t=0.01,
		process_mesh=True
		):
		
		self.name = 'HandlessCup'
		self.range = {
			"cup_radius" : [0.025, 0.05],
			"cup_height" : [0.05, 0.22],
			"cup_e1" : [0.01, 0.3],
			"cup_k" : [0.0, 0.3],
		}
		super().__init__(SE3, params, self.range, device, t, process_mesh)
		self.nonsymmetric_idx = [0,1,2,3]
		self.construct()
		
	def construct(self):
		cup_radius = self.params[...,[0]]
		cup_height = self.params[...,[1]]
		cup_e1 = self.params[...,[2]]
		cup_k = self.params[...,[3]]
		ones = torch.ones_like(cup_radius)
		
		SE3_cup = self.SE3.clone()
		SE3_cup[...,0:3,3] += self.SE3[...,0:3, 2] * cup_height
		params_cup = torch.cat([cup_radius, cup_radius, cup_height, cup_e1, ones*1.0, cup_k], dim=-1)
		self.cup = SuperQuadric(SE3_cup, params_cup, type="superparaboloid", t=self.t, process_mesh=self.process_mesh)
		
		self.quadrics = [self.cup]

	def get_grasp_poses(
			self, 
			desired_dir_sq=None, dir_angle_bound_sq=np.pi/2, flip_sq=False,
			desired_dir_sp=None, dir_angle_bound_sp=np.pi/2, flip_sp=False):
			
		# cup
		grasp_poses_cup = superparaboloid_grasp_planner(
			self.cup.SE3,
			self.cup.parameters,
			n_gripper=36,
			d=0.09,
			desired_dir=desired_dir_sp,
			dir_angle_bound=dir_angle_bound_sp,
			flip=flip_sp
		)

		return grasp_poses_cup

class Mug(Tableware):
	def __init__(
		self,
		SE3,
		params,
		device,
		t=0.01,
		process_mesh=True
		):
		
		self.name = 'Mug'
		self.range = {
			"cup_radius" : [0.025, 0.05],
			"cup_height" : [0.08, 0.12],
			"cup_e1" : [0.01, 0.3],
			"cup_k" : [-0.2, 0.2],
			"handle_width" : [0.002, 0.003],
			"handle_width_thickness_ratio" : [1.0, 2.0],
			"handle_length_ratio" : [0.5, 0.7],
			"handle_e2" : [0.2, 1.0],
			"handle_shear" : [-0.5, -0.0001],
		}
		super().__init__(SE3, params, self.range, device, t, process_mesh)
		self.nonsymmetric_idx = [0,1,2,3,4,5,6,7,8]
		self.construct()
		
	def construct(self):
		cup_radius = self.params[...,[0]]
		cup_height = self.params[...,[1]]
		cup_e1 = self.params[...,[2]]
		cup_k = self.params[...,[3]]
		handle_width = self.params[...,[4]]
		handle_width_thickness_ratio = self.params[...,[5]]
		handle_length_ratio = self.params[...,[6]]
		handle_e2 = self.params[...,[7]]
		handle_shear = self.params[...,[8]]
		ones = torch.ones_like(cup_radius)
		
		SE3_cup = self.SE3.clone()
		SE3_cup[...,0:3,3] += self.SE3[...,0:3, 2] * cup_height
		params_cup = torch.cat([cup_radius, cup_radius, cup_height, cup_e1, ones*1.0, cup_k], dim=-1)
		self.cup = SuperQuadric(SE3_cup, params_cup, type="superparaboloid", t=self.t, process_mesh=self.process_mesh)
		
		SE3_handle = SE3_cup.clone()
		handle_a3 = handle_length_ratio * cup_height / 4 * torch.pi
		handle_arc_angle = np.pi/2
		b = handle_arc_angle / handle_a3
		SE3_handle[...,0:3,3] -= self.SE3[...,0:3, 2] * (cup_height * 0.5 + handle_shear/b*(1-np.cos(handle_arc_angle)))
		SE3_handle[...,0:3,3] += self.SE3[...,0:3, 0] * ((1-cup_k*(0.5+handle_length_ratio/2))*cup_radius*(0.5-handle_length_ratio/2)**(cup_e1/2) + 1/b*(1-np.cos(handle_arc_angle)))
		SE3_handle[...,0:3,0:3] = torch.tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]).to(SE3_handle.device) @ SE3_handle[...,0:3,0:3].float()
		params_handle = torch.cat([handle_width, handle_width*handle_width_thickness_ratio, handle_a3, ones*0.2, handle_e2, ones*0.0, b, ones*0, handle_shear], dim=-1)
		self.handle = SuperQuadric(SE3_handle, params_handle, type="deformable_superellipsoid", process_mesh=self.process_mesh)

		self.quadrics = [self.cup, self.handle]

	def get_grasp_poses(
			self, 
			desired_dir_sq=None, dir_angle_bound_sq=np.pi/2, flip_sq=False,
			desired_dir_sp=None, dir_angle_bound_sp=np.pi/2, flip_sp=False):

		# cup
		grasp_poses_cup = superparaboloid_grasp_planner(
			self.cup.SE3,
			self.cup.parameters,
			n_gripper=24,
			d=0.09,
			desired_dir=desired_dir_sp,
			dir_angle_bound=dir_angle_bound_sp,
			flip=flip_sp
		)

		# handle
		grasp_poses_handle = bent_superellipsoid_grasp_planner(
			self.handle.SE3,
			self.handle.parameters,
			n_gripper=16,
			desired_dir=desired_dir_sq,
			dir_angle_bound=dir_angle_bound_sq,
			flip=flip_sq
		)

		return torch.cat([grasp_poses_cup, grasp_poses_handle], dim=0)

class Dish(Tableware):
	def __init__(
		self,
		SE3,
		params,
		device,
		t=0.01,
		process_mesh=True
		):
		
		self.name = 'Dish'
		self.range = {
			"dish_radius" : [0.08, 0.14],
			"dish_height" : [0.015, 0.03],
			"dish_e1" : [0.01, 0.3],
   			"dish_e2" : [0.5, 1.0],
			"dish_k" : [0.3, 0.6],
		}
		super().__init__(SE3, params, self.range, device, t, process_mesh)
		self.nonsymmetric_idx = [0,1,2,3,4]
		self.construct()
		
	def construct(self):
		dish_radius = self.params[...,[0]]
		dish_height = self.params[...,[1]]
		dish_e1 = self.params[...,[2]]
		dish_e2 = self.params[...,[3]]
		dish_k = self.params[...,[4]]
		
		SE3_bowl = self.SE3.clone()
		SE3_bowl[...,0:3,3] += self.SE3[...,0:3, 2] * dish_height
		params_bowl = torch.cat([dish_radius, dish_radius, dish_height, dish_e1, dish_e2, dish_k], dim=-1)
		self.bowl = SuperQuadric(SE3_bowl, params_bowl, type="superparaboloid", t=self.t, process_mesh=self.process_mesh)
		self.quadrics = [self.bowl]

	def get_grasp_poses(
			self, 
			desired_dir_sq=None, dir_angle_bound_sq=np.pi/2, flip_sq=False,
			desired_dir_sp=None, dir_angle_bound_sp=np.pi/2, flip_sp=False):
			
		# bowl
		grasp_poses_bowl = superparaboloid_grasp_planner(
			self.bowl.SE3,
			self.bowl.parameters,
			n_gripper=36,
			d=0.09,
			desired_dir=desired_dir_sp,
			dir_angle_bound=dir_angle_bound_sp,
			flip=flip_sp
		)

		return grasp_poses_bowl

name_to_class = {
	"WineGlass": WineGlass,
	"Bowl": Bowl,
	"Bottle": Bottle,
 	"BeerBottle": BeerBottle,
	"HandlessCup": HandlessCup,
	"Mug": Mug,
	"Dish": Dish,
}

idx_to_class = {
	0: WineGlass,
	1: Bowl,
	2: Bottle,
	3: BeerBottle,
	4: HandlessCup,
	5: Mug,
	6: Dish	
}

name_to_idx = {
	"WineGlass" : 0,
	"Bowl" : 1,
	"Bottle" : 2,
	"BeerBottle" : 3,
	"HandlessCup" : 4,
	"Mug" : 5,
	"Dish" : 6	
}

idx_to_name = {
	0: "WineGlass",
	1: "Bowl",
	2: "Bottle",
	3: "BeerBottle",
	4: "HandlessCup",
	5: "Mug",
	6: "Dish"	
}