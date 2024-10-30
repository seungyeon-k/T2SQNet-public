from tablewarenet.primitives import SuperQuadric
import numpy as np
import open3d as o3d
	
class Tableware():
	def __init__(self, SE3, params, range, device='cpu', t=0.01, process_mesh=True):
		self.quadrics = []
		self.SE3 = SE3
		self.params = params
		self.t = t
		self.process_mesh = process_mesh
		self.construct()
	
	def construct(self):
		pass
	
	def get_mesh(self, transform=True):
		for i, quadric in enumerate(self.quadrics):
			if i == 0:
				mesh = quadric.get_mesh()
			else:
				mesh += quadric.get_mesh()
		if not transform:
			mesh.transform(np.linalg.inv(self.SE3))
			mesh.triangle_normals = o3d.utility.Vector3dVector([])
		return mesh
  
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
		ones = np.ones_like(root_thickness)

		root_radius = cup_radius * radius_root_cup_ratio
		cup_height = total_height * height_pillar_cup_ratio
		pillar_height = total_height * (1-height_pillar_cup_ratio)
		
		SE3_root = self.SE3.clone()
		SE3_root[...,0:3,3] += self.SE3[...,0:3, 2] * root_thickness * 0.5
		params_root = np.concatenate([root_radius, root_radius, root_thickness * 0.5, ones*0.2, ones*1.0, ones*0], axis=-1)

		SE3_pillar = SE3_root.clone()
		SE3_pillar[...,0:3,3] += self.SE3[...,0:3, 2] * (root_thickness + pillar_height) * 0.5
		params_pillar = np.concatenate([pillar_radius, pillar_radius, pillar_height * 0.5, ones*0.2, ones*1.0, ones*0], axis=-1)
		
		SE3_cup = SE3_pillar.clone()
		SE3_cup[...,0:3,3] += self.SE3[...,0:3, 2] * (pillar_height * 0.5 + cup_height)
		params_cup = np.concatenate([cup_radius, cup_radius, cup_height, cup_e1, ones * 1.0, cup_k], axis=-1)
		
		self.root = SuperQuadric(SE3_root, params_root, type="superellipsoid")
		self.pillar = SuperQuadric(SE3_pillar, params_pillar, type="superellipsoid")
		self.cup = SuperQuadric(SE3_cup, params_cup, type="superparaboloid", t=self.t, process_mesh=self.process_mesh)
		
		self.quadrics = [self.root, self.pillar, self.cup]

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
		params_bowl = np.concatenate([bowl_a1, bowl_a2, bowl_height, bowl_e1, bowl_e2, bowl_k], axis=-1)
		self.bowl = SuperQuadric(SE3_bowl, params_bowl, type="superparaboloid", t=self.t, process_mesh=self.process_mesh)
		self.quadrics = [self.bowl]

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
		ones = np.ones_like(bottom_radius)
		
		SE3_bottom = self.SE3.clone()
		SE3_bottom[...,0:3,3] += self.SE3[...,0:3, 2] * bottom_height * 0.5
		params_bottom = np.concatenate([bottom_radius, bottom_radius, bottom_height * 0.5, ones*0.2, e2, ones*0], axis=-1)
		self.bottom = SuperQuadric(SE3_bottom, params_bottom, type="superellipsoid")
		
		SE3_middle = SE3_bottom.clone()
		SE3_middle[...,0:3,3] += self.SE3[...,0:3, 2] * (bottom_height + middle_height) * 0.5
		middle_radius = (bottom_radius + top_radius) * 0.5
		middle_k = -(middle_radius - top_radius) / middle_radius
		params_middle = np.concatenate([middle_radius, middle_radius, middle_height * 0.5, ones*0.2, e2, middle_k], axis=-1)
		self.middle = SuperQuadric(SE3_middle, params_middle, type="superellipsoid")
		
		SE3_top = SE3_middle.clone()
		SE3_top[...,0:3,3] += self.SE3[...,0:3, 2] * (middle_height + top_height) * 0.5
		params_top = np.concatenate([top_radius, top_radius, top_height * 0.5, ones*0.2, ones*1.0, ones*0], axis=-1)
		self.top = SuperQuadric(SE3_top, params_top, type="superellipsoid")
		
		self.quadrics = [self.bottom, self.middle, self.top]

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
		ones = np.ones_like(bottom_radius)
		
		SE3_bottom = self.SE3.clone()
		SE3_bottom[...,0:3,3] += self.SE3[...,0:3, 2] * bottom_height * 0.5
		params_bottom = np.concatenate([bottom_radius, bottom_radius, bottom_height * 0.5, ones*0.05, ones*1.0, ones*0], axis=-1)
		self.bottom = SuperQuadric(SE3_bottom, params_bottom, type="superellipsoid")
		
		SE3_middle = SE3_bottom.clone()
		SE3_middle[...,0:3,3] += self.SE3[...,0:3, 2] * (bottom_height + middle_height) * 0.5
		middle_radius = (bottom_radius + top_radius*(1-top_k)) * 0.5
		middle_k = -(middle_radius - top_radius*(1-top_k)) / middle_radius
		params_middle = np.concatenate([middle_radius, middle_radius, middle_height * 0.5, ones*0.05, ones*1.0, middle_k], axis=-1)
		self.middle = SuperQuadric(SE3_middle, params_middle, type="superellipsoid")
		
		SE3_top = SE3_middle.clone()
		SE3_top[...,0:3,3] += self.SE3[...,0:3, 2] * (middle_height + top_height) * 0.5
		params_top = np.concatenate([top_radius, top_radius, top_height * 0.5, ones*0.05, ones*1.0, top_k], axis=-1)
		self.top = SuperQuadric(SE3_top, params_top, type="superellipsoid")
		
		self.quadrics = [self.bottom, self.middle, self.top]

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
		ones = np.ones_like(cup_radius)
		
		SE3_cup = self.SE3.clone()
		SE3_cup[...,0:3,3] += self.SE3[...,0:3, 2] * cup_height
		params_cup = np.concatenate([cup_radius, cup_radius, cup_height, cup_e1, ones*1.0, cup_k], axis=-1)
		self.cup = SuperQuadric(SE3_cup, params_cup, type="superparaboloid", t=self.t, process_mesh=self.process_mesh)
		
		self.quadrics = [self.cup]

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
		ones = np.ones_like(cup_radius)
		
		SE3_cup = self.SE3.clone()
		SE3_cup[...,0:3,3] += self.SE3[...,0:3, 2] * cup_height
		params_cup = np.concatenate([cup_radius, cup_radius, cup_height, cup_e1, ones*1.0, cup_k], axis=-1)
		self.cup = SuperQuadric(SE3_cup, params_cup, type="superparaboloid", t=self.t, process_mesh=self.process_mesh)
		
		SE3_handle = SE3_cup.clone()
		handle_a3 = handle_length_ratio * cup_height / 4 * np.pi
		handle_arc_angle = np.pi/2
		b = handle_arc_angle / handle_a3
		SE3_handle[...,0:3,3] -= self.SE3[...,0:3, 2] * (cup_height * 0.5 + handle_shear/b*(1-np.cos(handle_arc_angle)))
		SE3_handle[...,0:3,3] += self.SE3[...,0:3, 0] * ((1-cup_k*(0.5+handle_length_ratio/2))*cup_radius*(0.5-handle_length_ratio/2)**(cup_e1/2) + 1/b*(1-np.cos(handle_arc_angle)))
		SE3_handle[...,0:3,0:3] = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]).to(SE3_handle.device) @ SE3_handle[...,0:3,0:3]
		params_handle = np.concatenate([handle_width, handle_width*handle_width_thickness_ratio, handle_a3, ones*0.2, handle_e2, ones*0.0, b, ones*0, handle_shear], axis=-1)
		self.handle = SuperQuadric(SE3_handle, params_handle, type="deformable_superellipsoid", process_mesh=self.process_mesh)

		self.quadrics = [self.cup, self.handle]

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
		params_bowl = np.concatenate([dish_radius, dish_radius, dish_height, dish_e1, dish_e2, dish_k], axis=-1)
		self.bowl = SuperQuadric(SE3_bowl, params_bowl, type="superparaboloid", t=self.t, process_mesh=self.process_mesh)
		self.quadrics = [self.bowl]

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