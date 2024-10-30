import numpy as np
import torch
import open3d as o3d
from copy import deepcopy
from tablewarenet.superquadrics import sq_distance

class SuperQuadric():
	def __init__(self, SE3, parameters, type, t=0.03, process_mesh=True):
		self.type = type
		self.SE3 = SE3
		self.parameters = parameters
		self.t = t
		self.process_mesh = process_mesh
	
	def get_mesh(self, color=[0.8, 0.8, 0.8], resolution=20, resolution_radial=30, resolution_height=20):

		if torch.is_tensor(self.parameters):
			self.parameters_numpy = self.parameters.detach().cpu().numpy()
		else:
			self.parameters_numpy = deepcopy(self.parameters)
		if torch.is_tensor(self.SE3):
			self.SE3_numpy = self.SE3.detach().cpu().numpy()
		else:
			self.SE3_numpy = deepcopy(self.SE3)

		if self.type == 'superellipsoid':
			mesh = mesh_superellipsoid(
				self.parameters_numpy, resolution)
		elif self.type == 'deformable_superellipsoid':
			mesh = mesh_deformable_superellipsoid(
				self.parameters_numpy, resolution,
				process_mesh=self.process_mesh)
		elif self.type == 'superparaboloid':
			mesh = mesh_superparaboloid(
				self.parameters_numpy, resolution_radial, 
				resolution_height, t=self.t, process_mesh=self.process_mesh)
		mesh.compute_vertex_normals()
		mesh.paint_uniform_color(color)
		mesh.transform(self.SE3_numpy)
		return mesh
	
	def get_point_cloud(self, number_of_points=1024):
		mesh = self.get_mesh()
		pc = mesh.sample_points_uniformly(number_of_points=number_of_points)
		pc_numpy = np.asarray(pc.points)
		return pc_numpy
	
	def get_differentiable_point_cloud(self):
		if self.type == 'superellipsoid':
			pc = diff_pc_superellipsoid(self.parameters)
		elif self.type == 'deformable_superellipsoid':
			pc = diff_pc_deformable_superellipsoid(self.parameters)
		elif self.type == 'superparaboloid':
			pc = diff_pc_superparaboloid(self.parameters)
		pc = (self.SE3.float() @ torch.cat([pc.transpose(-1, -2), torch.ones_like(pc.transpose(-1, -2)[..., [0], :])], dim=-2))[..., 0:3,:].transpose(-1, -2)
		return pc
	
	def get_sdf_value(self, pc, mode='e1'):
		return sq_distance(
			pc, self.SE3, self.parameters, type=self.type, mode=mode)
		

def mesh_superellipsoid(parameters, resolution=20):
	# parameters
	a1 = parameters[0]
	a2 = parameters[1]
	a3 = parameters[2]
	e1 = parameters[3]
	e2 = parameters[4]
	k = parameters[5]

	# make grids
	mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=resolution)
	vertices_numpy = np.asarray(mesh.vertices)
	eta = np.arcsin(vertices_numpy[:, 2:3])
	omega = np.arctan2(vertices_numpy[:, 1:2], vertices_numpy[:, 0:1])

	# make new vertices
	x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
	y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
	z = a3 * fexp(np.sin(eta), e1)
	
	f = k / a3 * z + 1
	x = f * x
	y = f * y

	# reconstruct point matrix
	points = np.concatenate((x, y, z), axis=1)

	mesh.vertices = o3d.utility.Vector3dVector(points)

	return mesh

def mesh_deformable_superellipsoid(parameters, resolution=30, process_mesh=True):

	# parameters
	a1 = parameters[0]
	a2 = parameters[1]
	a3 = parameters[2]
	e1 = parameters[3]
	e2 = parameters[4]
	k = parameters[5]
	b = parameters[6]
	alpha = parameters[7]
	s = parameters[8]

	# make grids
	mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=resolution)
	vertices_numpy = np.asarray(mesh.vertices)
	eta = np.arcsin(vertices_numpy[:, 2:3])
	omega = np.arctan2(vertices_numpy[:, 1:2], vertices_numpy[:, 0:1])

	# make new vertices
	x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
	y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
	z = a3 * fexp(np.sin(eta), e1)

	points = np.concatenate((x, y, z), axis=1)
	mesh.vertices = o3d.utility.Vector3dVector(points)
	if process_mesh:
		mesh.remove_duplicated_vertices()
		mesh.remove_degenerate_triangles()
		mesh = mesh.subdivide_midpoint(2)

	points = np.asarray(mesh.vertices)
	x = points[:, 0:1]
	y = points[:, 1:2]
	z = points[:, 2:3]

	# tapering
	f_x = k / a3 * z + 1
	f_y = k / a3 * z + 1
	x = f_x * x
	y = f_y * y

	# bending
	gamma = z * b
	r = np.cos(alpha - np.arctan2(y, x)) * np.sqrt(x ** 2 + y ** 2)
	R = 1 / b - np.cos(gamma) * (1 / b - r)
	x = x + np.cos(alpha) * (R - r)
	y = y + np.sin(alpha) * (R - r)
	z = np.sin(gamma) * (1 / b - r)
	
	# shearing
	z = z + s * x

	# reconstruct point matrix
	points = np.concatenate((x, y, z), axis=1)

	mesh.vertices = o3d.utility.Vector3dVector(points)

	if process_mesh:
		mesh.remove_duplicated_vertices()
		mesh.remove_degenerate_triangles()

	return mesh

def mesh_superparaboloid(
		parameters, resolution_radial, resolution_height, 
		t=0.01, process_mesh=True):

	# parameters
	a1 = parameters[0]
	a2 = parameters[1]
	a3 = parameters[2]
	e1 = parameters[3]
	e2 = parameters[4]
	k = parameters[5]

	# make pipe mesh
	cylinder_inner = create_open_cylinder(radius=1-t, height=1.0, resolution=resolution_radial, split=resolution_height, reverse_normal=True)
	cylinder_outer = create_open_cylinder(radius=1+t, height=1.0, resolution=resolution_radial, split=resolution_height)
	pipe = cylinder_inner + cylinder_outer
	pipe.translate([0, 0, 0.5])

	# make grids
	vertices_numpy = np.asarray(pipe.vertices)
	normxy = np.sqrt(vertices_numpy[:,1:2]**2 + vertices_numpy[:,0:1]**2)
	u = vertices_numpy[:,2:3]
	omega = np.arctan2(vertices_numpy[:,1:2], vertices_numpy[:,0:1])
	
	# make new vertices
	x = a1 * u * fexp(np.cos(omega), e2) * normxy
	y = a2 * u * fexp(np.sin(omega), e2) * normxy
	z = a3 * (fexp(u, 2/e1) - 1) 

	#######################
	####### TODO ##########
	#######################
	
	# tapering
	f = k / a3 * z + 1
	x = f * x
	y = f * y

	# obtain indices
	triangles_merge_numpy = np.asarray(pipe.triangles)
	n_vertices = len(vertices_numpy)
	n_unit = int(n_vertices / (2 * (resolution_height + 1)))
	n_big_unit = int(n_vertices / 2)
	top_inner = list(range(0, n_unit))
	top_outer = list(range(n_big_unit, n_big_unit+n_unit))

	# connect inner and outer
	top_connecting_triangle = connect_inner_outer_indices(top_inner, top_outer)
	connecting_triangle = np.array(top_connecting_triangle)
	new_triangles_merge_numpy = np.concatenate((triangles_merge_numpy, connecting_triangle), axis=0)

	# add new triangles
	pipe.triangles = o3d.utility.Vector3iVector(new_triangles_merge_numpy)  
	
	# reconstruct point matrix
	points = np.concatenate((x, y, z), axis=1)

	pipe.vertices = o3d.utility.Vector3dVector(points)

	if process_mesh:
		pipe.remove_duplicated_vertices()
		pipe.remove_degenerate_triangles()
	
	return pipe

def create_open_cylinder(radius=1.0, height=1.0, resolution=20, split=1, reverse_normal=False):
	cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution, split=split)
	cylinder.remove_vertices_by_index([0, 1])
	if reverse_normal:
		vertices_numpy = np.asarray(cylinder.triangles)
		vectices_reversed_numpy = vertices_numpy[:, [1, 0, 2]]
		cylinder.triangles = o3d.utility.Vector3iVector(vectices_reversed_numpy)

	return cylinder

def connect_inner_outer_indices(inner_indices, outer_indices, reverse_normal=False):

	n_points = len(inner_indices)
	inner_indices = inner_indices + [inner_indices[0]]
	outer_indices = outer_indices + [outer_indices[0]]

	if reverse_normal:
		connecting_triangle_1 = [[inner_indices[i], inner_indices[i+1], outer_indices[i+1]] for i in range(n_points)]
		connecting_triangle_2 = [[inner_indices[i], outer_indices[i+1], outer_indices[i]] for i in range(n_points)]
	else:
		connecting_triangle_1 = [[inner_indices[i+1], inner_indices[i], outer_indices[i+1]] for i in range(n_points)]
		connecting_triangle_2 = [[inner_indices[i], outer_indices[i], outer_indices[i+1]] for i in range(n_points)]
	connecting_triangle = connecting_triangle_1 + connecting_triangle_2

	return connecting_triangle 

def fexp(x, p):
	return np.sign(x)*(np.abs(x)**p)

def fexp_torch(x, p):
	return torch.sign(x)*(torch.abs(x)**p)

mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=4)
vertices_numpy = np.asarray(mesh.vertices)
eta_sph = torch.from_numpy(np.arcsin(vertices_numpy[:, 2:3])).float()
omega_sph = torch.from_numpy(np.arctan2(vertices_numpy[:, 1:2], vertices_numpy[:, 0:1])).float()

t= 0.01
cylinder_inner = create_open_cylinder(radius=1-t, height=1.0, resolution=4, split=8, reverse_normal=True)
cylinder_outer = create_open_cylinder(radius=1+t, height=1.0, resolution=4, split=8)
pipe = cylinder_inner + cylinder_outer
pipe.translate([0, 0, 0.5])

# make grids
vertices_numpy = np.asarray(pipe.vertices)
normxy_cyl = torch.from_numpy(np.sqrt(vertices_numpy[:,1:2]**2 + vertices_numpy[:,0:1]**2)).float()
u_cyl = torch.from_numpy(vertices_numpy[:,2:3]).float()
omega_cyl = torch.from_numpy(np.arctan2(vertices_numpy[:,1:2], vertices_numpy[:,0:1])).float()

def diff_pc_superellipsoid(parameters):
	# parameters
	
	a1 = parameters[...,[0]]
	a2 = parameters[...,[1]]
	a3 = parameters[...,[2]]
	e1 = parameters[...,[3]]
	e2 = parameters[...,[4]]
	k = parameters[...,[5]]

	device = parameters.device
	# make new vertices
	x = a1 * fexp_torch(torch.cos(eta_sph.to(device)).T, e1) * fexp_torch(torch.cos(omega_sph.to(device)).T, e2)
	y = a2 * fexp_torch(torch.cos(eta_sph.to(device)).T, e1) * fexp_torch(torch.sin(omega_sph.to(device)).T, e2)
	z = a3 * fexp_torch(torch.sin(eta_sph.to(device)).T, e1)
	
	f = k / a3 * z + 1
	x = f * x
	y = f * y

	# reconstruct point matrix
	points = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)
	return points

def diff_pc_deformable_superellipsoid(parameters):
	# parameters
	a1 = parameters[...,[0]]
	a2 = parameters[...,[1]]
	a3 = parameters[...,[2]]
	e1 = parameters[...,[3]]
	e2 = parameters[...,[4]]
	k = parameters[...,[5]]
	b = parameters[...,[6]]
	alpha = parameters[...,[7]]
	s = parameters[...,[8]]

	device = parameters.device
	
	# make new vertices
	x = a1 * (fexp_torch(torch.cos(eta_sph.to(device)).T, e1) * fexp_torch(torch.cos(omega_sph.to(device)).T, e2))
	y = a2 * (fexp_torch(torch.cos(eta_sph.to(device)).T, e1) * fexp_torch(torch.sin(omega_sph.to(device)).T, e2))
	z = a3 * fexp_torch(torch.sin(eta_sph.to(device)).T, e1)
	# tapering
	f_x = k / a3 * z + 1
	f_y = k / a3 * z + 1
	x = f_x * x
	y = f_y * y

	# bending
	gamma = z * b
	r = torch.cos(alpha - torch.arctan2(y, x)) * torch.sqrt(x ** 2 + y ** 2)
	R = 1 / b - torch.cos(gamma) * (1 / b - r)
	x = x + torch.cos(alpha) * (R - r)
	y = y + torch.sin(alpha) * (R - r)
	z = torch.sin(gamma) * (1 / b - r) + s * x #shear

	points = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)
	return points

def diff_pc_superparaboloid(parameters):

	# parameters
	a1 = parameters[...,[0]]
	a2 = parameters[...,[1]]
	a3 = parameters[...,[2]]
	e1 = parameters[...,[3]]
	e2 = parameters[...,[4]]
	k = parameters[...,[5]]
	
	device = parameters.device
	
	# make new vertices
	x = a1 * u_cyl.T.to(device) * fexp_torch(torch.cos(omega_cyl.to(device)).T, e2) * normxy_cyl.T.to(device)
	y = a2 * u_cyl.T.to(device) * fexp_torch(torch.sin(omega_cyl.to(device)).T, e2) * normxy_cyl.T.to(device)
	z = a3 * (fexp_torch(u_cyl.T.to(device), 2/e1) - 1) 
	
	# tapering
	f = k / a3 * z + 1
	x = f * x
	y = f * y

	points = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)
	
	return points