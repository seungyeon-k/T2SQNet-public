import numpy as np
import torch
from functions.utils_torch import get_SE3s_torch
from copy import deepcopy

def superellipsoid_grasp_planner(
		SE3, 
		parameters, 
		n_gripper_box=4,
		n_gripper_cyl_r=2,
		n_gripper_cyl_h=5,
		d=0.1, 
		max_width=0.07, 
		ratio=0.8, 
		desired_dir=None,
		dir_angle_bound=np.pi/4,
		flip=False,
  		augment_flip=False,
		tilt=False,
		augment_tilt=False,
	):

	# initialize
	a1 = parameters[0]
	a2 = parameters[1]
	a3 = parameters[2]
	e1 = parameters[3]
	e2 = parameters[4]
	k = parameters[5]
	if len(parameters) == 9:
		b = parameters[6]
		alpha = parameters[7]
		s = parameters[8]

	ps_list = []
	SO3s_list = []
	gripper_SE3s = []

	# box and non-cylinder
	if e2 < 0.6 or (e2 >= 0.6 and abs(a1 - a2) >= 0.01): 

		# default linspace
		grid = torch.linspace(0, 1, n_gripper_box)
		grid_2D_x, grid_2D_y = torch.meshgrid(grid, grid, indexing='ij')
		grid = grid.reshape(-1, 1)
		grid_2D_x = grid_2D_x.reshape(-1, 1)
		grid_2D_y = grid_2D_y.reshape(-1, 1)

		# side grasp
		if a2 < max_width / 2:
			ps = grid @ torch.tensor([[0, 0, a3 * 2 * ratio]]) + torch.tensor([[a1 + d, 0, -a3 * ratio]])
			theta = torch.tensor(torch.pi)
			SO3s = torch.tensor([[
				[0, torch.sin(theta), torch.cos(theta)],
				[0, torch.cos(theta), -torch.sin(theta)],
				[-1, 0, 0]
			]]).repeat(n_gripper_box, 1, 1)
			ps_list.append(ps)
			SO3s_list.append(SO3s)
			ps = grid @ torch.tensor([[0, 0, a3 * 2 * ratio]]) + torch.tensor([[-a1 - d, 0, -a3 * ratio]])
			theta = torch.tensor(0)
			SO3s = torch.tensor([[
				[0, torch.sin(theta), torch.cos(theta)],
				[0, torch.cos(theta), -torch.sin(theta)],
				[-1, 0, 0]
			]]).repeat(n_gripper_box, 1, 1)
			ps_list.append(ps)
			SO3s_list.append(SO3s)
		if a1 < max_width / 2:
			ps = grid @ torch.tensor([[0, 0, a3 * 2 * ratio]]) + torch.tensor([[0, a2 + d, -a3 * ratio]])
			theta = torch.tensor(torch.pi / 2)
			SO3s = torch.tensor([[
				[0, torch.sin(theta), torch.cos(theta)],
				[0, torch.cos(theta), -torch.sin(theta)],
				[-1, 0, 0]
			]]).repeat(n_gripper_box, 1, 1)
			ps_list.append(ps)
			SO3s_list.append(SO3s)
			ps = grid @ torch.tensor([[0, 0, a3 * 2 * ratio]]) + torch.tensor([[0, -a2 - d, -a3 * ratio]])
			theta = torch.tensor(-torch.pi / 2)
			SO3s = torch.tensor([[
				[0, torch.sin(theta), torch.cos(theta)],
				[0, torch.cos(theta), -torch.sin(theta)],
				[-1, 0, 0]
			]]).repeat(n_gripper_box, 1, 1)
			ps_list.append(ps)
			SO3s_list.append(SO3s)   			
		
	# cylinder
	else:
		# default linspace
		grid_r, grid_h = torch.linspace(0, 1, n_gripper_cyl_r+1), torch.linspace(0, 1, n_gripper_cyl_h)
		grid_r = grid_r[:-1]
		grid_2D_x, grid_2D_y = torch.meshgrid(grid_r, grid_h, indexing='ij')
		grid_2D_x = grid_2D_x.reshape(-1, 1)
		grid_2D_y = grid_2D_y.reshape(-1, 1)

		# side grasp
		if a2 < max_width / 2:
			ps = torch.cos(2 * torch.pi * grid_2D_x) @ torch.tensor([[d, 0, 0]]) + \
				torch.sin(2 * torch.pi * grid_2D_x) @ torch.tensor([[0, d, 0]]) + \
				grid_2D_y @ torch.tensor([[0, 0, a3 * 2 * ratio]]) + \
				torch.tensor([[0, 0, -a3 * ratio]])
			SO3s = torch.zeros((n_gripper_cyl_h * n_gripper_cyl_r, 3, 3))		
			SO3s[:, 0, 1] = torch.sin(2 * torch.pi * grid_2D_x).reshape(-1)
			SO3s[:, 0, 2] = - torch.cos(2 * torch.pi * grid_2D_x).reshape(-1)
			SO3s[:, 1, 1] = - torch.cos(2 * torch.pi * grid_2D_x).reshape(-1)
			SO3s[:, 1, 2] = - torch.sin(2 * torch.pi * grid_2D_x).reshape(-1)
			SO3s[:, 2, 0] = -1
			ps_list.append(ps)
			SO3s_list.append(SO3s)   

	# get gripper SE3s
	if len(ps_list) >= 1:
		gripper_ps = torch.cat(ps_list, dim=0)
		gripper_SO3s = torch.cat(SO3s_list, dim=0)
		gripper_SE3s = get_SE3s_torch(gripper_SO3s, gripper_ps)
		gripper_SE3s = SE3 @ gripper_SE3s.to(SE3)
	else:
		return torch.tensor([])

	# grasp pose augment
	if flip:
		flip_matrix = torch.tensor([[
			[-1, 0, 0, 0], 
			[0, -1, 0, 0], 
			[0, 0, 1, 0], 
			[0, 0, 0, 1]
		]]).to(gripper_SE3s)
		flipped_gripper_SE3s= gripper_SE3s @ flip_matrix
		if augment_flip:
			gripper_SE3s = torch.cat([gripper_SE3s, flipped_gripper_SE3s], dim=0)
		else:
			gripper_SE3s = flipped_gripper_SE3s

	# tilt grasp pose	
	if tilt:
		theta = torch.tensor(torch.pi/6)
		tilt_matrix = torch.tensor([[
			[torch.cos(theta),  0, -torch.sin(theta),  d*torch.sin(theta)], 
			[0,	     			1,  0,  			   0], 
			[torch.sin(theta),  0,  torch.cos(theta),  d-d*torch.cos(theta)], 
			[0, 	 			0,  0,  			   1]
		]]).to(gripper_SE3s)
		tilted_gripper_SE3s = gripper_SE3s @ tilt_matrix
		if augment_tilt:
			gripper_SE3s = torch.cat([gripper_SE3s, tilted_gripper_SE3s], dim=0)
		else:
			gripper_SE3s = tilted_gripper_SE3s

	# filter grasp poses
	if desired_dir is not None:
		projected_z_axis_of_gripper = deepcopy(gripper_SE3s[:,0:3,2])
		projected_z_axis_of_gripper[:, 2] = 0
		projected_z_axis_of_gripper = projected_z_axis_of_gripper/projected_z_axis_of_gripper.norm(dim=-1, keepdim=True)
		bool = gripper_SE3s[:,0:3,2] @ desired_dir.to(gripper_SE3s) >= np.cos(dir_angle_bound)
		gripper_SE3s = gripper_SE3s[bool]

	return gripper_SE3s

def superparaboloid_grasp_planner(
		SE3, 
		parameters, 
		n_gripper=10, 
		d=0.09, 
		ratio=0.1, 
		desired_dir=None,
		dir_angle_bound=np.pi/4,
		flip=False,
		augment_flip=False
	):

	# initialize
	a1 = parameters[0]
	a2 = parameters[1]
	a3 = parameters[2]
	e1 = parameters[3]
	e2 = parameters[4]
	k = parameters[5]
	ps_list = []
	SO3s_list = []
	gripper_SE3s = []

	# reference point
	X = a1 * ((1 - ratio) ** (e1/2))
	Y = torch.tensor(0)
	Z = - ratio * a3
	f = k / a3 * Z + 1
	dDtdx = torch.tensor([
		[f, 0, k / a3 * X],
		[0, f, k / a3 * Y],
		[0, 0, 1]
	])

	# normal vector and calculate tilt
	eps = 1e-10
	term_xy = (
		torch.abs(X/a1)**(2/e2) 
		+ torch.abs(Y/a2)**(2/e2)
		) + eps
	nx = (
		2/e1
		* torch.sign(X)/a1
		* (torch.abs(X/a1) + eps)**(2/e2-1)
		* term_xy**(e2/e1-1)
	)
	ny = (
		2/e1
		* torch.sign(Y)/a2
		* (torch.abs(Y/a2) + eps)**(2/e2-1)
		* term_xy**(e2/e1-1)
	)
	nz = -1 / a3
	normal = torch.tensor([nx, ny, nz])
	normal = normal / torch.norm(normal)
	normal = torch.inverse(dDtdx).transpose(0, 1) @ normal.unsqueeze(-1)
	normal = torch.det(dDtdx) * normal.squeeze(-1)
	phi = - torch.atan2(torch.abs(normal[2]), torch.abs(normal[0])) # - 0.1 * torch.pi

	# if phi < torch.pi / 8:
	if True:

		# default linspace
		# grid = torch.linspace(0, 1, n_gripper+1)[:-1]
		# theta = 2 * torch.pi * grid
		grid = sq_uniform_sampling_2D(e2, a1, a2, 0.005)
		if len(grid) >= n_gripper:
			grid = grid[::int(len(grid) / n_gripper)]
		theta = grid
		n_gripper = len(grid)

		# top-down grasp
		SO3s = torch.zeros((n_gripper, 3, 3))
		SO3s[:, 0, 0] = torch.sin(theta)
		SO3s[:, 1, 0] = -torch.cos(theta)
		SO3s[:, 0, 1] = -torch.cos(theta) * torch.cos(phi)
		SO3s[:, 1, 1] = -torch.sin(theta) * torch.cos(phi)
		SO3s[:, 2, 1] = -torch.sin(phi)
		SO3s[:, 0, 2] = torch.cos(theta) * torch.sin(phi)
		SO3s[:, 1, 2] = torch.sin(theta) * torch.sin(phi)
		SO3s[:, 2, 2] = -torch.cos(phi)
		ps = torch.cat([
			a1 * fexp_torch(torch.cos(theta), e2).unsqueeze(-1),
			a2 * fexp_torch(torch.sin(theta), e2).unsqueeze(-1),
			torch.zeros((n_gripper, 1))
		], dim=-1)
		ps -= (
			SO3s @ torch.tensor([0.0, 0.0, d]).unsqueeze(0).unsqueeze(-1).repeat(n_gripper, 1, 1)
		).squeeze(-1)

		ps_list.append(ps)
		SO3s_list.append(SO3s)  

	# get gripper SE3s
	if len(ps_list) >= 1:
		gripper_ps = torch.cat(ps_list, dim=0)
		gripper_SO3s = torch.cat(SO3s_list, dim=0)
		gripper_SE3s = get_SE3s_torch(gripper_SO3s, gripper_ps)
		gripper_SE3s = SE3 @ gripper_SE3s.to(SE3)
	else:
		return torch.tensor([])

	# grasp pose augment
	if flip:
		flip_matrix = torch.tensor([[
			[-1, 0, 0, 0], 
			[0, -1, 0, 0], 
			[0, 0, 1, 0], 
			[0, 0, 0, 1]
		]]).to(gripper_SE3s)
		flipped_gripper_SE3s= gripper_SE3s @ flip_matrix
		if augment_flip:
			gripper_SE3s = torch.cat([gripper_SE3s, flipped_gripper_SE3s], dim=0)
		else:
			gripper_SE3s = flipped_gripper_SE3s

	# filter grasp poses
	if desired_dir is not None:
		projected_z_axis_of_gripper = deepcopy(gripper_SE3s[:,0:3,2])
		projected_z_axis_of_gripper[:, 2] = 0
		projected_z_axis_of_gripper = projected_z_axis_of_gripper/projected_z_axis_of_gripper.norm(dim=-1, keepdim=True)
		bool = gripper_SE3s[:,0:3,0] @ desired_dir.to(gripper_SE3s) >= np.cos(dir_angle_bound)
		gripper_SE3s = gripper_SE3s[bool]

	return gripper_SE3s

def bent_superellipsoid_grasp_planner(
		SE3, 
		parameters, 
		n_gripper=5,
		d=0.1, 
		flip=False,
  		augment_flip=False,
		desired_dir=None,
		dir_angle_bound=np.pi/4
	):

	# initialize
	a1 = parameters[0]
	a2 = parameters[1]
	a3 = parameters[2]
	e1 = parameters[3]
	e2 = parameters[4]
	k = parameters[5]
	if len(parameters) == 9:
		b = parameters[6]
		alpha = parameters[7]
		s = parameters[8]

	ps_list = []
	SO3s_list = []
	gripper_SE3s = []

	# reference points
	grid = torch.linspace(-0.8, 0.8, n_gripper)
	X = -a1 * torch.ones_like(grid)
	Y = torch.zeros_like(grid)
	Z = a3 * grid
	dDbdx = get_bending_jabocian(X, Y, Z, b, alpha) 

	# normal vector
	eps = 1e-10
	term_xy = (
		torch.abs(X/a1)**(2/e2) 
		+ torch.abs(Y/a2)**(2/e2)
		) + eps
	nx = (
		2/e1
		* torch.sign(X)/a1
		* (torch.abs(X/a1) + eps)**(2/e2-1)
		* term_xy**(e2/e1-1)
	).unsqueeze(-1)
	ny = (
		2/e1
		* torch.sign(Y)/a2
		* (torch.abs(Y/a2) + eps)**(2/e2-1)
		* term_xy**(e2/e1-1)
	).unsqueeze(-1)
	nz = (
		2/e1
		* torch.sign(Z)/a3
		* (torch.abs(Z/a3) + eps)**(2/e1-1)
	).unsqueeze(-1)
	normal = torch.cat([nx, ny, nz], dim=1)
	normal = normal / torch.norm(normal, dim=1).unsqueeze(-1)
	normal = torch.inverse(dDbdx).transpose(1, 2) @ normal.unsqueeze(-1)
	normal = torch.det(dDbdx).unsqueeze(-1) * normal.squeeze(-1)

	# bending
	gamma = Z * b
	r = torch.cos(alpha - torch.atan2(Y, X)) * torch.sqrt(X ** 2 + Y ** 2)
	R = 1 / b - torch.cos(gamma) * (1 / b - r)
	X = X + torch.cos(alpha) * (R - r)
	Y = X + torch.sin(alpha) * (R - r)
	Z = torch.sin(gamma) * (1 / b - r)	
	dDsdx = get_shearing_jabocian(X, Y, Z, s) 

	# normal
	normal = torch.inverse(dDsdx).transpose(1, 2) @ normal.unsqueeze(-1)
	normal = torch.det(dDsdx).unsqueeze(-1) * normal.squeeze(-1)	

	# shearing
	Z = Z + s * X

	# side grasps
	theta = torch.acos(normal[:, 2])
	SO3s = torch.zeros(3, 3).squeeze(0).repeat(n_gripper, 1, 1)
	SO3s[:, 0, 0] = torch.cos(theta)
	SO3s[:, 0, 2] = torch.sin(theta)
	SO3s[:, 2, 0] = -torch.sin(theta)
	SO3s[:, 2, 2] = torch.cos(theta)
	SO3s[:, 1, 1] = 1
	ps = torch.cat([
		X.unsqueeze(-1), Y.unsqueeze(-1), Z.unsqueeze(-1)
	], dim=-1)
	ps -= SO3s[:, :3, 2] * d
	ps_list.append(ps)
	SO3s_list.append(SO3s)   

	# get gripper SE3s
	if len(ps_list) >= 1:
		gripper_ps = torch.cat(ps_list, dim=0)
		gripper_SO3s = torch.cat(SO3s_list, dim=0)
		gripper_SE3s = get_SE3s_torch(gripper_SO3s, gripper_ps)
		gripper_SE3s = SE3 @ gripper_SE3s.to(SE3)
	else:
		return torch.tensor([])

	# grasp pose augment
	if flip:
		flip_matrix = torch.tensor([[
			[-1, 0, 0, 0], 
			[0, -1, 0, 0], 
			[0, 0, 1, 0], 
			[0, 0, 0, 1]
		]]).to(gripper_SE3s)
		flipped_gripper_SE3s= gripper_SE3s @ flip_matrix
		if augment_flip:
			gripper_SE3s = torch.cat([gripper_SE3s, flipped_gripper_SE3s], dim=0)
		else:
			gripper_SE3s = flipped_gripper_SE3s

	# filter grasp poses
	if desired_dir is not None:
		projected_z_axis_of_gripper = deepcopy(gripper_SE3s[:,0:3,2])
		projected_z_axis_of_gripper[:, 2] = 0
		projected_z_axis_of_gripper = projected_z_axis_of_gripper/projected_z_axis_of_gripper.norm(dim=-1, keepdim=True)
		bool = gripper_SE3s[:,0:3,0] @ desired_dir.to(gripper_SE3s) >= np.cos(dir_angle_bound)
		gripper_SE3s = gripper_SE3s[bool]

	return gripper_SE3s

#############################################################
########################## UTILS ############################
#############################################################

def fexp_torch(x, p):
	return torch.sign(x)*(torch.abs(x)**p)

def get_bending_jabocian(X, Y, Z, b, alpha):
	# TODO: change to batchwise jacobian

	J_list = [] 
	for x, y, z in zip(X, Y, Z):
		alpha_atan2 = alpha - torch.atan2(y, x)
		R = torch.cos(alpha_atan2) * torch.sqrt(x ** 2 + y ** 2)
		dRdX = (-y * torch.sin(alpha_atan2) + x * torch.cos(alpha_atan2)) / torch.sqrt(x**2 + y**2)
		dRdY = (x * torch.sin(alpha_atan2) + y * torch.cos(alpha_atan2)) / torch.sqrt(x**2 + y**2)
		drdX = (1/b - R) / np.sqrt(z ** 2 + (1/b - R) ** 2) * dRdX
		drdY = (1/b - R) / np.sqrt(z ** 2 + (1/b - R) ** 2) * dRdY
		drdZ = - z / np.sqrt(z ** 2 + (1/b - R) ** 2)
		dgammadX = z / (z ** 2 + (1/b - R) ** 2) * dRdX
		dgammadY = z / (z ** 2 + (1/b - R) ** 2) * dRdY
		dgammadZ = (1/b - R) / (z ** 2 + (1/b - R) ** 2)
		J = torch.tensor([
			[1 - torch.cos(alpha) * (dRdX - drdX), -torch.cos(alpha) * (dRdY - drdY), torch.cos(alpha) * drdZ],
			[-torch.sin(alpha) * (dRdX - drdX), 1 - torch.sin(alpha) * (dRdY - drdY), torch.sin(alpha) * drdZ],
			[1/b * dgammadX, 1/b * dgammadY, 1/b * dgammadZ]])		
		J_list.append(J)

	J_list = torch.stack(J_list)
	return J_list

def get_shearing_jabocian(X, Y, Z, s):
	# TODO: change to barchwise jacobian

	J_list = []
	for x, y, z in zip(X, Y, Z):
		J = torch.tensor([
			[1, 0, 0],
			[0, 1, 0],
			[s, 0, 1]
		])
		J_list.append(J)

	J_list = torch.stack(J_list)
	return J_list	

def sq_uniform_sampling_2D(e, a, b, K):
		
	# initialize
	theta_forward = 0.0
	theta_backward = torch.pi / 2
	theta_list = []

	while True:
		theta_backward = theta_backward - delta_theta(
			theta_backward, e, a, b, K)
		if theta_backward < 0:
			break
		theta_list.append(theta_backward)

	while True:
		theta_forward = theta_forward + delta_theta(
			theta_forward, e, a, b, K)
		if theta_forward > torch.pi / 2:
			break
		theta_list.append(theta_forward)

	theta_list_sorted = torch.sort(torch.tensor(theta_list))[0]

	theta_final = torch.cat([
		torch.tensor([0.]),
		theta_list_sorted,
		torch.tensor([torch.pi / 2]),
		theta_list_sorted + torch.pi / 2,
		torch.tensor([torch.pi]),
		theta_list_sorted + torch.pi,
		torch.tensor([torch.pi * 3 / 2]),
		theta_list_sorted + torch.pi * 3 / 2,
	], dim=0)	
	
	return theta_final

def delta_theta(theta, e, a, b, K):
	
	# initialize
	eps = 1e-10

	# grad theta
	if theta < eps:
		d_theta = (K / b - theta ** e) ** (1/e) - theta
		
	elif theta > np.pi/2 - eps:
		d_theta = (
			(K / a - (torch.pi/2 - theta) ** e) ** (1/e) 
			- (np.pi / 2 - theta)
		)

	else:
		inner = (
			(torch.cos(theta) ** 2 * torch.sin(theta) ** 2) 
			/ (
				(a ** 2) * (np.cos(theta) ** (2*e)) 
				* (torch.sin(theta) ** 4) + (b ** 2) 
				* (torch.sin(theta) ** (2*e)) * (torch.cos(theta) ** 4)
			)
		)
		d_theta = K / e * torch.sqrt(inner)
	
	return d_theta