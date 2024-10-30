import numpy as np
import torch

def sq_distance(x, poses, sq_params, type, mode='1', eps=1e-4):
	'''
	input: x : (n x 3) pointcloud coordinates 
		poses : (n_sq x 4 x 4) superquadric poses
		sq_params : (n_sq x 5) superquadric parameters
		type : str, 'superellipsoid' or 'deformable_superellipsoid' or 'superparaboloid'
		mode : 'e1' or '1' or 'loss'
	output: pointcloud sdf values for each superquadrics (n x n_sq)
	'''

	# unsqueeze
	if poses.ndim == 2:
		poses = poses.unsqueeze(0)
		sq_params = sq_params.unsqueeze(0)

	# parameters
	n_sq = len(sq_params)
	a1 = sq_params[..., [0]] 
	a2 = sq_params[..., [1]] 
	a3 = sq_params[..., [2]] 
	e1 = sq_params[..., [3]]
	e2 = sq_params[..., [4]]
	k = sq_params[..., [5]]
	if type == 'deformable_superellipsoid':
		b = sq_params[..., [6]]
		alpha = sq_params[..., [7]]
		s = sq_params[..., [8]]

	# object positions
	positions = poses[..., 0:3, 3] 
	rotations = poses[..., 0:3, 0:3] 

	# repeat voxel coordinates
	x = x.unsqueeze(0).repeat(n_sq, 1, 1).transpose(1,2) # (n_sq x 3 x n)

	# coordinate transformation
	rotations_t = rotations.permute(0,2,1)
	x_transformed = (
		- rotations_t @ positions.unsqueeze(2) 
		+ rotations_t @ x
	) # (n_sq x 3 x n) 

	# coordinates
	X = x_transformed[:, 0, :]
	Y = x_transformed[:, 1, :]
	Z = x_transformed[:, 2, :]

	if type == 'superellipsoid':
		
		# inverse tapering
		f = k / a3 * Z + 1 
		X = X / torch.abs(f)
		Y = Y / torch.abs(f)
		
		# function value
		F = (
			torch.abs(X/a1)**(2/e2)
			+ torch.abs(Y/a2)**(2/e2)
		)**(e2/e1) + torch.abs(Z/a3)**(2/e1)

	elif type == "superparaboloid":
		
		# inverse tapering
		f = k / a3 * Z + 1 
		X = X / torch.abs(f)
		Y = Y / torch.abs(f)
		
		# function value
		F = (
			torch.abs(X/a1)**(2/e2)
			+ torch.abs(Y/a2)**(2/e2)
		)**(e2/e1) - Z/a3

		# clip
		F[Z >= 0] = 1.5

	elif type == 'deformable_superellipsoid':
		
		# inverse shearing
		Z = Z - s * X
		
		# inverse bending
		beta = torch.atan2(Y, X)
		R = (torch.cos(alpha) * torch.cos(beta) + torch.sin(alpha) * torch.sin(beta)) * (X ** 2 + Y ** 2) ** (1/2)
		r = 1 / b  - (Z ** 2 + (1 / b - R) ** 2) ** (1/2)
		gamma = torch.atan2(Z, 1 / b - R)
		X = X - torch.cos(alpha) * (R - r)
		Y = Y - torch.sin(alpha) * (R - r)
		Z = gamma * 1 / b
		
		# inverse tapering
		f = k / a3 * Z + 1 
		X = X / torch.abs(f)
		Y = Y / torch.abs(f)
		
		# function value
		F = (
			torch.abs(X/a1)**(2/e2)
			+ torch.abs(Y/a2)**(2/e2)
		)**(e2/e1) + torch.abs(Z/a3)**(2/e1)

	# calculate beta
	if mode == 'e1':
		F = F ** e1
	elif mode == 'loss':
		F = (1 - torch.abs(F) ** e1) ** 2
	F = F.T

	# squeeze
	if poses.shape[0] == 1:
		F = F.squeeze(-1)

	return F