import torch
from math import floor, ceil
from copy import deepcopy

def voxel_carving(
		mask_imgs, camera, bbox, voxel_size, 
		device='cuda:0', smoothed=False):
	
	# camera
	projections = camera['projection_matrices'].float()
	imgH, imgW = camera['camera_image_size']
	silhouettes = mask_imgs

	# function for calculation
	def pointer_at_batch(image, u, v):
		u = torch.clamp(u, 0, image.shape[1] - 1)
		v = torch.clamp(v, 0, image.shape[0] - 1)
		indices = (v * imgW + u).long()
		return image.view(-1)[indices]

	# parameters
	w = floor(2 * bbox[3] / voxel_size + 0.5)
	h = floor(2 * bbox[4] / voxel_size + 0.5)
	d = floor(2 * bbox[5] / voxel_size + 0.5)
	grid_x = torch.linspace(
		bbox[0] - bbox[3] + voxel_size * 0.5, bbox[0] + bbox[3] - voxel_size * 0.5, w)
	grid_y = torch.linspace(
		bbox[1] - bbox[4] + voxel_size * 0.5, bbox[1] + bbox[4] - voxel_size * 0.5, h)
	grid_z = torch.linspace(
		bbox[2] - bbox[5] + voxel_size * 0.5, bbox[2] + bbox[5] - voxel_size * 0.5, d)
	x, y, z = torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
	pts = torch.stack([x.flatten(), y.flatten(), z.flatten(), torch.ones_like(z.flatten())])

	# carving
	filled = []
	r = voxel_size * 0.5
	for P, im in zip(projections, silhouettes):
		
		# initialize
		P = P.to(device)
		im = im.to(device)

		# translate
		box_corners = torch.tensor([
				[-r, -r, -r, 0],
				[-r, -r, r, 0],
				[r, -r, -r, 0],
				[r, -r, r, 0],
				[-r, r, -r, 0],
				[-r, r, r, 0],
				[r, r, -r, 0],
				[r, r, r, 0]
			]).to(device)

		# initialize
		results = torch.zeros(pts.shape[-1], dtype=torch.int8).to(device)

		# augment values
		for i in range(len(box_corners)):
			pts_corner = deepcopy(pts) # 4 x n
			pts_corner = pts_corner.to(device) + box_corners[i].unsqueeze(-1)
			
			# get uvs
			uvs = P @ pts_corner
			z = deepcopy(uvs[2, :])
			uvs = uvs / z
			us = uvs[0, :]
			vs = uvs[1, :]

			# valid mask
			valid_mask = (us >= 0) & (us <= imgW - 1) & (vs >= 0) & (vs <= imgH - 1)

			ui = torch.clamp(us.long(), 0, imgW - 2)
			vi = torch.clamp(vs.long(), 0, imgH - 2)
			pu = us - ui.float()
			pv = vs - vi.float()

			value00 = pointer_at_batch(im, ui, vi)
			value01 = pointer_at_batch(im, ui, vi + 1)
			value10 = pointer_at_batch(im, ui + 1, vi)
			value11 = pointer_at_batch(im, ui + 1, vi + 1)

			interpolated_values = (value00 * (1 - pv) + value01 * pv) * (1 - pu) + \
									(value10 * (1 - pv) + value11 * pv) * pu

			results[valid_mask] = (
				(results[valid_mask] == 1)
				| (interpolated_values[valid_mask] > 0)
			).to(torch.int8)

		filled.append(results)

	# stack results
	filled = torch.stack(filled)
	filled = torch.sum(filled, dim=0).reshape(w, h, d)

	if smoothed:
		occupancy = filled / len(projections)
	else:
		occupancy = torch.zeros_like(filled)
		occupancy[filled >= len(projections)] = 1

	return occupancy
