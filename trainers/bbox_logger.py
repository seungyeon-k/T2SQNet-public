import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from omegaconf import OmegaConf
import wandb

from tablewarenet.tableware import idx_to_name, idx_to_class
from loss.chamfer_loss import ChamferLoss
from loss.metrics import averageMeter, average_precision

resize = torchvision.transforms.Resize((32, 32))
chamfer_loss = ChamferLoss()

class BBoxLogger:
	def __init__(self, prefix, items, best_model_metric=None, *args, **kwargs):
		self.prefix = prefix
		self.items = OmegaConf.to_container(items)
		self.best_model_metric = best_model_metric
		self.best_model_metric_val = None
		self.reset()

	def reset(self):
		for key, val in self.items.items():
			if val["type"] == 'average_meter':
				self.items[key]["value"] = averageMeter()
			elif self.items[key]["type"] == 'mAP':
				self.items[key]["value"] = {"gt_bbox": {
							'cls' : None,
							'bbox' : None,
						},
						"pred_bbox": {
							'cls' : None,
							'conf' : None,
							'bbox' : None,
						}}
			else:
				self.items[key]["value"] = None
		   
	def process_iter(self, data):
		for key, val in data.items():
			if key in self.items:
				if self.items[key]["type"] == 'average_meter':
					self.items[key]["value"].update(val)
				elif self.items[key]["type"] == 'mAP':
					self.items[key]["value"]['gt_bbox']['cls'] = val['gt_bbox']['cls'] if self.items[key]["value"]['gt_bbox']['cls'] is None\
						else torch.cat([self.items[key]["value"]['gt_bbox']['cls'], val['gt_bbox']['cls']], dim=0)
					self.items[key]["value"]['gt_bbox']['bbox'] = val['gt_bbox']['bbox'] if self.items[key]["value"]['gt_bbox']['bbox'] is None\
						else torch.cat([self.items[key]["value"]['gt_bbox']['bbox'], val['gt_bbox']['bbox']], dim=0)
					self.items[key]["value"]['pred_bbox']['cls'] = val['pred_bbox']['cls'] if self.items[key]["value"]['pred_bbox']['cls'] is None\
						else torch.cat([self.items[key]["value"]['pred_bbox']['cls'], val['pred_bbox']['cls']], dim=0)
					self.items[key]["value"]['pred_bbox']['conf'] = val['pred_bbox']['conf'] if self.items[key]["value"]['pred_bbox']['conf'] is None\
						else torch.cat([self.items[key]["value"]['pred_bbox']['conf'], val['pred_bbox']['conf']], dim=0)
					self.items[key]["value"]['pred_bbox']['bbox'] = val['pred_bbox']['bbox'] if self.items[key]["value"]['pred_bbox']['bbox'] is None\
						else torch.cat([self.items[key]["value"]['pred_bbox']['bbox'], val['pred_bbox']['bbox']], dim=0)
				else:
					self.items[key]["value"] = val
	
	def get_scalars(self):
		scalars = {}
		for key, val in self.items.items():
			if val["type"] == 'scalar':
				scalars[key] = val["value"]
			elif val["type"] == 'average_meter' :
				scalars[key] = val["value"].avg
		return scalars
	
	def log_by_interval(self, i):
		for key, _ in self.items.items():
			if i % self.items[key]["interval"] == 0 and i > 0:
				self.log_wandb_instance(key, i)
		
	def log_all(self, i):
		for key, _ in self.items.items():
			self.log_wandb_instance(key, i)
				
	def log_wandb_instance(self, key, i):
		type = self.items[key]["type"]
		val = self.items[key]["value"]
		dict = self.items[key]
		key = self.prefix + '_' + key
		
		if type == "scalar":
			wandb_dict = {key : val}
			wandb.log(wandb_dict, step=i)
			
		elif type == "average_meter":
			wandb_dict = {key : val.avg}
			wandb.log(wandb_dict, step=i)
			
		elif type == "image":
			wandb_dict = {key : [wandb.Image(val)]}
			wandb.log(wandb_dict, step=i)
			
		elif type == "grid_images":
			B, V, C, H, W = val.size()
			grid = make_grid(val.reshape(B*V, C, H, W), nrow=V).permute(1, 2, 0).cpu().detach().numpy()
			wandb_dict = {key : [wandb.Image(grid)]}
			wandb.log(wandb_dict, step=i)

		elif type == "bbox":
			pc = self.items[dict["with_pc"]]["value"]
			cls_pc, pose_pc, param_pc = pc["cls"], pc["pose"], pc["param"]
			B = len(cls_pc)
			wandb_pc_list = []
			pc_with_color = []
			for batch_idx in range(B):
				cls_temp = cls_pc[batch_idx]
				cls_temp = cls_temp[cls_temp != -1]
				N = len(cls_temp)
				temp_pc = []
				for obj_idx in range(N):
					obj = idx_to_class[cls_temp[obj_idx].item()](
						SE3=pose_pc[batch_idx][obj_idx],
						params=param_pc[batch_idx][obj_idx],
						device='cpu'
					)
					pc_numpy = obj.get_point_cloud()
					temp_pc.append(pc_numpy)
				temp_pc = np.concatenate(temp_pc, axis=0)
				temp_rgb = np.zeros_like(temp_pc)
				temp_rgb[...,:] = np.array(self.items[dict["with_pc"]]["color"])
				pc_with_color.append(np.concatenate([temp_pc, temp_rgb], axis=1))
			pc_box_list = []
			classes = val["cls"]
			confs = val['conf']
			bboxes = val['bbox']
			thld = dict["thld"]
			for classes_unbatched, confs_unbatched, bboxes_unbatched, pc_rgb in zip(classes, confs, bboxes, pc_with_color):
				all_boxes = []
				for cls, conf, bbox in zip(classes_unbatched, confs_unbatched, bboxes_unbatched):
					# get coordinates and labels
					if conf > thld and cls != -1:
						bbox = bbox.tolist()
						box_data = {
						"corners": [
							[bbox[0] - bbox[3], bbox[1] - bbox[4], bbox[2] - bbox[5]],
							[bbox[0] - bbox[3], bbox[1] + bbox[4], bbox[2] - bbox[5]],
							[bbox[0] - bbox[3], bbox[1] - bbox[4], bbox[2] + bbox[5]],
							[bbox[0] + bbox[3], bbox[1] - bbox[4], bbox[2] - bbox[5]],
							[bbox[0] + bbox[3], bbox[1] + bbox[4], bbox[2] - bbox[5]],
							[bbox[0] - bbox[3], bbox[1] + bbox[4], bbox[2] + bbox[5]],
							[bbox[0] + bbox[3], bbox[1] - bbox[4], bbox[2] + bbox[5]],
							[bbox[0] + bbox[3], bbox[1] + bbox[4], bbox[2] + bbox[5]],
						],
						"label": idx_to_name[cls.item()],
						"color": [int(255 * (1-(conf.item()-thld)/(1-thld))), int(255 * (conf.item()-thld)/(1-thld)), 0],
						}
						all_boxes.append(box_data)
				pc_box = wandb.Object3D({"type": "lidar/beta", "points": pc_rgb, "boxes": np.array(all_boxes)})
				pc_box_list.append(pc_box)
			wandb_dict = {key : pc_box_list}
			wandb.log(wandb_dict, step=i)
		
		elif type == "param_to_pc":
			cls, pose, param = val["cls"], val["pose"], val["param"]
			color = dict["color"]
			B = len(cls)
			wandb_pc_list = []
			for batch_idx in range(B):
				cls_temp = cls[batch_idx]
				cls_temp = cls_temp[cls_temp != -1]
				N = len(cls_temp)
				temp_pc = []
				for obj_idx in range(N):
					obj = idx_to_class[cls_temp[obj_idx].item()](
						SE3=pose[batch_idx][obj_idx],
						params=param[batch_idx][obj_idx],
						device='cpu'
					)
					pc_numpy = obj.get_point_cloud()
					temp_pc.append(pc_numpy)
				temp_pc = np.concatenate(temp_pc, axis=0)
				temp_rgb = np.zeros_like(temp_pc)
				temp_rgb[...,:] = np.array(color)
				pc = np.concatenate([temp_pc, temp_rgb], axis=1)
				wandb_pc_list.append(wandb.Object3D(pc))
			wandb_dict = {key : wandb_pc_list}
			wandb.log(wandb_dict, step=i)
		
		elif type == "ref_pos_3d":
			ref_pos_list, color = val, dict["color"]
			other_pc = self.items[dict["with_pc"]]["value"]
			other_cls, other_pose, other_param, other_color = other_pc["cls"], other_pc["pose"], other_pc["param"], self.items[dict["with_pc"]]["color"]
			B = len(other_cls)
			wandb_pc_list = []
			for batch_idx in range(B):
				# add pc of 'with_pc'
				other_cls_temp = other_cls[batch_idx]
				other_cls_temp = other_cls_temp[other_cls_temp != -1]
				N_other = len(other_cls_temp)
				other_temp_pc = []
				for obj_idx in range(N_other):
					obj = idx_to_class[other_cls_temp[obj_idx].item()](
						SE3=other_pose[batch_idx][obj_idx],
						params=other_param[batch_idx][obj_idx],
						device='cpu'
					)
					other_pc_numpy = obj.get_point_cloud()
					other_temp_pc.append(other_pc_numpy)
				other_temp_pc = np.concatenate(other_temp_pc, axis=0)
				other_temp_rgb = np.zeros_like(other_temp_pc)
				other_temp_rgb[...,:] = np.array(other_color)
				other_pc = np.concatenate([other_temp_pc, other_temp_rgb], axis=1)
				for ref_pos in ref_pos_list:
					ref_pc = ref_pos[batch_idx].reshape(-1, 3).detach().cpu().numpy()
					rgb = np.zeros_like(ref_pc)
					rgb[...,:] = np.array(color)
					ref_pc = np.concatenate([ref_pc, rgb], axis=1)
					all_pc = np.concatenate([ref_pc, other_pc], axis=0)
					wandb_pc_list.append(wandb.Object3D(all_pc))
			wandb_dict = {key : wandb_pc_list}
			wandb.log(wandb_dict, step=i)
				
		elif type == "ref_pos_2d":
			ref_pos_list, projection_matrices, img_size = val["ref_pos"], val["projection_matrices"], val["img_size"]
			other_pc = self.items[dict["with_pc"]]["value"]
			other_cls, other_pose, other_param = other_pc["cls"], other_pc["pose"], other_pc["param"]
			B = len(other_cls)
			wandb_img_list = []
			for batch_idx in range(B):
				# add pc of 'with_pc'
				other_cls_temp = other_cls[batch_idx]
				other_cls_temp = other_cls_temp[other_cls_temp != -1]
				N_other = len(other_cls_temp)
				obj_pc = []
				for obj_idx in range(N_other):
					obj = idx_to_class[other_cls_temp[obj_idx].item()](
						SE3=other_pose[batch_idx][obj_idx],
						params=other_param[batch_idx][obj_idx],
						device='cpu'
					)
					other_pc_numpy = obj.get_point_cloud()
					obj_pc.append(other_pc_numpy)
				obj_pc = np.concatenate(obj_pc, axis=0)
				obj_rgb = np.zeros_like(obj_pc)
				obj_rgb[:,2] = 1
				for ref_pos in ref_pos_list:
					ref_pc = ref_pos[batch_idx].reshape(-1, 3)
					projection_matrix = projection_matrices[batch_idx]
					ref_rgb = np.zeros_like(ref_pc.detach().cpu().numpy())
					ref_rgb[:,0] = 1
					all_pc = torch.cat([torch.from_numpy(obj_pc).to(ref_pc), ref_pc], axis=0)
					all_rgb = np.concatenate([obj_rgb, ref_rgb], axis=0)
					all_pc = torch.cat([all_pc, torch.ones_like(all_pc[...,[0]])], dim=-1)
					projected_position = (projection_matrix.unsqueeze(1) @ all_pc.unsqueeze(0).unsqueeze(-1)).squeeze(-1) # V x Q x 3
					projected_position = projected_position[...,:2] / torch.maximum(projected_position[...,[2]], torch.tensor(1e-5)) # V x Q x 2
					projected_position[..., 0] /= img_size[0,1] # w
					projected_position[..., 1] /= img_size[0,0] # h
					for img_pc in projected_position:
						fig = plt.figure()
						img_pc = img_pc.detach().cpu().numpy()
						plt.scatter(img_pc[:,0], 1-img_pc[:,1], color=all_rgb, marker='.')
						plt.xlim(0, 1)
						plt.ylim(0, 1)
						wandb_img_list.append(wandb.Image(fig))
						plt.close()
			wandb_dict = {key : wandb_img_list}
			wandb.log(wandb_dict, step=i)
		
		elif type == "param_with_conf_to_pc":
			cls, conf, pose, param = val["cls"], val["conf"], val["pose"], val["param"]
			thld = dict["thld"]
			color = dict["color"]
			pose = [list(x) for x in zip(*pose)]
			pose_new = []
			for i1 in pose:
				p = []
				for j1 in i1:
					p += [*j1]
				pose_new.append(p)
			pose = pose_new
			param = [list(x) for x in zip(*param)]
			param_new = []
			for i1 in param:
				p = []
				for j1 in i1:
					p += [*j1]
				param_new.append(p)
			param = param_new
			other_pc = self.items[dict["with_pc"]]["value"]
			other_cls, other_pose, other_param, other_color = other_pc["cls"], other_pc["pose"], other_pc["param"], self.items[dict["with_pc"]]["color"]
			B = len(cls)
			wandb_pc_list = []
			for batch_idx in range(B):
				# add pc of 'with_pc'
				other_cls_temp = other_cls[batch_idx]
				other_cls_temp = other_cls_temp[other_cls_temp != -1]
				N_other = len(other_cls_temp)
				other_temp_pc = []
				for obj_idx in range(N_other):
					obj = idx_to_class[other_cls_temp[obj_idx].item()](
						SE3=other_pose[batch_idx][obj_idx],
						params=other_param[batch_idx][obj_idx],
						device='cpu'
					)
					other_pc_numpy = obj.get_point_cloud()
					other_temp_pc.append(other_pc_numpy)
				other_temp_pc = np.concatenate(other_temp_pc, axis=0)
				other_temp_rgb = np.zeros_like(other_temp_pc)
				other_temp_rgb[...,:] = np.array(other_color)
				other_pc = np.concatenate([other_temp_pc, other_temp_rgb], axis=1)
				# add original pc
				cls_temp = cls[batch_idx]
				cls_temp = cls_temp[cls_temp != -1]
				N = len(cls_temp)
				temp_pc = []
				for obj_idx in range(N):
					if conf[batch_idx][obj_idx] > thld:
						obj = idx_to_class[cls_temp[obj_idx].item()](
							SE3=pose[batch_idx][obj_idx],
							params=param[batch_idx][obj_idx],
							device='cpu'
						)
						pc_numpy = obj.get_point_cloud()
						temp_pc.append(pc_numpy)
				if len(temp_pc) > 0:
					temp_pc = np.concatenate(temp_pc, axis=0)
					temp_rgb = np.zeros_like(temp_pc)
					temp_rgb[...,:] = np.array(color)
					pc = np.concatenate([temp_pc, temp_rgb], axis=1)
					# concate all pc
					all_pc = np.concatenate([other_pc, pc], axis=0)
				else:
					all_pc = other_pc
				wandb_pc_list.append(wandb.Object3D(all_pc))
			wandb_dict = {key : wandb_pc_list}
			wandb.log(wandb_dict, step=i)
			
		elif type == "mAP":
			pred_cls, pred_conf, pred_bbox = val["pred_bbox"]['cls'], val["pred_bbox"]['conf'], val["pred_bbox"]['bbox']
			gt_cls, gt_bbox  = val["gt_bbox"]['cls'], val["gt_bbox"]['bbox']
			iou_thlds = dict["iou_thld"]
			classes = gt_cls.unique()
			B = len(pred_conf)
			C = len(classes)
			for iou_thld in iou_thlds:
				ap = 0
				for c in classes:
					if c != -1:
						gt_mask = gt_cls == c.item()
						pred_mask = pred_cls == c.item()
						pred_bbox_ = pred_bbox[pred_mask].reshape(B, -1, 6)
						pred_conf_ = pred_conf[pred_mask].reshape(B, -1)
						ap += average_precision(pred_bbox_, pred_conf_, gt_bbox, gt_mask, iou_thld)
				map = ap / C
				wandb_dict = {key+'_'+str(iou_thld) : map.item()}
				wandb.log(wandb_dict, step=i)
				
	def get_best_model_booleans(self):
		key = self.best_model_metric
		type = self.items[key]["type"]
		val = self.items[key]["value"]
		dict = self.items[key]    
		
		if type == "scalar":
			if self.best_model_metric_val is None:
				bool = True
			elif dict['criterion'] == '↓':
				bool = self.best_model_metric_val > val
			elif dict['criterion'] == '↑':
				bool = self.best_model_metric_val < val
			if bool:
				self.best_model_metric_val = val
			return {key: bool}
			
		elif type == "average_meter":
			if self.best_model_metric_val is None:
				bool = True
			elif dict['criterion'] == '↓':
				bool = self.best_model_metric_val > val.avg
			elif dict['criterion'] == '↑':
				bool = self.best_model_metric_val < val.avg
			if bool:
				self.best_model_metric_val = val.avg
			return {key: bool}
		
		elif type== "mAP":
			pred_cls, pred_conf, pred_bbox = val["pred_bbox"]['cls'], val["pred_bbox"]['conf'], val["pred_bbox"]['bbox']
			gt_cls, gt_bbox  = val["gt_bbox"]['cls'], val["gt_bbox"]['bbox']
			iou_thlds = dict["iou_thld"]
			if self.best_model_metric_val is None:
				self.best_model_metric_val = {}
				for iou_thld in iou_thlds:
					self.best_model_metric_val[key+'_'+str(iou_thld)] = None
			classes = gt_cls.unique()
			B = len(pred_conf)
			C = len(classes)
			return_dict = {}
			for iou_thld in iou_thlds:
				ap = 0
				for c in classes:
					if c != -1:
						gt_mask = gt_cls == c.item()
						pred_mask = pred_cls == c.item()
						pred_bbox_ = pred_bbox[pred_mask].reshape(B, -1, 6)
						pred_conf_ = pred_conf[pred_mask].reshape(B, -1)
						ap += average_precision(pred_bbox_, pred_conf_, gt_bbox, gt_mask, iou_thld)
				map = ap / C
				if self.best_model_metric_val[key+'_'+str(iou_thld)] is None:
					bool = True
				else:
					bool = self.best_model_metric_val[key+'_'+str(iou_thld)] < map
				if bool:
					self.best_model_metric_val[key+'_'+str(iou_thld)] = map
				return_dict[key+'_'+str(iou_thld)] = bool
			return return_dict