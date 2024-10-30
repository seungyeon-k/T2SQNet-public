import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms
from copy import deepcopy

from models.resnet_fpn import get_resnet_fpn_backbone
from loss.hungarian_loss import hungarian_loss
from tablewarenet.tableware import *

# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class DERT3D(nn.Module):
    def __init__(
        self,
        resnet_fpn_backbone_args,
        transformer_layer_args,
        ref_pts_predictor,
        bounding_box_predictor,
        confidence_predictor,
        class_name_list,
        loss_weights,
        num_query=900,
        num_layers=6,
        embed_dims=256,
        image_normalize=None,
        *args,
        **kwargs
        ):
        super().__init__()
        
        self.resnet_fpn_backbone = get_resnet_fpn_backbone(**resnet_fpn_backbone_args)
        self.ref_pts_predictor = ref_pts_predictor
        self.transformer_layers = nn.ModuleList([])
        self.bounding_box_predictors = nn.ModuleList([])
        self.confidence_predictors = nn.ModuleList([])
        self.class_name_list = class_name_list
        for _ in range(num_layers):
            self.transformer_layers.append(DETR3DLayer(**transformer_layer_args))
            self.bounding_box_predictors.append(deepcopy(bounding_box_predictor))
            self.confidence_predictors.append(deepcopy(confidence_predictor))
        self.query_embedding = nn.Embedding(num_query, embed_dims)
        self.cls_loss = nn.BCELoss(reduce=False)
        self.l1loss = nn.L1Loss(reduce=False)
        self.mseloss = nn.MSELoss(reduce=False)
        self.loss_weights = loss_weights[:2]
        self.zero_conf_weight = loss_weights[2]
        if image_normalize is not None:
            self.normalize = transforms.Compose([
                transforms.Normalize(image_normalize['mean'], image_normalize['std']) # imagenet
            ])
        else:
            self.normalize = None
    
    def get_imgs_features_batch(self, imgs):
        
        assert len(imgs.shape) == 5 # Batch x View x Channel x Height x Width
        
        B, V, C, H, W = imgs.size()

        feature_pyramid = self.resnet_fpn_backbone(
            imgs.reshape(B*V, C, H, W) # Batch*View x Channel x Height x Width
        )
        
        for key, value in feature_pyramid.items():
            BV, C, H, W = value.size()
            feature_pyramid[key] = value.reshape(B, int(BV/B), C, H, W) # Batch x View x c x h x w
            
        return feature_pyramid
        
    def forward(
        self,
        imgs,
        camera_projection_matrices,
        img_size
        ):
        """

        Args:
            imgs (B x V x C x H x W torch tensor): 
            camera_projection_matrices (B x V x 3 x 4 torch tensor): 
        """
        
        batch_num = imgs.shape[0]
        
        feature_pyramid = self.get_imgs_features_batch(imgs)
        queries = self.query_embedding.weight.repeat(batch_num, 1, 1) # B x query_num x C
        bounding_box_list = []
        conf_list = []
        ref_pos_list = []
        
        for layer, bounding_box_predictor, confidence_predictor in zip(self.transformer_layers, self.bounding_box_predictors, self.confidence_predictors):
            queries, reference_position = layer(queries, feature_pyramid, self.ref_pts_predictor, camera_projection_matrices, img_size)
            
            ref_pos_list.append(reference_position)
            
            bounding_box = bounding_box_predictor(queries)
            conf = confidence_predictor(queries)
            
            bounding_box_list.append(bounding_box)
            conf_list.append(conf)
        
        return bounding_box_list, conf_list, ref_pos_list, feature_pyramid
    
    def bbox_loss(self, pred_bbox, gt_bbox):
        return self.l1loss(pred_bbox, gt_bbox).sum(dim=-1)
    
    def conf_loss(self, pred_conf, gt_conf):
        return self.cls_loss(pred_conf, gt_conf).squeeze()
        
    def train_step(self, data, optimizer, device, ddp_model=None, *args, **kwargs):
        
        mask_imgs = data['mask_imgs'].to(device).float()
        if self.normalize is not None:
            mask_imgs = self.normalize(mask_imgs)
        camera_projection_matrices = data['camera_projection_matrices'].to(device).float()
        img_size = data['img_size'].to(device)
        gt_bbox = data['gt_bbox'].to(device).float()
        gt_cls = data['gt_cls'].to(device).float()
        gt_pose = data['gt_pose']
        gt_param = data['gt_param']
        
        if ddp_model is not None:
            bounding_box_pred_list, conf_pred_list, ref_pos_list, _ = ddp_model(mask_imgs, camera_projection_matrices, img_size)
        else:
            bounding_box_pred_list, conf_pred_list, ref_pos_list, _ = self(mask_imgs, camera_projection_matrices, img_size)

        C = len(self.class_name_list)
        B = len(gt_cls)
        loss = 0
        # for layer
        for bounding_box_pred, conf_pred in zip(bounding_box_pred_list, conf_pred_list):
            Q = bounding_box_pred.shape[1]
            layer_loss = 0
            layer_bbox_loss = 0
            layer_conf_loss = 0
            layer_zero_conf_loss = 0
            cls_list = []
            # for class
            for idx, (bbox_per_cls, conf_per_cls) in enumerate(
                zip(
                    bounding_box_pred.split(int(Q/C), dim=1),
                    conf_pred.split(int(Q/C), dim=1),
                    )
                ):
                cls_idx = name_to_idx[self.class_name_list[idx]]
                cls_list.append(cls_idx)
                # for batch
                for gt_bbox_unbatched, \
                    gt_cls_unbatched, \
                    bbox_per_cls_unbatched, \
                    conf_per_cls_unbatched in zip(
                        gt_bbox,
                        gt_cls,
                        bbox_per_cls,
                        conf_per_cls,
                    ):
                    # find gt object with given class idx
                    mask = (gt_cls_unbatched == cls_idx)
                    # param_head to object's pose, param
                    if mask.sum() > 0:
                        gt_bbox_per_cls = gt_bbox_unbatched[mask]
                        gt_conf = torch.ones(mask.sum()).unsqueeze(-1).to(device).float()
                        matched_loss, losses, not_matched_query_indices, _, _ = hungarian_loss(
                            preds=[bbox_per_cls_unbatched, conf_per_cls_unbatched],
                            gts=[gt_bbox_per_cls, gt_conf],
                            criterions=[self.bbox_loss, self.conf_loss],
                            weights=self.loss_weights
                            )
                        not_matched_query_indices = not_matched_query_indices.to(device).squeeze()
                        not_matched_cls = conf_per_cls_unbatched[not_matched_query_indices]
                        not_obj = torch.zeros_like(not_matched_cls)
                        zero_conf_loss = self.cls_loss(not_matched_cls, not_obj).sum()
                        layer_loss += matched_loss + zero_conf_loss * self.zero_conf_weight
                        box_loss, conf_loss = losses
                        layer_bbox_loss += box_loss
                        layer_conf_loss += conf_loss
                        layer_zero_conf_loss += zero_conf_loss
                    else:
                        not_obj = torch.zeros_like(conf_per_cls_unbatched)
                        zero_conf_loss = self.cls_loss(conf_per_cls_unbatched, not_obj).sum()
                        layer_loss += zero_conf_loss * self.zero_conf_weight
                        layer_zero_conf_loss += zero_conf_loss
            loss += layer_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {"loss": loss.item()/B,
                "last_layer_loss": layer_loss.item()/B,
                "bbox_loss" : layer_bbox_loss.item()/B,
                "conf_loss" : layer_conf_loss.item()/B,
                "zero_conf_loss" : layer_zero_conf_loss.item()/B,
                "mask_imgs": mask_imgs,
                "gt_bbox": {
                    'cls' : gt_cls,
                    'conf' : torch.ones_like(gt_cls),
                    'bbox' : gt_bbox,
                    },
                "pred_bbox": {
                    'cls' : torch.tensor(cls_list).repeat_interleave(int(Q/C)).repeat(B,1),
                    'conf' : conf_pred.squeeze(-1),
                    'bbox' : bounding_box_pred,
                    },
                "gt_pc":{
                    'cls' : gt_cls,
                    'pose': gt_pose,
                    'param': gt_param,
                },
                "ref_pos_3d": [ref_pos_list[-1]],
                "ref_pos_2d": {
                    "ref_pos" : [ref_pos_list[-1]],
                    "projection_matrices" : camera_projection_matrices,
                    "img_size" : img_size,
                }
                }
            
    def validation_step(self, data, device, *args, **kwargs):
        with torch.no_grad():
            mask_imgs = data['mask_imgs'].to(device).float()
            if self.normalize is not None:
                mask_imgs = self.normalize(mask_imgs)
            camera_projection_matrices = data['camera_projection_matrices'].to(device).float()
            img_size = data['img_size'].to(device)
            gt_bbox = data['gt_bbox'].to(device).float()
            gt_cls = data['gt_cls'].to(device).float()
            gt_pose = data['gt_pose']
            gt_param = data['gt_param']
            bounding_box_pred_list, conf_pred_list, ref_pos_list, _ = self(mask_imgs, camera_projection_matrices, img_size)
            C = len(self.class_name_list)
            B = len(gt_cls)
            loss = 0
            # for layer
            for bounding_box_pred, conf_pred in zip(bounding_box_pred_list, conf_pred_list):
                Q = bounding_box_pred.shape[1]
                layer_loss = 0
                layer_bbox_loss = 0
                layer_conf_loss = 0
                layer_zero_conf_loss = 0
                cls_list = []
                # for class
                for idx, (bbox_per_cls, conf_per_cls) in enumerate(
                    zip(
                        bounding_box_pred.split(int(Q/C), dim=1),
                        conf_pred.split(int(Q/C), dim=1),
                        )
                    ):
                    cls_idx = name_to_idx[self.class_name_list[idx]]
                    cls_list.append(cls_idx)
                    # for batch
                    for gt_bbox_unbatched, \
                        gt_cls_unbatched, \
                        bbox_per_cls_unbatched, \
                        conf_per_cls_unbatched in zip(
                            gt_bbox,
                            gt_cls,
                            bbox_per_cls,
                            conf_per_cls,
                        ):
                        # find gt object with given class idx
                        mask = (gt_cls_unbatched == cls_idx)
                        # param_head to object's pose, param, diff_pc
                        if mask.sum() > 0:
                            gt_bbox_per_cls = gt_bbox_unbatched[mask]
                            gt_conf = torch.ones(mask.sum()).unsqueeze(-1).to(device).float()
                            matched_loss, losses, not_matched_query_indices, _, _ = hungarian_loss(
                                preds=[bbox_per_cls_unbatched, conf_per_cls_unbatched],
                                gts=[gt_bbox_per_cls, gt_conf],
                                criterions=[self.bbox_loss, self.conf_loss],
                                weights=self.loss_weights
                                )
                            not_matched_query_indices = not_matched_query_indices.to(device).squeeze()
                            not_matched_cls = conf_per_cls_unbatched[not_matched_query_indices]
                            not_obj = torch.zeros_like(not_matched_cls)
                            zero_conf_loss = self.cls_loss(not_matched_cls, not_obj).sum()
                            layer_loss += matched_loss + zero_conf_loss * self.zero_conf_weight
                            box_loss, conf_loss = losses
                            layer_bbox_loss += box_loss
                            layer_conf_loss += conf_loss
                            layer_zero_conf_loss += zero_conf_loss
                        else:
                            not_obj = torch.zeros_like(conf_per_cls_unbatched)
                            zero_conf_loss = self.cls_loss(conf_per_cls_unbatched, not_obj).sum() 
                            layer_loss += zero_conf_loss * self.zero_conf_weight
                            layer_zero_conf_loss += zero_conf_loss
                loss += layer_loss
                
            return {"loss": loss.item()/B,
                    "last_layer_loss": layer_loss.item()/B,
                    "bbox_loss" : layer_bbox_loss.item()/B,
                    "conf_loss" : layer_conf_loss.item()/B,
                    "zero_conf_loss" : layer_zero_conf_loss.item()/B,
                    "mask_imgs": mask_imgs,
                    "gt_bbox": {
                        'cls' : gt_cls,
                        'conf' : torch.ones_like(gt_cls),
                        'bbox' : gt_bbox,
                        },
                    "pred_bbox": {
                        'cls' : torch.tensor(cls_list).repeat_interleave(int(Q/C)).repeat(B,1),
                        'conf' : conf_pred.squeeze(-1),
                        'bbox' : bounding_box_pred,
                        },
                    "gt_pc":{
                        'cls' : gt_cls,
                        'pose': gt_pose,
                        'param': gt_param,
                    },
                    "ref_pos_3d": [ref_pos_list[-1]],
                    "ref_pos_2d": {
                        "ref_pos" : [ref_pos_list[-1]],
                        "projection_matrices" : camera_projection_matrices,
                        "img_size" : img_size,
                    },
                    "mAP":{
                        "gt_bbox": {
                            'cls' : gt_cls,
                            'bbox' : gt_bbox,
                        },
                        "pred_bbox": {
                            'cls' : torch.tensor(cls_list).repeat_interleave(int(Q/C)).repeat(B,1),
                            'conf' : conf_pred.squeeze(-1),
                            'bbox' : bounding_box_pred,
                        },
                    }
                    }
            
            
class DETR3DLayer(nn.TransformerDecoderLayer):
    def __init__(
        self,
        embed_dims,
        num_views,
        num_levels,
        workspace,
        use_attn_weights=True,
        *args,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        self.multihead_attn = DETR3DCrossAtnn(embed_dims, num_views, num_levels, use_attn_weights)
        self.workspace = workspace

    def _mha_block(self, query, img_feats, reference_position, camera_projection_matrices, img_size):
        x = self.multihead_attn(query, img_feats, reference_position, camera_projection_matrices, img_size)
        return self.dropout2(x)
    
    def forward(
        self,
        query: Tensor,
        img_feats: Tensor,
        ref_pts_predictor,
        camera_projection_matrices,
        img_size
    ):
        
        x = query
        x = self.norm1(x + self._sa_block(x, attn_mask=None, key_padding_mask=None))
        reference_position_sig = ref_pts_predictor(x)
        
        ref_pos_x = \
            reference_position_sig[..., [0]] * (self.workspace[0][1] - self.workspace[0][0]) + self.workspace[0][0]  # x
        ref_pos_y = \
            reference_position_sig[..., [1]] * (self.workspace[1][1] - self.workspace[1][0]) + self.workspace[1][0]  # y
        ref_pos_z = \
            reference_position_sig[..., [2]] * (self.workspace[2][1] - self.workspace[2][0]) + self.workspace[2][0]  # z
        reference_position = torch.cat([ref_pos_x, ref_pos_y, ref_pos_z], dim=-1)
            
        x = self.norm2(x + self._mha_block(x, img_feats, reference_position, camera_projection_matrices, img_size))
        x = self.norm3(x + self._ff_block(x))

        return x, reference_position

class DETR3DCrossAtnn(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_views,
        num_levels,
        use_attn_weights=True,
        *args,
        **kwargs
        ):
        super().__init__()
        self.attention_weights = nn.Linear(embed_dims,
                                           num_views * num_levels)
        self.num_levels = num_levels # number of pyramid feature layers
        self.num_views = num_views
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.embed_dims = embed_dims
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.use_attn_weights = use_attn_weights
        
    def forward(
        self,
        query,
        img_feats,
        reference_position,
        camera_projection_matrices,
        img_size,
        **kwargs
        ):

        bs, num_query, _ = query.size()

        output, mask = self.get_referred_features(
            img_feats, reference_position, camera_projection_matrices, img_size)
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)
        mask = mask.unsqueeze(1).unsqueeze(-1)
        if self.use_attn_weights:
            attention_weights = self.attention_weights(query).view(
            bs, self.num_levels, self.num_views, num_query, 1)
            attention_weights = attention_weights.sigmoid() * mask
            output = output * attention_weights
        else:
            output = output * mask
        output = output.sum([1,2])
        output = self.output_proj(output)
        pos_feat = self.position_encoder(reference_position)
        return output + pos_feat
    
    def get_referred_features(
        self,
        feature_pyramid,
        reference_position,
        camera_projection_matrices,
        img_size,
        ):
        """

        Args:
            feature_pyramid (OrderedDict): features with B x V x c x h x w
            reference_position (B x query_num x 3)
            camera_projection_matrices (B x V x 3 x 4)
        """
        eps = 1e-5
        Q = reference_position.shape[1]
        
        homogeneous_reference_position = torch.cat([reference_position, torch.ones_like(reference_position[...,[0]])], dim=-1)
        projected_position = (camera_projection_matrices.unsqueeze(2) @ homogeneous_reference_position.unsqueeze(3).unsqueeze(1)).squeeze(-1) # B x V x Q x 3
        mask = projected_position[...,2] >= eps
        projected_position = projected_position[...,:2] / torch.maximum(projected_position[...,[2]], torch.tensor(eps)) # B x V x Q x 2
        img_size = img_size.unsqueeze(1).unsqueeze(1)
        projected_position[..., 0] /= img_size[...,1] # w
        projected_position[..., 1] /= img_size[...,0] # h
        mask = (
        mask & (projected_position[..., 0] > 0.0) # B x V x Q
        & (projected_position[..., 0] < 1.0)
        & (projected_position[..., 1] > 0.0)
        & (projected_position[..., 1] < 1.0))
        
        projected_position = (projected_position - 0.5) * 2
        
        sampled_feats = []
        for lvl, feat in feature_pyramid.items():
            if lvl != 'pool':
                B, V, C, H, W = feat.size()
                feat = feat.view(B * V, C, H, W)
                projected_position_lvl = projected_position.view(B * V, Q, 1, 2)
                sampled_feat = F.grid_sample(feat, projected_position_lvl, align_corners=False)
                sampled_feat = sampled_feat.view(B, V, C, Q)
                sampled_feats.append(sampled_feat)
        sampled_feats = torch.stack(sampled_feats) # pyramid_lvl_num x B x V x C x Q
        sampled_feats = sampled_feats.permute(1, 0, 2, 4, 3) # B x pyramid_lvl_num x V x Q x C
        
        return sampled_feats, mask