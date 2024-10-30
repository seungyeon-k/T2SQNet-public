from scipy.optimize import linear_sum_assignment
import torch
import numpy as np

def hungarian_loss(preds, gts, criterions, weights=None):
    """
    preds : [(instance_num x pred_dim)] : model prediction
    gts : [(instance_num x gt_dim)] : labels
    criterions : [criterions], where criterion : (batch x output_dim), (batch x label_dim) --> batch size tensor
    weights : [float] : loss weights
    outputs : tuple([losses], not_selected_indices)
    note that num_instance > num_pred
    """
        
    if weights == None:
        weights = [1. for _ in range(len(preds))]
        
    cost_matrices = []
    for pred, gt, criterion in zip(preds, gts, criterions):
        num_instance_pred, _ = pred.size()
        num_instance_gt, _ = gt.size()
        pred_spanned = pred.repeat_interleave(num_instance_gt, dim=0)
        gt_spanned = gt.repeat(num_instance_pred, 1)
        cost_matrix = criterion(pred_spanned, gt_spanned).view(num_instance_pred, num_instance_gt)
        cost_matrices.append(cost_matrix)
    
    weights_tensor = torch.tensor(weights).unsqueeze(-1).unsqueeze(-1).to(pred.device)
    cost_matrix_temp = (weights_tensor * torch.stack(cost_matrices)).sum(dim=0).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix_temp)
    not_selected_index = torch.tensor(list(set(np.arange(num_instance_pred)) - set(row_ind))).long().to(pred.device)
    losses = []
    total_loss = 0
    for cost_matrix, weight in zip(cost_matrices, weights):
        loss = cost_matrix[row_ind, col_ind].sum()
        total_loss += loss * weight
        losses.append(loss)
    
    return total_loss, losses, not_selected_index, row_ind, col_ind

