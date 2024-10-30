import torch
import torch.nn as nn
import torch.nn.functional as F

class ChamferLoss(nn.Module):
    def __init__(self, use_mask=False, reduce=False):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.use_mask = use_mask
        self.reduce = reduce
        
    def batch_pairwise_dist(self, x, y):
        _, num_points_x, _ = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        if x.get_device() != -1:
            diag_ind_x = diag_ind_x.cuda(x.get_device())
            diag_ind_y = diag_ind_y.cuda(x.get_device())
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def forward(self, preds, gts):
        if self.use_mask:
            B = preds.shape[0]
            loss_1 = 0
            loss_2 = 0
            for idx in torch.unique(gts[...,3]):
                pred = preds[preds[...,3] == idx][...,0:3].view(B, -1, 3)
                gt = gts[gts[...,3] == idx][...,0:3].view(B, -1, 3)
                P = self.batch_pairwise_dist(gt, pred)
                mins, _ = torch.min(P, dim=1)
                loss_1 += torch.mean(mins, dim=-1)
                mins, _ = torch.min(P, dim=2)
                loss_2 += torch.mean(mins, dim=-1)
        else:
            P = self.batch_pairwise_dist(gts, preds)
            mins, _ = torch.min(P, dim=1)
            loss_1 = torch.mean(mins, dim=-1)
            mins, _ = torch.min(P, dim=2)
            loss_2 = torch.mean(mins, dim=-1)
        if self.reduce:
            return (loss_1 + loss_2).mean()
        else:
            return loss_1 + loss_2
