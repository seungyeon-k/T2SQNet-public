import numpy as np
import torch

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class runningScore_cls(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)


        return {"Overall Acc : \t": acc,
                "Mean Acc : \t": acc_cls,}


    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def depth_map_error(pred_depth, gt_depth, thld=0.05):
	"""
	Args:
		predicted_depth (B x W x H): 
		gt_depth (B x W x H):
		thld (float)
	"""
	return (((pred_depth - gt_depth).abs() / gt_depth) < thld).mean(dim=[1,2]).mean(dim=0)

def mask_iou(pred_mask, gt_mask):
	"""
	Args:
		predicted_depth (B x W x H): boolean tensor
		gt_depth (B x W x H): boolean tensor
	"""
	return ((pred_mask & gt_mask).sum(dim=[-1, -2]) / (pred_mask | gt_mask).sum(dim=[-1, -2])).mean(dim=0)

def IoU3D(pred, gt):
    """compute IoU of 3d bounding boxes

    Args:
        pred (B x 6 tensor): bounding boxes (x, y, z, w/2, h/2, d/2)
        gt (B x 6 tensor): bounding boxes (x, y, z, w/2, h/2, d/2)

    Returns:
        iou (B tensor)
    """
    pred[:, 3:6] = torch.relu(pred[:, 3:6])
    gt[:, 3:6] = torch.relu(gt[:, 3:6])
    
    pred_vol = 8 * pred[:, 3] * pred[:, 4] * pred[:, 5]
    gt_vol = 8 * gt[:, 3] * gt[:, 4] * gt[:, 5]
    intersection_rt = torch.min(
        torch.cat([pred[:, [0]] + pred[:, [3]], pred[:, [1]] + pred[:, [4]], pred[:, [2]] + pred[:, [5]]], dim=-1),
        torch.cat([gt[:, [0]] + gt[:, [3]], gt[:, [1]] + gt[:, [4]], gt[:, [2]] + gt[:, [5]]], dim=-1)
    )
    intersection_lb = torch.max(
        torch.cat([pred[:, [0]] - pred[:, [3]], pred[:, [1]] - pred[:, [4]], pred[:, [2]] - pred[:, [5]]], dim=-1),
        torch.cat([gt[:, [0]] - gt[:, [3]], gt[:, [1]] - gt[:, [4]], gt[:, [2]] - gt[:, [5]]], dim=-1)
    )
    intersection_size = torch.relu(intersection_rt - intersection_lb)
    intersection_vol = intersection_size[:, 0] * intersection_size[:, 1] * intersection_size[:, 2]
    iou = intersection_vol / (pred_vol + gt_vol - intersection_vol + 1e-12)
    return iou

def average_precision(pred_bbox, pred_conf, gt_bbox, gt_mask, iou_thld=0.75):
    """compute average precision of 3d bbox

    Args:
        pred_bbox (B x N_pred x 6 tensor): predicted bounding boxes
        pred_conf (B x N_pred tensor): predicted confidences; used for defining positive prediction
        gt_bbox (B x N_gt x 6 tensor): ground truth bounding boxes
        gt_mask (B x N_gt boolean tensor): True if the box is wanted class. False if the box is not interested class
        iou_thld (float, optional): Defaults to 0.75

    Returns:
        ap
    """
    # confidence thresholds
    conf_thld = torch.linspace(0.5, 0.95, 10).to(pred_bbox.device)
    # get all detection number and all gt number
    all_detection = (pred_conf.unsqueeze(-1) > conf_thld).sum(dim=[0, 1])
    all_gt = gt_mask.sum()
    # get iou score of all preds and gts
    B, num_instance_pred, _ = pred_bbox.size()
    _, num_instance_gt, _ = gt_bbox.size()
    pred_bbox_ = pred_bbox.repeat_interleave(num_instance_gt, dim=1).view(-1, 6)
    gt_bbox_ = gt_bbox.repeat(1, num_instance_pred, 1).view(-1, 6)
    iou_matrix = IoU3D(pred_bbox_, gt_bbox_).view(B, num_instance_pred, num_instance_gt)
    # count True Positive
    TP = ((pred_conf.unsqueeze(-1).unsqueeze(-1) > conf_thld) * gt_mask.unsqueeze(1).unsqueeze(-1) * (iou_matrix > iou_thld).unsqueeze(-1)).max(dim=1).values.sum(dim=[0,1])
    # get precision and recall
    precision = TP / (all_detection + 1e-12)
    recall = TP / (all_gt + 1e-12)
    # get AP
    precision_ = torch.tensor([torch.max(precision[:i+1]) for i in range(len(precision))]).to(pred_bbox.device)
    recall_diff = torch.flip(recall, dims=[0]).diff(prepend=torch.zeros_like(recall[[0]]))
    ap = (precision_ * recall_diff).sum()
    return ap