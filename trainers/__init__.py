from trainers.bbox_logger import BBoxLogger
from trainers.voxel_logger import VoxelLogger
from trainers.base import BaseTrainer

def get_trainer(cfg):
    trainer_type = cfg.get('type', None)
    device = cfg['device']
    if trainer_type == 'base':
        trainer = BaseTrainer(cfg, device=device)
    return trainer

def get_logger(cfg):
    if cfg.type == 'bboxlogger':
        logger = BBoxLogger(**cfg)
    elif cfg.type == 'voxellogger':
        logger = VoxelLogger(**cfg)
    return logger