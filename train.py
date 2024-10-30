import numpy as np
import argparse
import os
import random
import itertools
import wandb
import torch
import torch.distributed as dist
from torch.distributed import init_process_group
from omegaconf import OmegaConf
from datetime import datetime

from models import get_model
from trainers import get_trainer, get_logger
from loaders import get_dataloader
from trainers.optimizers import get_optimizer
from trainers.schedulers import get_scheduler
from utils.yaml_utils import save_yaml, parse_unknown_args, parse_nested_args, load_cfg_with_base

def run(cfg):
    # Setup seeds
    seed = cfg.get("seed", 1)
    print(f"running with random seed : {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(8)

    # Setup device
    device = cfg.device
    cfg['trainer']["device"] = device
    ddp = cfg["trainer"].get("ddp", False)
    if ddp:
        init_process_group(
            backend="nccl",
            rank=int(os.environ["RANK"]),
            world_size= int(os.environ["WORLD_SIZE"]),
            )
        torch.cuda.set_device(device)
        dist.barrier()
    
    # Setup dataloader
    d_dataloaders = {}
    loggers = {}
    for key, dataloader_cfg in cfg["data"].items():
        if key == 'training':
            d_dataloaders[key] = get_dataloader(dataloader_cfg, ddp=ddp)
        else:
            d_dataloaders[key] = get_dataloader(dataloader_cfg)
    
    # Setup logger
    for key, logger_cfg in cfg["logger"].items():
        loggers[key] = get_logger(logger_cfg)
    
    # Setup model
    model = get_model(cfg['model']).to(device)
    if ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [device])
    
    # Setup trainer
    trainer = get_trainer(cfg['trainer'])

    # Setup optimizer, lr_scheduler and loss function
    params = itertools.chain(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = get_optimizer(cfg["trainer"]["optimizer"], params)
    scheduler_cfg = cfg["trainer"].get("scheduler", None)
    if scheduler_cfg is not None:
        scheduler = get_scheduler(cfg["trainer"]["scheduler"], optimizer)
    else:
        scheduler = None
    model = trainer.train(
        model,
        optimizer,
        d_dataloaders,
        scheduler=scheduler,
        loggers=loggers,
        logdir=logdir,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--device", default="any")
    parser.add_argument("--logdir", default="results")
    parser.add_argument("--run", default=None)
    parser.add_argument('--local_rank', nargs='+', default=[])
    args, unknown = parser.parse_known_args()
    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)
    print(d_cmd_cfg)
    
    print(args.config)
    cfg = load_cfg_with_base(args.config)
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)
    print(OmegaConf.to_yaml(cfg))
    
    ddp = False
    if args.device == "cpu":
        cfg["device"] = f"cpu"
        rank = 0
    elif args.device == "any":
        cfg["device"] = "cuda"
        rank = 0
    elif args.device == 'ddp':
        ddp = True
        rank = int(os.environ["RANK"])
        cfg["trainer"]["ddp"] = True
        cfg["device"] = int(os.environ["LOCAL_RANK"])
    else:
        cfg["device"] = f"cuda:{args.device}"
        rank = 0

    if args.run is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
    else:
        run_id = args.run

    config_basename = os.path.basename(args.config).split(".")[0]

    if args.logdir is None:
        logdir = os.path.join(cfg['logdir'], config_basename, str(run_id))
    else:
        logdir = os.path.join(args.logdir, config_basename, str(run_id))
    if os.path.exists(logdir):
        logdir = logdir + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")
    
    print("Result directory: {}".format(logdir))
    # copy config file
    copied_yml = os.path.join(logdir, os.path.basename(args.config))
    print(f"config saved as {copied_yml}")
    if rank == 0:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        save_yaml(copied_yml, OmegaConf.to_yaml(cfg))
        wandb.init(
            entity=cfg['entity'],
            project=cfg['wandb_project_name'],
            config=OmegaConf.to_container(cfg),
            name=logdir
            )
    run(cfg)
