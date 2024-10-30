import os
import torch
from tqdm import tqdm, trange
import torch.distributed as dist

class BaseTrainer:
    """Trainer for a conventional iterative training of model"""
    def __init__(self, training_cfg, device):
        self.cfg = training_cfg
        self.device = device

    def train(self, model, opt, d_dataloaders, scheduler=None, loggers=None, logdir=''):
        cfg = self.cfg
        ddp = cfg.get("ddp", False)
        train_loader, val_loader = d_dataloaders['training'], d_dataloaders['validation']
        train_logger, val_logger = loggers['training'], loggers['validation']
        if ddp:
            is_main = dist.get_rank() == 0
        else:
            is_main = True
        if cfg["detect_nan"]:
            torch.autograd.detect_anomaly(check_nan=True)
            
        i = 0
        start_epoch = 0
        
        load_state_dir = cfg.get('load_state_dir', None)
        if load_state_dir is not None:
            model_state, opt_state, train_logger, val_logger, start_epoch, i = self.load_state(load_state_dir)
            if start_epoch is None:
                start_epoch = 0
            model.load_state_dict(model_state)
            opt.load_state_dict(opt_state)

        t_epoch = trange(start_epoch, cfg.n_epoch)
        for i_epoch in t_epoch:   
            t_epoch.set_description(f'Epoch - {i_epoch}')
            if ddp:
                train_loader.sampler.set_epoch(i_epoch)
            t_iter = tqdm(train_loader, leave=False)
            i_iter = 0
            model.train()
            for data in t_iter:
                if ddp:
                    data_train = model.module.train_step(data, optimizer=opt, device=self.device, ddp_model=model)
                else:
                    data_train = model.train_step(data, optimizer=opt, device=self.device)
                train_logger.process_iter(data_train)
                if is_main:
                    train_logger.log_by_interval(i)
                t_iter.set_description(f'Iter - {i_iter}')
                print_str = train_logger.get_scalars()
                t_iter.set_postfix(**print_str)
                i += 1
                i_iter += 1
            if is_main:
                model.eval()
                for val_data in val_loader:
                    if ddp:      
                        d_val = model.module.validation_step(val_data, device=self.device)
                    else:
                        d_val = model.validation_step(val_data, device=self.device)
                    val_logger.process_iter(d_val)
                val_logger.log_all(i)
            
                best_model_booleans = val_logger.get_best_model_booleans()
                for metric_name, is_best in best_model_booleans.items():
                    if is_best:
                        if ddp:
                            model_saved = model.module
                        else:
                            model_saved = model
                        self.save_model(model_saved, opt, train_logger, val_logger, logdir, best=True, metric_name=metric_name, i=i, i_epoch=i_epoch)
                val_logger.reset()
            
            if scheduler is not None:
                scheduler.step()
                
        if is_main:
            if ddp:
                model_saved = model.module
            else:
                model_saved = model
            self.save_model(model_saved, opt, train_logger, val_logger, logdir, best=False, i=i, i_epoch=i_epoch)
        if ddp:
            dist.destroy_process_group()
        
        return model
    
    def save_model(self, model, optimizer, train_logger, val_logger, logdir, best=False, metric_name=None, i=None, i_epoch=None):
        if best:
            pkl_name = "model_best_" + metric_name + ".pkl"
        else:
            if i is not None:
                pkl_name = "model_iter_{}.pkl".format(i)
            else:
                pkl_name = "model_epoch_{}.pkl".format(i_epoch)
        state = {
            "model_state": model.state_dict(),
            "opt": optimizer.state_dict(),
            "train_logger": train_logger,
            "val_logger": val_logger,
            "epoch": i_epoch, 
            'iter': i, 
            }
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)

    def load_state(self, logdir):
        file = torch.load(logdir)
        model_state = file["model_state"]
        opt_state = file["opt"]
        train_logger = file["train_logger"]
        val_logger = file["val_logger"]
        epoch = file["epoch"]
        iter = file["iter"]
        
        return model_state, opt_state, train_logger, val_logger, epoch, iter