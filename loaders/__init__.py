import torch
import os
from loaders.detr3d_loader import DETR3DDataset
from loaders.voxel_loader import VoxelDataset

loader_type_dict = {
    'detr3d': DETR3DDataset,
    'voxel': VoxelDataset
}

def get_dataloader(cfg_data, ddp=False):
    dataset = get_dataset(cfg_data)
    list_item = cfg_data.get('list_item', [])
    
    def collate_fn(datas):
        keys = list(datas[0].keys())
        batch = {}
        for key in keys:
            if key in list_item:
                batch[key] = [data[key] for data in datas]
            else:
                batch[key] = torch.stack([data[key] for data in datas])
        return batch

    # dataloader   
    if ddp:
        world_size = int(os.environ["WORLD_SIZE"])
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=cfg_data['shuffle'])
        loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size = cfg_data['batch_size']//world_size, 
                num_workers = cfg_data['num_workers']//world_size,
                sampler = sampler,
                collate_fn = collate_fn)
    else:
        loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size = cfg_data['batch_size'], 
                num_workers = cfg_data['num_workers'], 
                shuffle = cfg_data['shuffle'],
                collate_fn = collate_fn)
    
    return loader

def get_dataset(cfg_data):
    name = cfg_data['dataset']
    dataset = loader_type_dict[name](**cfg_data)
    return dataset


    