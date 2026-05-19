import os
import pandas as pd 
import fnmatch
import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange
from utils import timetensor_normalization, get_transform

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def split_into_patches(evs_tmap, patch_h, patch_w):
    evs_tmap_patches = rearrange(evs_tmap, '(h p1) (w p2) -> h w (p1 p2)', p1=patch_h, p2=patch_w)
    
    return 


class TrainDataset(Dataset):
    """
        TrainDataset
    """

    def __init__(self, data_dir='data', transform=None, training=True, patch_size=16):
        if training:
            sub_dir = 'train'
        else:
            sub_dir = 'test'
        self.flicker_dir = os.path.join(data_dir, 'flicker', sub_dir)
        self.gt_dir = os.path.join(data_dir, 'noflicker', sub_dir)
        self.names = sorted(fnmatch.filter(os.listdir(self.flicker_dir), '*.npy'))
        self.patch_h, self.patch_w = pair(patch_size)
        self.transform = None
        if transform:
            self.transform = get_transform(transform)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        flk_ev_tmap = np.load(os.path.join(self.flicker_dir, self.names[index]))
        noflk_ev_tmap = np.load(os.path.join(self.gt_dir, self.names[index]))
        height, width, channels = flk_ev_tmap.shape
        name = self.names[index].split('.')[0]
        
        # Need some normalization for flk_ev_tmap and noflk_ev_tmap
        # flk_ev_tmap, input_mean, input_stddev = timetensor_normalization(flk_ev_tmap, channels)
        # noflk_ev_tmap, gt_mean, gt_stddev = timetensor_normalization(noflk_ev_tmap, channels)
        
        flk_ev_tmap = torch.tensor(np.transpose(flk_ev_tmap, (2, 0, 1)), dtype=torch.float32)
        noflk_ev_tmap = torch.tensor(np.transpose(noflk_ev_tmap, (2, 0, 1)), dtype=torch.float32)

        if self.transform:
            flk_ev_tmap = self.transform(flk_ev_tmap)
            noflk_ev_tmap = self.transform(noflk_ev_tmap)

        return {'flk_ev_stack': flk_ev_tmap, 'noflk_ev_stack': noflk_ev_tmap, 
                # 'input_mean': input_mean, 'input_stddev': input_stddev, 
                # 'gt_mean': gt_mean, 'gt_stddev': gt_stddev, 
                'name': name}


class InferDataset(Dataset):
    """
        InferDataset
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir

        self.names = fnmatch.filter(os.listdir(self.data_dir), '*.npy')

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # (H, W, C)
        modulo = np.load(os.path.join(self.data_dir, self.names[index]))  # positive int, as float32

        name = self.names[index].split('.')[0]
        assert modulo.ndim == 3  # for RGB image

        # (C, H, W)
        modulo = torch.tensor(np.transpose(modulo, (2, 0, 1)), dtype=torch.float32)

        if self.transform:
            modulo = self.transform(modulo)

        return {'modulo': modulo, 'name': name}