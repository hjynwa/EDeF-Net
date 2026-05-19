import os
import pandas as pd 
import fnmatch
import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange
from utils import ev_stack_normalization, ev_stack_renormalization, get_transform

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def get_evs_tmap(evs_raw_arr, height, width):
    pols = evs_raw_arr[:, 3]
    pols[pols == 0] = -1
    evs_raw_arr[:,3] = pols

    evs_tmap =  np.empty((height, width), dtype=object)
    for i in range(height):
        for j in range(width):
            evs_tmap[i][j] = []
    
    for i, ev in enumerate(evs_raw_arr):
        t_pol = ev[0] * ev[3]
        evs_tmap[int(ev[2])][int(ev[1])].append(t_pol)

    return evs_tmap

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
        self.names = fnmatch.filter(os.listdir(self.flicker_dir), '*.txt')
        self.transform = get_transform(transform)
        self.patch_h, self.patch_w = pair(patch_size)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        height, width = 256, 256
        flk_evs_raw = pd.read_csv(os.path.join(self.flicker_dir, self.names[index]), 
                      header=None, 
                      names=['t', 'x', 'y', 'p'],
                      dtype={'t': np.int64, 'x': np.int16, 'y': np.int16, 'p': np.int16}, 
                      engine='c', nrows=None)
        noflk_evs_raw = pd.read_csv(os.path.join(self.gt_dir, self.names[index]), 
                      header=None, 
                      names=['t', 'x', 'y', 'p'],
                      dtype={'t': np.int64, 'x': np.int16, 'y': np.int16, 'p': np.int16}, 
                      engine='c', nrows=None)
        name = self.names[index].split('.')[0]
        
        flk_ev_tmap = get_evs_tmap(flk_evs_raw, height, width)
        noflk_ev_tmap = get_evs_tmap(noflk_evs_raw, height, width)
        
        
        
        
        flk_ev_stack = torch.tensor(np.transpose(flk_ev_stack, (2, 0, 1)), dtype=torch.float32)
        noflk_ev_stack = torch.tensor(np.transpose(noflk_ev_stack, (2, 0, 1)), dtype=torch.float32)

        if self.transform:
            flk_ev_stack = self.transform(flk_ev_stack)
            noflk_ev_stack = self.transform(noflk_ev_stack)

        return {'flk_ev_stack': flk_ev_stack, 'noflk_ev_stack': noflk_ev_stack, 
                'input_mean': input_mean, 'input_stddev': input_stddev, 
                'gt_mean': gt_mean, 'gt_stddev': gt_stddev, 
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