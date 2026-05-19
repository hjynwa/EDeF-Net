import os
import fnmatch
import h5py 

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import ev_stack_normalization, ev_stack_renormalization, get_transform

class TrainDataset(Dataset):
    """
        TrainDataset
    """

    def __init__(self, data_dir='data', transform=None, training=True):
        if training:
            sub_dir = 'train'
        else:
            sub_dir = 'test'
        self.flicker_dir = os.path.join(data_dir, 'flicker', sub_dir)
        self.gt_dir = os.path.join(data_dir, 'noflicker', sub_dir)
        self.names = sorted(fnmatch.filter(os.listdir(self.flicker_dir), '*.npy'))[:]
        self.transform = None
        if transform:
            self.transform = get_transform(transform)


    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # (H, W, C)
        flk_ev_stack = np.load(os.path.join(self.flicker_dir, self.names[index]))
        noflk_ev_stack = np.load(os.path.join(self.gt_dir, self.names[index]))
          
        # with h5py.File(os.path.join(self.flicker_dir, self.names[index]), "r") as f:
        #     keys = list(f.keys())
        #     random_key = np.random.choice(keys)
        #     # random_key = keys[3]
        #     flk_ev_stack = np.array(f[random_key])

        # with h5py.File(os.path.join(self.gt_dir, self.names[index]), "r") as f:
        #     noflk_ev_stack = np.array(f[random_key])
        
        ###############################
        # Add mask for the flicker stack
        mask = np.ones(flk_ev_stack.shape[:2])
        flk_ev_sum = np.sum(np.abs(flk_ev_stack), axis=2)
        mask[np.where(flk_ev_sum == 0)] = 0
        num_valid = np.sum(mask)
        ###############################
        
        name = self.names[index].split('.')[0]
        # flk_ev_stack, input_mean, input_stddev = ev_stack_normalization(flk_ev_stack)
        # noflk_ev_stack, gt_mean, gt_stddev = ev_stack_normalization(noflk_ev_stack)
        
        # Donot minus mean, only divide by stddev
        input_stddev = flk_ev_stack.std()
        flk_ev_stack_norm = flk_ev_stack / input_stddev
        gt_stddev = noflk_ev_stack.std()
        noflk_ev_stack_norm = noflk_ev_stack / gt_stddev
        
        ###############################
        # filter out the empty pixels
        # flk_ev_stack = mask[..., np.newaxis] * flk_ev_stack
        # noflk_ev_stack = mask[..., np.newaxis] * noflk_ev_stack
        
        # (C, H, W)
        flk_ev_stack_norm = torch.tensor(np.transpose(flk_ev_stack_norm, (2, 0, 1)), dtype=torch.float32)
        noflk_ev_stack_norm = torch.tensor(np.transpose(noflk_ev_stack_norm, (2, 0, 1)), dtype=torch.float32)

        if self.transform:
            flk_ev_stack_norm = self.transform(flk_ev_stack_norm)
            noflk_ev_stack_norm = self.transform(noflk_ev_stack_norm)

        return {
            'flk_ev_stack': flk_ev_stack_norm, 
            'flk_ev_stack_unnorm': flk_ev_stack,
            'noflk_ev_stack': noflk_ev_stack_norm, 
            'noflk_ev_stack_unnorm': noflk_ev_stack,
            'input_mean': [], 
            'input_stddev': input_stddev, 
            'gt_mean': [], 
            'gt_stddev': gt_stddev, 
            'name': name, 
            'mask': mask,
            'num_valid': num_valid
        }


class InferDataset(Dataset):
    """
        InferDataset
    """

    def __init__(self, data_dir='data', transform=None):
        sub_dir = 'infer'
        self.flicker_dir = os.path.join(data_dir, sub_dir)
        self.names = sorted(fnmatch.filter(os.listdir(self.flicker_dir), '*.npy'))[:]
        self.transform = None
        if transform:
            self.transform = get_transform(transform)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # (H, W, C)
        flk_ev_stack = np.load(os.path.join(self.flicker_dir, self.names[index]))  # [h, w, 2] array, positive and negativa channels
        name = self.names[index].split('.')[0]
        # flk_ev_stack, input_mean, input_stddev = ev_stack_normalization(flk_ev_stack)
        
        # Donot minus mean, only divide by stddev
        input_stddev = flk_ev_stack.std()
        flk_ev_stack_norm = flk_ev_stack / input_stddev

        # (C, H, W)
        flk_ev_stack_norm = torch.tensor(np.transpose(flk_ev_stack_norm, (2, 0, 1)), dtype=torch.float32)

        if self.transform:
            flk_ev_stack_norm = self.transform(flk_ev_stack_norm)

        return {
            'flk_ev_stack': flk_ev_stack_norm, 
            'flk_ev_stack_unnorm': flk_ev_stack,
            'input_mean': [], 
            'input_stddev': input_stddev, 
            'name': name
        }