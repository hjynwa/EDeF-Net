import json
import os
import cv2 
import torch
import torchvision.transforms as transforms
import numpy as np 
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from torch.optim import lr_scheduler
from PIL import Image
from einops import rearrange, reduce, repeat

def lmd_lr_lambda(decay_starts, total_epochs):
    return lambda epoch: 1.0 - max(0, epoch - decay_starts) / float((total_epochs - decay_starts) + 1)

def mtp_lr_lambda(decay_factor):
    return lambda epoch: decay_factor
    
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def normalize_tensor(input):
    normalized = (input-torch.min(input)) / (torch.max(input)-torch.min(input))
    return normalized

def normalize_voxel_grid(input):
    input = torch.sum(input, 0)
    normalized = (input-torch.min(input)) / (torch.max(input)-torch.min(input))
    return normalized[None, :, :]

def ev_stack_normalization(ev_stack):
    mean, stddev = ev_stack[ev_stack != 0].mean(), ev_stack[ev_stack != 0].std()
    ev_stack[ev_stack != 0] = (ev_stack[ev_stack != 0] - mean) / stddev
    
    return ev_stack, mean, stddev

def ev_stack_renormalization(ev_stack, mean, stddev):
    ev_stack[ev_stack != 0] = ev_stack[ev_stack != 0] * stddev + mean
    
    return ev_stack.astype(np.int)

def filter_max_value(ev_stack):
    unique, count = np.unique(ev_stack, return_counts=True)
    max_idx = np.where(count == np.max(count))[0][0]
    most_value = unique[max_idx]
    if most_value != 0:
        ev_stack = ev_stack - most_value
    return ev_stack

            
def re_distribution(ev_stack, f_path, stack_duration=10000):
    dir_path = os.path.dirname(f_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
            
    height, width, channel = ev_stack.shape
    
    interval = stack_duration // channel
    bin_low, bin_high = 1, interval
    with open(f_path, 'w') as f:
        for c in range(channel):
            events = []
            valid_pos = np.where(ev_stack[:, :, c] != 0)
            # print(valid_pos[0].shape)
            valid_values = ev_stack[:,:,c][valid_pos]
            for i, value in enumerate(valid_values):
                x, y = valid_pos[1][i], valid_pos[0][i]
                if value > 0:
                    p = 1
                else:
                    p = 0
                tlist = np.random.randint(bin_low, bin_high+1, np.abs(value).astype(int))
                for t in tlist:
                    events.append((t, x, y, p))
            events.sort(key=lambda k: k[0])
            for e in events:
                f.write("%d %d %d %d\n" % (e[0], e[1], e[2], e[3]))
            
            bin_low = bin_high
            bin_high += interval
            

def timetensor_normalization(ev_tt, channels):
    mean = np.mean(ev_tt, axis=-1)
    mean = repeat(mean, 'h w -> h w c', c=channels)
    stddev = np.std(ev_tt, axis=-1)
    stddev = repeat(stddev, 'h w -> h w c', c=channels)
    ev_tt[stddev != 0] = (ev_tt[stddev != 0] - mean[stddev != 0]) / stddev[stddev != 0]

    return ev_tt, mean, stddev

def get_transform(transform_params):
    transform_list = []

    if transform_params['resize']:
        osize = (transform_params['resize'][0], transform_params['resize'][1])
        interpolation_mode = Image.NEAREST
        if transform_params['interpolation'] == "NEAREST":
            interpolation_mode = Image.NEAREST
        transform_list.append(transforms.Resize(osize, interpolation_mode))
        
    return transforms.Compose(transform_list)

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        print(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
