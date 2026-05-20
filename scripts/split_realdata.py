
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:46:20 2023

@author: hanjin
"""
import numpy as np 
import pandas as pd 
import os 
import argparse
from tqdm import tqdm
import cv2 
import matplotlib.pyplot as plt
from metavision_core.event_io import RawReader

def events_to_stack(events, num_bins, height, width):
    """
    Build a events stack wich n bins in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis
    :param width, height: dimensions of the event stack
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    ev_stack = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    # events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / (deltaT)
    # To avoid the last bin to be empty
    events[:, 0] = num_bins * (events[:, 0] - first_stamp) / (deltaT+1)
    
    num_ev = events.shape[0]
    ts = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)

    valid_indices = np.ones(num_ev, dtype=np.bool_)
              
    np.add.at(ev_stack, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, pols[valid_indices])

    ev_stack = np.reshape(ev_stack, (num_bins, height, width))

    return ev_stack.transpose(1, 2, 0)


def split_events(input_dir, name, dest_dir, interval, start_time, end_time, height, width):    
    save_name = name.split('.')[0]
    if not os.path.exists(os.path.join(dest_dir, save_name, 'infer')):
        os.makedirs(os.path.join(dest_dir, save_name, 'infer'))
        
    filepath = os.path.join(input_dir, name)
    
    if filepath.endswith('.raw'):
        reader = RawReader(filepath, max_events=int(1e9))
        ev_stream = reader.load_delta_t(1e8)
    
    elif filepath.endswith('.csv'):
        ev_stream = pd.read_csv(filepath, 
                      header=None, 
                      names=['t', 'x', 'y', 'p'],
                      dtype={'t': np.int64, 'x': np.int16, 'y': np.int16, 'p': np.int16}, 
                      engine='c', nrows=None).values
    elif filepath.endswith('.txt'):
        # ev_stream = np.loadtxt(filepath, skiprows=1, delimiter=',')
        ev_stream = np.loadtxt(filepath)
    
    posi = ev_stream[:,0] > start_time 
    ev_stream = ev_stream[posi]
    posi = ev_stream[:,0] < end_time
    ev_stream = ev_stream[posi]
    
    time = ev_stream[:, 0]
    t_start = time[0]
    i = 0
    
    while t_start < time[-1]:
        t_end = t_start + interval
        valid_pos = np.where((time>=t_start) & (time<t_end))
        ev_stream_clip = ev_stream[valid_pos]
        if ev_stream_clip.shape[0] == 0:
            break
        ev_stack = events_to_stack(ev_stream_clip, num_bins=8, height=height, width=width)
        np.save(os.path.join(dest_dir, save_name, 'infer', save_name + '_%05d.npy' % i), ev_stack)
        
        t_start = t_end
        i = i+1
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', dest='input_dir', required=True, type=str, help='name of input txt directory')
    parser.add_argument('-o', '--output_dir', dest='output_dir', required=True, type=str, help='name of output txt directory')
    parser.add_argument('--interval', dest='interval', default=10000,  type=int, help='interval to split events stream')
    parser.add_argument('--start_time', dest='start_time', type=int, default=0, help='start time of events stream')
    parser.add_argument('--end_time', dest='end_time', type=int, default=100000, help='end time of events stream')
    parser.add_argument('--height', dest='height', type=int, default=360, help='height of the event stack')
    parser.add_argument('--width', dest='width', type=int, default=640, help='width of the event stack')
    args = parser.parse_args()
    
    names = sorted(os.listdir(args.input_dir))
    for i, name in enumerate(tqdm(names)):
        split_events(args.input_dir, name, args.output_dir, args.interval, args.start_time, args.end_time, args.height, args.width)
        print("Finish %s...(%d / %d)" % (name, i+1, len(names)))
 
