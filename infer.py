import argparse
import collections
import numpy as np 
import torch
import cv2 
import os 
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParserTest
from utils import ev_stack_normalization, ev_stack_renormalization, re_distribution, filter_max_value


def batch_ev_stack_renormalization(ev_stack, stddev):
    renorm_ev_stack = torch.zeros_like(ev_stack)
    for i in range(ev_stack.shape[0]):
        renorm_ev_stack[i] = ev_stack[i] * stddev[i]

    return renorm_ev_stack

def renormalize_stack(pred, gt):
    pred_renorm = np.zeros_like(pred)
    a = np.quantile(gt, 0.00000001)
    b = np.quantile(gt, 0.99999999)
    c = np.quantile(pred, 0.001)
    d = np.quantile(pred, 0.999)
    
    # for non-flicker scenes:
    # c = np.quantile(pred, 0.002)
    # d = np.quantile(pred, 0.998)
    
    # print('a:, %.2f, b: %.2f, c: %.2f, d: %.2f' % (a, b, c, d))

    if c > -1:
        c = -1.0
    if d < 1:
        d = 1.0
    pred_renorm[pred < 0] = pred[pred < 0] / np.abs(c)
    pred_renorm[pred > 0] = pred[pred > 0] / np.abs(d)

    pred_renorm = np.clip(pred_renorm, a, b)

    return np.trunc(pred_renorm).astype(int)

def save_sequence(evs_stack, save_path, height, width, ith):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    pos_color = [63, 169, 245]
    neg_color = [212, 20, 90]
    for i in range(2):
        if i == 0:
            stack = np.sum(evs_stack[:,:,:4], axis=2)
        else:
            stack = np.sum(evs_stack[:,:,4:], axis=2)
        # stack = np.stack([stack, stack, stack], axis=2)
        
        pos_posi = np.where(stack > 0)
        neg_posi = np.where(stack < 0)
        init_img = np.ones((height, width, 3)) * 255
        init_img[pos_posi[0], pos_posi[1]] = pos_color
        init_img[neg_posi[0], neg_posi[1]] = neg_color
        cv2.imwrite(os.path.join(save_path, '%05d.png' % (ith*2+i)), init_img[:,:,::-1])
    
        

def main(config):

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        transform=config['data_loader']['args']['transform'],
        num_workers=2
    )
    
    # build model architecture
    model = config.init_obj('arch', module_arch)

    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # save re-distribution events stream
    results_dir = config.results_dir

    # num_test = config['data_loader']['num_test']
    num_test = config.num_test
    with torch.no_grad():
        for i, data_sample in enumerate(tqdm(data_loader)):
            if i >= num_test:  # only apply our model to num_test data.
                break
            data = data_sample['flk_ev_stack'].to(device)
            filename = data_sample['name']
            
            model_output = model(data)
            pred = model_output['pred']
            t_res = model_output['t_res']
            s_res = model_output['s_res']
            
            pred_np = pred[0].clone().cpu().numpy().transpose(1,2,0).astype(np.float32)
            unnorm_np = data_sample['flk_ev_stack_unnorm'][0].clone().numpy().transpose(1,2,0).astype(np.float32)
            pred_renorm = renormalize_stack(pred_np, unnorm_np)
            
            results_dir_list = config['data_loader']['args']['data_dir'].split('/')
            output_stack_path = os.path.join(results_dir, results_dir_list[-2], results_dir_list[-1], 'evs_stack', filename[0]+'.npy')
            dir_path = os.path.dirname(output_stack_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
            np.save(output_stack_path, pred_renorm)
            np.save(output_stack_path.replace('.npy', '_unnorm.npy'), pred[0].clone().cpu().numpy().transpose(1,2,0).astype(np.float32))
            
            sequence_path = os.path.join(results_dir, results_dir_list[-2], results_dir_list[-1], 'ev_sequences')
            save_sequence(pred_renorm, sequence_path, 512, 512, i)
            
            re_distribution_path = os.path.join(results_dir, results_dir_list[-2], results_dir_list[-1], 'evs_txt', filename[0]+'.txt')
            re_distribution(pred_renorm, re_distribution_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--run_id', default=None, type=str,
                      help='unique Identifier for saving directory')
    args.add_argument('--num_test', default=1000, type=int,
                    help='number of test data (default: 1000)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--data_dir'], type=str, target='data_loader;args;data_dir'),
        CustomArgs(['--data_loader_type'], type=str, target='data_loader;type')
    ]
    
    for opt in options:
        if opt.flags == ['--data_loader_type']:
            args.add_argument(*opt.flags, default='EvstackInferDataLoader', type=opt.type)
        else:
            args.add_argument(*opt.flags, default=None, type=opt.type)
    if not isinstance(args, tuple):
        args = args.parse_args()
    config = ConfigParserTest.from_args(args, options)
    main(config)