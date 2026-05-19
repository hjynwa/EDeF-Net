import argparse
import numpy as np 
import collections
import torch
import math
import os 
import time 
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParserTest
from utils import ev_stack_normalization, ev_stack_renormalization, re_distribution
from skimage import metrics
from sklearn.metrics import mean_absolute_error


def MAE(img1, img2):
    maes = []
    num_c = img1.shape[-1]
    for i in range(num_c):
        mae = mean_absolute_error(img1[:,:,i], img2[:,:,i], multioutput='uniform_average')
        maes.append(mae)
    return np.mean(maes)

def PSNR(img1, img2):
    mse_ = np.mean( (img1 - img2) ** 2 )
    if mse_ == 0:
        return 100
    return 10 * math.log10(1 / mse_)

def SSIM(img1, img2):
    return metrics.structural_similarity(img1, img2, data_range=1, channel_axis=2)


def compute_mse(pred, gt, squared=True, nozero=False):
    if nozero:
        valid_pos = np.where(gt != 0)
        output_errors = np.average((gt[valid_pos] - pred[valid_pos]) ** 2)
    else:
        output_errors = np.average((gt - pred) ** 2)
    if not squared:
        output_errors = np.sqrt(output_errors)
        
    return output_errors

def compute_snr(pred, gt):
    noise = gt - pred
    signal_power = np.mean(pred**2)
    noise_power = np.mean(noise**2)

    snr = signal_power / noise_power
    # snr_db = 10 * np.log10(snr)

    return snr

def renormalize_stack(pred, gt):
    pred_renorm = np.zeros_like(pred)
    a = np.quantile(gt, 0.0001)
    b = np.quantile(gt, 0.9999)
    c = np.quantile(pred, 0.0001)
    d = np.quantile(pred, 0.9999)
    # print('a:, %.2f, b: %.2f, c: %.2f, d: %.2f' % (a, b, c, d))
    pred_renorm[pred < -1] = pred[pred < -1] / c * a
    pred_renorm[pred > 1] = pred[pred > 1] / d * b

    c_ = np.quantile(pred, 0.01)
    d_ = np.quantile(pred, 0.99)

    if c_ > -1:
        c_ = -2.0
    if d_ < 1:
        d_ = 2.0
    pred_renorm[pred < 0] = pred[pred < 0] / c_ * a
    pred_renorm[pred > 0] = pred[pred > 0] / d_ * b

    pred_renorm = np.clip(pred_renorm, a, b)

    return np.trunc(pred_renorm).astype(int)


def main(config):
    # logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        training=False,
        transform=config['data_loader']['args']['transform'],
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: %.3f M" % (total_params / 1e6))
    # logger.info(model)

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    # total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns)).to(device)
    metrics_names = []
    for i, metric in enumerate(metric_fns):
        metrics_names.append(metric.__name__)
    
    # save re-distribution events stream
    results_dir = config.results_dir

    num_test = config.num_test
    if num_test > len(data_loader):
        num_test = len(data_loader)
    cost_times = []
    mses, psnrs, ssims, maes, snrs = [], [], [], [], []

    with torch.no_grad():
        for i, data_sample in enumerate(tqdm(data_loader)):
            if i >= num_test:  # only apply our model to num_test data.
                break
            # start time stamp
            start_ts = time.time()
            data, target = data_sample['flk_ev_stack'].to(device), data_sample['noflk_ev_stack'].to(device)
            
            filename = data_sample['name']
            
            model_output = model(data)
            pred = model_output['pred']
            t_res = model_output['t_res']
            s_res = model_output['s_res']
            
            # end time stamp
            end_ts = time.time()
            cost_times.append(end_ts - start_ts)
            
            batch_size = data.shape[0]
            pred_arr = pred[0].clone().cpu().numpy().astype(np.float32).transpose(1,2,0)
            target_arr = target[0].clone().cpu().numpy().astype(np.float32).transpose(1,2,0)
            
            gt_unnorm_np = data_sample['noflk_ev_stack_unnorm'][0].clone().numpy().astype(np.float32)
            flk_unnorm_np = data_sample['flk_ev_stack_unnorm'][0].clone().numpy().astype(np.float32)
            pred_renorm = renormalize_stack(pred_arr, gt_unnorm_np)
            output_stack_path = os.path.join(results_dir, config['data_loader']['args']['data_dir'].split('/')[-1], 'evs_stack', filename[0] + '_pred.npy')
            dir_path = os.path.dirname(output_stack_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            np.save(output_stack_path, pred_renorm)
            # np.save(output_stack_path.replace('_pred.npy', '_nonorm_pred.npy'), pred_arr)
            
            re_distribution_path = os.path.join(results_dir, config['data_loader']['args']['data_dir'].split('/')[-1], 'evs_txt', filename[0]+'.txt')
            re_distribution(pred_renorm, re_distribution_path)
            
            mse = compute_mse(pred_renorm, gt_unnorm_np, nozero=False)
            mses.append(mse)

            if pred_renorm.max() - pred_renorm.min() != 0:
                pred_renorm_4psnr = (pred_renorm - pred_renorm.min()) / (pred_renorm.max() - pred_renorm.min())
            else:
                pred_renorm_4psnr = pred_renorm - pred_renorm.min()
            gt_unnorm_np_4psnr = (gt_unnorm_np - gt_unnorm_np.min()) / (gt_unnorm_np.max() - gt_unnorm_np.min())
            psnrs.append(PSNR(pred_renorm_4psnr, gt_unnorm_np_4psnr))
            maes.append(MAE(pred_renorm, gt_unnorm_np))
            ssims.append(SSIM(pred_renorm, gt_unnorm_np))
            snrs.append(compute_snr(pred_renorm, gt_unnorm_np))
            
        print(config['name'])
        # for k in range(len(metric_fns)):
        #     print(metrics_names[k], ": ", total_metrics[k] / num_test)
        print("Average time cost: %.4f s" % np.mean(cost_times))
        mse_avg, psnr_avg, mae_avg, ssim_avg, snr_avg = np.mean(mses), np.mean(psnrs), np.mean(maes), np.mean(ssims), np.mean(snrs)
        print('MSE: %.4f, PSNR: %.4f, MAE: %.4f, SSIM: %.4f, SNR: %.4f' % (mse_avg, psnr_avg, mae_avg, ssim_avg, snr_avg))


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
        CustomArgs(['--data_dir'], type=str, target='data_loader;args;data_dir')
    ]
    
    for opt in options:
        args.add_argument(*opt.flags, default=None, type=opt.type)
    if not isinstance(args, tuple):
        args = args.parse_args()
    config = ConfigParserTest.from_args(args, options)
    main(config)
