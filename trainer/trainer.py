from pyexpat import model
import numpy as np
import torch
from torchvision.utils import make_grid
import torchvision as tv 
from base import BaseTrainer
from utils import inf_loop, MetricTracker, normalize_tensor, normalize_voxel_grid, ev_stack_renormalization
from tqdm import tqdm


def batch_ev_stack_renormalization(ev_stack, stddev):
    renorm_ev_stack = torch.zeros_like(ev_stack)
    for i in range(ev_stack.shape[0]):
        renorm_ev_stack[i] = ev_stack[i] * stddev[i]

    return renorm_ev_stack

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
    
    def _eval_metrics(self, pred, gt):
        acc_metrics = np.zeros(len(self.metric_ftns))
        for i, metric in enumerate(self.metric_ftns):
            acc_metrics[i] += metric(pred, gt)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _compute_losses(self, pred, gt):
        losses = torch.zeros(len(self.criterion), dtype=torch.float32).cuda()
        for i, loss in enumerate(self.criterion):
            losses[i] += loss(pred, gt)
            self.writer.add_scalar('{}'.format(loss.__name__), losses[i].item())
        return torch.sum(losses)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        
        total_loss = 0
        total_metrics = np.zeros(len(self.metric_ftns))
        
        for batch_idx, data_sample in enumerate(tqdm(self.data_loader, ascii=True)):
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            
            data, target = data_sample['flk_ev_stack'].to(self.device), data_sample['noflk_ev_stack'].to(self.device)
            num_valid = data_sample['num_valid']
            
            self.optimizer.zero_grad()
            model_output = self.model(data)
            pred = model_output['pred']
            t_res = model_output['t_res']
            s_res = model_output['s_res']
            data_renorm = batch_ev_stack_renormalization(data, data_sample['input_stddev'].to(self.device))
            pred_renorm = batch_ev_stack_renormalization(pred, data_sample['input_stddev'].to(self.device))
            target_renorm = batch_ev_stack_renormalization(target, data_sample['gt_stddev'].to(self.device))
            
            loss = self._compute_losses(pred_renorm, target_renorm)
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('total_loss', loss.item())
            
            total_loss += loss.item()
            total_metrics += self._eval_metrics(pred_renorm, target_renorm)           

            if batch_idx % self.log_step == 0:
                self.logger.debug('\nTrain Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx), loss.item()))

                show_grid = [
                    normalize_voxel_grid(data_renorm[0]), 
                    normalize_voxel_grid(pred_renorm[0]), 
                    normalize_voxel_grid(target_renorm[0]), 
                    normalize_voxel_grid(s_res[0])
                ]
                self.writer.add_image('images', make_grid(show_grid, nrow=3, normalize=False))
                
                # input
                show_grid = [
                    data_renorm[0][i].unsqueeze(0) for i in range(data_renorm.shape[1]) 
                ]
                self.writer.add_image('input_chan', make_grid(show_grid, nrow=4, normalize=True))
                
                # output
                show_grid = [
                    pred_renorm[0][i].unsqueeze(0) for i in range(pred_renorm.shape[1]) 
                ]
                self.writer.add_image('prediction', make_grid(show_grid, nrow=4, normalize=True))
                
                # target
                show_grid = [
                    target_renorm[0][i].unsqueeze(0) for i in range(target_renorm.shape[1]) 
                ]
                self.writer.add_image('target', make_grid(show_grid, nrow=4, normalize=True))

                
            if batch_idx == self.len_epoch:
                break
            
        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metric_ftns))
        
        with torch.no_grad():
            for batch_idx, data_sample in enumerate(self.valid_data_loader):
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                
                data, target = data_sample['flk_ev_stack'].to(self.device), data_sample['noflk_ev_stack'].to(self.device)

                model_output = self.model(data)
                pred, t_res, s_res = model_output['pred'], model_output['t_res'], model_output['s_res']
                loss = self._compute_losses(pred, target)
            
                self.writer.add_scalar('total_loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(pred, target)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        val_log = {
            'loss': total_val_loss / len(self.valid_data_loader),
            'metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
                        
        return val_log


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
