import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def l1_loss(output, target, **kwargs):
    l1_loss_lambda = kwargs.get('l1_loss_lambda', 1)
    return F.l1_loss(output, target) * l1_loss_lambda

def l2_loss(output, target, **kwargs):
    l2_loss_lambda = kwargs.get('l2_loss_lambda', 1)
    return F.mse_loss(output, target) * l2_loss_lambda

def l1_and_l2(output, target, **kwargs):
    l1_loss_lambda = kwargs.get('l1_loss_lambda', 1)
    l1_loss = F.l1_loss(output, target) * l1_loss_lambda

    l2_loss_lambda = kwargs.get('l2_loss_lambda', 1)
    l2_loss = F.mse_loss(output, target) * l2_loss_lambda

    return l1_loss + l2_loss

def num_ev_loss(output, target, **kwargs):
    num_ev_loss_lambda = kwargs.get('num_ev_loss_lambda', 1)
    num_ev_output = torch.sum(torch.abs(output))
    num_ev_target = torch.sum(torch.abs(target))
    return F.l1_loss(num_ev_output, num_ev_target) * num_ev_loss_lambda