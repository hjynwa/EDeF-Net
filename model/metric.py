import torch
from .metric_utils.per_pixel import accuracy, precision, recall, f1_score
from .metric_utils.psnr import PSNR
from .metric_utils.ssim import SSIM, MS_SSIM

def psnr(output, target, data_range=1.):
    """
    Peak Signal to Noise Ratio
    """
    with torch.no_grad():
        mse = torch.mean((target - output) ** 2)
    return 10 * torch.log10(data_range ** 2 / mse)

def mse(output, target):
    """
    Mean Square Error
    """
    with torch.no_grad():
        mse = torch.mean((target - output) ** 2)
    return mse

def top_acc(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


