import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp


class IntensityLoss(nn.Module):
    def __init__(self):
        super(IntensityLoss, self).__init__()

    def forward(self, prediction, target):
        # return torch.mean(torch.abs((prediction - target) ** 2))    # it's mine
        return torch.mean(torch.pow(torch.abs(prediction - target), 2))  # PyAnomaly


class GradientLoss(nn.Module):
    def __init__(self, channels=3):
        super(GradientLoss, self).__init__()

        pos = torch.from_numpy(np.identity(channels, dtype=np.float32))
        neg = -1 * pos

        # Note: when doing conv2d, the channel order is different from tensorflow, so do permutation.
        filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1)  # .cuda()
        filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1)  # .cuda()

        # https://github.com/lv-tuan/semantic-segmentation/blob/b4fc685bb35d9b7547b805b1395c515876ec48db/sdcnet/models/sdc_net2d.py#L76
        self.register_buffer('filter_x', filter_x)
        self.register_buffer('filter_y', filter_y)

    def forward(self, prediction, target):
        # https://github.com/NVIDIA/vid2vid/blob/2e6d13755fc2e33200e7d4c0c44f2692d6ab0898/models/flownet.py#L27
        filter_x = self.filter_x
        filter_y = self.filter_y

        # Do padding to match the  result of the original tensorflow implementation
        gen_frames_x = nn.functional.pad(prediction, [0, 1, 0, 0])
        gen_frames_y = nn.functional.pad(prediction, [0, 0, 0, 1])
        gt_frames_x = nn.functional.pad(target, [0, 1, 0, 0])
        gt_frames_y = nn.functional.pad(target, [0, 0, 0, 1])

        gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, filter_x))
        gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, filter_y))
        gt_dx = torch.abs(nn.functional.conv2d(gt_frames_x, filter_x))
        gt_dy = torch.abs(nn.functional.conv2d(gt_frames_y, filter_y))

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x + grad_diff_y)


class L2Loss(nn.Module):
    def __init__(self, eps=1e-8):   # 1 x 10^(-8) = 0.00000001
        super(L2Loss, self).__init__()
        self.eps = eps

    def forward(self, prediction, target):
        error = torch.mean(torch.pow((prediction-target), 2))
        error = torch.sqrt(error + self.eps)
        return error


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    cs = (cs + 1) / 2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
        ret = (ret + 1) / 2
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
        ret = (ret + 1) / 2

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


class MultiLossFunction(nn.Module):
    def __init__(self, config):
        super(MultiLossFunction, self).__init__()
        self.intensity_loss = IntensityLoss()
        self.gradient_loss = GradientLoss(channels=config.DATASET.NUM_INCHANNELS)
        self.msssim_loss = MSSSIM()
        self.l2_loss = L2Loss()

    def forward(self, prediction, target):
        inte_loss = self.intensity_loss(prediction, target)
        grad_loss = self.gradient_loss(prediction, target)
        msssim = (1 - self.msssim_loss(prediction, target))/2
        l2_loss = self.l2_loss(prediction, target)

        return inte_loss, grad_loss, msssim, l2_loss