import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class ConvBnRelu(nn.Module):
    # https://github.com/lingtengqiu/Deeperlab-pytorch/blob/master/seg_opr/seg_oprs.py
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ConvTransposeBnRelu(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=2):
        super(ConvTransposeBnRelu, self).__init__()
        if stride != 2:     # ConvTranspose2d with factor = 4
            if kernel_size == 4:    # stride == 4
                padding = 0
                output_padding = 0
        else:       # ConvTranspose2d with factor = 2
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
        self.ConvTranspose = nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels,
                                                kernel_size=kernel_size, stride=stride, padding=padding,
                                                output_padding=output_padding, bias=False)
        self.bn = nn.BatchNorm2d(output_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.ConvTranspose(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

