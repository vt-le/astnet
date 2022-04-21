import logging
import torch
import torch.nn as nn
from networks.wider_resnet import wresnet
from networks.helper import ConvBnRelu, ConvTransposeBnRelu, initialize_weights

from networks.helper import TemporalShift, ChannelAttention

logger = logging.getLogger(__name__)


class ASTNet(nn.Module):
    def get_name(self):
        return self.model_name

    def __init__(self, config):
        super(ASTNet, self).__init__()
        frames = config.MODEL.ENCODED_FRAMES
        final_conv_kernel = config.MODEL.EXTRA.FINAL_CONV_KERNEL
        self.model_name = config.MODEL.NAME

        logger.info('=> ' + self.model_name + '_1024: (CATTN + TSM) - Ped2')

        self.wrn38 = wresnet(config, self.model_name, pretrained=False)

        channels = [4096, 2048, 1024, 512, 256, 128]

        self.conv_x8 = nn.Conv2d(channels[0] * frames, channels[1], kernel_size=1, bias=False)
        self.conv_x2 = nn.Conv2d(channels[4] * frames, channels[4], kernel_size=1, bias=False)
        self.conv_x1 = nn.Conv2d(channels[5] * frames, channels[5], kernel_size=1, bias=False)

        self.up8 = ConvTransposeBnRelu(channels[1], channels[2], kernel_size=2)   # 2048          -> 1024
        self.up4 = ConvTransposeBnRelu(channels[2] + channels[4], channels[3], kernel_size=2)   # 1024  +   256 -> 512
        self.up2 = ConvTransposeBnRelu(channels[3] + channels[5], channels[4], kernel_size=2)   # 512   +   128 -> 256

        self.tsm_left = TemporalShift(n_segment=4, n_div=16, direction='left')

        self.attn8 = ChannelAttention(channels[2])
        self.attn4 = ChannelAttention(channels[3])
        self.attn2 = ChannelAttention(channels[4])

        self.final = nn.Sequential(
            ConvBnRelu(channels[4], channels[5], kernel_size=1, padding=0),
            ConvBnRelu(channels[5], channels[5], kernel_size=3, padding=1),
            nn.Conv2d(channels[5], 3,
                      kernel_size=final_conv_kernel,
                      padding=1 if final_conv_kernel == 3 else 0,
                      bias=False)
        )

        initialize_weights(self.conv_x1, self.conv_x2, self.conv_x8)
        initialize_weights(self.up2, self.up4, self.up8)
        initialize_weights(self.attn2, self.attn4, self.attn8)
        initialize_weights(self.final)

    def forward(self, x):
        x1s, x2s, x8s = [], [], []
        for xi in x:
            x1, x2, x8 = self.wrn38(xi)
            x8s.append(x8)
            x2s.append(x2)
            x1s.append(x1)

        x8 = self.conv_x8(torch.cat(x8s, dim=1))
        x2 = self.conv_x2(torch.cat(x2s, dim=1))
        x1 = self.conv_x1(torch.cat(x1s, dim=1))

        left = self.tsm_left(x8)
        x8 = x8 + left

        x = self.up8(x8)                            # 2048          -> 1024, 24, 40
        x = self.attn8(x)

        x = self.up4(torch.cat([x2, x], dim=1))     # 1024 + 256    -> 512, 48, 80
        x = self.attn4(x)

        x = self.up2(torch.cat([x1, x], dim=1))     # 512 + 128     -> 256, 96, 160
        x = self.attn2(x)

        return self.final(x)

