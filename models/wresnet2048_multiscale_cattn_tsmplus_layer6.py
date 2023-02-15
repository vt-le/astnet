import logging
import torch
import torch.nn as nn
from networks.wider_resnet import wresnet
from networks.helper import ConvBnRelu, ConvTransposeBnRelu, initialize_weights

logger = logging.getLogger(__name__)


class ASTNet(nn.Module):
    def get_name(self):
        return self.model_name

    def __init__(self, config):
        super(ASTNet, self).__init__()
        frames = config.MODEL.ENCODED_FRAMES
        final_conv_kernel = config.MODEL.EXTRA.FINAL_CONV_KERNEL
        self.model_name = config.MODEL.NAME

        logger.info(self.model_name + ' (AM + TSM) - WiderResNet_layer6')

        self.wrn = wresnet(config, self.model_name, pretrained=True)

        channels = [4096, 2048, 1024, 512, 256, 128]

        self.conv_x7 = nn.Conv2d(channels[1] * frames, channels[0], kernel_size=1, bias=False)
        self.conv_x3 = nn.Conv2d(channels[4] * frames, channels[3], kernel_size=1, bias=False)
        self.conv_x2 = nn.Conv2d(channels[5] * frames, channels[4], kernel_size=1, bias=False)

        self.tsm_left = TemporalShift(n_segment=4, n_div=16, direction='left', split=False)

        self.up8 = ConvTransposeBnRelu(channels[0], channels[1], kernel_size=2)
        self.up4 = ConvTransposeBnRelu(channels[2] + channels[3], channels[2], kernel_size=2)
        self.up2 = ConvTransposeBnRelu(channels[3] + channels[4], channels[3], kernel_size=2)

        lReLU = nn.LeakyReLU(0.2, True)
        self.attn8 = RCAB(channels[1], channels[2], kernel_size=3, reduction=16, norm='BN', act=lReLU, downscale=True)
        self.attn4 = RCAB(channels[2], channels[3], kernel_size=3, reduction=16, norm='BN', act=lReLU, downscale=True)
        self.attn2 = RCAB(channels[3], channels[4], kernel_size=3, reduction=16, norm='BN', act=lReLU, downscale=True)

        self.final = nn.Sequential(
            ConvBnRelu(channels[4], channels[5], kernel_size=3, padding=1),
            ConvBnRelu(channels[5], channels[5], kernel_size=5, padding=2),
            nn.Conv2d(channels[5], 3,
                      kernel_size=final_conv_kernel,
                      padding=(final_conv_kernel-1)//2,
                      bias=False)
        )

        initialize_weights(self.conv_x2, self.conv_x3, self.conv_x7)
        initialize_weights(self.up2, self.up4, self.up8)
        initialize_weights(self.attn2, self.attn4, self.attn8)
        initialize_weights(self.final)

    def forward(self, x):
        x2s, x3s, x7s = [], [], []
        for xi in x:
            x2, x3, x7 = self.wrn(xi)
            x7s.append(x7)
            x3s.append(x3)
            x2s.append(x2)

        x7s = self.conv_x7(torch.cat(x7s, dim=1))
        x3s = self.conv_x3(torch.cat(x3s, dim=1))
        x2s = self.conv_x2(torch.cat(x2s, dim=1))

        left = self.tsm_left(x7s)

        x7s = x7s + left
        x = self.up8(x7s)
        x = self.attn8(x)

        x = self.up4(torch.cat([x3s, x], dim=1))
        x = self.attn4(x)

        x = self.up2(torch.cat([x2s, x], dim=1))
        x = self.attn2(x)

        return self.final(x)












