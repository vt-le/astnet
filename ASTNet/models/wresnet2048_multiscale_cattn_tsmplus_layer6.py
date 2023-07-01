import logging
import torch
import torch.nn as nn
from models.wider_resnet import wresnet
from models.basic_modules import ConvBnRelu, ConvTransposeBnRelu, initialize_weights

logger = logging.getLogger(__name__)


class ASTNet(nn.Module):
    def get_name(self):
        return self.model_name

    def __init__(self, config, pretrained=True):
        super(ASTNet, self).__init__()
        frames = config.MODEL.ENCODED_FRAMES
        final_conv_kernel = config.MODEL.EXTRA.FINAL_CONV_KERNEL
        self.model_name = config.MODEL.NAME

        logger.info(self.model_name + ' (AM + TSM) - WiderResNet_layer6')

        self.wrn = wresnet(config, self.model_name, pretrained=pretrained)

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
            ConvBnRelu(channels[4], channels[5], kernel_size=3, padding=1),  # TODO: kernel_size=3
            ConvBnRelu(channels[5], channels[5], kernel_size=5, padding=2),  # TODO: kernel_size=3
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

        x = self.up4(torch.cat([x3s, x], dim=1))     # 1024 + 512    -> 512, 48, 80
        x = self.attn4(x)

        x = self.up2(torch.cat([x2s, x], dim=1))     # 512 + 256     -> 256, 96, 160
        x = self.attn2(x)

        return self.final(x)


class TemporalShift(nn.Module):
    def __init__(self, n_segment=4, n_div=8, direction='left', split=False):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.direction = direction
        self.split = split

        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, direction=self.direction, split=self.split)
        return x

    @staticmethod
    def shift(x, n_segment=4, fold_div=8, direction='left', split=False):
        bz, nt, h, w = x.size()
        c = nt // n_segment
        x = x.view(bz, n_segment, c, h, w)

        fold = c // fold_div

        out = torch.zeros_like(x)
        if direction == 'left':
            out[:, :-1, :fold] = x[:, 1:, :fold]
            out[:, :, fold:] = x[:, :, fold:]
        elif direction == 'right':
            out[:, 1:, :fold] = x[:, :-1, :fold]
            out[:, :, fold:] = x[:, :, fold:]
        else:
            out[:, :-1, :fold] = x[:, 1:, :fold]
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        if split:
            p1, _ = out.split([fold * 2, c - (fold * 2)], dim=2)
            p1 = p1.reshape(bz, n_segment * fold * 2, h, w)
            return p1
        else:
            return out.view(bz, nt, h, w)


class RCAB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, reduction,
            norm=False, act=nn.ReLU(True), downscale=False, return_ca=False):
        super(RCAB, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
            act,
            ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
        )
        self.CA = CALayer(out_feat, reduction)
        self.sig = nn.Sigmoid()

        self.downscale = downscale
        if downscale:
            self.downConv = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1)
        self.return_ca = return_ca

    def forward(self, x):
        res = x
        out = self.body(x)
        ca = self.CA(out)

        if self.downscale:
            res = self.downConv(res)

        return res + (out * self.sig(ca))


class ConvNorm(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super(ConvNorm, self).__init__()

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, bias=True)

        self.norm = norm
        if norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_feat, track_running_stats=True)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(out_feat)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y











