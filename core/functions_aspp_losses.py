from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tqdm

import torch
import torch.nn as nn

from utils import utils


logger = logging.getLogger(__name__)


def decode_input(input, train=True):
    video = input['video']
    video_name = input['video_name']

    if train:
        inputs = video[:-1]
        target = video[-1]
        return inputs, target
        # return video, video_name
    else:   # TODO: bo sung cho test
        return video, video_name


def inference(config, data_loader, model):
    loss_func_mse = nn.MSELoss(reduction='none')

    model.eval()
    psnr_list = []
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            print('[{}/{}]'.format(i+1, len(data_loader)))
            psnr_video = []

            video, video_name = decode_input(input=data, train=False)
            video = [frame.to(device=config.GPUS[0]) for frame in video]
            for f in tqdm.tqdm(range(len(video) - fp)):
                inputs = video[f:f + fp]
                output = model(inputs)
                target = video[f + fp:f + fp + 1][0]

                # compute PSNR for each frame
                mse_imgs = torch.mean(loss_func_mse((output[0] + 1) / 2, (target[0] + 1) / 2)).item()
                psnr = utils.psnr_park(mse_imgs)
                psnr_video.append(psnr)

            psnr_list.append(psnr_video)
    return psnr_list

