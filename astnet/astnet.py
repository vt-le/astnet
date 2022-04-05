import pprint
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import datasets
import models as models
from utils import utils
from config import config, update_config
import core

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():

    parser = argparse.ArgumentParser(description='ASTNet for Anomaly Detection')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    # update_config(config, args)
    return args


def main():

    args = parse_args()
    update_config(config, args)

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    gpus = list(config.GPUS)
    model = models.get_net(config)
    logger.info('Model: {}'.format(model.get_name()))
    model = nn.DataParallel(model, device_ids=gpus).cuda(device=gpus[0])
    logger.info('Epoch: '.format(args.model_file))

    # load model
    state_dict = torch.load(args.model_file)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)

    test_dataset = eval('datasets.get_test_data')(config)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    mat_loader = datasets.get_label(config)
    mat = mat_loader()

    psnr_list = core.inference(config, test_loader, model)
    assert len(psnr_list) == len(mat), f'Ground truth has {len(mat)} videos, BUT got {len(psnr_list)} detected videos!'

    auc, fpr, tpr = utils.calculate_auc(config, psnr_list, mat)

    logger.info('AUC: %f' % auc)


if __name__ == '__main__':
    main()
