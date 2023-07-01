import os
import pprint
import argparse
import torch.backends.cudnn as cudnn

import torch
import torch.nn as nn

from config.defaults import _C as config, update_config
from utils import train_util, log_util, loss_util, optimizer_util, anomaly_util
import models as models
from models.wresnet1024_cattn_tsm import ASTNet as get_net1
from models.wresnet2048_multiscale_cattn_tsmplus_layer6 import ASTNet as get_net2
import datasets


def parse_args():

    parser = argparse.ArgumentParser(description='ASTNet')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        default='config/shanghaitech_wresnet.yaml', type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        log_util.create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    if config.DATASET.DATASET == "ped2":
        model = get_net1(config)
    else:
        model = get_net2(config)

    logger.info('Model: {}'.format(model.get_name()))

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    losses = loss_util.MultiLossFunction(config=config).cuda()

    optimizer = optimizer_util.get_optimizer(config, model)

    scheduler = optimizer_util.get_scheduler(config, optimizer)

    train_dataset = eval('datasets.get_data')(config)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True
    )
    logger.info('Number videos: {}'.format(len(train_dataset)))

    last_epoch = config.TRAIN.BEGIN_EPOCH
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        train(config, train_loader, model, losses, optimizer, epoch, logger)

        scheduler.step()

        if (epoch + 1) % config.SAVE_CHECKPOINT_FREQ == 0:
            logger.info('=> saving model state epoch_{}.pth to {}\n'.format(epoch+1, final_output_dir))
            torch.save(model.module.state_dict(), os.path.join(final_output_dir,
                                                               'epoch_{}.pth'.format(epoch + 1)))
    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)


def train(config, train_loader, model, loss_functions, optimizer, epoch, logger):
    loss_func_mse = nn.MSELoss(reduction='none')

    model.train()

    for i, data in enumerate(train_loader):
        # decode input
        inputs, target = train_util.decode_input(input=data, train=True)
        output = model(inputs)

        # compute loss
        target = target.cuda(non_blocking=True)
        inte_loss, grad_loss, msssim_loss, l2_loss = loss_functions(output, target)
        loss = inte_loss + grad_loss + msssim_loss + l2_loss

        # compute PSNR
        mse_imgs = torch.mean(loss_func_mse((output + 1) / 2, (target + 1) / 2)).item()
        psnr = anomaly_util.psnr_park(mse_imgs)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_lr = optimizer.param_groups[0]['lr']
        if (i + 1) % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Lr {lr:.6f}\t' \
                  '[inte {inte:.5f} + grad {grad:.4f} + msssim {msssim:.4f} + L2 {l2:.4f}]\t' \
                  'PSNR {psnr:.2f}'.format(epoch+1, i+1, len(train_loader),
                                             lr=cur_lr,
                                             inte=inte_loss, grad=grad_loss, msssim=msssim_loss, l2=l2_loss,
                                             psnr=psnr)
            logger.info(msg)


if __name__ == '__main__':
    main()
