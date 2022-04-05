import os
import logging
import time
import math
import pickle
import numpy as np
from pathlib import Path
from sklearn import metrics


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
                          (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def psnr_park(mse):
    return 10 * math.log10(1 / mse)


def anomaly_score(psnr, max_psnr, min_psnr):
    return (psnr - min_psnr) / (max_psnr - min_psnr)


def calculate_auc(config, psnr_list, mat):
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df  # number of frames to process

    scores = np.array([], dtype=np.float)
    labels = np.array([], dtype=np.int)

    for i in range(len(psnr_list)):
        score = anomaly_score(psnr_list[i], np.max(psnr_list[i]), np.min(psnr_list[i]))

        scores = np.concatenate((scores, score), axis=0)
        labels = np.concatenate((labels, mat[i][fp:]), axis=0)
    assert scores.shape == labels.shape, f'Ground truth has {labels.shape[0]} frames, BUT got {scores.shape[0]} detected frames!'
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)

    return auc, fpr, tpr

