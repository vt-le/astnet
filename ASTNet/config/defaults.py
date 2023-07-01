from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = 'output'
_C.LOG_DIR = 'log'
_C.GPUS = (0, 1, 2, 3)
_C.WORKERS = 4
_C.PRINT_FREQ = 50
_C.SAVE_CHECKPOINT_FREQ = 1
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = '../../datasets'
_C.DATASET.DATASET = 'ped2'
_C.DATASET.TRAINSET = 'training/frames'
_C.DATASET.TESTSET = 'testing/frames'
_C.DATASET.NUM_INCHANNELS = 3
_C.DATASET.NUM_FRAMES = 4
_C.DATASET.FRAME_STEPS = 1
_C.DATASET.LOWER_BOUND = 500

# train
_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE_PER_GPU = 16
_C.TRAIN.SHUFFLE = True

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 200
_C.TRAIN.RESUME = True
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.OPTIMIZER = 'adam'

# sgd and
_C.TRAIN.MOMENTUM = 0.0
_C.TRAIN.WD = 0.0
_C.TRAIN.NESTEROV = False

_C.TRAIN.LR_TYPE = 'linear'
_C.TRAIN.LR = 0.0002
_C.TRAIN.LR_STEP = [40, 70]
_C.TRAIN.LR_FACTOR = 0.5


# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 1


# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'ASTNet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.IMAGE_SIZE = [256, 256]
_C.MODEL.MEMORY_SIZE = 3
_C.MODEL.ENCODED_FRAMES = 2
_C.MODEL.DECODED_FRAMES = 1
# _C.MODEL.SIGMA = 1.5


_C.MODEL.EXTRA = CN()
_C.MODEL.EXTRA.FINAL_CONV_KERNEL = 1


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
