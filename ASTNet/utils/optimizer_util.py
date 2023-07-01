import torch.optim as optim


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            # betas=(0.5, 0.999)
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer


def get_scheduler(config, optimizer):
    if config.TRAIN.LR_TYPE == 'linear':
        epoch_count = 0  # the starting epoch count, eg: 0
        n_epochs = config.TRAIN.LR_STEP[0]  # eg: 99
        n_epochs_decay = config.TRAIN.LR_STEP[1]  # eg: 100

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
            return lr_l

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif config.TRAIN.LR_TYPE == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.TRAIN.LR_STEP, gamma=config.TRAIN.LR_FACTOR)
    elif config.TRAIN.LR_TYPE == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config.TRAIN.LR_TYPE)

    return scheduler


def update_learning_rate(config, optimizer):
    """Update learning rates for all the networks; called at the end of every epoch"""
    old_lr = optimizer.param_groups[0]['lr']
    scheduler = get_scheduler(config, optimizer)
    scheduler.step()

    lr = optimizer.param_groups[0]['lr']
    print('learning rate %.7f -> %.7f' % (old_lr, lr))


def update_learning_rate_linear(config, optimizer, epoch):
    epoch_count = 0  # the starting epoch count, eg: 0
    n_epochs = config.TRAIN.LR_STEP[0]  # eg: 99
    n_epochs_decay = config.TRAIN.LR_STEP[1]  # eg: 100
    base_lr = config.TRAIN.LR

    rule = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay)
    lr = base_lr * rule
    optimizer.param_groups[0]['lr'] = lr
