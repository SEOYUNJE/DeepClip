import torch
import torch.nn as nn

def build_optimizer(model: nn.Module, cfg):
    """
    Two parameter groups with the same lr (open_clip convention):
        decay    : weights with 2 or more dimensions (conv, linear kernels)          -> cfg.train.weight_decay
        no_decay : 1 dimensional and scalar parameters (norm weights, biases,
                   BatchNorm, t_itc, b_itc, t_cls, s_itc, s_cls)                      -> 0
    The frozen text encoder is not a part of DeepClip, so it is never in the optimizer.
    """

    decay, no_decay = [], []

    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim < 2:
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": cfg.train.weight_decay, "name": "decay"},
        {"params": no_decay, "weight_decay": 0., "name": "no_decay"},
    ]

    return getattr(torch.optim, cfg.train.optimizer)(param_groups, lr=cfg.train.lr)
