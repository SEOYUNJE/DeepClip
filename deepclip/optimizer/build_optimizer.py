import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, cfg):
    """
    Parameter groups (open_clip convention):
        decay     : weights with 2 or more dimensions (conv, linear kernels)          -> cfg.train.weight_decay, lr
        no_decay  : 1 dimensional and scalar parameters (norm weights, biases,
                    BatchNorm, t_itc, b_itc)                                           -> 0, lr
        text_lora : LoRA parameters of the text encoder (DeepClipTextFT only)          -> 0, cfg.train.text_lr (default lr)
    Frozen parameters (the CLIP text weights) are never in the optimizer.
    """

    decay, no_decay, text_lora = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if ".lora_" in name:
            text_lora.append(param)
        elif param.ndim < 2:
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": cfg.train.weight_decay, "name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "name": "no_decay"},
    ]

    if text_lora:
        text_lr = cfg.train.get("text_lr", None) or cfg.train.lr
        param_groups.append(
            {
                "params": text_lora,
                "weight_decay": 0.0,
                "lr": text_lr,
                "name": "text_lora",
            }
        )

    return getattr(torch.optim, cfg.train.optimizer)(param_groups, lr=cfg.train.lr)
