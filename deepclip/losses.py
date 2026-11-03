import torch
import torch.nn.functional as F

def itc_loss(logits, Y, W):
    """
    Sigmoid image-text contrastive loss with multiple positives (SigLIP normalization).

    logits: (B,U) t_itc * cos(img_i, txt_j) + b_itc
    Y:      (B,U) 1 if caption j is true for image i
    W:      (B,U) weight of every cell

    The weighted BCE of the U cells of an image are summed, then averaged over the B images,
    so the loss of an image does not depend on U.
    """

    loss = F.binary_cross_entropy_with_logits(logits.float(), Y, weight=W, reduction="none")
    return loss.sum(dim=1).mean()

def cls_loss(logits, labels):
    """
    Real/fake loss. logits = t_cls * (cos_fake - cos_real), which is the same as the 2-way softmax
    of the REAL_TEXT and FAKE_TEXT logits.

    logits: (B,), labels: (B,) 0: REAL, 1: FAKE
    """

    return F.binary_cross_entropy_with_logits(logits.float(), labels)

def uncertainty_weighting(itc, cls, s_itc, s_cls):
    """
    Uncertainty weighting (Kendall et al., CVPR 2018): exp(-s) * L + s for each loss.
    The weight exp(-s) is learned and cannot collapse to 0, because s grows when the weight shrinks.
    """

    return torch.exp(-s_itc) * itc + s_itc + torch.exp(-s_cls) * cls + s_cls
