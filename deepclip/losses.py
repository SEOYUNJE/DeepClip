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
