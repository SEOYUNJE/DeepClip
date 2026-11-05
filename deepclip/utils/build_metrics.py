import torch
from torchmetrics import (
                            MetricCollection,
                            Accuracy,
                            AUROC,
                            F1Score
                        )
from torchmetrics.classification import MultilabelAveragePrecision
from deepclip.data.text import REGIONS, TYPES, PAIR_ORDER

# ap_{region}: mean AP of the 5 pairs of the region, ap_{type}: mean AP of the 4 pairs of the type
AP_NAMES = [f"ap_{region}" for region in REGIONS] + [f"ap_{type_}" for type_ in TYPES]

def build_metrics(device: str = "cpu"):
    """
    Real/fake metrics. The inputs are the 2-class probabilities: to_two_class(p_fake) and the int labels.
    With 2 classes, acc and auc are the same as the binary ones, and f1 is the macro F1 of REAL and FAKE.
    """
    metrics = MetricCollection({
        'acc': Accuracy(task="multiclass", num_classes=2),
        'auc': AUROC(task="multiclass", num_classes=2),
        'f1': F1Score(task="multiclass", num_classes=2, average="macro"),
    })

    return metrics.to(device)

def to_two_class(p_fake):
    """(B,) probability of FAKE -> (B,2) probabilities of REAL and FAKE"""
    return torch.stack([1 - p_fake, p_fake], dim=-1)

class PairAveragePrecision:
    """
    Average Precision of the 20 (region, type) pairs, reported as
        map           : mean AP of the 20 pairs
        ap_{region}   : mean AP of the 5 pairs of each region (4 values)
        ap_{type}     : mean AP of the 4 pairs of each type (5 values)
    Pairs with no positive image are left out of every mean (their AP is not defined).
    The scores and targets of the whole epoch are kept, and compute() is called at the end of the epoch.
    """

    def __init__(self, device: str = "cpu"):
        self.metric = MultilabelAveragePrecision(num_labels=len(PAIR_ORDER), average=None).to(device)
        self.num_pos = torch.zeros(len(PAIR_ORDER), dtype=torch.long, device=device)

    def reset(self):
        self.metric.reset()
        self.num_pos.zero_()

    def update(self, scores, targets):
        """scores: (B,20) probabilities, targets: (B,20) 0/1"""
        self.metric.update(scores, targets)
        self.num_pos += targets.sum(dim=0)

    def compute(self):
        """-> (map, {AP name: mean AP}) with the 9 AP_NAMES. nan when no pair of a group has a positive image"""
        ap = self.metric.compute().float()
        ap[self.num_pos == 0] = float("nan")

        grid = ap.view(len(REGIONS), len(TYPES))
        groups = [grid[r] for r in range(len(REGIONS))] + [grid[:, t] for t in range(len(TYPES))]

        return torch.nanmean(ap).item(), {name: torch.nanmean(g).item() for name, g in zip(AP_NAMES, groups)}
