import numpy as np
import torch
from torch.utils.data import get_worker_info
from deepclip.data.text import REAL_TEXT, parse_pair_label, sample_selection, render_caption

class DeepClipCollate:
    """
    Makes the captions of a batch and the targets of the loss.

    For every image: REAL -> REAL_TEXT, FAKE -> the caption of the pair chosen by sample_selection
    (a new choice at every step). The B captions are reduced to the U unique captions (in order).

    Returns a dict:
        images       (B,3,H,W)
        labels       (B,)    0: REAL, 1: FAKE
        text_idx     (U,)    rows of text_table of the unique captions
        Y            (B,U)   1 if caption j is true for image i
                             (REAL_TEXT column: image i is REAL, FAKE column: the pair of caption j is on in image i)
        W            (B,U)   real_text_weight for the REAL_TEXT column, fake_text_weight for the FAKE columns
        pair_targets (B,20)  pair_label of every image (all 0 for REAL), target of map and the AP
    """

    def __init__(
            self,
            caption_to_idx: dict,
            type_weights: np.ndarray,
            region_weights: np.ndarray,
            real_text_weight: float = 1.0,
            fake_text_weight: float = 1.0,
            seed: int | None = None,
            ):
        """
        Args:
            caption_to_idx: caption -> row of text_table (DeepClip.caption_to_idx).
            type_weights: (4,5) weights of the types (type_weights of the train FAKE frames).
            region_weights: (4,) weights of the regions (region_weights of the train FAKE frames).
            real_text_weight, fake_text_weight: Weights of the REAL_TEXT column and the FAKE columns of the loss.
            seed: Fixed seed for the same captions at every epoch (valid). None for new captions at every epoch (train).
        """

        self.caption_to_idx = caption_to_idx
        self.type_weights = type_weights
        self.region_weights = region_weights
        self.real_text_weight = real_text_weight
        self.fake_text_weight = fake_text_weight
        self.seed = seed
        self.reset()

    def reset(self):
        """Restarts the random captions (with seed: the same captions as the previous epoch when num_workers=0)"""
        self.rng = np.random.default_rng(self.seed if self.seed is not None else np.random.randint(2**31))
        self._worker_id = None

    def _update_rng(self):
        # Every DataLoader worker gets a copy of this object, so the rng is set again in each worker
        info = get_worker_info()

        if info is not None and self._worker_id != info.id:
            self._worker_id = info.id
            self.rng = np.random.default_rng([self.seed, info.id] if self.seed is not None else info.seed)

    def __call__(self, batch):
        self._update_rng()

        imgs, labels, pair_labels = zip(*batch)
        images = torch.stack(imgs) # (B,3,H,W)
        labels = torch.stack(labels).float() # (B,)

        lab = np.stack([parse_pair_label(p) for p in pair_labels]) # (B,4,5)
        is_fake = labels.numpy() == 1

        # ====== Caption of every image ======
        captions, selections = [], []
        for i in range(len(batch)):
            if is_fake[i]:
                sel = sample_selection(lab[i], self.type_weights, self.region_weights, self.rng)
                captions.append(render_caption(sel))
            else:
                sel = np.zeros_like(lab[i])
                captions.append(REAL_TEXT)
            selections.append(sel)

        # ====== Unique captions (in order) ======
        uniq = list(dict.fromkeys(captions))
        first = [captions.index(c) for c in uniq]
        text_idx = torch.tensor([self.caption_to_idx[c] for c in uniq], dtype=torch.long) # (U,)

        # ====== Targets Y and weights W ======
        is_real_col = np.array([c == REAL_TEXT for c in uniq]) # (U,)
        sel_cols = np.stack([selections[i] for i in first]) # (U,4,5)

        # the pair of caption j is on in image i
        subset = ~(sel_cols[None] & ~lab[:, None]).any(axis=(2, 3)) # (B,U)
        Y = np.where(is_real_col[None], ~is_fake[:, None], is_fake[:, None] & subset)
        W = np.where(is_real_col, self.real_text_weight, self.fake_text_weight)

        return {
            "images": images,
            "labels": labels,
            "text_idx": text_idx,
            "Y": torch.from_numpy(Y).float(),
            "W": torch.from_numpy(np.broadcast_to(W, Y.shape).copy()).float(),
            "pair_targets": torch.from_numpy(lab.reshape(len(batch), -1)).long(),
        }
