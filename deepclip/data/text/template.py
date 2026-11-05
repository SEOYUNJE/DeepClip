"""
Face forgery text template.

Every rule about captions is kept in this file: the order of the 20 (region, type) pairs of `pair_label`
(also used by preprocess/raw_annotation.py), the caption phrases, the sampling of the pairs described by a
caption and the list of every caption a model can see (the rows of its text table).
"""

import numpy as np

REGIONS = ("face", "eyes", "nose", "mouth")
TYPES = ("blur", "color", "texture", "structure", "blend")
# Order of the 20 characters of pair_label: index = 5 * region + type
PAIR_ORDER = [(region, type_) for region in REGIONS for type_ in TYPES]

REGION_PHRASES = {
    "face": "the whole face",
    "eyes": "the eyes",
    "nose": "the nose",
    "mouth": "the mouth",
}
TYPE_PHRASES = {
    "blur": "unnatural blur",
    "color": "unnatural color variations",
    "texture": "unusual texture patterns",
    "structure": "structural distortion",
    "blend": "blending artifacts",
}

# REAL caption of the loss, also the caption of the real/fake score (p_fake = 1 - p_real)
REAL_TEXT = "a photo of a real face."

def parse_pair_label(pair_label: str) -> np.ndarray:
    """'0101...' (20 characters) -> (4, 5) bool array"""

    if len(pair_label) != len(PAIR_ORDER):
        raise ValueError(f"pair_label must have {len(PAIR_ORDER)} characters: {pair_label!r}")

    return (np.frombuffer(pair_label.encode(), dtype=np.uint8) == ord("1")).reshape(len(REGIONS), len(TYPES))

def _stack(pair_labels) -> np.ndarray:
    return np.stack([parse_pair_label(s) for s in pair_labels])

def pair_frequency(pair_labels) -> np.ndarray:
    """
    pair_label of the FAKE frames -> (4, 5) frequency.
    freq(region, type) = frames where the type flag of the region is on / frames where the region is forged
    """

    labels = _stack(pair_labels)
    forged = labels.any(axis=2).sum(axis=0)

    return labels.sum(axis=0) / np.maximum(forged, 1)[:, None]

def region_frequency(pair_labels) -> np.ndarray:
    """pair_label of the FAKE frames -> (4,) ratio of the frames where each region is forged"""

    return _stack(pair_labels).any(axis=2).mean(axis=0)

def type_weights(freq: np.ndarray, alpha: float) -> np.ndarray:
    """(4, 5) w = freq^-alpha (0 for a pair that never appears)"""

    return np.where(freq > 0, np.power(np.maximum(freq, 1e-12), -alpha), 0.0)

def region_weights(freq: np.ndarray, beta: float) -> np.ndarray:
    """(4,) rw = freq^-beta (0 for a region that is never forged)"""

    return np.where(freq > 0, np.power(np.maximum(freq, 1e-12), -beta), 0.0)

def _choice(rng: np.random.Generator, items: np.ndarray, size: int, weights: np.ndarray) -> np.ndarray:
    """
    Weighted choice without replacement (uniform when every weight is 0).
    An item with weight 0 (a pair or region never seen in the train FAKE frames) gets the smallest nonzero
    weight, otherwise there can be fewer nonzero weights than size and rng.choice fails.
    """

    nonzero = weights > 0
    if not nonzero.all():
        weights = np.where(nonzero, weights, weights[nonzero].min() if nonzero.any() else 1.0)

    return rng.choice(items, size=size, replace=False, p=weights / weights.sum())

def sample_selection(label: np.ndarray, weights: np.ndarray, region_weights: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Chooses the (region, type) pair described by the caption of one FAKE image in this step.

    1. One region is chosen among the forged regions with the probability region_weights[region]
       (normalized among them).
    2. One type is chosen among the types which are on in that region with the probability
       weights[region, type] (normalized among them).

    label: (4, 5) bool, the pairs which are on in the image
    return: (4, 5) bool with one True, always a subset of label
    """

    active_regions = np.flatnonzero(label.any(axis=1))

    if len(active_regions) == 0:
        raise ValueError("A FAKE image needs at least one pair which is on")

    selection = np.zeros_like(label, dtype=bool)

    region = _choice(rng, active_regions, 1, region_weights[active_regions])[0]
    active_types = np.flatnonzero(label[region])
    type_ = _choice(rng, active_types, 1, weights[region, active_types])[0]
    selection[region, type_] = True

    return selection

def render_caption(selection: np.ndarray) -> str:
    """Chosen pair ((4, 5) bool with one True) -> caption. No pair gives REAL_TEXT."""

    pairs = np.argwhere(selection)

    if len(pairs) == 0:
        return REAL_TEXT
    if len(pairs) != 1:
        raise ValueError(f"A caption describes one (region, type) pair, not {len(pairs)}")

    region, type_ = pairs[0]
    return f"a photo of a fake face with {TYPE_PHRASES[TYPES[type_]]} in {REGION_PHRASES[REGIONS[region]]}."

def all_captions() -> list:
    """Every caption the collate can make, in a fixed order: REAL_TEXT, then the 20 captions of PAIR_ORDER"""

    return [REAL_TEXT] + [single_pair_caption(region, type_) for region, type_ in PAIR_ORDER]

def single_pair_caption(region: str, type_: str) -> str:
    selection = np.zeros((len(REGIONS), len(TYPES)), dtype=bool)
    selection[REGIONS.index(region), TYPES.index(type_)] = True

    return render_caption(selection)
