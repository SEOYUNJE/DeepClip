"""
Face forgery text template.

Every rule about captions is kept in this file: the order of the 20 (region, type) pairs of `pair_label`
(also used by preprocess/raw_annotation.py), the caption phrases, the sampling of the pairs described by a
caption and the list of every caption a model can see (the rows of its text table).
"""

from itertools import combinations, product
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

REAL_TEXT = "a photo of a real face."   # REAL caption of itc_loss and real caption of cls_loss
FAKE_TEXT = "a photo of a fake face."   # fake caption of cls_loss (not used by itc_loss)

MAX_REGIONS = 2
MAX_TYPES = 2

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

def _check_limits(max_regions: int, max_types: int):
    if not (1 <= max_regions <= MAX_REGIONS and 1 <= max_types <= MAX_TYPES):
        raise ValueError(f"max_regions and max_types must be in [1, {MAX_REGIONS}] and [1, {MAX_TYPES}]: "
                         f"{max_regions}, {max_types}")

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
                     rng: np.random.Generator, max_regions: int, max_types: int) -> np.ndarray:
    """
    Chooses the pairs described by the caption of one FAKE image in this step.

    1. The number of regions (1 to max_regions) is uniform, and the regions are chosen among the forged
       regions with the probability region_weights[region] (normalized among them).
    2. In each chosen region, the number of types (1 to max_types) is uniform, and the types are chosen among
       the types which are on with the probability weights[region, type] (normalized among them).

    label: (4, 5) bool, the pairs which are on in the image
    return: (4, 5) bool, always a subset of label
    """

    _check_limits(max_regions, max_types)

    active_regions = np.flatnonzero(label.any(axis=1))

    if len(active_regions) == 0:
        raise ValueError("A FAKE image needs at least one pair which is on")

    selection = np.zeros_like(label, dtype=bool)

    k_r = rng.integers(1, min(max_regions, len(active_regions)) + 1)
    regions = _choice(rng, active_regions, k_r, region_weights[active_regions])

    for region in regions:
        active_types = np.flatnonzero(label[region])
        k_t = rng.integers(1, min(max_types, len(active_types)) + 1)
        types = _choice(rng, active_types, k_t, weights[region, active_types])
        selection[region, types] = True

    return selection

def render_caption(selection: np.ndarray) -> str:
    """
    Chosen pairs -> caption. Regions follow REGIONS and types follow TYPES, so the same pairs always give
    the same caption. No pair gives REAL_TEXT.
    """

    parts = []

    for r, region in enumerate(REGIONS):
        types = [TYPE_PHRASES[t] for i, t in enumerate(TYPES) if selection[r, i]]

        if types:
            parts.append(f"{' and '.join(types)} in {REGION_PHRASES[region]}")

    if not parts:
        return REAL_TEXT

    return f"a photo of a fake face with {', and '.join(parts)}."

def all_captions(max_regions: int = MAX_REGIONS, max_types: int = MAX_TYPES) -> list:
    """
    Every caption which sample_selection + render_caption can make, in a fixed order:
    REAL_TEXT, FAKE_TEXT, the 20 captions with one pair in PAIR_ORDER, then the other captions.
    """

    _check_limits(max_regions, max_types)

    type_sets = [types for k in range(1, max_types + 1) for types in combinations(range(len(TYPES)), k)]

    captions = [REAL_TEXT, FAKE_TEXT]
    captions += [single_pair_caption(region, type_) for region, type_ in PAIR_ORDER]
    seen = set(captions)

    for k in range(1, max_regions + 1):
        for regions in combinations(range(len(REGIONS)), k):
            for type_combo in product(type_sets, repeat=k):
                selection = np.zeros((len(REGIONS), len(TYPES)), dtype=bool)

                for region, types in zip(regions, type_combo):
                    selection[region, list(types)] = True

                caption = render_caption(selection)

                if caption not in seen:
                    seen.add(caption)
                    captions.append(caption)

    return captions

def single_pair_caption(region: str, type_: str) -> str:
    selection = np.zeros((len(REGIONS), len(TYPES)), dtype=bool)
    selection[REGIONS.index(region), TYPES.index(type_)] = True

    return render_caption(selection)
