"""
Forgery feature analysis of VLFFD (Face Forgery Text Generator, raw annotation step).

This module re-implements the analysis of VLFFD's raw_annotation.py (https://github.com/skJack/VLFFD)
for a pair of aligned face crops (real / fake) and the 68 facial landmarks of the fake crop.

Faithful to the original implementation:
    - Images are BGR arrays read by cv2.imread and are passed as-is to cv2.COLOR_RGB2LAB / COLOR_RGB2GRAY,
      like the original code does (the channel order is not swapped).
    - Features are computed on the whole image multiplied by the region mask (pixels outside of the
      region are 0), not only on the pixels inside of the region.
    - The thresholds of the forgery mask, blend gradient (0.5) are the values of the original code.

Differences from the original implementation:
    - The thresholds of blur, color, texture, structure, blend edge and blend frequency are adjusted to the FF++
      score distribution. The original values (100, 2.0, 0.7, 0.97, 0.4, 0.5) turn the blur and color flags on too
      rarely and the texture, structure and edge flags on almost always for FF++, and the frequency flag never.
      The thresholds are the same for all regions.
    - The same region mask (built from the landmarks of the fake crop) is applied to the real and fake
      images. The original builds the real region from the landmarks of the real image.
    - Every forged region is analyzed and the scores are returned, not only the flags.
    - skimage's removed `multichannel=True` argument is replaced by `channel_axis=-1`.
    - scipy's binary_dilation fills the whole image if `iterations < 1`; no dilation is applied instead.
"""

import cv2
import numpy as np
from scipy.ndimage import binary_dilation
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity

REGIONS = ("face", "eyes", "nose", "mouth")

FACE_POINTS = list(range(17, 68))

MASK_THRESHOLD = 55
BLUR_THRESHOLD = 25
COLOR_THRESHOLD = 1.0
TEXTURE_THRESHOLD = 2.8
STRUCTURE_THRESHOLD = 0.985
GRADIENT_THRESHOLD = 0.5
EDGE_THRESHOLD = 4
FREQUENCY_THRESHOLD = 0.15
BOUNDARY_WIDTH = 10

# Names of the values computed for each forged region
FEATURE_FIELDS = (
    "blur_score", "blur_flag",
    "color_score", "color_flag",
    "texture_score", "texture_flag",
    "structure_score", "structure_flag",
    "blend_gradient", "blend_edge", "blend_freq", "blend_flag",
)


def compute_forgery_mask(real_image: np.ndarray, fake_image: np.ndarray) -> np.ndarray:
    """
    Forgery mask from the structural similarity of a real / fake pair
    (generate_mask_from_diff of the original implementation).

    Args:
        real_image, fake_image: BGR images of the same shape
    Returns:
        (H, W) uint8 mask. The more the fake image differs from the real image, the larger the value
    """

    if real_image.shape != fake_image.shape:
        raise ValueError(f"Shape mismatch: {real_image.shape} vs {fake_image.shape}")

    _, ssim_map = structural_similarity(real_image, fake_image, channel_axis=-1, full=True)

    diff = 1 - ssim_map

    # Pixels with a negative similarity exceed 255 here, like in the original implementation
    with np.errstate(invalid="ignore"):
        mask = (diff * 255).astype(np.uint8)

    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


def _dist(p1, p2) -> float:
    return float(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))


def _dilate(line: np.ndarray, iterations: int) -> np.ndarray:
    # binary_dilation fills the whole image if iterations < 1
    if iterations < 1:
        return line.astype(np.uint8)

    return binary_dilation(line, iterations=iterations).astype(np.uint8)


def _get_five_key(landmarks: np.ndarray):
    """Left eye center, right eye center, nose tip, left mouth corner, right mouth corner"""

    leye_center = (landmarks[36] + landmarks[39]) * 0.5
    reye_center = (landmarks[42] + landmarks[45]) * 0.5

    keys = [leye_center, reye_center, landmarks[33], landmarks[48], landmarks[54]]

    return [tuple(int(v) for v in k.astype("int32")) for k in keys]


def get_region_mask(image_shape, landmarks: np.ndarray, region: str) -> np.ndarray:
    """
    Region mask built from the 68 landmarks.

    Args:
        image_shape: shape of the image (H, W, ...)
        landmarks: (68, 2) landmarks of the image
        region: one of REGIONS
    Returns:
        (H, W) uint8 mask of 0 / 1
    """

    landmarks = np.asarray(landmarks).astype(np.int32)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    if region == "face":
        points = cv2.convexHull(landmarks[FACE_POINTS])
        return cv2.fillConvexPoly(mask, points, color=1)

    leye, reye, nose, lmouth, rmouth = _get_five_key(landmarks)

    if region == "eyes":
        (x1, y1), (x2, y2) = leye, reye
        line = cv2.line(mask, (x1, y1), (x2, y2), color=1, thickness=2)
        return _dilate(line, int(_dist((x1, y1), (x2, y2)) // 4))

    if region == "nose":
        (x1, y1), (x2, y2) = leye, reye
        x3, y3 = nose
        x4 = int((x1 + x2) / 2)
        y4 = int((y1 + y2) / 2)
        line = cv2.line(mask, (x3, y3), (x4, y4), color=1, thickness=2)
        return _dilate(line, int(_dist((x1, y1), (x2, y2)) // 4))

    if region == "mouth":
        (x1, y1), (x2, y2) = lmouth, rmouth
        line = cv2.line(mask, (x1, y1), (x2, y2), color=1, thickness=2)
        return _dilate(line, int(_dist((x1, y1), (x2, y2)) // 3))

    raise ValueError(f"Unknown region: {region}")


def get_forgery_degree(region_mask: np.ndarray, forgery_mask: np.ndarray) -> float:
    """Average value of the forgery mask inside of the region (0 ~ 255)"""

    area = np.count_nonzero(region_mask)

    if area == 0:
        return 0.0

    return float(np.sum(region_mask * forgery_mask, dtype=np.float64)) / area


def extract_region(image: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    """Image whose pixels outside of the region are 0"""

    return image * np.expand_dims(region_mask, -1)


def get_blur_score(real_part: np.ndarray, fake_part: np.ndarray) -> float:
    """Laplacian variance of the real image minus that of the fake image"""

    real_var = cv2.Laplacian(real_part, cv2.CV_64F).var()
    fake_var = cv2.Laplacian(fake_part, cv2.CV_64F).var()

    return float(real_var - fake_var)


def get_color_score(real_part: np.ndarray, fake_part: np.ndarray) -> float:
    """Mean of the differences of the mean and std in the Lab color space"""

    real_lab = cv2.cvtColor(real_part, cv2.COLOR_RGB2LAB)
    fake_lab = cv2.cvtColor(fake_part, cv2.COLOR_RGB2LAB)

    real_mean = np.mean(real_lab, axis=(0, 1))
    fake_mean = np.mean(fake_lab, axis=(0, 1))
    real_std = np.std(real_lab, axis=(0, 1))
    fake_std = np.std(fake_lab, axis=(0, 1))

    mean_diff = np.mean(np.abs(real_mean - fake_mean))
    std_diff = np.mean(np.abs(real_std - fake_std))

    return float((mean_diff + std_diff) / 2)


def _glcm_contrast(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)

    return float(graycoprops(glcm, 'contrast')[0, 0])


def get_texture_score(real_part: np.ndarray, fake_part: np.ndarray) -> float:
    """Absolute difference of the GLCM contrast"""

    return abs(_glcm_contrast(real_part) - _glcm_contrast(fake_part))


def get_structure_score(real_part: np.ndarray, fake_part: np.ndarray) -> float:
    """Structural similarity (lower means more deformed)"""

    return float(structural_similarity(real_part, fake_part, channel_axis=-1, data_range=255))


def _get_boundary_region(mask: np.ndarray, width: int = BOUNDARY_WIDTH):
    binary_mask = (mask > 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    inner_mask = cv2.erode(binary_mask, kernel, iterations=width)
    outer_mask = cv2.dilate(binary_mask, kernel, iterations=width)

    return binary_mask - inner_mask, outer_mask - binary_mask


def _mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if values.size > 0 else float("nan")


def get_blend_scores(image: np.ndarray, fake_region: np.ndarray):
    """
    Blending artifacts around the boundary of the region (only the fake image is used).

    Args:
        image: the whole fake image
        fake_region: the fake image multiplied by the region mask
    Returns:
        gradient, edge, frequency scores of the boundary
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # The original implementation uses the first channel of the extracted region as the mask
    inner_boundary, outer_boundary = _get_boundary_region(fake_region[..., 0])
    total_boundary = inner_boundary + outer_boundary

    # Gradient discontinuity between the inner and outer boundary
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    inner_gradient = _mean_or_nan(gradient_magnitude[inner_boundary > 0])
    outer_gradient = _mean_or_nan(gradient_magnitude[outer_boundary > 0])
    gradient_score = abs(inner_gradient - outer_gradient) / (outer_gradient + 1e-6)

    # Edge density at the boundary
    edges = cv2.Canny(gray, 100, 200)
    boundary_edges = edges * total_boundary
    edge_score = np.sum(boundary_edges, dtype=np.float64) / (np.sum(total_boundary, dtype=np.float64) + 1e-6)

    # High / low frequency ratio at the boundary
    dct = cv2.dct(np.float32(gray * total_boundary))
    h, w = dct.shape
    high_freq = dct[h // 2:, w // 2:]
    low_freq = dct[:h // 2, :w // 2]
    freq_score = np.sum(np.abs(high_freq)) / (np.sum(np.abs(low_freq)) + 1e-6)

    return float(gradient_score), float(edge_score), float(freq_score)


def analyze_region(real_image: np.ndarray, fake_image: np.ndarray, region_mask: np.ndarray) -> dict:
    """
    Scores and flags of the 5 forgery types (blur, color, texture, structure, blending) of a region.

    Returns:
        dict whose keys are FEATURE_FIELDS
    """

    real_part = extract_region(real_image, region_mask)
    fake_part = extract_region(fake_image, region_mask)

    blur_score = get_blur_score(real_part, fake_part)
    color_score = get_color_score(real_part, fake_part)
    texture_score = get_texture_score(real_part, fake_part)
    structure_score = get_structure_score(real_part, fake_part)
    gradient_score, edge_score, freq_score = get_blend_scores(fake_image, fake_part)

    # NaN comparisons are False
    blend_evidence = (
        (gradient_score > GRADIENT_THRESHOLD)
        + (edge_score > EDGE_THRESHOLD)
        + (freq_score > FREQUENCY_THRESHOLD)
    )

    return {
        "blur_score": blur_score,
        "blur_flag": blur_score > BLUR_THRESHOLD,
        "color_score": color_score,
        "color_flag": color_score > COLOR_THRESHOLD,
        "texture_score": texture_score,
        "texture_flag": texture_score > TEXTURE_THRESHOLD,
        "structure_score": structure_score,
        "structure_flag": structure_score < STRUCTURE_THRESHOLD,
        "blend_gradient": gradient_score,
        "blend_edge": edge_score,
        "blend_freq": freq_score,
        "blend_flag": bool(blend_evidence >= 2),
    }


def analyze_frame(
                  real_image: np.ndarray,
                  fake_image: np.ndarray,
                  landmarks: np.ndarray,
                  forgery_mask: np.ndarray
                  ) -> dict:
    """
    Analyzes every region of a real / fake pair.

    Args:
        real_image, fake_image: BGR images of the same shape
        landmarks: (68, 2) landmarks of the fake image
        forgery_mask: output of compute_forgery_mask
    Returns:
        {region: None if the region is not forged else analyze_region result}
    """

    results = {}

    for region in REGIONS:
        region_mask = get_region_mask(fake_image.shape, landmarks, region)

        if get_forgery_degree(region_mask, forgery_mask) > MASK_THRESHOLD:
            results[region] = analyze_region(real_image, fake_image, region_mask)
        else:
            results[region] = None

    return results
