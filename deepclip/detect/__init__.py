# deepclip.detect needs the optional extra "detect" (ultralytics, for the YOLO face detection).
# Without it, `from deepclip.detect import ...` fails here with how to install it.
try:
    import ultralytics  # noqa: F401
except ModuleNotFoundError as e:
    if e.name != "ultralytics":
        raise
    raise ModuleNotFoundError(
        "deepclip.detect needs ultralytics for face detection, which is not installed.\n"
        "Install deepclip with the 'detect' extra:\n\n"
        '    pip install "deepclip[detect]"\n\n'
        "(or install it on its own: pip install ultralytics)",
        name="ultralytics",
    ) from None

from .utils import to_artifact_scores
from .image_predictor import ClipImagePredictor
from .video_predictor import ClipVideoPredictor

__all__ = [
    "ClipImagePredictor",
    "ClipVideoPredictor",
    "to_artifact_scores",
]
