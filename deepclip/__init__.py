import warnings
import logging
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("deepclip") # pyproject.toml
except PackageNotFoundError: # Not installed (e.g. run from the repository)
    __version__ = "0.0.0"

# ------- 로그 및 경고 제어 ---------
warnings.filterwarnings("ignore") # 일반 경고 숨기기
logging.getLogger("timm").setLevel(logging.ERROR) # timm 내부 로그 숨기기
logging.getLogger("torch").setLevel(logging.ERROR) # torch 관련 로그 숨기기

from .models import image_encoder, text_encoder, deepclip
from .models.image_encoder import DeepImageEncoder, DeepImageEncoderV2
from .models.text_encoder import DeepTextEncoder
from .models.deepclip import (
    DeepClip,
    DeepClipV2,
    deepclip_b0,
    deepclip_b5,
    deepclip_b0_v2,
    deepclip_b5_v2,
)

__all__ = [
    "__version__",
    "DeepImageEncoder",
    "DeepImageEncoderV2",
    "DeepTextEncoder",
    "DeepClip",
    "DeepClipV2",
    "deepclip_b0",
    "deepclip_b5",
    "deepclip_b0_v2",
    "deepclip_b5_v2",
]
