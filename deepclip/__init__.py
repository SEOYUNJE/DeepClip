import warnings
import logging

# ------- 로그 및 경고 제어 ---------
warnings.filterwarnings("ignore") # 일반 경고 숨기기
logging.getLogger("timm").setLevel(logging.ERROR) # timm 내부 로그 숨기기
logging.getLogger("torch").setLevel(logging.ERROR) # torch 관련 로그 숨기기

from .models import image_encoder, text_encoder, deepclip
from .models.image_encoder import DeepImageEncoder
from .models.text_encoder import DeepTextEncoder
from .models.deepclip import (
    DeepClip,
    deepclip_b0_r1_t1,
    deepclip_b0_r1_t2,
    deepclip_b0_r2_t2,
    deepclip_b5_r1_t1,
    deepclip_b5_r1_t2,
    deepclip_b5_r2_t2,
)

__all__ = [
    "DeepImageEncoder",
    "DeepTextEncoder",
    "DeepClip",
    "deepclip_b0_r1_t1",
    "deepclip_b0_r1_t2",
    "deepclip_b0_r2_t2",
    "deepclip_b5_r1_t1",
    "deepclip_b5_r1_t2",
    "deepclip_b5_r2_t2",
]
