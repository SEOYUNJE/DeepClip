from typing import Dict, List, Optional, Tuple
import numpy as np
import timm
import torch
from deepclip.data.text import PAIR_ORDER
from deepclip.models.deepclip import available_models, default_cfgs
from .face_detector import FaceDetector


def to_artifact_scores(artifact_probs: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    (20,) artifact probabilities of a DeepClip model -> {region: {artifact type: probability}}

        region        : face, eyes, nose, mouth
        artifact type : blur, color, texture, structure, blend
    """
    scores = {}
    for (region, type_), prob in zip(PAIR_ORDER, artifact_probs):
        scores.setdefault(region, {})[type_] = float(prob)
    return scores


class ClipPredictorBase:
    """
    Parts shared by ClipImagePredictor and ClipVideoPredictor.

    Besides the fake probability, DeepClip gives artifact scores: for each face region
    (face, eyes, nose, mouth), how likely each forgery artifact (blur, color, texture, structure, blend) is visible.
    """

    def __init__(
        self,
        margin_ratio: float,
        conf_thres: float,
        min_face_ratio: float,
        model_name: str,
        device: Optional[str] = None,
    ):
        if model_name not in available_models():
            raise ValueError(
                f"Unknown DeepClip model {model_name!r}. Available: {available_models()}"
            )

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.margin_ratio = margin_ratio
        self.face_detector = FaceDetector(conf_thres, min_face_ratio)
        self.model_name = model_name
        self.img_size = list(default_cfgs[model_name]["input_size"][1:])  # [H, W]

        # Model Inference Mode
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def _forward(self, batch: torch.Tensor) -> np.ndarray:
        """(B,3,H,W) -> (B,) fake probabilities"""
        with torch.no_grad():
            return torch.sigmoid(self.model(batch).view(-1)).cpu().numpy()

    def _forward_artifacts(self, batch: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """(B,3,H,W) -> (B,) fake probabilities, (B,20) artifact probabilities"""
        with torch.no_grad():
            img_emb = self.model.encode_image(batch)
            probs = torch.sigmoid(self.model.real_fake_logits(img_emb))
            artifact_probs = torch.sigmoid(self.model.pair_logits(img_emb))
        return probs.cpu().numpy(), artifact_probs.cpu().numpy()

    def _crop_face(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Crops the face with margin_ratio around the bbox, clipped to the image"""
        xmin, ymin, xmax, ymax = bbox

        w = xmax - xmin
        h = ymax - ymin
        pad_w = int(w * self.margin_ratio)
        pad_h = int(h * self.margin_ratio)

        y1 = max(int(ymin - pad_h), 0)
        y2 = min(int(ymax + pad_h), img.shape[0])
        x1 = max(int(xmin - pad_w), 0)
        x2 = min(int(xmax + pad_w), img.shape[1])

        return img[y1:y2, x1:x2]

    def __repr__(self):
        return (
            f"{type(self).__name__}(model_name='{self.model_name}', "
            f"margin_ratio={self.margin_ratio}, "
            f"device='{self.device}', "
            f"img_size={self.img_size})"
        )
