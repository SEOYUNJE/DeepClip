from typing import Optional
import cv2
import numpy as np
from deepclip.data import get_test_transforms
from .utils import ClipPredictorBase, to_artifact_scores


class ClipImagePredictor(ClipPredictorBase):
    """
    Detects the largest face in an image, crops it and predicts with a DeepClip model.

    Example:
        >>> from deepclip.detect import ClipImagePredictor
        >>> predictor = ClipImagePredictor(model_name="deepclip_b0")
        >>> predictor.predict_img("face.jpg")            # fake probability
        >>> predictor.predict_img_artifacts("face.jpg")  # {"prob": ..., "artifacts": {...}}
    """

    def __init__(
        self,
        margin_ratio: float = 0.2,
        conf_thres: float = 0.5,
        min_face_ratio: float = 0.01,
        model_name: str = "deepclip_b0",
        device: Optional[str] = None,
    ):
        super().__init__(margin_ratio, conf_thres, min_face_ratio, model_name, device)

    def _preprocess_img(self, img_path: str) -> Optional[np.ndarray]:
        """Returns the cropped RGB face, or None when the image is invalid or no face is found"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[ImageError] Image not found or invalid: {img_path}")
                return None

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            bbox = self.face_detector.detect_batch([img], [1.0])[0]
            if len(bbox) == 0:
                return None

            cropped = self._crop_face(img, bbox)
            if cropped.size == 0:
                print("[CropError] Resulting crop size is zero.")
                return None

            return cropped

        except Exception as e:
            print(f"[Error] Preprocess Failed for {img_path}: {e}")
            return None

    def _to_batch(self, img: np.ndarray, tta_hflip: float):
        transforms = get_test_transforms(img_size=self.img_size, tta_hflip=tta_hflip)
        return transforms(image=img)["image"].unsqueeze(0).to(self.device)  # (1,3,H,W)

    def predict_img(self, img_path: str, tta_hflip: float = 0) -> float:
        """Returns the fake probability (0.5 when no face is found)"""
        try:
            img = self._preprocess_img(img_path)
            if img is None:
                return 0.5

            return float(self._forward(self._to_batch(img, tta_hflip))[0])

        except Exception as e:
            print(f"[Critical Skip] Inference failed for {img_path}: {e}")
            return 0.5

    def predict_img_artifacts(self, img_path: str, tta_hflip: float = 0) -> dict:
        """
        Returns {"prob": fake probability, "artifacts": {region: {artifact type: probability}}}.
        artifacts is None when no face is found (prob is 0.5, same as predict_img).
        """
        try:
            img = self._preprocess_img(img_path)
            if img is None:
                return {"prob": 0.5, "artifacts": None}

            probs, artifact_probs = self._forward_artifacts(
                self._to_batch(img, tta_hflip)
            )

            return {
                "prob": float(probs[0]),
                "artifacts": to_artifact_scores(artifact_probs[0]),
            }

        except Exception as e:
            print(f"[Critical Skip] Inference failed for {img_path}: {e}")
            return {"prob": 0.5, "artifacts": None}
