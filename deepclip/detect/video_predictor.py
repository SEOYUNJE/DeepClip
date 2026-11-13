import os
from typing import Dict, List, Optional
import cv2
import numpy as np
import torch
from deepclip.data import get_test_transforms
from .utils import ClipPredictorBase, to_artifact_scores


class ClipVideoPredictor(ClipPredictorBase):
    """
    Samples num_frames frames from a video, crops the largest face of each frame,
    predicts each frame with a DeepClip model and aggregates the frames.

    Example:
        >>> from deepclip.detect import ClipVideoPredictor
        >>> predictor = ClipVideoPredictor(model_name="deepclip_b0")
        >>> predictor.predict_video("video.mp4")            # fake probability
        >>> predictor.predict_video_artifacts("video.mp4")  # {"prob": ..., "artifacts": {...}}
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

    def _get_frame_indices(self, frame_cnt: int, num_frames: int) -> List[int]:
        frame_indices = np.linspace(
            0, frame_cnt - 1, num=num_frames, endpoint=True, dtype=np.int32
        )
        return sorted(set(frame_indices.tolist()))

    def _extract_frames(
        self, video_path: str, num_frames: int
    ) -> Dict[int, np.ndarray]:
        """
        Extracts frames from the video and returns a dictionary in the format {frame_index: RGB frame}.
        Returns an empty dictionary {} if the extraction fails.
        """
        cap = None
        try:
            if not os.path.exists(video_path):
                print(f"[VideoError] File not found: {video_path}")
                return {}

            cap = cv2.VideoCapture(filename=video_path)
            if not cap.isOpened():
                print(f"[VideoError] Open failed: {video_path}")
                return {}

            frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_cnt <= 0:
                print(f"[VideoError] Invalid frame count: {video_path}")
                return {}

            frame_indices = self._get_frame_indices(frame_cnt, num_frames)
            frames = {}
            target_idx = 0

            for frame_idx in range(frame_indices[-1] + 1):
                if not cap.grab():
                    break

                if frame_idx == frame_indices[target_idx]:
                    ret, frame = cap.retrieve()
                    if ret and frame is not None:
                        frames[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    target_idx += 1
                    if target_idx >= len(frame_indices):
                        break

            return frames

        except Exception as e:
            print(f"[UnexpectedError] {video_path}: {e}")
            return {}
        finally:
            if cap is not None and cap.isOpened():
                cap.release()

    def _preprocess_frames(self, video_path: str, num_frames: int) -> List[np.ndarray]:
        """
        Detects and crops faces from extracted frames, returning them as a list.
        Returns an empty list [] if no faces are detected or an error occurs.
        """
        try:
            frames = list(self._extract_frames(video_path, num_frames).values())
            if not frames:
                return []

            bboxes = self.face_detector.detect_batch(frames, [1.0] * len(frames))
            cropped_frames = [
                self._crop_face(frame, bbox)
                for frame, bbox in zip(frames, bboxes)
                if len(bbox) > 0
            ]

            if not cropped_frames:
                print(f"[Info] No faces detected in: {video_path}")

            return cropped_frames

        except Exception as e:
            print(f"[Error] Preprocess Failed for {video_path}: {e}")
            return []

    def _to_batch(self, frames: List[np.ndarray], tta_hflip: float) -> torch.Tensor:
        transforms = get_test_transforms(img_size=self.img_size, tta_hflip=tta_hflip)
        return torch.stack([transforms(image=f)["image"] for f in frames]).to(
            self.device
        )  # (B,3,H,W)

    def _frame_aggregate(self, preds: np.ndarray, agg_mode: str) -> float:
        """
        mean : mean of the frames
        vote : ratio of frames over 0.5
        conf : mean of the confident frames (> 0.8 or < 0.2) when enough frames agree, otherwise mean
        """
        sz = len(preds)

        if agg_mode == "mean":
            return float(np.mean(preds))

        elif agg_mode == "vote":
            return float(np.sum(preds > 0.5) / sz)

        elif agg_mode == "conf":
            t_fake = 0.8
            t_real = 0.2

            fakes = preds[preds > t_fake]

            if len(fakes) > (sz // 2.5):
                return float(np.mean(fakes))
            elif np.count_nonzero(preds < t_real) > 0.9 * sz:
                return float(np.mean(preds[preds < t_real]))
            else:
                return float(np.mean(preds))

        raise ValueError(
            f"Unknown agg_mode {agg_mode!r}. Available: ['mean', 'vote', 'conf']"
        )

    def predict_video(
        self,
        video_path: str,
        num_frames: int = 20,
        agg_mode: str = "conf",
        tta_hflip: float = 0,
    ) -> float:
        """Returns the fake probability aggregated over the frames (0.5 when no face is found)"""
        try:
            frames = self._preprocess_frames(video_path, num_frames)
            if not frames:
                return 0.5

            return self._frame_aggregate(
                self._forward(self._to_batch(frames, tta_hflip)), agg_mode
            )

        except Exception as e:
            print(f"Error in predict_video: {e}")
            return 0.5

    def predict_video_artifacts(
        self,
        video_path: str,
        num_frames: int = 20,
        agg_mode: str = "conf",
        tta_hflip: float = 0,
    ) -> dict:
        """
        Returns {"prob": fake probability, "artifacts": {region: {artifact type: probability}}}.
        prob is aggregated over the frames with agg_mode (same as predict_video), artifacts is the mean over the frames.
        artifacts is None when no face is found (prob is 0.5, same as predict_video).
        """
        try:
            frames = self._preprocess_frames(video_path, num_frames)
            if not frames:
                return {"prob": 0.5, "artifacts": None}

            probs, artifact_probs = self._forward_artifacts(
                self._to_batch(frames, tta_hflip)
            )  # (B,), (B,20)

            return {
                "prob": self._frame_aggregate(probs, agg_mode),
                "artifacts": to_artifact_scores(artifact_probs.mean(axis=0)),
            }

        except Exception as e:
            print(f"Error in predict_video_artifacts: {e}")
            return {"prob": 0.5, "artifacts": None}
