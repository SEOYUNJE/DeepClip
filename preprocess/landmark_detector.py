import urllib.request
from pathlib import Path
from typing import List, Optional
import numpy as np

WEIGHTS_URL = "https://github.com/SEOYUNJE/DeepClip/releases/download/v0.0.1/shape_predictor_68_face_landmarks.dat"
DEFAULT_WEIGHTS_PATH = Path.home() / ".cache" / "deepclip" / "shape_predictor_68_face_landmarks.dat"

class LandmarkDetector:
    """
    dlib-based Face Landmark Detector.

    This class performs batch inference to detect 68 facial landmarks.
    """

    def __init__(
                self,
                predictor_path: Optional[str] = None,
                use_full_crop_rect: bool = False
                ):
        """
            Args:
                predictor_path: Path of shape_predictor_68_face_landmarks.dat. If the file does not exist,
                                it is downloaded from the DeepClip release
                                (default: ~/.cache/deepclip/shape_predictor_68_face_landmarks.dat)
                use_full_crop_rect: If True, the whole cropped image is used as the face rectangle
                                    instead of running dlib's frontal face detector
        """

        import dlib

        predictor_path = Path(predictor_path) if predictor_path else DEFAULT_WEIGHTS_PATH

        if not predictor_path.exists():
            predictor_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = predictor_path.with_name(predictor_path.name + ".part")
            print(f"Downloading {WEIGHTS_URL} -> {predictor_path}")
            urllib.request.urlretrieve(WEIGHTS_URL, tmp_path)
            tmp_path.replace(predictor_path)

        self.dlib = dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(predictor_path))
        self.use_full_crop_rect = use_full_crop_rect

    def detect_batch(
                    self,
                    frames: List[np.ndarray],
                    ):
        """
        Detects 68 facial landmarks for a batch of face images.

        Args:
            frames: List of Cropped Face Images (Numpy arrays, RGB format, uint8)

        Returns:
            List of landmarks.
            - Each landmark: (68, 2) float32 array of [x, y] coordinates if detected.
            - If no face detected, an empty list []
        """

        batch_landmarks = []

        for frame in frames:
            frame = np.ascontiguousarray(frame)

            if self.use_full_crop_rect:
                h, w = frame.shape[:2]
                rect = self.dlib.rectangle(0, 0, w - 1, h - 1)
            else:
                rects = self.detector(frame, 1)

                if len(rects) == 0:
                    batch_landmarks.append([])
                    continue

                # Use the largest face
                rect = max(rects, key=lambda r: r.area())

            shape = self.predictor(frame, rect)

            keypoints = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)

            batch_landmarks.append(keypoints)

        return batch_landmarks
