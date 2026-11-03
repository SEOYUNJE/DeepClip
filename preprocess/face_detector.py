import urllib.request
from pathlib import Path
from ultralytics import YOLO
from typing import List, Optional
import numpy as np

WEIGHTS_URL = "https://github.com/SEOYUNJE/DeepClip/releases/download/v0.0.1/yolov8n-face.pt"
DEFAULT_WEIGHTS_PATH = Path.home() / ".cache" / "deepclip" / "yolov8n-face.pt"

class FaceDetector:
    """
    YOLO-based Face Detector.

    This class performs batch inference to detect FACE bounding boxes.
    It uses a custom YOLO model trained on face datasets (e.g., yolov8n-face.pt).
    """

    def __init__(
                self,
                conf_thres: float,
                min_face_ratio: float,
                weights_path: Optional[str] = None
                ):
        """
            Args:
                conf_thres: Confidence threshold for YOLO face detection
                min_face_ratio: Minimum face bounding box area ratio relative to the full frame area.
                weights_path: Path of yolov8n-face.pt. If the file does not exist, it is downloaded
                              from the DeepClip release (default: ~/.cache/deepclip/yolov8n-face.pt)
        """

        weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH

        if not weights_path.exists():
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = weights_path.with_name(weights_path.name + ".part")
            print(f"Downloading {WEIGHTS_URL} -> {weights_path}")
            urllib.request.urlretrieve(WEIGHTS_URL, tmp_path)
            tmp_path.replace(weights_path)

        self.model = YOLO(str(weights_path), task='detect')
        self.conf_thres = conf_thres
        self.min_face_ratio = min_face_ratio

    def detect_batch(
                    self,
                    frames: List[np.ndarray],
                    scales: List[float]
                    ):

        results = self.model(frames, conf=self.conf_thres, verbose=False)

        batch_boxes = []

        for i, result in enumerate(results):
            scale_factor = scales[i]

            if len(result.boxes) == 0:
                batch_boxes.append([])
                continue

            # (N, 6) -> [xmin, ymin, xmax, ymax, conf, cls]
            data = result.boxes.data.cpu().numpy()

            face_boxes = data[data[:, 5] == 0] # Category: Face

            if len(face_boxes) == 0:
                batch_boxes.append([])
                continue

            ws = face_boxes[:, 2] - face_boxes[:, 0]
            hs = face_boxes[:, 3] - face_boxes[:, 1]
            areas = ws * hs

            max_area_idx = np.argmax(areas)
            max_area = areas[max_area_idx]

            frame_h, frame_w = frames[i].shape[:2]
            frame_area = frame_h * frame_w

            if (max_area / frame_area) < self.min_face_ratio:
                batch_boxes.append([])
            else:
                best_face_box = face_boxes[max_area_idx, :4]

                final_box = best_face_box / scale_factor

                batch_boxes.append(final_box.tolist())

        return batch_boxes
