# DeepClip

<p align="center">
  <img src="https://img.shields.io/github/license/SEOYUNJE/DeepClip?style=flat-square&color=555555&logo=github&logoColor=white" alt="License">
  <img src="https://img.shields.io/github/stars/SEOYUNJE/DeepClip?style=flat-square&color=FFD700&logo=github&logoColor=white" alt="Stars">
  <img src="https://img.shields.io/github/v/release/SEOYUNJE/DeepClip?style=flat-square&color=orange&logo=github&logoColor=white" alt="Release">
  <img src="https://img.shields.io/pypi/v/deepclip?style=flat-square&color=3775A9&logo=pypi&logoColor=white" alt="PyPI">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
</p>

<p align="center">
  <b>CLIP for deepfakes</b> — face forgery detection aligned with CLIP text embeddings
</p>

## 📌 Contents

- [💡 What DeepClip Predicts](#-what-deepclip-predicts) - Fake probability plus 20 artifact scores (4 face regions × 5 artifact types)
- [🧬 Model Zoo](#-model-zoo) - `deepclip_b0` and `deepclip_b5`
- [📦 Installation](#-installation) - `pip install deepclip` for inference, or the full repo for training
- [🔮 Predict Image & Video](#-predict-image--video) - Detect deepfakes and their artifacts with `deepclip.detect`
- [🚀 Training](#-training) - Train DeepClip with `train_deepclip.py` and W&B experiment tracking
- [📈 Model Evaluation](#-model-evaluation) - Benchmark a pretrained model on deepfake videos with `test_deepclip.py`
- [⚖️ License](#-license) - MIT license

## 💡 What DeepClip Predicts

DeepClip scores a face image against CLIP text captions. Besides the **fake probability**, it gives an
**artifact score** for each pair of a face region and a forgery artifact type, so you can see *where* and *what kind of*
forgery the model sees.

| | Values |
| --- | --- |
| **Region** | `face`, `eyes`, `nose`, `mouth` |
| **Artifact type** | `blur`, `color`, `texture`, `structure`, `blend` |

- **Fake probability**: `1 - p_real`, where `p_real` is the score of the caption *"a photo of a real face."*
- **Artifact scores**: 4 regions × 5 types = 20 captions such as *"... blending artifacts on the mouth"*, each a probability in `[0, 1]`.

## 🧬 Model Zoo

| Model | Resolution | Image Backbone | Text Encoder | Model Config | Download |
| ----- | ---------- | -------------- | ------------ | ------------ | -------- |
| ⚡ deepclip_b0 | 224 X 224 | tf_efficientnet_b0 + GCViT | CLIP ViT-B/16 (frozen) | [spec](deepclip/config/deepclip_b0.yaml) | [model](https://github.com/SEOYUNJE/DeepClip/releases/download/v1.0.0/deepclip_b0.pt) |
| 🔥 deepclip_b5 | 384 X 384 | tf_efficientnet_b5 + GCViT | CLIP ViT-L/14 (frozen) | [spec](deepclip/config/deepclip_b5.yaml) | [model](https://github.com/SEOYUNJE/DeepClip/releases/download/v1.0.0/deepclip_b5.pt) |

The pretrained weights are downloaded automatically with `timm.create_model(model_name, pretrained=True)`.

## 📦 Installation

### Option A: Pip (Inference)

```bash
pip install "deepclip[detect]"
```

The `detect` extra installs [`ultralytics`](https://github.com/ultralytics/ultralytics) for the YOLO face detection
used by `deepclip.detect`. If you only need the models (`timm.create_model("deepclip_b0", pretrained=True)`),
`pip install deepclip` is enough.

> [!NOTE]
> Importing `deepclip.detect` without the extra raises an error that tells you to run `pip install "deepclip[detect]"`.

### Option B: Full Repository (Training & Evaluation)

`train_deepclip.py` and `test_deepclip.py` run against the full repo, not the pip package.
Clone it and install the full dev environment:

```bash
git clone https://github.com/SEOYUNJE/DeepClip.git
cd DeepClip
pip install -r requirements.txt
```

## 🔮 Predict Image & Video

The predictors detect the largest face with YOLO (`yolov8n-face`, downloaded to `~/.cache/deepclip` on the first run),
crop it with a margin and run DeepClip on the crop.

#### Predict DeepFake Image

```python
from deepclip.detect import ClipImagePredictor

# Initialize the predictor
predictor = ClipImagePredictor(
            margin_ratio = 0.2, # Margin ratio around the detected face crop
            conf_thres = 0.5, # Confidence threshold for face detection
            min_face_ratio = 0.01, # Minimum face-to-frame size ratio to process
            model_name = "deepclip_b0", # deepclip_b0, deepclip_b5
            )

# Fake probability only
prob = predictor.predict_img(
            img_path = "path/to/image.jpg",
            tta_hflip = 0.0 # Horizontal Flip for Test-Time Augmentation
            )
print(f"Deepfake Probability: {prob:.4f}")

# Fake probability + artifact scores
result = predictor.predict_img_artifacts(img_path = "path/to/image.jpg")
print(result["prob"])                       # fake probability
print(result["artifacts"]["mouth"]["blend"]) # blending artifacts on the mouth
```

#### Predict DeepFake Video

```python
from deepclip.detect import ClipVideoPredictor

# Initialize the predictor
predictor = ClipVideoPredictor(
            margin_ratio = 0.2, # Margin ratio around the detected face crop
            conf_thres = 0.5, # Confidence threshold for face detection
            min_face_ratio = 0.01, # Minimum face-to-frame size ratio to process
            model_name = "deepclip_b0", # deepclip_b0, deepclip_b5
            )

# Fake probability only
prob = predictor.predict_video(
            video_path = "path/to/video.mp4",
            num_frames = 20, # Number of frames to sample per video
            agg_mode = "conf", # Aggregation Method: 'conf', 'mean', 'vote'
            tta_hflip = 0.0 # Horizontal Flip for Test-Time Augmentation
            )
print(f"Deepfake Probability: {prob:.4f}")

# Fake probability + artifact scores (mean over the frames)
result = predictor.predict_video_artifacts(video_path = "path/to/video.mp4", num_frames = 20)
```

**Output of `predict_*_artifacts`**

```python
{
    "prob": 0.9634,               # fake probability (aggregated with agg_mode for videos)
    "artifacts": {                # None when no face is found (prob is 0.5)
        "face":  {"blur": 0.12, "color": 0.08, "texture": 0.21, "structure": 0.05, "blend": 0.64},
        "eyes":  {...},
        "nose":  {...},
        "mouth": {...},
    },
}
```

| `agg_mode` | Frame Aggregation |
| ---------- | ----------------- |
| `conf` | Mean of the confident frames (> 0.8 or < 0.2) when enough frames agree, otherwise mean |
| `mean` | Mean of all frames |
| `vote` | Ratio of frames with probability > 0.5 |

## 🚀 Training

DeepClip is trained on the frame-level FaceForensics++ dataset with its forgery annotation.
We recommend **Weights & Biases (W&B)** for experiment tracking (`--wandb-api-key` is required).

```python
!python train_deepclip.py \
    --root-dir DATA_ROOT \
    --model-name "deepclip_b0" \ # deepclip_b0, deepclip_b5
    --output-dir "checkpoints" \ # Directory of the best checkpoint
    --seed 2025 \ # for reproducibility
    --wandb-api-key "your-api-key" # Write your own api key
```

`DATA_ROOT` must contain the frame metadata and the face crops:

```Plaintext
DATA_ROOT/
├── crops/
│   └── {vid}/
│       ├── 12.png
│       └── ...
└── train_clip_frame_metadata.csv   # vid, frame_idx, label (REAL/FAKE), ori_vid, pair_label
```

`pair_label` is a string of 20 `0`/`1` characters, one per (region, type) pair in the order
`face-blur, face-color, ..., mouth-blend` (index = 5 × region + type).

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--root-dir` | (required) | Root directory of the dataset |
| `--model-name` | `deepclip_b0` | `deepclip_b0` or `deepclip_b5` |
| `--config` | `deepclip/config/deepclip_{b0,b5}.yaml` | Training recipe (model, text sampling, loss, optimizer) |
| `--output-dir` | `.` | Directory of the best checkpoint |
| `--seed` | `2025` | Random seed |
| `--num-workers` | `cpu_count - 1` | DataLoader workers |
| `--debug` | off | Train on a small subset |
| `--wandb-api-key` | (required) | W&B API key |
| `--metadata` | `train_clip_frame_metadata.csv` | Metadata file under `--root-dir` |
| `--crop-dir` | `crops` | Face crop directory under `--root-dir` |

## 📈 Model Evaluation

`test_deepclip.py` evaluates a pretrained DeepClip model on deepfake videos and prints AUC, accuracy, log loss,
macro F1, the confusion matrix and per-class precision / recall / F1.

```python
!python test_deepclip.py \
    --root-dir DATA_ROOT \
    --dataset "celeb_df_v2" \ # celeb_df_v2, ff++, kodf
    --model-name "deepclip_b0" \ # deepclip_b0, deepclip_b5
    --margin-ratio 0.2 \
    --conf-thres 0.5 \
    --min-face-ratio 0.01 \
    --num-frames 20 \
    --agg-mode "conf" \ # conf, mean, vote
    --tta-hflip 0.0 \
    --top-k 3 # Number of top artifact scores used by the rule-based fake probability (1-20)
```

The final fake probability of a video is rule-based: `max(prob, S_art)`, where `S_art` is the sigmoid of the mean
of the top-k artifact logits. Artifacts can only raise the fake probability, never lower it.

`DATA_ROOT` must contain the test metadata of each dataset (`label` is `REAL` / `FAKE`):

```Plaintext
DATA_ROOT/
├── Celeb_DF(v2)/
│   ├── test_metadata.csv          # vid, source, label
│   └── {source}/{vid}.mp4
├── FF++/
│   ├── test_metadata.csv
│   └── {source}/{vid}.mp4
└── kodf/
    ├── test_metadata.csv
    └── {source}/{pid}/{vid}.mp4
```

## ⚖️ License

This project is licensed under the terms of the MIT license.
