# FaceForensics++ Preprocessing

Builds the frame-level dataset of FaceForensics++ (FF++) used by DeepClip and generates the
**raw annotation** of the [Face Forgery Text Generator (FFTG)](https://github.com/skJack/VLFFD) of
*Towards General Visual-Linguistic Face Forgery Detection* (CVPR 2025).

FF++ has 5 manipulation methods (`Deepfakes`, `Face2Face`, `FaceShifter`, `FaceSwap`, `NeuralTextures`) and the
`Original` videos. Only the **train** split (`train.json`) is processed.

## Pipeline

```
split_data.py            train_metadata.csv, test_metadata.csv
detect_original_face.py  boxes/{ori_vid}.json                 face boxes of the real videos (YOLO)
crop_face.py             crops/ + train_frame_metadata.csv    face crops of real and fake videos
generate_landmark.py     landmarks_68/ -> landmarks_68.tar    68 landmarks of the fake crops (dlib)
                         + train_clip_frame_metadata.csv
raw_annotation.py        forgery_masks/ -> forgery_masks.tar  forgery analysis of the fake crops
                         + raw_annotation.parquet
```

`split_data`, `detect_original_face` and `crop_face` come from the preprocessing of
[DeepGuard](https://github.com/HanMoonSub/DeepGuard) (without its YOLO landmarks, which DeepClip does not use).
Fake videos reuse the face boxes of their real video, so a fake crop and the real crop with the same
`frame_idx` are pixel-aligned.

## Requirements

```bash
pip install -r requirements.txt
```

Two weight files are **not stored in the repository**. They are downloaded automatically to
`~/.cache/deepclip/` the first time they are needed (no checksum is verified):

| File | Used by | Source |
| ---- | ------- | ------ |
| `yolov8n-face.pt` | `detect_original_face.py` | [v0.0.1 release](https://github.com/SEOYUNJE/DeepClip/releases/tag/v0.0.1), from [derronqi/yolov8-face](https://github.com/derronqi/yolov8-face) (GPL-3.0) |
| `shape_predictor_68_face_landmarks.dat` | `generate_landmark.py` | [v0.0.1 release](https://github.com/SEOYUNJE/DeepClip/releases/tag/v0.0.1), from [dlib](http://dlib.net/files/) |

> The dlib 68-point model was trained on the iBUG 300-W dataset, whose license may restrict commercial use.
> Check the terms before redistributing or using it commercially.

## Usage

`DATA_ROOT` is the directory with the FF++ videos (`original/`, `Deepfakes/`, ...) and `csv/FF++_Metadata.csv`.
On Colab, work on the local disk (`/content`) and copy the results to Drive. Writing and reading many small files
directly on Drive is slow.

### 1. Split data

```bash
python -m preprocess.split_data --root-dir DATA_ROOT
```

### 2. Detect original face

```bash
python -m preprocess.detect_original_face --root-dir DATA_ROOT --num-frames 20 --conf-thres 0.5 --min-face-ratio 0.01 --jitter 5
```

### 3. Crop face

```bash
python -m preprocess.crop_face --root-dir DATA_ROOT --margin-ratio 0.2 --margin-jitter 0.0
```

Output: `crops/{video_id}/{frame_idx}.png` and `train_frame_metadata.csv`
(`vid, ori_vid, source, frame_idx, label`), which records every saved crop. It is the input of the next step and
is not needed to train a model.

### 4. Generate landmark

```bash
python -m preprocess.generate_landmark --root-dir DATA_ROOT
```

Detects 68 landmarks with dlib on the **fake** crops only (the landmarks are only used to build the region masks
of the fake frames).

| Output | Description |
| ------ | ----------- |
| `landmarks_68/{vid}/{frame_idx}.npy` | `(68, 2)` landmarks of a fake crop |
| `landmarks_68.tar` | tar of `landmarks_68/` |
| `train_clip_frame_metadata.csv` | Same columns as `train_frame_metadata.csv`. **All REAL rows and the FAKE rows whose landmarks were detected** |

Already generated `.npy` files are reused when the script is run again. Use `--use-full-crop-rect` to use the whole
crop as the face rectangle instead of dlib's face detector, and `--no-tar` to skip the tar file.

### 5. Raw annotation

```bash
python -m preprocess.raw_annotation --root-dir DATA_ROOT --num-workers 2
```

If `landmarks_68/` does not exist, it is extracted from `landmarks_68.tar`.

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--output` | `raw_annotation.parquet` | Output file name (in `--root-dir`) |
| `--limit-per-source` | 0 | Only process the first N frames of each source (pilot run). 0 means all |
| `--num-workers` | 1 | Number of worker processes |
| `--flush-every` | 2000 | Save the parquet file after every N frames |
| `--no-masks` | | Do not save the forgery masks |
| `--no-tar` | | Do not create `forgery_masks.tar` |

Run a pilot first to check the results:

```bash
python -m preprocess.raw_annotation --root-dir DATA_ROOT --output pilot.parquet --limit-per-source 20 --no-masks --no-tar
```

The parquet file is saved while running, so a stopped run resumes from the frames which are not yet in the
parquet file. Skipped frames are not recorded, so they are analyzed again. The analysis took about 0.3 s (224 px) to 0.75 s (400 px) per frame on one CPU core in a synthetic benchmark
(SSIM is more than 90% of it), so use `--num-workers`.

| Output | Description |
| ------ | ----------- |
| `forgery_masks/{vid}/{frame_idx}.png` | 8-bit forgery mask of the fake crop (kept frames only) |
| `forgery_masks.tar` | tar of `forgery_masks/` |
| `raw_annotation.parquet` | One row per kept fake frame |
| `train_clip_frame_metadata.csv` | **Overwritten** with all REAL rows and the kept FAKE rows. Keep a copy of the original if it is needed |

Only the fake frames with `status` `ok` are kept: at least one region is forged **and** at least one forgery flag
(`blur`, `color`, `texture`, `structure`, `blend`) of the forged regions is on. The other frames are skipped
(`no_forged_region`, `no_flag`, `no_landmark`, `no_pair`, `error`) and get no forgery mask. REAL rows are always
kept. A pilot run (`--limit-per-source`) does not change `train_clip_frame_metadata.csv`.

## Raw annotation

For every fake frame, each facial region (`face`, `eyes`, `nose`, `mouth`) is judged as forged when the average of the
forgery mask inside of the region exceeds 55. The forgery mask is `1 - SSIM` of the real / fake pair. Five forgery
types are then measured in **every forged region** and stored with their scores and flags. Nothing is selected or
turned into text here: choosing which regions and types to describe is left to the text template.

| Column | Description |
| ------ | ----------- |
| `vid`, `ori_vid`, `frame_idx`, `source` | Identify the fake frame. The real frame is `crops/{ori_vid}/{frame_idx}.png` |
| `status` | `ok`: at least one forged region and one forgery flag (the only status saved), `no_forged_region`, `no_flag`: forged region(s) without any flag, `no_landmark`: the landmark file is missing, `no_pair`: a crop image is missing, `error`: unexpected error (message printed) |
| `{region}_forged` | Whether the region is forged |
| `{region}_blur_score`, `_blur_flag` | Laplacian variance of the real minus the fake image. Flag: score above the threshold |
| `{region}_color_score`, `_color_flag` | Mean of the differences of the mean and std in the Lab color space. Flag: above the threshold |
| `{region}_texture_score`, `_texture_flag` | Absolute difference of the GLCM contrast. Flag: above the threshold |
| `{region}_structure_score`, `_structure_flag` | SSIM of the real / fake pair. Flag: below the threshold |
| `{region}_blend_gradient`, `_blend_edge`, `_blend_freq` | Gradient, edge and frequency scores at the region boundary |
| `{region}_blend_flag` | At least two of the three blending scores exceed their thresholds |

The scores and flags are empty when the region is not forged or `status` is not `ok`.

### Thresholds

The thresholds are the same for all regions (constants in `forgery_features.py`). Only the forgery mask threshold (55) and
the blend gradient threshold (0.5) are the original values. The others are adjusted to the FF++ score distribution
because the original values turned the blur and color flags on for only 1 to 8% of the forged regions, the texture,
structure (`face`) and edge flags on for about 90% or more, and the frequency flag never.

| Type | Flag is on if | Original |
| ---- | ------------- | -------- |
| blur | score > 25 | 100 |
| color | score > 1.0 | 2.0 |
| texture | score > 2.8 | 0.7 |
| structure | score < 0.985 | 0.97 |
| blend | at least two of gradient > 0.5, edge > 4, frequency > 0.15 | 0.5, 0.4, 0.5 |

`structure` is an SSIM (a similarity, 1 means identical), so a lower score means a more deformed region and the flag uses `<`.
The other scores are differences, so their flags use `>`.

Known limits on FF++ (forged regions): the structure score depends on the area of the region, so a single threshold
cannot fit all regions. At 0.985 its flag is on for 100% of `face`, so a forged `face` always has a flag, and for 85% of
`eyes`, 48% of `nose` and 61% of `mouth`. The blending frequency scores are only about 0.13 to 0.17, so a threshold of
0.15 separates them by small differences. Use the stored scores if you need the magnitude.

### Differences from the original implementation

The analysis follows `raw_annotation.py` of [VLFFD](https://github.com/skJack/VLFFD) (images are BGR arrays passed as-is to
`RGB2LAB`/`RGB2GRAY`, features are computed on the whole image multiplied by the region mask), except for:

- The same region mask (from the landmarks of the **fake** crop) is applied to the real and fake images. The original
  builds the real region from the landmarks of the real image.
- Every forged region is analyzed, and scores are stored. The original only keeps the flags and describes the forged regions.
- The original reads existing forgery masks if they exist (FF++ mask files, which are not used here), otherwise it falls back
  to the SSIM-based mask used here.
- `multichannel=True` of scikit-image is replaced by `channel_axis=-1`, and a dilation of 0 iterations (which fills the
  whole image in scipy) is skipped.
- Pixels with a negative SSIM give a mask value above 255, which wraps around in `uint8`, as in the original.

## Citation

```
@inproceedings{roessler2019faceforensicspp,
    author = {Andreas R\"ossler and Davide Cozzolino and Luisa Verdoliva and Christian Riess and Justus Thies and Matthias Nie{\ss}ner},
    title = {Face{F}orensics++: Learning to Detect Manipulated Facial Images},
    booktitle= {International Conference on Computer Vision (ICCV)},
    year = {2019}
}
```
