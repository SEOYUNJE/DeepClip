import argparse
import shutil
import tarfile
import time
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from pathlib import Path
from colorama import Fore, Style
from .landmark_detector import LandmarkDetector

c_ = Fore.BLUE
g_ = Fore.GREEN
r_ = Fore.RED
y_ = Fore.YELLOW
s_ = Style.BRIGHT
rs_ = Style.RESET_ALL

def process_landmarks(
                      meta_df: pd.DataFrame,
                      root_dir: str,
                      landmark_detector: LandmarkDetector,
):
    """
    Detects 68 facial landmarks for every FAKE crop and saves them as .npy files.

    REAL crops are not processed, because the landmarks are only used to build region masks
    of the FAKE frames. The returned metadata keeps every REAL row and only the FAKE rows
    whose landmarks were detected.

    Args:
        meta_df: train_frame_metadata.csv (vid, ori_vid, source, frame_idx, label)
    Returns:
        stats, filtered metadata (same columns and row order as meta_df)
    """

    root_path = Path(root_dir)
    crops_dir = root_path / "crops"
    landmarks_dir = root_path / "landmarks_68"
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    fake_df = meta_df[meta_df['label'] == 'FAKE']

    stats = {
        "total_fake_frames": len(fake_df),
        "landmarks_found": 0,
        "landmarks_failed": 0,
        "missing_crops": 0,
    }

    valid_index = []

    groups = fake_df.groupby('vid', sort=False)

    for vid, group in tqdm(groups, total=groups.ngroups, desc="Generate Landmark"):
        save_dir = landmarks_dir / vid
        save_dir.mkdir(parents=True, exist_ok=True)

        batch_frames = []
        batch_rows = []

        for row_idx, frame_idx in zip(group.index, group['frame_idx']):
            save_path = save_dir / f"{frame_idx}.npy"

            # Reuse landmarks that were already generated
            if save_path.exists():
                valid_index.append(row_idx)
                stats["landmarks_found"] += 1
                continue

            crop_bgr = cv2.imread(str(crops_dir / vid / f"{frame_idx}.png"))

            if crop_bgr is None:
                stats["missing_crops"] += 1
                continue

            batch_frames.append(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            batch_rows.append((row_idx, save_path))

        results = landmark_detector.detect_batch(batch_frames) if batch_frames else []

        for (row_idx, save_path), result in zip(batch_rows, results):
            if len(result) == 0:
                stats["landmarks_failed"] += 1
                continue

            np.save(save_path, result)
            valid_index.append(row_idx)
            stats["landmarks_found"] += 1

        # Remove directories without any landmark
        if not any(save_dir.iterdir()):
            shutil.rmtree(save_dir)

    keep_mask = (meta_df['label'] == 'REAL') | meta_df.index.isin(valid_index)

    return stats, meta_df[keep_mask].reset_index(drop=True)

def make_tar(root_dir: str):
    root_path = Path(root_dir)
    tar_path = root_path / "landmarks_68.tar"

    with tarfile.open(tar_path, "w") as tar:
        tar.add(root_path / "landmarks_68", arcname="landmarks_68")

    return tar_path

def main():
    parser = argparse.ArgumentParser(description="Generate 68 Facial Landmarks for FAKE crops")
    parser.add_argument("--root-dir", required=True, help="Root Directory with crops/ and train_frame_metadata.csv")
    parser.add_argument("--predictor-path", default=None, help="Path of shape_predictor_68_face_landmarks.dat (downloaded if missing)")
    parser.add_argument("--use-full-crop-rect", action="store_true", help="Use the whole crop as the face rectangle instead of dlib's face detector")
    parser.add_argument("--no-tar", action="store_true", help="Do not create landmarks_68.tar")

    args = parser.parse_args()

    root_path = Path(args.root_dir)
    csv_path = root_path / "train_frame_metadata.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {csv_path}")

    start_time = time.time()

    print(f"\n{c_}{s_}>>> Step 1: Loading metadata...{rs_}")
    meta_df = pd.read_csv(csv_path, dtype={'vid': str, 'ori_vid': str})

    print(f"{c_}{s_}>>> Step 2: Initializing Landmark Detector...{rs_}")
    landmark_detector = LandmarkDetector(args.predictor_path, args.use_full_crop_rect)

    print(f"{c_}{s_}>>> Step 3: Starting Process...{rs_}\n")
    stats, clip_meta_df = process_landmarks(meta_df, args.root_dir, landmark_detector)

    save_path = root_path / "train_clip_frame_metadata.csv"
    clip_meta_df.to_csv(save_path, index=False)

    if not args.no_tar:
        print(f"\n{c_}{s_}>>> Step 4: Making landmarks_68.tar...{rs_}")
        tar_path = make_tar(args.root_dir)

    total_time = time.time() - start_time

    print(f"\n{g_}{s_}{'='*50}{rs_}")
    print(f"{g_}{s_}Processing Summary Report{rs_}")
    print(f"{g_}{s_}{'='*50}{rs_}")
    print(f" - Total FAKE Frames      : {stats['total_fake_frames']}")
    print(f" - Landmarks Found        : {stats['landmarks_found']}")
    print(f" - Landmarks Failed       : {r_}{stats['landmarks_failed']}{rs_}")
    print(f" - Missing Crops          : {r_}{stats['missing_crops']}{rs_}")
    print(f" - Total Time Taken       : {y_}{total_time:.2f}s{rs_}")

    fake_all = meta_df[meta_df['label'] == 'FAKE'].groupby('source').size()
    fake_kept = clip_meta_df[clip_meta_df['label'] == 'FAKE'].groupby('source').size()

    print("\nLandmark success rate by source (FAKE):")
    for source, total in fake_all.items():
        kept = int(fake_kept.get(source, 0))
        print(f"   {source:<16}: {kept}/{total} ({kept / total * 100:.2f}%)")

    print(f"\n{y_}Metadata CSV saved to: {save_path} "
          f"(REAL {int((clip_meta_df['label'] == 'REAL').sum())} + "
          f"FAKE {int((clip_meta_df['label'] == 'FAKE').sum())}){rs_}")

    if not args.no_tar:
        print(f"{y_}Tar saved to: {tar_path}{rs_}")

    print(f"{g_}{s_}{'='*50}{rs_}\n")

if __name__ == "__main__":
    main()
