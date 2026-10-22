import argparse
import time
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from pathlib import Path
from typing import List
import json
from colorama import Fore, Style
from .frame_extractor import FrameExtractor
from .utils import get_video_paths

c_ = Fore.BLUE
g_ = Fore.GREEN
r_ = Fore.RED
y_ = Fore.YELLOW
s_ = Style.BRIGHT
rs_ = Style.RESET_ALL

label2id = {
    'Deepfakes': '0_',
    'Face2Face': '1_',
    'FaceShifter': '2_',
    'FaceSwap': '3_',
    'NeuralTextures': '4_'
}

def process_crops(
                  video_paths: List[Path],
                  ori_video_paths: List[Path],
                  root_dir: str,
                  margin_ratio: float,
                  margin_jitter: float,
):
    root_path = Path(root_dir)
    boxes_dir = root_path / "boxes"
    crops_dir = root_path / "crops"

    crops_dir.mkdir(parents=True, exist_ok=True)

    all_frame_meta = []

    stats = {
        "total_videos": len(video_paths),
        "processed_videos": 0,
        "skipped_videos": 0,
        "total_frames_saved": 0,
    }

    pbar = tqdm(zip(video_paths, ori_video_paths), total=len(video_paths), desc="Crop Face")

    for (video_path, ori_video_path) in pbar:
        video_id = video_path.stem
        source = video_path.parent.name
        ori_video_id = ori_video_path.stem
        label = "REAL" if video_id == ori_video_id else "FAKE"

        frame_extractor = FrameExtractor()
        json_path = boxes_dir / f"{ori_video_id}.json"

        if not json_path.exists():
            stats["skipped_videos"] += 1
            continue

        with open(json_path, "r") as f:
            ori_box_dict = json.load(f)

        ori_frame_indices = sorted([int(x) for x in list(ori_box_dict.keys())])
        frames_dict = frame_extractor.extract_frames_for_crop(str(video_path), ori_frame_indices)

        if len(frames_dict) == 0:
            stats["skipped_videos"] += 1
            continue

        new_video_id = label2id.get(str(source), '') + str(video_id)

        save_video_dir = crops_dir / new_video_id
        save_video_dir.mkdir(parents=True, exist_ok=True)

        for idx, frame in frames_dict.items():
            ori_box = ori_box_dict[str(idx)]
            xmin, ymin, xmax, ymax = ori_box

            current_margin = margin_ratio
            if margin_jitter > 0:
                jitter_scale = margin_ratio * margin_jitter
                noise = np.random.uniform(-jitter_scale, jitter_scale)
                current_margin = max(0.0, margin_ratio + noise)

            w = xmax - xmin
            h = ymax - ymin
            pad_w = int(w * current_margin)
            pad_h = int(h * current_margin)

            y1 = max(int(ymin - pad_h), 0)
            y2 = min(int(ymax + pad_h), frame.shape[0])
            x1 = max(int(xmin - pad_w), 0)
            x2 = min(int(xmax + pad_w), frame.shape[1])

            crop = frame[y1:y2, x1:x2]

            save_video_path = save_video_dir / f"{idx}.png"
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_video_path), crop_bgr)
            stats["total_frames_saved"] += 1

            all_frame_meta.append({
                'vid': str(new_video_id),
                'ori_vid': str(ori_video_id),
                'source': str(source),
                'frame_idx': int(idx),
                'label': label
            })

        stats["processed_videos"] += 1

    return stats, pd.DataFrame(all_frame_meta)

def main():
    parser = argparse.ArgumentParser(description="Crop Face")
    parser.add_argument("--root-dir", required=True, help="Root Directory with FaceForenscis++")
    parser.add_argument("--margin-ratio", default=0.2, type=float, help="Margin Ratio for Detected Face")
    parser.add_argument("--margin-jitter", default=0.0, type=float, help="noise for margin ratio")

    args = parser.parse_args()

    start_time = time.time()

    print(f"\n{c_}{s_}>>> Step 1: Scanning video paths...{rs_}")
    video_paths, ori_video_paths = get_video_paths(args.root_dir)

    print(f"{c_}{s_}>>> Step 2: Starting Process...{rs_}\n")
    summary, meta_df = process_crops(video_paths, ori_video_paths, args.root_dir, args.margin_ratio,
                            args.margin_jitter)

    total_time = time.time() - start_time

    print(f"\n{g_}{s_}{'='*50}{rs_}")
    print(f"{g_}{s_}Processing Summary Report{rs_}")
    print(f"{g_}{s_}{'='*50}{rs_}")
    print(f" - Total Videos Found     : {summary['total_videos']}")
    print(f" - Successfully Processed : {summary['processed_videos']}")
    print(f" - Skipped (No Box/Frame) : {r_}{summary['skipped_videos']}{rs_}")
    print(f" - Total Face Crops Saved : {summary['total_frames_saved']}")
    print(f" - Total Time Taken       : {y_}{total_time:.2f}s{rs_}")

    if summary['processed_videos'] > 0:
        avg_time = total_time / summary['processed_videos']
        print(f" - Average Time per Video : {avg_time:.2f}s")

    if not meta_df.empty:
        save_path = Path(args.root_dir) / "train_frame_metadata.csv"
        meta_df.to_csv(save_path, index=False)
        print(f"\n{y_}📂 Metadata CSV saved to: {save_path}{rs_}")
        print(f"\n📝 Sample Data (Top 3):\n{meta_df.head(3)}")

    print(f"{g_}{s_}{'='*50}{rs_}\n")

if __name__ == "__main__":
    main()
