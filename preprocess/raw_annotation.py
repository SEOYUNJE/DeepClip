import argparse
import tarfile
import time
import numpy as np
import pandas as pd
import cv2
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
from colorama import Fore, Style
from .forgery_features import REGIONS, FEATURE_FIELDS, compute_forgery_mask, analyze_frame
from .utils import get_frame_pairs

c_ = Fore.BLUE
g_ = Fore.GREEN
r_ = Fore.RED
y_ = Fore.YELLOW
s_ = Style.BRIGHT
rs_ = Style.RESET_ALL

ID_COLUMNS = ["vid", "ori_vid", "frame_idx", "source", "status"]
COLUMNS = (
    ID_COLUMNS
    + [f"{region}_forged" for region in REGIONS]
    + [f"{region}_{field}" for region in REGIONS for field in FEATURE_FIELDS]
)
BOOL_COLUMNS = [col for col in COLUMNS if col.endswith("_forged") or col.endswith("_flag")]
FLOAT_COLUMNS = [col for col in COLUMNS if col not in ID_COLUMNS and col not in BOOL_COLUMNS]

def process_frame(row: dict, root_path: Path, save_masks: bool) -> dict:
    """
    Analyzes one FAKE frame and returns one row of raw_annotation.parquet.

    status:
        ok: at least one region is forged
        no_forged_region: no region is forged
        no_landmark: crop exists, but its landmark file does not
        no_pair: fake or real crop image does not exist
        error: unexpected error while analyzing (the message is printed)
    """

    result = {col: None for col in COLUMNS}
    result.update({
        "vid": row["vid"],
        "ori_vid": row["ori_vid"],
        "frame_idx": int(row["frame_idx"]),
        "source": row["source"],
    })

    fake_image = cv2.imread(str(row["fake_path"]))
    real_image = cv2.imread(str(row["real_path"]))

    if fake_image is None or real_image is None:
        result["status"] = "no_pair"
        return result

    landmark_path = root_path / "landmarks_68" / row["vid"] / f"{row['frame_idx']}.npy"

    if not landmark_path.exists():
        result["status"] = "no_landmark"
        return result

    try:
        landmarks = np.load(landmark_path)

        if real_image.shape != fake_image.shape:
            real_image = cv2.resize(real_image, (fake_image.shape[1], fake_image.shape[0]))

        forgery_mask = compute_forgery_mask(real_image, fake_image)

        if save_masks:
            mask_dir = root_path / "forgery_masks" / row["vid"]
            mask_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_dir / f"{row['frame_idx']}.png"), forgery_mask)

        analysis = analyze_frame(real_image, fake_image, landmarks, forgery_mask)
    except Exception as e:
        print(f"[error] {row['vid']}/{row['frame_idx']}: {type(e).__name__}: {e}")
        result["status"] = "error"
        return result

    for region, features in analysis.items():
        result[f"{region}_forged"] = features is not None

        if features is not None:
            for field, value in features.items():
                result[f"{region}_{field}"] = value

    result["status"] = "ok" if any(features is not None for features in analysis.values()) else "no_forged_region"

    return result

def to_dataframe(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)

    df = df.astype({col: "boolean" for col in BOOL_COLUMNS})
    df = df.astype({col: "float32" for col in FLOAT_COLUMNS})
    df = df.astype({"frame_idx": "int32"})

    return df

def save_parquet(rows: list, previous_df: pd.DataFrame, output_path: Path):
    df = to_dataframe(rows)

    if previous_df is not None:
        df = pd.concat([previous_df, df], ignore_index=True)

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)

def extract_landmarks_if_needed(root_path: Path):
    """raw_annotation reads landmarks_68/{vid}/{frame_idx}.npy, so landmarks_68.tar is extracted if the folder does not exist"""

    if (root_path / "landmarks_68").exists():
        return

    tar_path = root_path / "landmarks_68.tar"

    if not tar_path.exists():
        raise FileNotFoundError(f"Neither landmarks_68/ nor landmarks_68.tar found in: {root_path}")

    print(f"{c_}{s_}>>> Extracting {tar_path}...{rs_}")
    with tarfile.open(tar_path) as tar:
        # "data" filter is only available in recent Python versions
        if hasattr(tarfile, "data_filter"):
            tar.extractall(root_path, filter="data")
        else:
            tar.extractall(root_path)

def make_tar(root_path: Path):
    tar_path = root_path / "forgery_masks.tar"

    with tarfile.open(tar_path, "w") as tar:
        tar.add(root_path / "forgery_masks", arcname="forgery_masks")

    return tar_path

def main():
    parser = argparse.ArgumentParser(description="VLFFD Raw Annotation (forgery mask and forgery features of FAKE frames)")
    parser.add_argument("--root-dir", required=True, help="Root Directory with crops/, landmarks_68/ and train_clip_frame_metadata.csv")
    parser.add_argument("--output", default="raw_annotation.parquet", help="Output file name (saved in root-dir)")
    parser.add_argument("--limit-per-source", default=0, type=int, help="Only process the first N frames of each source (pilot run). 0 means all")
    parser.add_argument("--num-workers", default=1, type=int, help="Number of worker processes")
    parser.add_argument("--flush-every", default=2000, type=int, help="Save the parquet file after every N frames")
    parser.add_argument("--no-masks", action="store_true", help="Do not save forgery masks")
    parser.add_argument("--no-tar", action="store_true", help="Do not create forgery_masks.tar")

    args = parser.parse_args()

    root_path = Path(args.root_dir)
    output_path = root_path / args.output

    start_time = time.time()

    print(f"\n{c_}{s_}>>> Step 1: Loading FAKE frame pairs...{rs_}")
    pairs_df = get_frame_pairs(args.root_dir)

    if args.limit_per_source > 0:
        pairs_df = pairs_df.groupby('source').head(args.limit_per_source).reset_index(drop=True)

    extract_landmarks_if_needed(root_path)

    previous_df = None

    if output_path.exists():
        previous_df = pd.read_parquet(output_path)
        done = set(zip(previous_df['vid'], previous_df['frame_idx']))

        pairs_df = pairs_df[[(vid, idx) not in done for vid, idx in zip(pairs_df['vid'], pairs_df['frame_idx'])]]
        print(f"Resuming: {len(previous_df)} frames are already processed")

    rows_to_process = pairs_df.to_dict('records')

    print(f"{c_}{s_}>>> Step 2: Analyzing {len(rows_to_process)} frames...{rs_}\n")
    worker = partial(process_frame, root_path=root_path, save_masks=not args.no_masks)

    new_rows = []
    pending = []

    def flush():
        nonlocal previous_df, pending

        if pending:
            save_parquet(pending, previous_df, output_path)
            previous_df = pd.read_parquet(output_path)
            new_rows.extend(pending)
            pending = []

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            results = pool.imap(worker, rows_to_process, chunksize=16)

            for result in tqdm(results, total=len(rows_to_process), desc="Raw Annotation"):
                pending.append(result)

                if len(pending) >= args.flush_every:
                    flush()
    else:
        for row in tqdm(rows_to_process, desc="Raw Annotation"):
            pending.append(worker(row))

            if len(pending) >= args.flush_every:
                flush()

    flush()

    if not args.no_masks and not args.no_tar and (root_path / "forgery_masks").exists():
        print(f"\n{c_}{s_}>>> Step 3: Making forgery_masks.tar...{rs_}")
        tar_path = make_tar(root_path)
    else:
        tar_path = None

    total_time = time.time() - start_time

    if previous_df is None:
        print(f"\n{r_}No frames were processed.{rs_}")
        return

    df = previous_df

    print(f"\n{g_}{s_}{'='*50}{rs_}")
    print(f"{g_}{s_}Processing Summary Report{rs_}")
    print(f"{g_}{s_}{'='*50}{rs_}")
    print(f" - Total Frames           : {len(df)}")
    print(f" - Newly Processed        : {len(new_rows)}")
    print(f" - Total Time Taken       : {y_}{total_time:.2f}s{rs_}")

    print("\nStatus by source:")
    print(pd.crosstab(df['source'], df['status']).to_string())

    print("\nForged regions (ratio of frames with status ok or no_forged_region):")
    analyzed = df[df['status'].isin(['ok', 'no_forged_region'])]

    for region in REGIONS:
        forged = int(analyzed[f"{region}_forged"].sum())
        print(f"   {region:<6}: {forged}/{len(analyzed)}")

    print(f"\n{y_}Parquet saved to: {output_path}{rs_}")

    if tar_path is not None:
        print(f"{y_}Tar saved to: {tar_path}{rs_}")

    print(f"{g_}{s_}{'='*50}{rs_}\n")

if __name__ == "__main__":
    main()
