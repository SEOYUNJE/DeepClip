#!/usr/bin/env bash
# Runs the FF++ preprocessing steps in order.
#
#   STEPS        steps to run, comma separated (default: 1,2,3,4,5)
#   DATA_ROOT    FF++ root directory (default: /data)
#   NUM_WORKERS  worker processes of raw_annotation (default: nproc)
#   *_ARGS       arguments of each step, replacing the defaults below (e.g. RAW_ANNOTATION_ARGS="--num-workers 8 --no-tar")
#
set -euo pipefail

STEPS="${STEPS:-1,2,3,4,5}"
DATA_ROOT="${DATA_ROOT:-/data}"
NUM_WORKERS="${NUM_WORKERS:-$(nproc)}"

SPLIT_DATA_ARGS="${SPLIT_DATA_ARGS:-}"
DETECT_FACE_ARGS="${DETECT_FACE_ARGS:---num-frames 20 --conf-thres 0.5 --min-face-ratio 0.01 --jitter 5}"
CROP_FACE_ARGS="${CROP_FACE_ARGS:---margin-ratio 0.2 --margin-jitter 0.0}"
LANDMARK_ARGS="${LANDMARK_ARGS:-}"
RAW_ANNOTATION_ARGS="${RAW_ANNOTATION_ARGS:---num-workers ${NUM_WORKERS}}"

cd "$(dirname "$0")/.."

run_step() {
    local step="$1" title="$2"
    shift 2

    if [[ ",${STEPS}," != *",${step},"* ]]; then
        return
    fi

    echo ">>> [${step}/5] ${title}"
    python -m "$@"
}

# shellcheck disable=SC2086
run_step 1 "Split data"          preprocess.split_data           --root-dir "${DATA_ROOT}" ${SPLIT_DATA_ARGS}
# shellcheck disable=SC2086
run_step 2 "Detect original face" preprocess.detect_original_face --root-dir "${DATA_ROOT}" ${DETECT_FACE_ARGS}
# shellcheck disable=SC2086
run_step 3 "Crop face"           preprocess.crop_face            --root-dir "${DATA_ROOT}" ${CROP_FACE_ARGS}
# shellcheck disable=SC2086
run_step 4 "Generate landmark"   preprocess.generate_landmark    --root-dir "${DATA_ROOT}" ${LANDMARK_ARGS}
# shellcheck disable=SC2086
run_step 5 "Raw annotation"      preprocess.raw_annotation       --root-dir "${DATA_ROOT}" ${RAW_ANNOTATION_ARGS}

echo ">>> Done"
