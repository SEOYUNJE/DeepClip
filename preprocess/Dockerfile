# FF++ preprocessing pipeline of DeepClip (split -> detect -> crop -> landmark -> raw annotation)
#
# Build from the repository root (the scripts run as `python -m preprocess.xxx`):
#   docker build -f preprocess/Dockerfile -t deepclip-preprocess:gpu .
#   docker build -f preprocess/Dockerfile --build-arg BASE_IMAGE=python:3.11-slim -t deepclip-preprocess:cpu .
#
# The weights (yolov8n-face.pt, shape_predictor_68_face_landmarks.dat) are not in the image: they are downloaded to
# /root/.cache/deepclip on the first run, so mount a volume there to keep them.

# CUDA runtime for the YOLO face detection. Use a CPU image with --build-arg BASE_IMAGE=python:3.11-slim
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# build-essential, cmake: dlib is built from source
# libgl1, libglib2.0-0: OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY preprocess/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY preprocess/ ./preprocess/
RUN chmod +x preprocess/run_pipeline.sh

ENTRYPOINT ["/app/preprocess/run_pipeline.sh"]
