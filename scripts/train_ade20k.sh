#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/segmentation/ade20k_small.yaml}"
DATA_PATH="${2:-/path/to/ADEChallengeData2016}"
PRETRAINED="${3:-outputs/imagenet/egcvmamba_small/best.pth}"
OUTPUT="${4:-outputs/ade20k/egcvmamba_small}"

torchrun --standalone --nproc_per_node=4 tools/train_segmentation.py \
  --config "${CONFIG}" \
  --data-path "${DATA_PATH}" \
  --pretrained "${PRETRAINED}" \
  --output "${OUTPUT}"
