#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/classification/egcvmamba_small.yaml}"
DATA_PATH="${2:-/path/to/imagenet}"
OUTPUT="${3:-outputs/imagenet/egcvmamba_small}"

torchrun --standalone --nproc_per_node=4 tools/train_classification.py \
  --config "${CONFIG}" \
  --data-path "${DATA_PATH}" \
  --output "${OUTPUT}"
