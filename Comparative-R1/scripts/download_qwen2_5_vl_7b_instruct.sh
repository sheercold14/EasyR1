#!/bin/bash
# set -euo pipefail

# Download Qwen/Qwen2.5-VL-7B-Instruct to a local folder using huggingface_hub.
#
# Usage:
#   OUT_DIR=/mnt/cache/wuruixiao/users/lsc/models/Qwen2.5-VL-7B-Instruct \
#     bash EasyR1/Comparative-R1/scripts/download_qwen2_5_vl_7b_instruct.sh
#
# If you need auth:
#   export HF_TOKEN=...

REPO_ID="${REPO_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUT_DIR="${OUT_DIR:-/tmp/shared-storage/cache/}"
REVISION="${REVISION:-}"

ARGS=(--repo_id "$REPO_ID" --out_dir "$OUT_DIR")
if [[ -n "${REVISION}" ]]; then
  ARGS+=(--revision "$REVISION")
fi

python3 ./download_qwen2_5_vl_7b_instruct.py "${ARGS[@]}"

