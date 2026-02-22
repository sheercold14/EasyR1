#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
OUT_DIR="${PROJECT_ROOT}/data/offline_rft/isic/16shot"
SPLIT_SCRIPT="${SCRIPT_DIR}/split_isic_task_types.py"

bash "${SCRIPT_DIR}/generate_isic_16shot_all.sh"

FILES=(
  "${OUT_DIR}/16shot_nothinking.jsonl"
  "${OUT_DIR}/test_16shot_nothinking.jsonl"
  "${OUT_DIR}/MCQ_16shot_nothinking.jsonl"
  "${OUT_DIR}/MCQ_test_16shot_nothinking.jsonl"
  "${OUT_DIR}/MCQ_16shot_nothinking.full_labels.jsonl"
  "${OUT_DIR}/MCQ_test_16shot_nothinking.full_labels.jsonl"
  "${OUT_DIR}/MCQ_16shot_nothinking.full_labels_text.jsonl"
  "${OUT_DIR}/MCQ_test_16shot_nothinking.full_labels_text.jsonl"
  "${OUT_DIR}/MCQ_16shot_nothinking.4labels_text.jsonl"
  "${OUT_DIR}/MCQ_test_16shot_nothinking.4labels_text.jsonl"
)

for file in "${FILES[@]}"; do
  python3 "${SPLIT_SCRIPT}" \
    --input "${file}" \
    --out-binary "${file%.jsonl}.binary.jsonl" \
    --out-8class "${file%.jsonl}.8class.jsonl" \
    --strict
done

