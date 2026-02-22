#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

CONVERT_SCRIPT="${PROJECT_ROOT}/Comparative-R1/scripts/ReMAP/convert_fewshot_dtd_to_offline_rft.py"
EXPERT_V2_SCRIPT="${PROJECT_ROOT}/Comparative-R1/scripts/ominimed_expertv2.py"
MCQ_DIR="${PROJECT_ROOT}/Comparative-R1/scripts/MCQ"
OUT_DIR="${PROJECT_ROOT}/data/offline_rft/isic/16shot"
QA_OPEN_ACCESS_DIR="${QA_OPEN_ACCESS_DIR:-/mnt/cache/wuruixiao/users/lsc/data/OmniMedVQA/QA_information/Open-access}"
OMNI_ROOT="${OMNI_ROOT:-/mnt/cache/wuruixiao/users/lsc/data/OmniMedVQA}"
SEED="${SEED:-42}"
DTD_TMP_DIR="${OUT_DIR}/source_from_expertv2"

mkdir -p "${OUT_DIR}"
mkdir -p "${DTD_TMP_DIR}"

python3 "${EXPERT_V2_SCRIPT}" \
  --omni_root "${OMNI_ROOT}" \
  build-fewshot-dtd \
  --dataset-regex "ISIC" \
  --question-type "Disease Diagnosis" \
  --min-option-count 2 \
  --max-option-count 4 \
  --shots 16 \
  --candidate-source original_options \
  --seed "${SEED}" \
  --skip-missing-images \
  --out-dir "${DTD_TMP_DIR}" \
  --out-stem "ISIC"

python3 "${CONVERT_SCRIPT}" \
  --input "${DTD_TMP_DIR}/ISIC_fewshot_16.jsonl" \
  --output "${OUT_DIR}/16shot_nothinking.jsonl"

python3 "${CONVERT_SCRIPT}" \
  --input "${DTD_TMP_DIR}/ISIC_fewshot_test.jsonl" \
  --output "${OUT_DIR}/test_16shot_nothinking.jsonl"

python3 "${MCQ_DIR}/build_isic_mcq_from_4shot.py" \
  --input "${OUT_DIR}/16shot_nothinking.jsonl" \
  --qa-open-access-dir "${QA_OPEN_ACCESS_DIR}" \
  --output "${OUT_DIR}/MCQ_16shot_nothinking.jsonl" \
  --strict

python3 "${MCQ_DIR}/build_isic_mcq_from_4shot.py" \
  --input "${OUT_DIR}/test_16shot_nothinking.jsonl" \
  --qa-open-access-dir "${QA_OPEN_ACCESS_DIR}" \
  --output "${OUT_DIR}/MCQ_test_16shot_nothinking.jsonl" \
  --strict

python3 "${MCQ_DIR}/build_isic_mcq_from_4shot_full_labels.py" \
  --input "${OUT_DIR}/16shot_nothinking.jsonl" \
  --qa-open-access-dir "${QA_OPEN_ACCESS_DIR}" \
  --output "${OUT_DIR}/MCQ_16shot_nothinking.full_labels.jsonl" \
  --strict

python3 "${MCQ_DIR}/build_isic_mcq_from_4shot_full_labels.py" \
  --input "${OUT_DIR}/test_16shot_nothinking.jsonl" \
  --qa-open-access-dir "${QA_OPEN_ACCESS_DIR}" \
  --output "${OUT_DIR}/MCQ_test_16shot_nothinking.full_labels.jsonl" \
  --strict

python3 "${MCQ_DIR}/build_isic_mcq_from_4shot_full_labels_text.py" \
  --input "${OUT_DIR}/16shot_nothinking.jsonl" \
  --qa-open-access-dir "${QA_OPEN_ACCESS_DIR}" \
  --output "${OUT_DIR}/MCQ_16shot_nothinking.full_labels_text.jsonl" \
  --strict

python3 "${MCQ_DIR}/build_isic_mcq_from_4shot_full_labels_text.py" \
  --input "${OUT_DIR}/test_16shot_nothinking.jsonl" \
  --qa-open-access-dir "${QA_OPEN_ACCESS_DIR}" \
  --output "${OUT_DIR}/MCQ_test_16shot_nothinking.full_labels_text.jsonl" \
  --strict

python3 "${MCQ_DIR}/build_isic_mcq_from_4shot_4labels_text.py" \
  --input "${OUT_DIR}/16shot_nothinking.jsonl" \
  --qa-open-access-dir "${QA_OPEN_ACCESS_DIR}" \
  --output "${OUT_DIR}/MCQ_16shot_nothinking.4labels_text.jsonl" \
  --mcq-option-source "${OUT_DIR}/MCQ_16shot_nothinking.jsonl" \
  --strict

python3 "${MCQ_DIR}/build_isic_mcq_from_4shot_4labels_text.py" \
  --input "${OUT_DIR}/test_16shot_nothinking.jsonl" \
  --qa-open-access-dir "${QA_OPEN_ACCESS_DIR}" \
  --output "${OUT_DIR}/MCQ_test_16shot_nothinking.4labels_text.jsonl" \
  --mcq-option-source "${OUT_DIR}/MCQ_test_16shot_nothinking.jsonl" \
  --strict
