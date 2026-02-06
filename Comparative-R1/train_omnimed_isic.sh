#!/bin/bash
# OminiMedExpert ISIC (mcq_letter + B1–B7) training launcher.
#
# Usage (recommended from repo root):
#   bash EasyR1/Comparative-R1/train_omnimed_isic.sh
#
# Override examples:
#   MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct NUM_GPUS=2 bash EasyR1/Comparative-R1/train_omnimed_isic.sh

set -e

# export NCCL_CUMEM_HOST_ENABLE=0

# ===== Locate roots =====
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYR1_ROOT="/mnt/cache/wuruixiao/users/lsc/EasyR1"
REPO_ROOT="/mnt/cache/wuruixiao/users/lsc/"
# ===== Paths =====
MODEL_PATH=${MODEL_PATH:-"/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b"}
DATA_DIR=${DATA_DIR:-"${EASYR1_ROOT}/data/OminiMedExpert/isic_disease_diagnosis_v1_0.05/comparative"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/train_b1_tasks_800.jsonl"}
VAL_FILE=${VAL_FILE:-"${DATA_DIR}/test_b1_tasks_200.jsonl"}
OMNI_ROOT=${OMNI_ROOT:-"${REPO_ROOT}/data/OmniMedVQA"}

CONFIG=${CONFIG:-"${EASYR1_ROOT}/Comparative-R1/configs/omnimed_isic_gspo.yaml"}
FORMAT_PROMPT=${FORMAT_PROMPT:-"${EASYR1_ROOT}/Comparative-R1/prompts/omnimed_isic.jinja"}
REWARD_FUNCTION=${REWARD_FUNCTION:-"${EASYR1_ROOT}/Comparative-R1/reward/omnimed_isic_reward.py:compute_score"}

# ===== Training params =====
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-32}
N_SAMPLES=${N_SAMPLES:-4}
TEMPERATURE=${TEMPERATURE:-0.7}
SAVE_FREQ=${SAVE_FREQ:-5}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-"omnimed_isic_v1_b1_800_n${N_SAMPLES}_t${TEMPERATURE}"}

echo "=========================================="
echo "OminiMedExpert ISIC Training (GSPO/GRPO)"
echo "=========================================="
echo "Config:      ${CONFIG}"
echo "Model:       ${MODEL_PATH}"
echo "Train file:  ${TRAIN_FILE}"
echo "Val file:    ${VAL_FILE}"
echo "Image root:  ${OMNI_ROOT}"
echo "GPUs:        ${NUM_GPUS}"
echo "Batch size:  ${BATCH_SIZE}"
echo "N samples:   ${N_SAMPLES}"
echo "Temperature: ${TEMPERATURE}"
echo "Experiment:  ${EXPERIMENT_NAME}"
echo ""

cd "${EASYR1_ROOT}"
python3 -m verl.trainer.main \
  config="${CONFIG}" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.image_dir="${OMNI_ROOT}" \
  data.format_prompt="${FORMAT_PROMPT}" \
  worker.reward.reward_function="${REWARD_FUNCTION}" \
  data.rollout_batch_size="${BATCH_SIZE}" \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.rollout.n="${N_SAMPLES}" \
  worker.rollout.temperature="${TEMPERATURE}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.save_freq="${SAVE_FREQ}"
