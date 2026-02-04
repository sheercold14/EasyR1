#!/bin/bash
# OminiMedExpert ISIC (mcq_letter + B1–B7) training launcher (task-aware version).
#
# Usage:
#   bash EasyR1/Comparative-R1/train_omnimed_isic_taskaware.sh
#
# Overrides:
#   MODEL_PATH=/path/to/Qwen2.5-VL-7B NUM_GPUS=4 bash EasyR1/Comparative-R1/train_omnimed_isic_taskaware.sh

set -e

EASYR1_ROOT="/mnt/cache/wuruixiao/users/lsc/EasyR1"
REPO_ROOT="/mnt/cache/wuruixiao/users/lsc/"

# ===== Paths =====
MODEL_PATH=${MODEL_PATH:-"/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b"}
DATA_DIR=${DATA_DIR:-"${EASYR1_ROOT}/data/OminiMedExpert/isic_disease_diagnosis_v0_0.05"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/train_mix_fewshot0.05_plus_btasks_relpath.jsonl"}
VAL_FILE=${VAL_FILE:-"${DATA_DIR}/val.jsonl"}
OMNI_ROOT=${OMNI_ROOT:-"${REPO_ROOT}/data/OmniMedVQA"}

CONFIG=${CONFIG:-"${EASYR1_ROOT}/Comparative-R1/configs/omnimed_isic_gspo_taskaware.yaml"}
FORMAT_PROMPT=${FORMAT_PROMPT:-"${EASYR1_ROOT}/Comparative-R1/prompts/omnimed_isic.jinja"}
REWARD_FUNCTION=${REWARD_FUNCTION:-"${EASYR1_ROOT}/Comparative-R1/reward/omnimed_isic_reward_v2.py:compute_score"}

# ===== Training params =====
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-32}
N_SAMPLES=${N_SAMPLES:-4}
TEMPERATURE=${TEMPERATURE:-0.7}
SAVE_FREQ=${SAVE_FREQ:-5}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-"omnimed_isic_btasks_n${N_SAMPLES}_t${TEMPERATURE}_taskaware"}

echo "=========================================="
echo "OminiMedExpert ISIC Training (Task-Aware)"
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

