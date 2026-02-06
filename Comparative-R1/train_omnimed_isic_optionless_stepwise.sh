#!/usr/bin/env bash
# OminiMedExpert ISIC optionless + B-tasks stepwise training launcher.
#
# Usage:
#   bash Comparative-R1/train_omnimed_isic_optionless_stepwise.sh
#
# Common overrides:
#   MODEL_PATH=/path/to/model \
#   OMNI_ROOT=/path/to/OmniMedVQA \
#   NUM_GPUS=4 \
#   bash Comparative-R1/train_omnimed_isic_optionless_stepwise.sh

set -euo pipefail

EASYR1_ROOT="/mnt/cache/wuruixiao/users/lsc/EasyR1"
REPO_ROOT="/mnt/cache/wuruixiao/users/lsc"

pick_existing_dir() {
  for d in "$@"; do
    if [[ -n "${d}" && -d "${d}" ]]; then
      echo "${d}"
      return 0
    fi
  done
  echo ""
  return 0
}

# ===== Paths =====
MODEL_PATH=${MODEL_PATH:-"/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b"}
DATA_DIR=${DATA_DIR:-"${EASYR1_ROOT}/data/OminiMedExpert/isic_disease_diagnosis_v1_0.05"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/train_fewshot_0.5_optionless.jsonl"}
VAL_FILE=${VAL_FILE:-"${DATA_DIR}/test_optionless.jsonl"}


OMNI_ROOT=${OMNI_ROOT:-"${REPO_ROOT}/data/OmniMedVQA"}

CONFIG=${CONFIG:-"${EASYR1_ROOT}/Comparative-R1/configs/omnimed_isic_gspo_taskaware.yaml"}
FORMAT_PROMPT=${FORMAT_PROMPT:-"${EASYR1_ROOT}/Comparative-R1/prompts/omnimed_isic.jinja"}
REWARD_FUNCTION=${REWARD_FUNCTION:-"${EASYR1_ROOT}/Comparative-R1/reward/omnimed_isic_optionless_mixed_stepwise_reward_v1.py:compute_score"}

# ===== Training params =====
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-32}
N_SAMPLES=${N_SAMPLES:-4}
TEMPERATURE=${TEMPERATURE:-0.7}
SAVE_FREQ=${SAVE_FREQ:-5}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"omnimed_isic_optionless_single_n${N_SAMPLES}_t${TEMPERATURE}"}

# ===== Task-aware advantage (override config in this script) =====
# Recommended: keep enabled for multi-task stability.
TASK_AWARE_ENABLE=${TASK_AWARE_ENABLE:-true}
TASK_WEIGHT_SINGLE=${TASK_WEIGHT_SINGLE:-1.0}
TASK_WEIGHT_B1=${TASK_WEIGHT_B1:-1.0}
TASK_WEIGHT_B2=${TASK_WEIGHT_B2:-1.0}
TASK_WEIGHT_B3=${TASK_WEIGHT_B3:-1.0}
TASK_WEIGHT_B4=${TASK_WEIGHT_B4:-1.0}
TASK_WEIGHT_B5=${TASK_WEIGHT_B5:-1.0}
TASK_WEIGHT_B6=${TASK_WEIGHT_B6:-1.0}
TASK_WEIGHT_B7=${TASK_WEIGHT_B7:-1.0}

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Train file not found: ${TRAIN_FILE}" >&2
  exit 1
fi
if [[ ! -f "${VAL_FILE}" ]]; then
  echo "Val file not found: ${VAL_FILE}" >&2
  exit 1
fi
if [[ -z "${OMNI_ROOT}" || ! -d "${OMNI_ROOT}" ]]; then
  echo "OMNI_ROOT is missing or invalid. Please set OMNI_ROOT=/path/to/OmniMedVQA" >&2
  exit 1
fi

echo "==============================================="
echo "OminiMedExpert ISIC Train (Optionless+Stepwise)"
echo "==============================================="
echo "Root:        ${EASYR1_ROOT}"
echo "Config:      ${CONFIG}"
echo "Model:       ${MODEL_PATH}"
echo "Train file:  ${TRAIN_FILE}"
echo "Val file:    ${VAL_FILE}"
echo "Image root:  ${OMNI_ROOT}"
echo "Reward:      ${REWARD_FUNCTION}"
echo "GPUs:        ${NUM_GPUS}"
echo "Batch size:  ${BATCH_SIZE}"
echo "N samples:   ${N_SAMPLES}"
echo "Temperature: ${TEMPERATURE}"
echo "Experiment:  ${EXPERIMENT_NAME}"
echo "Task aware:  ${TASK_AWARE_ENABLE}"
echo ""

cd "${EASYR1_ROOT}"
CMD=(python3 -m verl.trainer.main \
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
  trainer.save_freq="${SAVE_FREQ}")

if [[ "${TASK_AWARE_ENABLE}" == "true" ]]; then
  CMD+=(
    algorithm.task_adv_ops_enable=true
    algorithm.task_adv_weights.mcq_optionless_text="${TASK_WEIGHT_SINGLE}"
    algorithm.task_adv_weights.B1_target_search="${TASK_WEIGHT_B1}"
    algorithm.task_adv_weights.B2_odd_one_out="${TASK_WEIGHT_B2}"
    algorithm.task_adv_weights.B3_label_corruption="${TASK_WEIGHT_B3}"
    algorithm.task_adv_weights.B4_exemplar_match="${TASK_WEIGHT_B4}"
    algorithm.task_adv_weights.B5_same_different="${TASK_WEIGHT_B5}"
    algorithm.task_adv_weights.B6_pair_finding="${TASK_WEIGHT_B6}"
    algorithm.task_adv_weights.B7_support_set_nway="${TASK_WEIGHT_B7}"
  )
else
  CMD+=(algorithm.task_adv_ops_enable=false)
fi

"${CMD[@]}"
