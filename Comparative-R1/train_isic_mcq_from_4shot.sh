#!/bin/bash

set -x

MODEL_PATH=${MODEL_PATH:-/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b}
TRAIN_FILES=${TRAIN_FILES:-data/offline_rft/isic/v1/MCQ_4shot_nothinking.jsonl}
VAL_FILES=${VAL_FILES:-data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.jsonl}
IMAGE_DIR=${IMAGE_DIR:-/mnt/cache/wuruixiao/users/lsc/data/OmniMedVQA}
REWARD_FUNCTION=${REWARD_FUNCTION:-Comparative-R1/reward/omnimed_isic_reward_v2.py:compute_score}
CONFIG=${CONFIG:-Comparative-R1/configs/isic_offline_rft_remap.yaml}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-isic_mcq_4shot}

python3 -m verl.trainer.main \
  config=${CONFIG} \
  data.image_dir=${IMAGE_DIR} \
  data.train_files=${TRAIN_FILES} \
  data.val_files=${VAL_FILES} \
  worker.actor.model.model_path=${MODEL_PATH} \
  worker.reward.reward_function=${REWARD_FUNCTION} \
  worker.rollout.limit_images=1 \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.n_gpus_per_node=4
