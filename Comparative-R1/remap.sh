#!/bin/bash

set -x

# EASYR1_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# cd "${EASYR1_ROOT}"

# Path to your base model (local path or HF-style id if your environment supports it).
MODEL_PATH=/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b
# REWARD_FUNCTION=/mnt/cache/wuruixiao/users/lsc/EasyR1/Comparative-R1/reward/dtd_noformat.py
# EXPERIMENT_NAME="qwen2_5_7b_dtd_b2n_gspo_thinking"
TRAIN_FILES=/mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/train_mix_all.jsonl
VAL_FILES=/mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/test_4shot_thinking.jsonl
EXPERIMENT_NAME="remap_mixall_thinking_isic_v0.6_cls"
IMAGE_DIR=/mnt/cache/wuruixiao/users/lsc/data/OmniMedVQA
# Optional: resume from an existing checkpoint (leave empty to train from MODEL_PATH).
# LOAD_CHECKPOINT_PATH=/mnt/cache/wuruixiao/users/lsc/EasyR1/checkpoints/ReMAP/remap_curriculum_isic_v0.6/global_step_100

python3 -m verl.trainer.main \
    data.image_dir=${IMAGE_DIR} \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    config=Comparative-R1/configs/isic_offline_rft_remap.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXPERIMENT_NAME}\
    trainer.n_gpus_per_node=4 \
    # trainer.load_checkpoint_path=${LOAD_CHECKPOINT_PATH} \
    # worker.reward.reward_function=${REWARD_FUNCTION}:compute_score \

