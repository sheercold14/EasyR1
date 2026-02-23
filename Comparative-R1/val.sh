#!/bin/bash

set -x

# EASYR1_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# cd "${EASYR1_ROOT}"

# Path to your base model (local path or HF-style id if your environment supports it).
MODEL_PATH=/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b
# REWARD_FUNCTION=/mnt/cache/wuruixiao/users/lsc/EasyR1/Comparative-R1/reward/dtd_noformat.py
REWARD_FUNCTION=/mnt/cache/wuruixiao/users/lsc/EasyR1/Comparative-R1/reward/omnimed_isic_reward.py
# EXPERIMENT_NAME="qwen2_5_7b_dtd_b2n_gspo_thinking"
# TRAIN_FILES=/mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/train_mix_all.jsonl

TRAIN_FILES=/mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/MCQ_4shot_nothinking.4labels_text.jsonl
VAL_FILES=/mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.jsonl
# TRAIN_FILES=/mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/MCQ_4shot_nothinking.jsonl
# VAL_FILES=/mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.jsonl

EXPERIMENT_NAME="4label_text_list_eval_MCQ4_nothinking_isic_rollout_log"
# EXPERIMENT_NAME="pretrain_eval_MCQ4_nothinking_isic_rollout_log"

IMAGE_DIR=/mnt/cache/wuruixiao/users/lsc/data/OmniMedVQA
# Optional: resume from an existing checkpoint (leave empty to train from MODEL_PATH).
# LOAD_CHECKPOINT_PATH=/mnt/cache/wuruixiao/users/lsc/EasyR1/checkpoints/ReMAP/remap_curriculum_isic_v0.6/global_step_100
FORMAT_PROMPT=/mnt/cache/wuruixiao/users/lsc/EasyR1/Comparative-R1/prompts/omnimed_isic.jinja
python3 -m verl.trainer.main \
    data.image_dir=${IMAGE_DIR} \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    config=Comparative-R1/configs/MCQ.yaml \
    trainer.experiment_name=${EXPERIMENT_NAME}\
    trainer.n_gpus_per_node=2 \
    trainer.train_rollout_groups_to_log=20 \
    trainer.train_rollout_log_only_all_wrong=true \
    worker.reward.reward_function=${REWARD_FUNCTION}:compute_score \
    data.format_prompt=${FORMAT_PROMPT} \
    trainer.val_before_train=true \
    trainer.val_only=true \
    trainer.find_last_checkpoint=false \
    trainer.load_checkpoint_path=null \
    worker.actor.model.model_path=/mnt/cache/wuruixiao/users/lsc/EasyR1/checkpoints/MCQ_failure/4label_text_list_eval2_nothinking_isic_rollout_log/global_step_170/actor/huggingface/ \
        # worker.actor.model.model_path=${MODEL_PATH} \