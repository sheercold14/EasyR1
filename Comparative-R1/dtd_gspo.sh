#!/bin/bash

set -x

# EASYR1_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# cd "${EASYR1_ROOT}"

# Path to your base model (local path or HF-style id if your environment supports it).
MODEL_PATH=/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b
FORMAT_PROMPT=/mnt/cache/wuruixiao/users/lsc/EasyR1/Comparative-R1/prompts/dtd_nothinking.jinja
REWARD_FUNCTION=/mnt/cache/wuruixiao/users/lsc/EasyR1/Comparative-R1/reward/dtd_noformat.py
# EXPERIMENT_NAME="qwen2_5_7b_dtd_b2n_gspo_thinking"
EXPERIMENT_NAME="qwen2_5_7b_isic_gspo"
# Optional: resume from an existing checkpoint (leave empty to train from MODEL_PATH).
# LOAD_CHECKPOINT_PATH=

python3 -m verl.trainer.main \
    config=Comparative-R1/configs/dtd_config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXPERIMENT_NAME}\
    trainer.n_gpus_per_node=4 \
    worker.reward.reward_function=${REWARD_FUNCTION}:compute_score \
    data.format_prompt=${FORMAT_PROMPT} \
