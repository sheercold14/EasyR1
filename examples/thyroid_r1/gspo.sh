#!/bin/bash

set -x

# Path to your base model (local path or HF-style id if your environment supports it).
MODEL_PATH=/tmp/shared-storage/lishichao/cache/models--qwen--qwen2.5-vl-7b-instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5

# Optional: resume from an existing checkpoint (leave empty to train from MODEL_PATH).
LOAD_CHECKPOINT_PATH=/tmp/shared-storage/lishichao/EasyR1/checkpoints/easy_r1/pathology_1211_batchsize=32_n=4_t=0.7_p=0.9_correct_rewards/global_step_855

python3 -m verl.trainer.main \
    config=examples/thyroid_r1/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.loss_type=gspo_token \
    worker.actor.loss_avg_mode=seq \
    worker.actor.clip_ratio_low=3e-4 \
    worker.actor.clip_ratio_high=4e-4 \
    algorithm.disable_kl=True \
    trainer.experiment_name=pathology_1220_gspo_rollout4_fold0_promptv2.3_rewardv2.3_batchsize=32_n=4_t=0.7_p0.9 \
    trainer.n_gpus_per_node=4