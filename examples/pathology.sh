#!/bin/bash

set -x

MODEL_PATH=/tmp/shared-storage/lishichao/cache/models--qwen--qwen2.5-vl-7b-instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5

python3 -m verl.trainer.main \
    config=examples/pathology_config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=pathology_1213_gspo_promptv1_rewardv1_batchsize=32_n=4_t=0.7_p=0.9 \
    trainer.load_checkpoint_path=/tmp/shared-storage/lishichao/EasyR1/checkpoints/easy_r1/pathology_1211_batchsize=32_n=4_t=0.7_p=0.9_correct_rewards/global_step_855 \
    trainer.n_gpus_per_node=4