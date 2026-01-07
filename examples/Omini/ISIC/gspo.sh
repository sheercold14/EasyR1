#!/bin/bash

set -x

# Path to your base model (local path or HF-style id if your environment supports it).
MODEL_PATH=/tmp/shared-storage/lishichao/cache/models--qwen--qwen2.5-vl-7b-instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5

# Optional: resume from an existing checkpoint (leave empty to train from MODEL_PATH).
LOAD_CHECKPOINT_PATH=

python3 -m verl.trainer.main \
    config=examples/Omini/ISIC/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.load_checkpoint_path=${LOAD_CHECKPOINT_PATH} \
    trainer.experiment_name=omini/isic_gspo_kl_1e-2_16_shot \
    trainer.n_gpus_per_node=4
