#!/bin/bash

set -x

EASYR1_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${EASYR1_ROOT}"

# Path to your base model (local path or HF-style id if your environment supports it).
MODEL_PATH=/tmp/shared-storage/lishichao/cache/models--qwen--qwen2.5-vl-7b-instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5

# Optional: resume from an existing checkpoint (leave empty to train from MODEL_PATH).
LOAD_CHECKPOINT_PATH=

python3 -m verl.trainer.main \
    config=Comparative-R1/configs/dtd_config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.load_checkpoint_path=${LOAD_CHECKPOINT_PATH} \
    trainer.experiment_name=comparative_r1/dtd_b2n_gspo_kl_1e-2_4shot \
    trainer.n_gpus_per_node=4
