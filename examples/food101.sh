#!/bin/bash

set -x

MODEL_PATH=/tmp/shared-storage/lishichao/cache/models--qwen--qwen2.5-vl-7b-instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5

python3 -m verl.trainer.main \
    config=examples/food101_config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=food101_1210_n=4_t=0.7_p=0.9_answer_only \
    trainer.n_gpus_per_node=4