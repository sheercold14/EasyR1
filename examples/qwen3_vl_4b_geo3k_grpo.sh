#!/bin/bash

set -x

MODEL_PATH=/data/shichao/models/Qwen3-VL-2B-Instruct  # replace it with your local file path
export NCCL_CUMEM_HOST_ENABLE=0


# export CUDA_VISIBLE_DEVICES=0,1
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export CUDA_DEVICE_ORDER=PCI_BUS_ID


python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_vl_2b_geo_grpo \
    trainer.n_gpus_per_node=4
