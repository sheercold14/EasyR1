#!/bin/bash

set -x

MODEL_PATH=${MODEL_PATH:-/tmp/shared-storage/lishichao/cache/models--qwen--qwen2.5-vl-7b-instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}

export ORACLE_REDIS_URL=${ORACLE_REDIS_URL:-redis://localhost:6381/0}
export ORACLE_QUEUE_KEY=${ORACLE_QUEUE_KEY:-oracle_queue}
export ORACLE_WEAKNESS_KEY=${ORACLE_WEAKNESS_KEY:-student_weakness}
export ORACLE_EPOCH_SIZE=${ORACLE_EPOCH_SIZE:-10000}
export ORACLE_REDIS_TIMEOUT=${ORACLE_REDIS_TIMEOUT:-2}
export ORACLE_REDIS_MAX_RETRIES=${ORACLE_REDIS_MAX_RETRIES:-3}
export ORACLE_REDIS_RETRY_SLEEP=${ORACLE_REDIS_RETRY_SLEEP:-0.2}
export ORACLE_FALLBACK_SAMPLE_PROB=${ORACLE_FALLBACK_SAMPLE_PROB:-0.1}
export ORACLE_DATASET_NUM_WORKERS=${ORACLE_DATASET_NUM_WORKERS:-0}
export ORACLE_WEAKNESS_TOPK=${ORACLE_WEAKNESS_TOPK:-3}
export ORACLE_WEAKNESS_THRESHOLD=${ORACLE_WEAKNESS_THRESHOLD:--0.1}

python3 -m oracle_grpo.run_grpo_online \
    config=oracle_grpo/configs/isic_online.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_7b_isic_oracle_grpo
