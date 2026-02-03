#!/bin/bash
# Comparative-R1 Training Pipeline
#
# This script:
# 1. Preprocesses data to generate mixed/comparative samples
# 2. Trains the model using GSPO
#
# Usage:
#   bash train_comparative.sh

set -e

export NCCL_CUMEM_HOST_ENABLE=0
# ===== Configuration =====
# MODEL_PATH=/data/shichao/models/Qwen3-VL-2B-Instruct
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
DATA_PATH=/data/shichao/EasyR1/data/thyroid/all.jsonl
OUTPUT_DIR=/data/shichao/EasyR1/data/thyroid/comparative_r1
IMAGE_ROOT=/data/shichao/hospital3/hospital3_thyroid_merged 

# Data generation parameters
TRAIN_SPLITS=${TRAIN_SPLITS:-"0,1,2"}
VAL_SPLITS=${VAL_SPLITS:-"3,4"}
NUM_IMAGES=${NUM_IMAGES:-2}
TARGET_STRATEGY=${TARGET_STRATEGY:-"random"}
NUM_COMP_SAMPLES=${NUM_COMP_SAMPLES:-100}

# Training parameters
NUM_GPUS=${NUM_GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-32}
N_SAMPLES=${N_SAMPLES:-4}
TEMPERATURE=${TEMPERATURE:-0.7}
SAVE_FREQ=${SAVE_FREQ:-5}

# Optional: resume from checkpoint
LOAD_CHECKPOINT_PATH=${LOAD_CHECKPOINT_PATH:-""}

# Experiment name
EXPERIMENT_NAME="thyroid_mixed_train${TRAIN_SPLITS//,/_}_val${VAL_SPLITS//,/_}_k${NUM_IMAGES}_${TARGET_STRATEGY}_n${NUM_COMP_SAMPLES}_t${TEMPERATURE}"

# ===== Step 1: Generate Data =====
echo "=========================================="
echo "Step 1: Generating Training Data"
echo "=========================================="
echo "Train splits: $TRAIN_SPLITS"
echo "Val splits: $VAL_SPLITS"
echo "Images per comparison: $NUM_IMAGES"
echo "Target strategy: $TARGET_STRATEGY"
echo ""

python3 /data/shichao/EasyR1/Comparative-R1/scripts/generate_comparative_data.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --train_splits "$TRAIN_SPLITS" \
    --val_splits "$VAL_SPLITS" \
    --num_images "$NUM_IMAGES" \
    --target_class_strategy "$TARGET_STRATEGY" \
    --num_comparative_samples "$NUM_COMP_SAMPLES"

echo ""
echo "Data generation complete!"
echo ""

# ===== Step 2: Train Model =====
echo "=========================================="
echo "Step 2: Training with GSPO"
echo "=========================================="
echo "Experiment: $EXPERIMENT_NAME"
echo ""

PYTHONPATH=/data/shichao/EasyR1/Comparative-R1/verl_patches:/data/shichao/EasyR1:$PYTHONPATH python3 -m verl.trainer.main \
    config=Comparative-R1/configs/comparative_gspo.yaml \
    data.train_files=${OUTPUT_DIR}/train.jsonl \
    data.val_files=${OUTPUT_DIR}/val.jsonl \
    data.rollout_batch_size=${BATCH_SIZE} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.limit_images=$((NUM_IMAGES + 1)) \
    worker.rollout.n=${N_SAMPLES} \
    worker.rollout.temperature=${TEMPERATURE} \
    worker.reward.reward_function=Comparative-R1/reward/comparative_reward.py:compute_score \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.save_freq=${SAVE_FREQ} \
    ${LOAD_CHECKPOINT_PATH:+trainer.load_checkpoint_path=${LOAD_CHECKPOINT_PATH}}

echo ""
echo "=========================================="
echo "Training complete: ${EXPERIMENT_NAME}"
echo "=========================================="
