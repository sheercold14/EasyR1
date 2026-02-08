#!/bin/bash

set -x

MODEL_PATH=${MODEL_PATH:-/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
CONFIG_PATH=${CONFIG_PATH:-Comparative-R1/configs/dtd_config.yaml}
EVAL_FILE=${EVAL_FILE:-}
MODE=${MODE:-multi}  # single | multi
EXPERIMENT_NAME=${EXPERIMENT_NAME:-dtd_eval_${MODE}}
N_GPUS=${N_GPUS:-4}

# single/multi unified reward:
# - single reads ground_truth.label
# - multi  reads ground_truth.correct_answer
REWARD_FUNCTION=${REWARD_FUNCTION:-Comparative-R1/reward/dtd_direct_mixed_reward.py}

# only used in MODE=single (for MODE=multi we force no jinja)
FORMAT_PROMPT=${FORMAT_PROMPT:-Comparative-R1/prompts/dtd_nothinking.jinja}

if [ -z "${EVAL_FILE}" ]; then
    echo "Please set EVAL_FILE"
    echo "Example: MODE=multi EVAL_FILE=/path/to/multi.jsonl CHECKPOINT_PATH=/path/to/ckpt bash Comparative-R1/dtd_eval.sh"
    exit 1
fi

if [[ "${REWARD_FUNCTION}" != *:* ]]; then
    REWARD_FUNCTION="${REWARD_FUNCTION}:compute_score"
fi

if [ "${MODE}" = "single" ]; then
    PROMPT_KEY=problem
    ANSWER_KEY=label
    IMAGE_KEY=image
    FORMAT_OVERRIDE="data.format_prompt=${FORMAT_PROMPT}"
elif [ "${MODE}" = "multi" ]; then
    PROMPT_KEY=prompt
    ANSWER_KEY=answer
    IMAGE_KEY=images
    # direct multi prompt already has answer format instruction
    FORMAT_OVERRIDE="data.format_prompt=null"
else
    echo "Unknown MODE=${MODE}, expected single|multi"
    exit 1
fi

EXTRA_CKPT_ARG=""
if [ -n "${CHECKPOINT_PATH}" ]; then
    EXTRA_CKPT_ARG="trainer.load_checkpoint_path=${CHECKPOINT_PATH}"
fi

python3 -m verl.trainer.main \
    config=${CONFIG_PATH} \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.val_only=true \
    trainer.val_before_train=true \
    trainer.find_last_checkpoint=false \
    trainer.logger='[file]' \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    data.val_files=${EVAL_FILE} \
    data.prompt_key=${PROMPT_KEY} \
    data.answer_key=${ANSWER_KEY} \
    data.image_key=${IMAGE_KEY} \
    ${FORMAT_OVERRIDE} \
    ${EXTRA_CKPT_ARG}
