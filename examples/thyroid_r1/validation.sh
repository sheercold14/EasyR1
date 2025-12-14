
set -x

MODEL_PATH=/tmp/shared-storage/lishichao/cache/models--qwen--qwen2.5-vl-7b-instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5

python -m verl.trainer.main \
  config=examples/pathology_config.yaml \
  worker.actor.model.model_path=${MODEL_PATH} \
  data.val_files=/tmp/shared-storage/lishichao/EasyR1/data/thyroid/val_with_img.jsonl \
  trainer.val_only=true \
  trainer.load_checkpoint_path=/tmp/shared-storage/lishichao/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7b_geo_grpo_pathology_1210_n=4_t=0.7_p=0.9_correct_rewards/global_step_1225 \
  trainer.save_checkpoint_path=/tmp/shared-storage/lishichao/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7b_geo_grpo_pathology_1210_n=4_t=0.7_p=0.9_correct_rewards/global_step_1225 \
  trainer.val_freq=1 \
  data.val_batch_size=4  \
  data.format_prompt=/tmp/shared-storage/lishichao/EasyR1/examples/format_prompt/pathology.jinja \
  trainer.val_generations_to_log=200 