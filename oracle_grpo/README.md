Oracle-Guided VQ GRPO (Online Active Teaching)

This folder adds an online data injection and logic reward loop without modifying the core EasyR1 code.

Key components
- `oracle_grpo/online_dataset.py`: Redis-backed IterableDataset with local fallback.
- `oracle_grpo/online_dataloader.py`: StatefulDataLoader wrapper for online training.
- `oracle_grpo/run_grpo_online.py`: GRPO entrypoint using the online dataloader.
- `oracle_grpo/reward/logic_reward.py`: Logic-based reward + weakness feedback.
- `oracle_grpo/teacher/teacher_oracle.py`: Simple Redis producer example.

Redis payload format (JSON)
{
  "prompt": "<your prompt>",
  "images": ["/abs/or/rel/path.jpg"],
  "meta": {
    "class_label": "sparrow",
    "must_include": ["conical", "beak"],
    "must_not_include": ["striped"],
    "avoid_in_think": ["guess"],
    "answer": "sparrow"
  }
}

Environment variables
- `ORACLE_REDIS_URL`: redis://host:port/db
- `ORACLE_QUEUE_KEY`: Redis list key for incoming samples
- `ORACLE_WEAKNESS_KEY`: Redis key for weakness feedback
- `ORACLE_EPOCH_SIZE`: epoch length used by the dataloader
- `ORACLE_REDIS_TIMEOUT`: Redis socket timeout (seconds)
- `ORACLE_REDIS_MAX_RETRIES`: retries for Redis operations
- `ORACLE_REDIS_RETRY_SLEEP`: sleep between retries
- `ORACLE_FALLBACK_SAMPLE_PROB`: chance to sample from fallback dataset
- `ORACLE_DATASET_NUM_WORKERS`: dataset worker count
- `ORACLE_WEAKNESS_TOPK`: number of weak classes to report
- `ORACLE_WEAKNESS_THRESHOLD`: minimum score considered weak

Quick start (ISIC)
1) Start Redis and ensure `pip install redis` is available.
2) Run teacher:
   `python3 oracle_grpo/teacher/teacher_oracle.py --seed-jsonl /tmp/shared-storage/lishichao/EasyR1/data/omnimedvqa/ISIC/few_shot/16-shot/train.jsonl`
3) Run training:
   `bash oracle_grpo/scripts/run_isic_online_grpo.sh`

Notes
- Redis only stores image paths, not image bytes.
- If Redis is empty, the dataset samples from `data.train_files` to avoid GPU stalls.
- The reward function accepts either JSON `ground_truth` or plain string answers.
