"""Online dataloader that pulls samples from Redis with fallback."""

from __future__ import annotations

import os
from typing import Optional

from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset import RLHFDataset, collate_fn

from .online_dataset import OnlineDataset


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def create_online_dataloader(config, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]):
    fallback_dataset = None
    if config.train_files:
        fallback_dataset = RLHFDataset(
            data_path=config.train_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            video_key=config.video_key,
            image_dir=config.image_dir,
            video_fps=config.video_fps,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
            filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
        )

    epoch_size = _get_env_int("ORACLE_EPOCH_SIZE", 10000)
    redis_timeout = _get_env_int("ORACLE_REDIS_TIMEOUT", 2)
    redis_retries = _get_env_int("ORACLE_REDIS_MAX_RETRIES", 3)
    redis_retry_sleep = _get_env_float("ORACLE_REDIS_RETRY_SLEEP", 0.2)
    fallback_prob = _get_env_float("ORACLE_FALLBACK_SAMPLE_PROB", 0.0)

    train_dataset = OnlineDataset(
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        fallback_dataset=fallback_dataset,
        epoch_size=epoch_size,
        redis_url=config.oracle_redis_url,
        redis_queue_key=config.oracle_queue_key,
        redis_timeout=redis_timeout,
        redis_max_retries=redis_retries,
        redis_retry_sleep=redis_retry_sleep,
        fallback_sample_prob=fallback_prob,
        filter_overlong_prompts=config.filter_overlong_prompts,
    )

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=_get_env_int("ORACLE_DATASET_NUM_WORKERS", 0),
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_dataset = RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
    )

    val_batch_size = len(val_dataset) if config.val_batch_size == -1 else config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=_get_env_int("ORACLE_VAL_NUM_WORKERS", 0),
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
