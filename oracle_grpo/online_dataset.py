"""Online dataset backed by Redis with local fallback."""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Optional

import torch
from jinja2 import Template
from torch.utils.data import IterableDataset, get_worker_info

from verl.utils import torch_functional as VF
from verl.utils.dataset import process_image, process_video


class OverlongPromptError(RuntimeError):
    """Raised when a prompt exceeds max_prompt_length in online mode."""


class OnlineDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        processor,
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        fallback_dataset=None,
        epoch_size: int = 10000,
        redis_url: Optional[str] = None,
        redis_queue_key: Optional[str] = None,
        redis_timeout: int = 2,
        redis_max_retries: int = 3,
        redis_retry_sleep: float = 0.2,
        fallback_sample_prob: float = 0.0,
        filter_overlong_prompts: bool = True,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.fallback_dataset = fallback_dataset
        self.epoch_size = epoch_size
        self.redis_url = redis_url
        self.redis_queue_key = redis_queue_key
        self.redis_timeout = redis_timeout
        self.redis_max_retries = redis_max_retries
        self.redis_retry_sleep = redis_retry_sleep
        self.fallback_sample_prob = fallback_sample_prob
        self.filter_overlong_prompts = filter_overlong_prompts
        self.dropped_overlong = 0
        self.total_seen = 0

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

    def __len__(self) -> int:
        return int(self.epoch_size)

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        if self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]

        return [{"role": "user", "content": prompt_str}]

    def _get_redis_client(self):
        if self.redis_url is None or self.redis_queue_key is None:
            return None
        try:
            import redis  # type: ignore
        except ImportError as exc:
            raise RuntimeError("redis package is required for online mode") from exc

        return redis.Redis.from_url(
            self.redis_url,
            socket_timeout=self.redis_timeout,
            socket_connect_timeout=self.redis_timeout,
            decode_responses=False,
        )

    def _pull_from_redis(self, client) -> Optional[dict[str, Any]]:
        for _ in range(self.redis_max_retries):
            try:
                item = client.blpop(self.redis_queue_key, timeout=self.redis_timeout)
                if item is None:
                    return None
                _, raw = item
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                return json.loads(raw)
            except Exception:
                time.sleep(self.redis_retry_sleep)
        return None

    def _fallback_sample(self, rng: random.Random) -> Optional[dict[str, Any]]:
        if self.fallback_dataset is None:
            return None
        dataset = getattr(self.fallback_dataset, "dataset", self.fallback_dataset)
        if len(dataset) == 0:
            return None
        idx = rng.randrange(len(dataset))
        return dict(dataset[idx])

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = payload.get(self.prompt_key) or payload.get("prompt") or payload.get("question")
        if prompt is None:
            raise ValueError("payload missing prompt")

        example: dict[str, Any] = {self.prompt_key: prompt}

        images = payload.get(self.image_key) or payload.get("images")
        if images is None:
            image_path = payload.get("image_path")
            if image_path:
                images = [image_path]
        if isinstance(images, str):
            images = [images]
        if images is not None:
            example[self.image_key] = images

        meta: dict[str, Any] = {}
        meta_src = payload.get("meta")
        if isinstance(meta_src, dict):
            meta.update(meta_src)
        for key in (
            "class_label",
            "must_include",
            "must_not_include",
            "logic_constraints",
            "avoid_in_think",
            "hallucination",
        ):
            if key in payload:
                meta[key] = payload[key]

        answer = payload.get(self.answer_key) or payload.get("answer") or payload.get("ground_truth")
        if answer is not None and "answer" not in meta:
            meta["answer"] = answer

        if meta:
            example[self.answer_key] = json.dumps(meta, ensure_ascii=True)
        else:
            example[self.answer_key] = answer if answer is not None else ""

        return example

    def _example_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        example = self._normalize_payload(payload)
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            if self.filter_overlong_prompts and input_ids.size(0) > self.max_prompt_length:
                raise OverlongPromptError(
                    f"Prompt length {input_ids.size(0)} is longer than {self.max_prompt_length}."
                )
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            if self.filter_overlong_prompts and input_ids.size(0) > self.max_prompt_length:
                raise OverlongPromptError(
                    f"Prompt length {input_ids.size(0)} is longer than {self.max_prompt_length}."
                )
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            if self.filter_overlong_prompts and input_ids.size(0) > self.max_prompt_length:
                raise OverlongPromptError(
                    f"Prompt length {input_ids.size(0)} is longer than {self.max_prompt_length}."
                )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        return example

    def __iter__(self):
        breakpoint()
        worker = get_worker_info()
        seed_offset = 0 if worker is None else worker.id
        rng = random.Random(1234 + seed_offset)
        client = None

        while True:
            if client is None:
                client = self._get_redis_client()

            payload = None
            use_fallback = self.fallback_dataset is not None and rng.random() < self.fallback_sample_prob
            if not use_fallback and client is not None:
                payload = self._pull_from_redis(client)

            if payload is None:
                payload = self._fallback_sample(rng)

            if payload is None:
                time.sleep(self.redis_retry_sleep)
                continue

            try:
                example = self._example_from_payload(payload)
                self.total_seen += 1
                yield example
            except OverlongPromptError:
                self.dropped_overlong += 1
                self.total_seen += 1
            except Exception:
                time.sleep(self.redis_retry_sleep)

    def get_drop_stats(self) -> dict[str, float]:
        if self.total_seen <= 0:
            drop_rate = 0.0
        else:
            drop_rate = self.dropped_overlong / self.total_seen
        return {
            "dropped_overlong": float(self.dropped_overlong),
            "total_seen": float(self.total_seen),
            "drop_rate": drop_rate,
        }
