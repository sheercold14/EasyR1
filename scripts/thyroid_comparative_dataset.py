# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Flexible Mixed Dataset for Medical Image Few-Shot Learning.

Supports two training modes:
1. Multi-image comparative: K images, "Which image shows [target class]?"
2. Single-image classification: 1 image, "Classify this image"

Features:
- Random target class selection for comparative task
- Support for N classes (not just binary)
- Flexible class name mapping
- Configurable data generation interface
"""

import json
import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils import torch_functional as VF


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    """Process and resize image according to pixel constraints."""
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for DataLoader."""
    from collections import defaultdict

    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


class ClassConfig:
    """
    Configuration for class labels and prompts.

    This interface allows flexible customization for different datasets.
    Extend or modify this class for your specific dataset.
    """

    def __init__(
        self,
        class_names: list[str] | None = None,
        class_descriptions: dict[str, str] | None = None,
        comparative_prompt_template: str | None = None,
        single_prompt_template: str | None = None,
    ):
        """
        Initialize class configuration.

        Args:
            class_names: List of class names (e.g., ["benign", "malignant"])
            class_descriptions: Optional descriptions for each class
            comparative_prompt_template: Template for comparative prompts
            single_prompt_template: Template for single-image prompts
        """
        self.class_names = class_names or ["benign", "malignant"]
        self.class_descriptions = class_descriptions or {}

        # Default comparative prompt template
        self.comparative_prompt_template = comparative_prompt_template or (
            "You are a professional pathology assistant for thyroid OCT-embedded frozen blocks.\n\n"
            "Below are {num_images} images, labeled {labels}.\n\n"
            "{image_descriptions}\n\n"
            "Carefully compare these images. Your task is to identify which image shows a **{target_class}** case.\n\n"
            "Provide your analysis:\n"
            "1. Describe the key features of each image\n"
            "2. Compare the differences between images\n"
            "3. Explain which features indicate different characteristics\n"
            "4. Conclude with your answer\n\n"
            "Follow the exact output format:\n\n"
            "<thinking>\n"
            "[Your comparative analysis]\n"
            "</thinking>\n\n"
            "<answer> {answer_options} </answer>"
        )

        # Default single-image prompt template
        self.single_prompt_template = single_prompt_template or (
            "You are a professional pathology assistant for thyroid OCT-embedded frozen blocks.\n\n"
            "<image>\n\n"
            "Analyze this image and classify it.\n\n"
            "Provide your analysis:\n"
            "1. Describe the key features\n"
            "2. Explain what these features indicate\n"
            "3. Conclude with your diagnosis\n\n"
            "Follow the exact output format:\n\n"
            "<thinking>\n"
            "[Your analysis]\n"
            "</thinking>\n\n"
            "<answer> {class_options} </answer>"
        )

    def get_class_options(self) -> str:
        """Get formatted class options for single-image task."""
        return " or ".join(self.class_names)

    def get_answer_options(self, num_images: int) -> str:
        """Get formatted answer options for comparative task."""
        letters = [chr(ord("A") + i) for i in range(num_images)]
        if len(letters) == 2:
            return f"{letters[0]} or {letters[1]}"
        return ", ".join(letters[:-1]) + f", or {letters[-1]}"

    def get_comparative_prompt(self, target_class: str, num_images: int) -> str:
        """Generate comparative prompt for given target class."""
        letters = [chr(ord("A") + i) for i in range(num_images)]
        labels = ", ".join(letters)
        image_descriptions = "\n".join([f"({letter}) Image {letter}: <image>" for letter in letters])

        return self.comparative_prompt_template.format(
            num_images=num_images,
            labels=labels,
            target_class=target_class,
            image_descriptions=image_descriptions,
            answer_options=self.get_answer_options(num_images),
        )

    def get_single_prompt(self) -> str:
        """Generate single-image classification prompt."""
        return self.single_prompt_template.format(
            class_options=self.get_class_options()
        )


class FlexibleMixedDataset(Dataset):
    """
    Flexible mixed dataset for medical image classification.

    Supports:
    - Random target class selection for comparative tasks
    - N-class classification (not just binary)
    - Configurable class names and prompts

    Args:
        data_path: Path to jsonl file
        tokenizer: Pre-trained tokenizer
        processor: VLM processor for handling images
        split: Which fold to use (e.g., "1")
        mode: "train" or "val"
        num_images: Number of images per comparison
        single_image_ratio: Fraction of single-image samples (0.0 to 1.0)
        target_class_strategy: How to select target for comparative task
            - "random": Randomly choose from all classes
            - "fixed_first": Always use first class
            - "rotate": Rotate through classes
        class_config: ClassConfig instance for customization
        label_key: Key in data for label (default: "answer.label")
        max_prompt_length: Maximum prompt length in tokens
        min_pixels: Minimum pixels for image resizing
        max_pixels: Maximum pixels for image resizing
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        split: str | list[str] = "1",
        mode: Literal["train", "val"] = "train",
        num_images: int = 2,
        single_image_ratio: float = 0.3,
        target_class_strategy: Literal["random", "fixed_first", "rotate"] = "random",
        class_config: ClassConfig | None = None,
        label_key: str = "answer.label",
        max_prompt_length: int = 2048,
        min_pixels: Optional[int] = 262144,
        max_pixels: Optional[int] = 4194304,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        # Support multiple splits: "0,1,2" or ["0", "1", "2"] or "1"
        if isinstance(split, str):
            self.splits = [s.strip() for s in split.split(",")]
        else:
            self.splits = [str(s) for s in split]
        self.mode = mode
        self.num_images = num_images
        self.single_image_ratio = np.clip(single_image_ratio, 0.0, 1.0)
        self.target_class_strategy = target_class_strategy
        self.label_key = label_key
        self.max_prompt_length = max_prompt_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.rng = np.random.default_rng(seed)

        # Class configuration
        self.class_config = class_config or ClassConfig()
        self.class_names = self.class_config.class_names

        # Validation always uses single-image mode
        if mode == "val":
            self.effective_single_ratio = 1.0
        else:
            self.effective_single_ratio = self.single_image_ratio

        # Track rotation state
        self._current_target_idx = 0

        # Load and filter data
        self.samples_by_class, self.all_samples = self._load_and_filter_data(data_path)

        # Calculate dataset size
        self._dataset_size = len(self.all_samples) * 100

        print(f"FlexibleMixedDataset initialized:")
        print(f"  Mode: {mode}")
        print(f"  Splits: {self.splits}")
        print(f"  Classes: {self.class_names}")
        for name in self.class_names:
            print(f"    - {name}: {len(self.samples_by_class.get(name, []))} samples")
        if mode == "train":
            print(f"  Single-image ratio: {self.single_image_ratio:.1%}")
            print(f"  Target class strategy: {target_class_strategy}")
            print(f"  Expected mix: {self.effective_single_ratio:.1%} single, {1-self.effective_single_ratio:.1%} comparative")

    def _get_label(self, data: dict) -> str:
        """Extract label from data using label_key path."""
        keys = self.label_key.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, "unknown")
            else:
                value = getattr(value, key, "unknown")
        return str(value)

    def _load_and_filter_data(self, data_path: str) -> tuple[dict[str, list[dict]], list[dict]]:
        """Load jsonl and filter by split, group by class."""
        samples_by_class = defaultdict(list)
        all_samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())

                # Filter by split
                if str(data.get("split")) not in self.splits:
                    continue

                label = self._get_label(data)
                if label not in self.class_names:
                    continue

                sample = {
                    "image": data["images"][0] if data.get("images") else None,
                    "label": label,
                    "case_text": data.get("answer", {}).get("case_text", ""),
                }

                samples_by_class[label].append(sample)
                all_samples.append(sample)

        return dict(samples_by_class), all_samples

    def _select_target_class(self) -> str:
        """Select target class based on strategy."""
        available_classes = [c for c in self.class_names if len(self.samples_by_class.get(c, [])) > 0]

        if not available_classes:
            return self.class_names[0]  # Fallback

        if self.target_class_strategy == "random":
            return self.rng.choice(available_classes)
        elif self.target_class_strategy == "fixed_first":
            return available_classes[0]
        elif self.target_class_strategy == "rotate":
            target = available_classes[self._current_target_idx % len(available_classes)]
            self._current_target_idx += 1
            return target
        else:
            return available_classes[0]

    def _get_single_sample(self, index: int) -> dict[str, Any]:
        """Get a single-image classification sample."""
        sample = self.all_samples[index % len(self.all_samples)]

        # Build prompt
        prompt = self.class_config.get_single_prompt()

        # Build messages
        content_list = [
            {"type": "text", "text": prompt.replace("<image>", "")},
            {"type": "image"},
        ]

        messages = [{"role": "user", "content": content_list}]
        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # Process image
        image_path = sample["image"]
        processed_image = process_image(image_path, self.min_pixels, self.max_pixels)

        model_inputs = self.processor([processed_image], [prompt_text], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]

        # Position IDs
        if "Qwen3VLProcessor" in self.processor.__class__.__name__:
            from verl.models.transformers.qwen3_vl import get_rope_index
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        # Postprocess
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="right",
        )

        raw_prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "multi_modal_data": {"images": [image_path]},
            "ground_truth": {
                "task_type": "single",
                "label": sample["label"],
                "class_names": self.class_names,
            },
        }

    def _get_comparative_sample(self, index: int) -> dict[str, Any]:
        """Get a multi-image comparative sample."""
        # Select target class
        target_class = self._select_target_class()

        # Sample 1 target + (num_images - 1) non-target
        target_samples = self.samples_by_class.get(target_class, [])
        if not target_samples:
            # Fallback if no samples for target class
            target_class = self.class_names[0]
            target_samples = self.samples_by_class.get(target_class, [])

        target_idx = self.rng.integers(0, len(target_samples))
        sampled_target = [target_samples[target_idx]]

        # Sample non-target images
        non_target_classes = [c for c in self.class_names if c != target_class]
        sampled_distractors = []

        for i in range(self.num_images - 1):
            # Cycle through non-target classes
            class_for_distractor = non_target_classes[i % len(non_target_classes)]
            samples = self.samples_by_class.get(class_for_distractor, [])
            if samples:
                idx = self.rng.integers(0, len(samples))
                sampled_distractors.append(samples[idx])

        # Combine and shuffle
        all_samples = sampled_target + sampled_distractors
        shuffle_idx = self.rng.permutation(len(all_samples))
        shuffled_samples = [all_samples[i] for i in shuffle_idx]

        # Find target position
        target_position = None
        for i, sample in enumerate(shuffled_samples):
            if sample["label"] == target_class:
                target_position = i
                break

        correct_answer = chr(ord("A") + target_position)

        # Build prompt
        prompt = self.class_config.get_comparative_prompt(target_class, len(shuffled_samples))

        # Build messages with images
        content_list = [{"type": "text", "text": prompt}]

        for _ in range(len(shuffled_samples)):
            for j, content in enumerate(content_list):
                if content["type"] == "text" and "<image>" in content["text"]:
                    parts = content["text"].split("<image>", 1)
                    content_list[j] = {"type": "text", "text": parts[0]}
                    content_list.insert(j + 1, {"type": "image"})
                    if parts[1]:
                        content_list.insert(j + 2, {"type": "text", "text": parts[1]})
                    break

        messages = [{"role": "user", "content": content_list}]
        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # Process images
        image_paths = [img["image"] for img in shuffled_samples]
        processed_images = [process_image(img, self.min_pixels, self.max_pixels) for img in image_paths]

        model_inputs = self.processor(processed_images, [prompt_text], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]

        # Position IDs
        if "Qwen3VLProcessor" in self.processor.__class__.__name__:
            from verl.models.transformers.qwen3_vl import get_rope_index
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        # Postprocess
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="right",
        )

        raw_prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "multi_modal_data": {"images": image_paths},
            "ground_truth": {
                "task_type": "comparative",
                "target_class": target_class,
                "correct_answer": correct_answer,
                "num_images": len(shuffled_samples),
                "labels": [img["label"] for img in shuffled_samples],
                "class_names": self.class_names,
            },
        }

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, index):
        is_single = (index % 100) < (self.effective_single_ratio * 100)

        if is_single:
            return self._get_single_sample(index)
        else:
            return self._get_comparative_sample(index)


# Backward compatibility aliases
ThyroidMixedDataset = FlexibleMixedDataset
ThyroidComparativeDataset = FlexibleMixedDataset


def create_mixed_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin],
    split: str | list[str] = "1",
    val_splits: str | list[str] | None = None,
    num_images: int = 2,
    single_image_ratio: float = 0.3,
    target_class_strategy: Literal["random", "fixed_first", "rotate"] = "random",
    class_config: ClassConfig | None = None,
    batch_size: int = 32,
    max_prompt_length: int = 2048,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Create DataLoader for mixed dataset.

    Args:
        data_path: Path to jsonl file
        tokenizer: Pre-trained tokenizer
        processor: VLM processor
        split: Which fold(s) to use for training (e.g., "0,1,2" or ["0", "1", "2"])
        val_splits: Which fold(s) to use for validation (default: auto-compute from train splits)
        num_images: Number of images per comparison
        single_image_ratio: Fraction of single-image samples
        target_class_strategy: How to select target class ("random", "fixed_first", "rotate")
        class_config: Optional ClassConfig for customization
        batch_size: Batch size
        max_prompt_length: Max prompt length
        shuffle: Whether to shuffle
        seed: Random seed

    Returns:
        train_dataloader, val_dataloader
    """
    from torchdata.stateful_dataloader import StatefulDataLoader
    from torch.utils.data import RandomSampler, SequentialSampler

    # Training dataset
    train_dataset = FlexibleMixedDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        processor=processor,
        split=split,
        mode="train",
        num_images=num_images,
        single_image_ratio=single_image_ratio,
        target_class_strategy=target_class_strategy,
        class_config=class_config,
        max_prompt_length=max_prompt_length,
        seed=seed,
    )

    if shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    # Validation dataset (single-image only)
    # Auto-compute val_splits if not provided
    if val_splits is None:
        # Infer all available splits (0-4) and use ones not in train
        if isinstance(split, str):
            train_split_set = set(s.strip() for s in split.split(","))
        else:
            train_split_set = set(str(s) for s in split)
        all_splits = {str(i) for i in range(5)}
        val_split_set = all_splits - train_split_set
        val_splits = list(val_split_set) if val_split_set else ["0"]

    val_dataset = FlexibleMixedDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        processor=processor,
        split=val_splits,
        mode="val",
        num_images=num_images,
        single_image_ratio=1.0,
        class_config=class_config,
        max_prompt_length=max_prompt_length,
        seed=seed,
    )

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    print(f"Train dataloader size: {len(train_dataloader)}")
    print(f"Val dataloader size (splits={val_splits}, single-image only): {len(val_dataloader)}")

    return train_dataloader, val_dataloader


# Backward compatibility
create_comparative_dataloader = create_mixed_dataloader
