#!/usr/bin/env python3
"""
Preprocessing script to generate structured data for comparative training.

This script:
1. Extracts only the filename from absolute image paths
2. Embeds all template variables directly in the prompt string (no separate fields)
3. Outputs clean JSONL with fully rendered prompts

The jinja template only needs to output {{ content }} since all variables
are embedded during data generation.
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal


def load_data(data_path: str) -> tuple[dict[str, list], list]:
    """Load all.jsonl and split by class."""
    samples_by_split_class = defaultdict(lambda: defaultdict(list))
    all_samples = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            split = str(data.get("split", "0"))
            label = data.get("answer", {}).get("label", "unknown")
            image = data.get("images", [None])[0]

            if label not in ["benign", "malignant"]:
                continue

            # Extract only the filename from the absolute path
            if image:
                filename = os.path.basename(image)
            else:
                filename = None

            sample = {
                "images": [filename],
                "label": label,
                "split": split,
            }
            samples_by_split_class[split][label].append(sample)
            all_samples.append(sample)

    return samples_by_split_class, all_samples


def generate_single_sample(sample: dict) -> dict:
    """Generate single-image sample with fully rendered prompt."""
    # Get the filename
    filename = sample["images"][0] if sample.get("images") else None

    # Generate fully rendered prompt (all variables embedded)
    prompt = """You are a professional pathology assistant for thyroid OCT-embedded frozen blocks.

<image>
Analyze this image and classify it.

Provide your analysis:
1. Describe the key features (margin, shape, echogenicity, etc.)
2. Explain what these features indicate
3. Conclude with your diagnosis

Follow the exact output format:

<thinking>
[Your analysis]
</thinking>

<answer> benign or malignant </answer>"""

    return {
        "prompt": prompt,
        "images": [filename],  # Only filename, not full path
        "answer": {"label": sample["label"]},
        "split": sample["split"],
    }


def generate_comparative_sample(
    benign_samples: list,
    malignant_samples: list,
    num_images: int,
    target_class: Literal["benign", "malignant"],
    split: str,
) -> dict:
    """Generate multi-image comparative sample with fully rendered prompt."""
    # Always 1 target + (num_images - 1) distractors
    target_samples = benign_samples if target_class == "benign" else malignant_samples
    distractor_samples = malignant_samples if target_class == "benign" else benign_samples

    # Sample 1 target
    target = random.choice(target_samples)

    # Sample distractors
    distractors = []
    for i in range(num_images - 1):
        distractor = random.choice(distractor_samples)
        distractors.append(distractor)

    # Combine and shuffle
    all_samples = [target] + distractors
    random.shuffle(all_samples)

    # Find target position
    target_idx = all_samples.index(target)
    correct_answer = chr(ord("A") + target_idx)

    # Generate labels string
    labels = [s["label"] for s in all_samples]
    letters = [chr(ord("A") + i) for i in range(num_images)]
    labels_str = ", ".join([f"({letters[i]}) {labels[i]}" for i in range(num_images)])

    # Build image placeholders
    image_placeholders = "\n".join([f"({letters[i]}) Image {letters[i]}: <image>" for i in range(num_images)])

    # Generate fully rendered prompt (all variables embedded)
    prompt = f"""You are a professional pathology assistant for thyroid OCT-embedded frozen blocks.

Below are {num_images} images, labeled {labels_str}.

{image_placeholders}

Carefully compare these images. Your task is to identify which image shows a **{target_class}** case.

Provide your analysis:
1. Describe the key features of each image (margin, shape, echogenicity, etc.)
2. Compare the differences between images
3. Explain which features indicate different characteristics
4. Conclude with your answer

Follow the exact output format:

<thinking>
[Your comparative analysis]
</thinking>

<answer> {' or '.join(letters)} </answer>"""

    # Get filenames only (not full paths)
    filenames = [s["images"][0] for s in all_samples]

    return {
        "prompt": prompt,
        "images": filenames,  # Only filenames, not full paths
        "labels": labels,
        "target_class": target_class,
        "correct_answer": correct_answer,
        "answer": {"correct_answer": correct_answer, "target_class": target_class},
        "split": split,
    }


def generate_dataset(
    data_path: str,
    train_splits: list[str],
    val_splits: list[str],
    num_images: int,
    target_class_strategy: Literal["random", "fixed_first"] = "random",
    num_comparative_samples: int = 1000,
    seed: int = 42,
) -> tuple[list, list]:
    """Generate train and val datasets with fully rendered prompts."""
    random.seed(seed)

    # Load data
    samples_by_split_class, _ = load_data(data_path)

    print(f"Loaded data from {data_path}")
    for split in train_splits + val_splits:
        for label in ["benign", "malignant"]:
            count = len(samples_by_split_class.get(split, {}).get(label, []))
            print(f"  Split {split} - {label}: {count} samples")

    # Generate training samples
    train_samples = []
    print(f"\nGenerating training data...")

    for split in train_splits:
        print(f"  Split {split}:")

        split_data = samples_by_split_class.get(split, {})
        benign_samples = split_data.get("benign", [])
        malignant_samples = split_data.get("malignant", [])

        # Single-image samples (all samples)
        for sample in benign_samples + malignant_samples:
            train_samples.append(generate_single_sample(sample))
        print(f"    Single-image: {len(benign_samples) + len(malignant_samples)} samples")

        # Multi-image comparative samples
        num_comp = num_comparative_samples
        for _ in range(num_comp):
            if len(benign_samples) == 0 or len(malignant_samples) == 0:
                continue

            if target_class_strategy == "random":
                target_class = random.choice(["benign", "malignant"])
            else:
                target_class = "benign"

            train_samples.append(generate_comparative_sample(
                benign_samples, malignant_samples, num_images, target_class, split
            ))
        print(f"    Comparative: {num_comp} samples")

    # Shuffle training data
    random.shuffle(train_samples)

    # Generate validation samples (single-image only)
    val_samples = []
    print(f"\nGenerating validation data (single-image only)...")

    for split in val_splits:
        print(f"  Split {split}:")

        split_data = samples_by_split_class.get(split, {})
        for label in ["benign", "malignant"]:
            for sample in split_data.get(label, []):
                val_samples.append(generate_single_sample(sample))
        print(f"    Total: {len(split_data.get('benign', [])) + len(split_data.get('malignant', []))} samples")

    return train_samples, val_samples


def save_dataset(samples: list, output_path: str):
    """Save dataset to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"  Saved {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate structured data for comparative training")
    parser.add_argument("--data_path", type=str, default="/data/shichao/EasyR1/data/thyroid/all.jsonl")
    parser.add_argument("--output_dir", type=str, default="/data/shichao/EasyR1/data/thyroid")
    parser.add_argument("--train_splits", type=str, default="0,1,2")
    parser.add_argument("--val_splits", type=str, default="3,4")
    parser.add_argument("--num_images", type=int, default=2)
    parser.add_argument("--target_class_strategy", type=str, default="random", choices=["random", "fixed_first"])
    parser.add_argument("--num_comparative_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_splits = args.train_splits.split(",")
    val_splits = args.val_splits.split(",")

    print("=" * 60)
    print("Comparative-R1 Data Generation")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Train splits: {train_splits}")
    print(f"Val splits: {val_splits}")
    print(f"Images per comparison: {args.num_images}")
    print(f"Target class strategy: {args.target_class_strategy}")
    print(f"Comparative samples per split: {args.num_comparative_samples}")
    print("=" * 60)

    # Generate datasets
    train_samples, val_samples = generate_dataset(
        data_path=args.data_path,
        train_splits=train_splits,
        val_splits=val_splits,
        num_images=args.num_images,
        target_class_strategy=args.target_class_strategy,
        num_comparative_samples=args.num_comparative_samples,
        seed=args.seed,
    )

    # Save datasets
    print(f"\nSaving datasets...")
    train_path = Path(args.output_dir) / "train.jsonl"
    val_path = Path(args.output_dir) / "val.jsonl"

    save_dataset(train_samples, train_path)
    save_dataset(val_samples, val_path)

    print("\nDone!")
    print("\nNote: All template variables are embedded in the prompt field.")
    print("The jinja template only needs to output {{ content }}")


if __name__ == "__main__":
    main()
