#!/usr/bin/env python3
"""
Generate verifiable multi-image contrast tasks for OmniMedVQA-ISIC.

Motivation:
  - Reduce reliance on text priors / shortcut label mapping by turning a
    single-image 4-way MCQ dataset into multi-image index-selection tasks.
  - Keep rewards fully verifiable (exact match on a short discrete answer).

Input schema (JSONL):
  {
    "prompt": "...",                 # usually question + options
    "images": [".../xxx.jpg"],       # single image path
    "answer": {"label": "...", ...}  # ground-truth label text
  }

Output schema (JSONL):
  {
    "prompt": "... <image> ...",     # contains N occurrences of "<image>"
    "images": ["..."],              # N image paths
    "answer": { ... }               # includes "task_type" and "correct_answer"
  }

This script intentionally makes prompts self-contained (it injects "<image>"
tokens directly), so you can set `format_prompt=None` in training configs.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal


@dataclass(frozen=True)
class IsicItem:
    image: str
    label: str
    prompt: str
    answer: dict

    @property
    def question_id(self) -> str:
        qid = self.answer.get("question_id")
        return str(qid) if qid is not None else ""


TaskName = Literal[
    "single_vqa",
    "target_search",
    "odd_one_out",
    "label_corruption",
    "exemplar_match",
    "same_different",
]


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def load_isic_items(path: Path) -> list[IsicItem]:
    items: list[IsicItem] = []
    for obj in _read_jsonl(path):
        images = obj.get("images") or []
        if not isinstance(images, list) or not images or not isinstance(images[0], str):
            continue

        answer = obj.get("answer") or {}
        if not isinstance(answer, dict):
            continue

        label = answer.get("label")
        if label is None:
            continue

        prompt = obj.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue

        items.append(
            IsicItem(
                image=images[0],
                label=str(label).strip(),
                prompt=prompt.strip(),
                answer=answer,
            )
        )
    if not items:
        raise ValueError(f"No valid items loaded from {path}")
    return items


def _letters(n: int) -> list[str]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if n > 26:
        raise ValueError("n too large for A-Z labeling (max 26)")
    return [chr(ord("A") + i) for i in range(n)]


def _letter_options(letters: list[str]) -> str:
    if len(letters) == 1:
        return letters[0]
    if len(letters) == 2:
        return f"{letters[0]} or {letters[1]}"
    return ", ".join(letters[:-1]) + f", or {letters[-1]}"


def _render_images(letters: list[str]) -> str:
    return "\n".join([f"({l}) Image {l}: <image>" for l in letters])


def _render_labeled_images(letters: list[str], shown_labels: list[str]) -> str:
    if len(letters) != len(shown_labels):
        raise ValueError("letters and shown_labels must have same length")
    lines: list[str] = []
    for ltr, lbl in zip(letters, shown_labels, strict=True):
        lines.append(f"({ltr}) Claimed label: {lbl}\n({ltr}) Image {ltr}: <image>")
    return "\n".join(lines)


def _sample_distinct(rng: random.Random, items: list[IsicItem], k: int) -> list[IsicItem]:
    if k <= 0:
        return []
    if k == 1:
        return [rng.choice(items)]
    if len(items) >= k:
        return rng.sample(items, k)
    # Fall back to sampling with replacement if the class is too small.
    return [rng.choice(items) for _ in range(k)]


def gen_single_vqa(rng: random.Random, items: list[IsicItem]) -> dict:
    it = rng.choice(items)
    prompt = (
        "You are a medical VQA assistant. Read the image carefully and answer the question.\n\n"
        "<image>\n"
        f"{it.prompt}\n\n"
        "Follow the exact output format:\n"
        "<answer> exact option text </answer>"
    )
    answer = dict(it.answer)
    answer["task_type"] = "single_vqa"
    return {
        "prompt": prompt,
        "images": [it.image],
        "answer": answer,
    }


def gen_target_search(
    rng: random.Random, by_label: dict[str, list[IsicItem]], k: int, *, balance_labels: bool = True
) -> dict:
    labels = sorted(by_label.keys())
    target_label = rng.choice(labels) if balance_labels else rng.choice([it.label for its in by_label.values() for it in its])

    target_item = rng.choice(by_label[target_label])

    other_labels = [l for l in labels if l != target_label]
    if not other_labels:
        raise ValueError("Need at least 2 labels for target_search")

    distractor_labels = (
        rng.sample(other_labels, k - 1) if (k - 1) <= len(other_labels) else [rng.choice(other_labels) for _ in range(k - 1)]
    )
    distractors = [rng.choice(by_label[l]) for l in distractor_labels]

    chosen: list[IsicItem] = [target_item, *distractors]
    rng.shuffle(chosen)

    letters = _letters(k)
    correct_idx = next(i for i, it in enumerate(chosen) if it.image == target_item.image)
    correct_answer = letters[correct_idx]

    prompt = (
        "You are a medical VQA assistant.\n\n"
        f"Below are {k} dermoscopy images, labeled {_letter_options(letters)}.\n\n"
        f"{_render_images(letters)}\n\n"
        f"Task: Exactly one image shows **{target_label}**. Which image is it?\n\n"
        "Answer with only one letter.\n"
        f"<answer> {_letter_options(letters)} </answer>"
    )

    return {
        "prompt": prompt,
        "images": [it.image for it in chosen],
        "answer": {
            "task_type": "target_search",
            "target_label": target_label,
            "correct_answer": correct_answer,
            "labels": [it.label for it in chosen],
            "question_ids": [it.question_id for it in chosen],
        },
    }


def gen_odd_one_out(
    rng: random.Random, by_label: dict[str, list[IsicItem]], k: int, *, balance_labels: bool = True
) -> dict:
    labels = sorted(by_label.keys())
    majority_label = rng.choice(labels) if balance_labels else rng.choice([it.label for its in by_label.values() for it in its])
    other_labels = [l for l in labels if l != majority_label]
    if not other_labels:
        raise ValueError("Need at least 2 labels for odd_one_out")

    odd_label = rng.choice(other_labels)

    majority_items = _sample_distinct(rng, by_label[majority_label], k - 1)
    odd_item = rng.choice(by_label[odd_label])

    chosen: list[IsicItem] = [*majority_items, odd_item]
    rng.shuffle(chosen)

    letters = _letters(k)
    correct_idx = next(i for i, it in enumerate(chosen) if it.image == odd_item.image)
    correct_answer = letters[correct_idx]

    prompt = (
        "You are a medical VQA assistant.\n\n"
        f"Below are {k} dermoscopy images, labeled {_letter_options(letters)}.\n\n"
        f"{_render_images(letters)}\n\n"
        f"Task: Exactly {k-1} images depict the same condition and 1 image depicts a different condition.\n"
        "Which image is the odd one out?\n\n"
        "Answer with only one letter.\n"
        f"<answer> {_letter_options(letters)} </answer>"
    )

    return {
        "prompt": prompt,
        "images": [it.image for it in chosen],
        "answer": {
            "task_type": "odd_one_out",
            "correct_answer": correct_answer,
            "majority_label": majority_label,
            "odd_label": odd_label,
            "labels": [it.label for it in chosen],
            "question_ids": [it.question_id for it in chosen],
        },
    }


def gen_label_corruption(
    rng: random.Random, by_label: dict[str, list[IsicItem]], k: int, *, enforce_distinct_labels: bool = True
) -> dict:
    labels = sorted(by_label.keys())
    if enforce_distinct_labels and k >= len(labels):
        raise ValueError(
            f"label_corruption with distinct shown labels requires k <= {len(labels)-1} (got k={k})."
        )

    # Prefer unique true labels to avoid text-only shortcuts.
    true_labels = rng.sample(labels, k) if enforce_distinct_labels else [rng.choice(labels) for _ in range(k)]
    chosen: list[IsicItem] = [rng.choice(by_label[lbl]) for lbl in true_labels]

    corrupt_idx = rng.randrange(k)
    disallowed = set(true_labels)
    disallowed.remove(true_labels[corrupt_idx])

    candidate_corrupt_labels = [l for l in labels if l != true_labels[corrupt_idx] and l not in disallowed]
    if not candidate_corrupt_labels:
        # Fall back (may introduce duplicate shown labels if enforce_distinct_labels=False)
        candidate_corrupt_labels = [l for l in labels if l != true_labels[corrupt_idx]]

    corrupt_to = rng.choice(candidate_corrupt_labels)
    shown_labels = list(true_labels)
    shown_labels[corrupt_idx] = corrupt_to

    letters = _letters(k)
    correct_answer = letters[corrupt_idx]

    prompt = (
        "You are a medical VQA assistant.\n\n"
        "Below are image-label pairs. Exactly one claimed label is incorrect.\n"
        "Identify which position has the corrupted (wrong) label.\n\n"
        f"Pairs are labeled {_letter_options(letters)}.\n\n"
        f"{_render_labeled_images(letters, shown_labels)}\n\n"
        "Answer with only one letter.\n"
        f"<answer> {_letter_options(letters)} </answer>"
    )

    return {
        "prompt": prompt,
        "images": [it.image for it in chosen],
        "answer": {
            "task_type": "label_corruption",
            "correct_answer": correct_answer,
            "true_labels": true_labels,
            "shown_labels": shown_labels,
            "corrupt_from": true_labels[corrupt_idx],
            "corrupt_to": corrupt_to,
            "question_ids": [it.question_id for it in chosen],
        },
    }


def gen_exemplar_match(
    rng: random.Random,
    by_label: dict[str, list[IsicItem]],
    num_candidates: int,
    *,
    balance_labels: bool = True,
) -> dict:
    if num_candidates < 1:
        raise ValueError("num_candidates must be >= 1")

    labels = sorted(by_label.keys())
    anchor_label = rng.choice(labels) if balance_labels else rng.choice([it.label for its in by_label.values() for it in its])

    anchor_pool = by_label[anchor_label]
    if len(anchor_pool) < 2:
        raise ValueError(f"Need >=2 samples for label {anchor_label!r} for exemplar_match")

    ref_item, pos_item = _sample_distinct(rng, anchor_pool, 2)

    negative_labels = [l for l in labels if l != anchor_label]
    if not negative_labels and num_candidates > 1:
        raise ValueError("Need at least 2 labels for exemplar_match with negatives")

    neg_labels = (
        rng.sample(negative_labels, num_candidates - 1)
        if (num_candidates - 1) <= len(negative_labels)
        else [rng.choice(negative_labels) for _ in range(num_candidates - 1)]
    )
    neg_items = [rng.choice(by_label[lbl]) for lbl in neg_labels]

    candidates: list[IsicItem] = [pos_item, *neg_items]
    rng.shuffle(candidates)

    # Total images = 1 reference + N candidates
    letters = _letters(1 + num_candidates)
    ref_letter = letters[0]
    cand_letters = letters[1:]

    correct_idx = next(i for i, it in enumerate(candidates) if it.image == pos_item.image)
    correct_answer = cand_letters[correct_idx]

    prompt = (
        "You are a medical VQA assistant.\n\n"
        f"Reference image ({ref_letter}): <image>\n\n"
        f"Candidate images, labeled {_letter_options(cand_letters)}:\n\n"
        + "\n".join([f"({ltr}) Candidate {ltr}: <image>" for ltr in cand_letters])
        + "\n\n"
        f"Task: Exactly one candidate depicts the same condition as the reference image {ref_letter}.\n"
        "Which candidate is it?\n\n"
        "Answer with only one letter.\n"
        f"<answer> {_letter_options(cand_letters)} </answer>"
    )

    return {
        "prompt": prompt,
        "images": [ref_item.image, *[it.image for it in candidates]],
        "answer": {
            "task_type": "exemplar_match",
            "correct_answer": correct_answer,
            "anchor_label": anchor_label,
            "labels": [ref_item.label, *[it.label for it in candidates]],
            "question_ids": [ref_item.question_id, *[it.question_id for it in candidates]],
        },
    }


def gen_same_different(
    rng: random.Random, by_label: dict[str, list[IsicItem]], *, same_prob: float = 0.5, balance_labels: bool = True
) -> dict:
    labels = sorted(by_label.keys())
    want_same = rng.random() < same_prob

    if want_same:
        lbl = rng.choice(labels) if balance_labels else rng.choice([it.label for its in by_label.values() for it in its])
        pool = by_label[lbl]
        a, b = _sample_distinct(rng, pool, 2)
        correct = "same"
        used = [a, b]
    else:
        a_lbl, b_lbl = rng.sample(labels, 2) if len(labels) >= 2 else (labels[0], labels[0])
        a = rng.choice(by_label[a_lbl])
        b = rng.choice(by_label[b_lbl])
        correct = "different" if a_lbl != b_lbl else "same"
        used = [a, b]

    prompt = (
        "You are a medical VQA assistant.\n\n"
        "(A) Image A: <image>\n"
        "(B) Image B: <image>\n\n"
        "Task: Do these two images depict the same condition?\n\n"
        "Answer with exactly one token: 'same' or 'different'.\n"
        "<answer> same or different </answer>"
    )
    return {
        "prompt": prompt,
        "images": [used[0].image, used[1].image],
        "answer": {
            "task_type": "same_different",
            "correct_answer": correct,
            "labels": [used[0].label, used[1].label],
            "question_ids": [used[0].question_id, used[1].question_id],
        },
    }


def _parse_task_specs(raw_specs: list[str], default_size: int) -> dict[TaskName, int]:
    if not raw_specs:
        return {
            "single_vqa": default_size,
            "target_search": default_size,
            "odd_one_out": default_size,
            "label_corruption": default_size,
            "exemplar_match": default_size,
        }

    out: dict[TaskName, int] = {}
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"Invalid --task {raw!r}. Expected TASK=NUM.")
        name, num = raw.split("=", 1)
        name = name.strip()
        num = num.strip()
        if not name or not num:
            raise ValueError(f"Invalid --task {raw!r}. Expected TASK=NUM.")
        try:
            count = int(num)
        except ValueError as exc:
            raise ValueError(f"Invalid --task {raw!r}. NUM must be an int.") from exc
        if count < 0:
            raise ValueError(f"Invalid --task {raw!r}. NUM must be >= 0.")
        out[name] = count  # type: ignore[assignment]
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ISIC verifiable contrast tasks (JSONL)")
    parser.add_argument("--input", type=Path, required=True, help="Input ISIC JSONL (single-image OmniMedVQA format)")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=4, help="Number of images per task (target_search/odd_one_out/label_corruption)")
    parser.add_argument("--exemplar_candidates", type=int, default=3, help="Num candidates (plus 1 reference) for exemplar_match")
    parser.add_argument("--same_prob", type=float, default=0.5, help="P(same) for same_different")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help=(
            "Task spec TASK=NUM. Available: single_vqa,target_search,odd_one_out,label_corruption,exemplar_match,same_different. "
            "Repeatable."
        ),
    )
    parser.add_argument("--no_shuffle", action="store_true", help="Do not shuffle the final mixed dataset")
    parser.add_argument("--show_stats", action="store_true", help="Print label/task stats to stdout")

    args = parser.parse_args()

    rng = random.Random(args.seed)
    items = load_isic_items(args.input)

    by_label: dict[str, list[IsicItem]] = defaultdict(list)
    for it in items:
        by_label[it.label].append(it)

    labels = sorted(by_label.keys())
    if args.show_stats:
        ctr = Counter([it.label for it in items])
        print(f"Loaded {len(items)} items from {args.input}")
        print(f"Labels ({len(labels)}): {labels}")
        print("Top label counts:", ctr.most_common(20))

    if args.k < 2:
        raise ValueError("--k must be >= 2 for multi-image tasks")

    task_counts = _parse_task_specs(args.task, default_size=len(items))

    generators: dict[TaskName, Callable[[], dict]] = {
        "single_vqa": lambda: gen_single_vqa(rng, items),
        "target_search": lambda: gen_target_search(rng, by_label, args.k),
        "odd_one_out": lambda: gen_odd_one_out(rng, by_label, args.k),
        "label_corruption": lambda: gen_label_corruption(rng, by_label, args.k),
        "exemplar_match": lambda: gen_exemplar_match(rng, by_label, args.exemplar_candidates),
        "same_different": lambda: gen_same_different(rng, by_label, same_prob=args.same_prob),
    }

    rows: list[dict] = []
    for task_name, count in task_counts.items():
        if count == 0:
            continue
        if task_name not in generators:
            raise ValueError(f"Unknown task {task_name!r}. Available: {', '.join(sorted(generators.keys()))}")
        for _ in range(count):
            rows.append(generators[task_name]())

    if not args.no_shuffle:
        rng.shuffle(rows)

    if args.show_stats:
        task_ctr = Counter([r.get("answer", {}).get("task_type", "unknown") for r in rows])
        print("Generated task counts:", dict(task_ctr))
        print(f"Writing {len(rows)} rows to {args.output}")

    _write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()

