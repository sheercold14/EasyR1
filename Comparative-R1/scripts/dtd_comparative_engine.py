#!/usr/bin/env python3
"""
DTD multi-image comparative data engine (B1-B7).

Input format (JSONL):
  {"image": "banded/banded_0060.jpg", "label": "banded", "split": "test", "problem": "..."}

Output format (JSONL): EasyR1 comparative rows
  {"prompt": "...", "images": [...], "answer": {...}}

Example:
  python3 Comparative-R1/scripts/dtd_comparative_engine.py \
    --input /data/shichao/datasets_b2n/DescribableTextures_b2n_base_test.jsonl \
    --output /data/shichao/EasyR1/data/CLS/DTD/B-tasks/DescribableTextures_b2n_B700.jsonl \
    --split test \
    --prompt-style direct \
    --num-per-task 100 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from ominimed_expertv2 import generate_b_tasks  # type: ignore

TASKS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7")


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield obj


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _first_line(text: str) -> str:
    s = text.strip()
    if not s:
        return "What type of texture is in the photo?"
    return s.splitlines()[0].strip() or "What type of texture is in the photo?"


def _normalize_path(image: str, image_prefix: str) -> str:
    p = image.strip()
    if not image_prefix:
        return p
    return str(Path(image_prefix) / p)


def _parse_task_specs(raw_specs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"Invalid --task {raw!r}. Expected like B1=1000.")
        name, num = raw.split("=", 1)
        name = name.strip().upper()
        num = num.strip()
        if name not in TASKS:
            raise ValueError(f"Unknown task {name!r}. Expected one of B1..B7.")
        count = int(num)
        if count < 0:
            raise ValueError(f"Task count must be >= 0 (got {count})")
        out[name] = count
    return out


def _uniform_task_counts(tasks_csv: str, num_per_task: int) -> dict[str, int]:
    if num_per_task < 0:
        raise ValueError("--num-per-task must be >= 0")
    tasks = [t.strip().upper() for t in tasks_csv.split(",") if t.strip()]
    if not tasks:
        tasks = list(TASKS)
    out: dict[str, int] = {}
    for t in tasks:
        if t not in TASKS:
            raise ValueError(f"Unknown task in --tasks: {t!r}")
        out[t] = num_per_task
    return out


def _dtd_to_base_row(
    item: dict,
    *,
    image_prefix: str,
    dataset_name: str,
    default_question_type: str,
    default_modality_type: str,
    fallback_question: str,
) -> dict | None:
    image = item.get("image")
    label = item.get("label")
    if not isinstance(image, str) or not image.strip():
        return None
    if not isinstance(label, str) or not label.strip():
        return None

    problem = item.get("problem")
    if isinstance(problem, str) and problem.strip():
        prompt = problem.strip()
        question = _first_line(problem)
    else:
        prompt = fallback_question
        question = fallback_question

    split = item.get("split")
    split_text = str(split).strip() if split is not None else ""
    qid = item.get("id") or item.get("question_id") or f"{dataset_name}:{image}"

    return {
        "prompt": prompt,
        "images": [_normalize_path(image, image_prefix)],
        "answer": {
            "label": label.strip(),
            "question_id": str(qid),
            "dataset": dataset_name,
            "question_type": default_question_type,
            "question": question,
            "modality_type": default_modality_type,
            "source_split": split_text,
            # Keep options empty so option_count=0; DTD source has no MCQ options.
            "option_A": None,
            "option_B": None,
            "option_C": None,
            "option_D": None,
            "task_type": "single_dtd",
        },
    }


def build_rows_from_dtd(
    input_path: Path,
    *,
    split: str,
    max_rows: int | None,
    seed: int,
    image_prefix: str,
    dataset_name: str,
    question_type: str,
    modality_type: str,
    fallback_question: str,
):
    raw = list(_read_jsonl(input_path))

    if split != "all":
        raw = [x for x in raw if str(x.get("split", "")).strip() == split]

    rows = []
    skipped = 0
    for x in raw:
        row = _dtd_to_base_row(
            x,
            image_prefix=image_prefix,
            dataset_name=dataset_name,
            default_question_type=question_type,
            default_modality_type=modality_type,
            fallback_question=fallback_question,
        )
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    if max_rows is not None and max_rows >= 0 and len(rows) > max_rows:
        rng = random.Random(seed)
        rows = rng.sample(rows, max_rows)

    label_counter = Counter(r.get("answer", {}).get("label", "") for r in rows)
    info = {
        "input": str(input_path),
        "rows_after_split": len(raw),
        "rows_valid": len(rows),
        "rows_skipped_invalid": skipped,
        "labels": len(label_counter),
        "labels_top50": dict(label_counter.most_common(50)),
    }
    return rows, info


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DTD B1-B7 comparative data generator")
    p.add_argument("--input", type=Path, required=True, help="DTD JSONL input path")
    p.add_argument("--output", type=Path, required=True, help="Output comparative JSONL path")
    p.add_argument("--summary", type=Path, default=None, help="Optional summary JSON path")

    p.add_argument("--split", type=str, default="all", help="Use only this split (e.g. train/test), or all")
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap for source rows after split filter")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--image-prefix", type=str, default="", help="Prefix joined with each image path")
    p.add_argument("--dataset-name", type=str, default="DTD", help="dataset field written into answer")
    p.add_argument("--question-type", type=str, default="texture_classification")
    p.add_argument("--modality-type", type=str, default="natural_image")
    p.add_argument("--fallback-question", type=str, default="What type of texture is in the photo?")

    p.add_argument(
        "--label-space-by",
        type=str,
        default="question_type+optioncount",
        choices=[
            "question_type",
            "question_type+modality",
            "question_type+optioncount",
            "question_type+modality+optioncount",
        ],
    )
    p.add_argument("--task", action="append", default=[], help="Task count like B1=1000 (repeatable)")
    p.add_argument("--num-per-task", type=int, default=None, help="If set and --task is empty, use this count for each task")
    p.add_argument("--tasks", type=str, default="B1,B2,B3,B4,B5,B6,B7", help="Used with --num-per-task")

    p.add_argument("--k", type=int, default=4, help="K images for B1/B2/B3/B6")
    p.add_argument("--b4-candidates", type=int, default=3, help="Num candidates (excluding reference) for B4")
    p.add_argument("--b5-same-prob", type=float, default=0.5, help="P(same) for B5")
    p.add_argument("--b7-nway", type=int, default=5, help="N-way for B7 support-set")
    p.add_argument(
        "--prompt-style",
        type=str,
        default="stepwise",
        choices=["stepwise", "direct"],
        help="Prompt style for B tasks",
    )
    p.add_argument("--no-shuffle", action="store_true", help="Do not shuffle generated rows")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.task:
        task_counts = _parse_task_specs(args.task)
    elif args.num_per_task is not None:
        task_counts = _uniform_task_counts(args.tasks, args.num_per_task)
    else:
        raise SystemExit("Please provide --task B1=... (repeatable) or --num-per-task.")

    if args.k < 2:
        raise SystemExit("--k must be >= 2")
    if args.k > 26:
        raise SystemExit("--k must be <= 26")
    if (1 + args.b4_candidates) > 26:
        raise SystemExit("--b4-candidates too large (1+N <= 26)")
    if args.b7_nway > 26:
        raise SystemExit("--b7-nway must be <= 26")
    if not (0.0 <= args.b5_same_prob <= 1.0):
        raise SystemExit("--b5-same-prob must be in [0,1]")

    base_rows, source_info = build_rows_from_dtd(
        args.input,
        split=args.split,
        max_rows=args.max_rows,
        seed=args.seed,
        image_prefix=args.image_prefix,
        dataset_name=args.dataset_name,
        question_type=args.question_type,
        modality_type=args.modality_type,
        fallback_question=args.fallback_question,
    )
    if not base_rows:
        raise SystemExit("No valid source rows after filtering.")

    out_rows, gen_info = generate_b_tasks(
        base_rows,
        seed=args.seed,
        label_space_by=args.label_space_by,  # type: ignore[arg-type]
        task_counts=task_counts,  # type: ignore[arg-type]
        k=args.k,
        b4_candidates=args.b4_candidates,
        b5_same_prob=args.b5_same_prob,
        b7_nway=args.b7_nway,
        prompt_style=args.prompt_style,  # type: ignore[arg-type]
        shuffle=not args.no_shuffle,
    )

    _write_jsonl(args.output, out_rows)

    task_counter = Counter(r.get("answer", {}).get("task_type", "unknown") for r in out_rows)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "split": args.split,
        "task_counts_requested": task_counts,
        "generated_rows": len(out_rows),
        "task_type_counts": dict(task_counter),
        "source_info": source_info,
        "generation_info": gen_info,
    }

    summary_path = args.summary if args.summary is not None else args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(args.output), "generated": len(out_rows), "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
