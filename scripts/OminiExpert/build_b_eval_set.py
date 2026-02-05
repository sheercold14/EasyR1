#!/usr/bin/env python3
"""
Generate ONLY multi-image comparative tasks (B1-B7) from a base single-image JSONL.

This is meant for offline evaluation:
- Keep your original single-image eval file unchanged (use it as-is).
- Generate a separate multi-image eval file with a specified number of tasks per B1..B7.

We reuse the exact generation logic from `omnimed_expert.py` (`generate_b_tasks()`), so the
prompt/answer schema matches training.

Example:
  python EasyR1/scripts/OminiExpert/build_b_eval_set.py \
    --input  EasyR1/data/OminiMedExpert/isic_disease_diagnosis_v0_0.05/val.jsonl \
    --output EasyR1/data/OminiMedExpert/isic_disease_diagnosis_v0_0.05/val_btasks_210.jsonl \
    --task B2=30 --task B3=30 --task B4=30 --task B5=30 --task B6=30 --task B7=30 \
    --k 4 --b7-nway 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Reuse existing builder logic.
from omnimed_expert import _read_jsonl, _write_jsonl, generate_b_tasks  # type: ignore


def _parse_task_specs(raw_specs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"Invalid --task {raw!r}. Expected like B6=100.")
        name, num = raw.split("=", 1)
        name = name.strip().upper()
        num = num.strip()
        if name not in {"B1", "B2", "B3", "B4", "B5", "B6", "B7"}:
            raise ValueError(f"Unknown task {name!r}. Expected one of B1..B7.")
        count = int(num)
        if count < 0:
            raise ValueError(f"Task count must be >= 0 (got {count})")
        out[name] = count
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Base single-image JSONL (mcq_letter).")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL containing only B tasks.")
    ap.add_argument("--summary", type=Path, default=None, help="Optional JSON summary path.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
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
    ap.add_argument("--task", action="append", default=[], help="Task count like B6=100. Repeatable.")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--b4-candidates", type=int, default=3)
    ap.add_argument("--b5-same-prob", type=float, default=0.5)
    ap.add_argument("--b7-nway", type=int, default=5)
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    base_rows = list(_read_jsonl(args.input))
    if not base_rows:
        raise SystemExit(f"Empty input: {args.input}")

    task_counts = _parse_task_specs(args.task)
    if not task_counts:
        raise SystemExit("No --task provided. Example: --task B6=200")

    out_rows, info = generate_b_tasks(
        base_rows,
        seed=args.seed,
        label_space_by=args.label_space_by,  # type: ignore[arg-type]
        task_counts=task_counts,  # type: ignore[arg-type]
        k=args.k,
        b4_candidates=args.b4_candidates,
        b5_same_prob=args.b5_same_prob,
        b7_nway=args.b7_nway,
        shuffle=args.shuffle,
    )

    _write_jsonl(args.output, out_rows)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "requested": task_counts,
        "generated_rows": len(out_rows),
        "generation_info": info,
    }
    summary_path = args.summary if args.summary is not None else args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

