#!/usr/bin/env python3
"""
Merge multiple JSONL files and shuffle rows.

Typical use (your case):
  python EasyR1/scripts/OminiExpert/merge_shuffle_jsonl.py \\
    --input EasyR1/data/OminiMedExpert/isic_disease_diagnosis_v1_0.05/comparative/train_b_tasks.jsonl \\
    --input EasyR1/data/OminiMedExpert/isic_disease_diagnosis_v1_0.05/train_fewshot_0.5.jsonl \\
    --output EasyR1/data/OminiMedExpert/isic_disease_diagnosis_v1_0.05/train_btasks+fewshot0.5_shuf_seed42.jsonl \\
    --seed 42

Notes:
  - Keeps all rows by default (no dedup).
  - Optionally deduplicate by a JSON key like "question_id" via --dedupe-key.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON in {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise SystemExit(f"Expected JSON object in {path}:{line_no}, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _default_output(inputs: list[Path], seed: int) -> Path:
    parent = inputs[0].parent
    stems = [p.stem for p in inputs]
    safe = "+".join(s.replace(" ", "_") for s in stems)
    return parent / f"{safe}_shuf_seed{seed}.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", type=Path, required=True, help="Input JSONL. Repeatable.")
    ap.add_argument("--output", type=Path, default=None, help="Output JSONL path.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedupe-key", type=str, default=None, help='Optional JSON key to dedupe by (e.g. "question_id").')
    ap.add_argument("--no-shuffle", action="store_true", help="Disable shuffle (default: shuffle enabled).")
    ap.add_argument("--summary", type=Path, default=None, help="Optional JSON summary path.")
    args = ap.parse_args()

    inputs: list[Path] = args.input
    for p in inputs:
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")

    out_path: Path = args.output if args.output is not None else _default_output(inputs, seed=args.seed)

    all_rows: list[dict[str, Any]] = []
    per_input_counts: dict[str, int] = {}
    for p in inputs:
        rows = _read_jsonl(p)
        per_input_counts[str(p)] = len(rows)
        all_rows.extend(rows)

    before = len(all_rows)
    dedupe_dropped = 0
    if args.dedupe_key:
        key = args.dedupe_key
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for row in all_rows:
            v = row.get(key, None)
            if v is None:
                # Treat missing key as unique.
                deduped.append(row)
                continue
            hv = json.dumps(v, ensure_ascii=False, sort_keys=True)
            if hv in seen:
                dedupe_dropped += 1
                continue
            seen.add(hv)
            deduped.append(row)
        all_rows = deduped

    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(all_rows)

    _write_jsonl(out_path, all_rows)

    # Lightweight summary for reproducibility/debugging.
    question_type_counter: Counter[str] = Counter()
    dataset_counter: Counter[str] = Counter()
    for row in all_rows:
        gt = row.get("ground_truth")
        if isinstance(gt, dict):
            qt = gt.get("question_type")
            ds = gt.get("dataset")
            if isinstance(qt, str):
                question_type_counter[qt] += 1
            if isinstance(ds, str):
                dataset_counter[ds] += 1

    summary = {
        "inputs": [str(p) for p in inputs],
        "output": str(out_path),
        "seed": args.seed,
        "shuffle": not args.no_shuffle,
        "dedupe_key": args.dedupe_key,
        "rows_before": before,
        "rows_after": len(all_rows),
        "dedupe_dropped": dedupe_dropped,
        "per_input_counts": per_input_counts,
        "question_type_counts": dict(question_type_counter),
        "dataset_counts": dict(dataset_counter),
    }

    summary_path = args.summary if args.summary is not None else out_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path} ({len(all_rows)} rows)")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

