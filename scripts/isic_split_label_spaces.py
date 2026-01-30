#!/usr/bin/env python3
"""
Split OmniMedVQA-ISIC JSONL into separate label spaces.

ISIC in this repo mixes two different tasks:
  1) Malignancy (2-option): labels {"Benign image.", "Malignant"}
  2) Diagnosis (4-option): 8 disease names (e.g., "Melanoma", "Basal cell carcinoma", ...)

This script writes:
  - <out_root>/comparative/{train,val}.jsonl  (diagnosis / disease-name subset)
  - <out_root>/malignancy/{train,val}.jsonl  (2-option benign/malignant subset)
  - <out_root>/{comparative,malignancy}/summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


MALIGNANCY_LABELS = {"Benign image.", "Malignant"}
OPTION_KEYS = ("option_A", "option_B", "option_C", "option_D")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def num_options(answer: dict) -> int:
    n = 0
    for k in OPTION_KEYS:
        v = answer.get(k, None)
        if v is None:
            continue
        if str(v).strip() == "":
            continue
        n += 1
    return n


def bucket_for(row: dict) -> str:
    answer = row.get("answer", {})
    if not isinstance(answer, dict):
        return "unknown"
    label = str(answer.get("label", "")).strip()
    if label in MALIGNANCY_LABELS:
        return "malignancy"
    return "comparative"


def summarize(rows: list[dict]) -> dict:
    labels = Counter()
    opt_counts = Counter()
    datasets = Counter()
    for r in rows:
        a = r.get("answer", {})
        if not isinstance(a, dict):
            continue
        labels[str(a.get("label", "")).strip()] += 1
        opt_counts[num_options(a)] += 1
        datasets[str(a.get("dataset", "")).strip()] += 1
    return {
        "total": len(rows),
        "labels": dict(labels.most_common()),
        "num_options": dict(opt_counts.most_common()),
        "datasets": dict(datasets.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split OmniMedVQA-ISIC into diagnosis vs malignancy subsets")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument(
        "--out_root",
        type=Path,
        required=True,
        help="Directory containing ISIC (writes comparative/ and malignancy/ under it)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_comp = args.out_root / "comparative"
    out_mal = args.out_root / "malignancy"

    def process_split(src: Path) -> tuple[list[dict], list[dict], dict]:
        comp: list[dict] = []
        mal: list[dict] = []
        unknown = Counter()
        for row in read_jsonl(src):
            b = bucket_for(row)
            if b == "comparative":
                comp.append(row)
            elif b == "malignancy":
                mal.append(row)
            else:
                unknown[b] += 1
        return comp, mal, dict(unknown)

    comp_train, mal_train, unk_train = process_split(args.train)
    comp_val, mal_val, unk_val = process_split(args.val)

    targets = [
        (out_comp / "train.jsonl", comp_train),
        (out_comp / "val.jsonl", comp_val),
        (out_mal / "train.jsonl", mal_train),
        (out_mal / "val.jsonl", mal_val),
    ]
    if not args.overwrite:
        for p, _ in targets:
            if p.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {p} (pass --overwrite)")

    for p, rows in targets:
        write_jsonl(p, rows)

    (out_comp / "summary.json").write_text(
        json.dumps(
            {
                "train": summarize(comp_train),
                "val": summarize(comp_val),
                "unknown_bucket_counts": {"train": unk_train, "val": unk_val},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_mal / "summary.json").write_text(
        json.dumps(
            {
                "train": summarize(mal_train),
                "val": summarize(mal_val),
                "unknown_bucket_counts": {"train": unk_train, "val": unk_val},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Wrote:")
    print(f"  {out_comp / 'train.jsonl'} ({len(comp_train)})")
    print(f"  {out_comp / 'val.jsonl'}   ({len(comp_val)})")
    print(f"  {out_mal / 'train.jsonl'} ({len(mal_train)})")
    print(f"  {out_mal / 'val.jsonl'}   ({len(mal_val)})")


if __name__ == "__main__":
    main()

