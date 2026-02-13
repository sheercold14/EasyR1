#!/usr/bin/env python3
"""
Mix one or more JSONL datasets into a single JSONL file.

Common use case (OminiMedExpert):
  - Mix single-image MCQ data (train/train_fewshot) with comparative B-tasks.

Examples
--------
1) Simple concatenate (then shuffle):
  python EasyR1/Comparative-R1/scripts/mix_jsonl_datasets.py \
    --inputs /data/shichao/EasyR1/data/offline_rft/isic/v1/4shot_nothinking.jsonl \
    --inputs /data/shichao/EasyR1/data/offline_rft/isic/v1/train_attr.jsonl \
    --inputs /data/shichao/EasyR1/data/offline_rft/isic/v1/train_text_rule.jsonl \
    --output /data/shichao/EasyR1/data/offline_rft/isic/v1/train_mix_all.jsonl \
    --shuffle --seed 42

2) Sample a fixed total with a ratio (60% single, 40% comparative):
  python EasyR1/Comparative-R1/scripts/mix_jsonl_datasets.py \
    --inputs single.jsonl --inputs comparative.jsonl \
    --weights 0.6 0.4 \
    --total 20000 \
    --output train_mix_20k.jsonl \
    --shuffle --seed 123
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class InputSpec:
    path: Path
    weight: float


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _iter_nonempty_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line.rstrip("\n")


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_lines(path: Path, lines: Iterable[str]) -> int:
    _ensure_parent_dir(path)
    n = 0
    with path.open("w", encoding="utf-8") as out:
        for line in lines:
            out.write(line)
            out.write("\n")
            n += 1
    return n


def _in_memory_shuffle(lines: List[str], seed: int) -> List[str]:
    rng = random.Random(seed)
    rng.shuffle(lines)
    return lines


def _bucket_shuffle_to_output(
    input_paths: List[Path],
    output_path: Path,
    seed: int,
    buckets: int,
    bucket_in_memory_limit: int,
) -> int:
    """
    Approximate shuffle without holding the full dataset in memory:
      1) assign each line to a random bucket file
      2) shuffle each bucket in memory and write buckets sequentially

    If a bucket exceeds bucket_in_memory_limit, it is written without in-bucket shuffle
    (still randomized across buckets).
    """
    rng = random.Random(seed)
    _ensure_parent_dir(output_path)

    with tempfile.TemporaryDirectory(prefix="mix_jsonl_shuffle_", dir=str(output_path.parent)) as tmpdir:
        tmpdir_p = Path(tmpdir)
        bucket_files = [tmpdir_p / f"bucket_{i:04d}.jsonl" for i in range(buckets)]
        bucket_fhs = [bf.open("w", encoding="utf-8") for bf in bucket_files]
        try:
            for ip in input_paths:
                for line in _iter_nonempty_lines(ip):
                    b = rng.randrange(buckets)
                    bucket_fhs[b].write(line)
                    bucket_fhs[b].write("\n")
        finally:
            for fh in bucket_fhs:
                fh.close()

        written = 0
        with output_path.open("w", encoding="utf-8") as out:
            for bf in bucket_files:
                if not bf.exists():
                    continue
                with bf.open("r", encoding="utf-8") as f:
                    lines = [ln.rstrip("\n") for ln in f if ln.strip()]
                if not lines:
                    continue
                if len(lines) <= bucket_in_memory_limit:
                    rng.shuffle(lines)
                for ln in lines:
                    out.write(ln)
                    out.write("\n")
                    written += 1
        return written


def _plan_sample_counts(total: int, specs: List[InputSpec]) -> List[int]:
    weights = [max(0.0, s.weight) for s in specs]
    if sum(weights) <= 0:
        raise ValueError("All weights are <= 0; cannot sample.")
    weights = [w / sum(weights) for w in weights]

    # Largest-remainder method to make counts sum exactly to total.
    raw = [total * w for w in weights]
    floor = [int(x) for x in raw]
    remain = total - sum(floor)
    rema = sorted([(raw[i] - floor[i], i) for i in range(len(specs))], reverse=True)
    counts = floor[:]
    for _, idx in rema[:remain]:
        counts[idx] += 1
    return counts


def _sample_without_replacement(path: Path, k: int, seed: int) -> List[str]:
    if k <= 0:
        return []
    n = _count_lines(path)
    if k > n:
        raise ValueError(f"Cannot sample k={k} without replacement from {path} with only n={n} lines.")
    rng = random.Random(seed)
    indices = set(rng.sample(range(n), k))
    picked: List[str] = []
    i = 0
    for line in _iter_nonempty_lines(path):
        if i in indices:
            picked.append(line)
        i += 1
    return picked


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", action="append", required=True, help="Input JSONL path (repeatable).")
    p.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="Sampling weights aligned with --inputs (only used with --total).",
    )
    p.add_argument("--total", type=int, default=None, help="If set, sample this many total rows from inputs.")
    p.add_argument("--output", required=True, help="Output JSONL path.")
    p.add_argument("--shuffle", action="store_true", help="Shuffle output lines.")
    p.add_argument("--seed", type=int, default=123, help="Random seed.")
    p.add_argument(
        "--in-memory-shuffle-limit",
        type=int,
        default=200_000,
        help="If total lines <= this, shuffle fully in memory; otherwise use bucket shuffle.",
    )
    p.add_argument(
        "--shuffle-buckets",
        type=int,
        default=64,
        help="Number of buckets for external shuffle (used when dataset is large).",
    )
    p.add_argument(
        "--bucket-in-memory-limit",
        type=int,
        default=50_000,
        help="Max lines per bucket to shuffle in memory.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned counts and exit without writing.")

    args = p.parse_args(argv)

    input_paths = [Path(x) for x in args.inputs]
    for ip in input_paths:
        if not ip.exists():
            raise FileNotFoundError(ip)

    out_path = Path(args.output)

    if args.total is None:
        # Simple concat (and optional shuffle).
        if args.dry_run:
            total = sum(_count_lines(p) for p in input_paths)
            print(f"[dry-run] concat total_lines={total} -> {out_path}")
            return 0

        if not args.shuffle:
            def _all_lines() -> Iterable[str]:
                for ip in input_paths:
                    yield from _iter_nonempty_lines(ip)

            n = _write_lines(out_path, _all_lines())
            print(f"Wrote {n} lines -> {out_path}")
            return 0

        # Shuffle mode: decide in-memory vs bucket shuffle.
        total_lines = sum(_count_lines(p) for p in input_paths)
        if total_lines <= args.in_memory_shuffle_limit:
            lines: List[str] = []
            for ip in input_paths:
                lines.extend(list(_iter_nonempty_lines(ip)))
            _in_memory_shuffle(lines, args.seed)
            n = _write_lines(out_path, lines)
            print(f"Wrote {n} lines (in-memory shuffled) -> {out_path}")
            return 0

        n = _bucket_shuffle_to_output(
            input_paths=input_paths,
            output_path=out_path,
            seed=args.seed,
            buckets=args.shuffle_buckets,
            bucket_in_memory_limit=args.bucket_in_memory_limit,
        )
        print(f"Wrote {n} lines (bucket shuffled) -> {out_path}")
        return 0

    # Sample mode (fixed total).
    if args.total <= 0:
        raise ValueError("--total must be > 0")

    if args.weights is None or len(args.weights) == 0:
        weights = [1.0 for _ in input_paths]
    else:
        if len(args.weights) != len(input_paths):
            raise ValueError("--weights length must match number of --inputs.")
        weights = args.weights

    specs = [InputSpec(path=p, weight=w) for p, w in zip(input_paths, weights)]
    per_input = _plan_sample_counts(args.total, specs)

    if args.dry_run:
        for spec, k in zip(specs, per_input):
            n = _count_lines(spec.path)
            print(f"[dry-run] sample {k} / {n} from {spec.path} (weight={spec.weight})")
        print(f"[dry-run] total={sum(per_input)} -> {out_path} (shuffle={args.shuffle})")
        return 0

    picked_all: List[str] = []
    for i, (spec, k) in enumerate(zip(specs, per_input)):
        # Derive a stable per-file seed so runs are reproducible even if input order changes.
        file_seed = (args.seed * 1000003 + i * 10007) & 0xFFFFFFFF
        picked_all.extend(_sample_without_replacement(spec.path, k, file_seed))

    if args.shuffle:
        _in_memory_shuffle(picked_all, args.seed)

    n = _write_lines(out_path, picked_all)
    print(f"Wrote {n} sampled lines -> {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # allow piping to `head` etc.
        raise SystemExit(0)

