#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Tuple

"""
Fix HuggingFace `datasets` / PyArrow schema errors caused by mixed types in nested fields.

Common failure:
  ArrowInvalid: Column(/answer/correct_answer) changed from string to array ...

This script normalizes `answer.correct_answer` to ALWAYS be a string:
  - if it is a list, join items with a separator (default: ", ")
  - if it is a scalar, cast to string

Usage:
  python3 Comparative-R1/scripts/ReMAP/fix_offline_rft_correct_answer_type.py \
    --input  data/offline_rft/isic/v1/train_text_rule.jsonl \
    --output data/offline_rft/isic/v1/train_text_rule.fixed.jsonl
"""


def _read_nonempty_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            yield line_no, s


def _normalize_correct_answer(value: Any, *, join_sep: str) -> Tuple[str, bool]:
    """
    Returns: (normalized_value, changed)
    """
    if value is None:
        return "", True

    if isinstance(value, str):
        return value, False

    if isinstance(value, list):
        parts: list[str] = []
        for x in value:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                parts.append(s)
        return join_sep.join(parts), True

    # Fallback: cast scalars (bool/int/float/dict/...) to string.
    return str(value), True


def main() -> None:
    p = argparse.ArgumentParser(description="Normalize offline_rft JSONL answer.correct_answer type for PyArrow.")
    p.add_argument("--input", required=True, type=Path, help="Input JSONL path.")
    p.add_argument("--output", type=Path, default=None, help="Output JSONL path (default: <input>.fixed.jsonl).")
    p.add_argument("--join-sep", type=str, default=", ", help="Separator for list -> string join.")
    p.add_argument("--dry-run", action="store_true", help="Only print summary; do not write output.")
    args = p.parse_args()

    in_path: Path = args.input
    out_path: Path = args.output or in_path.with_suffix(in_path.suffix + ".fixed.jsonl")

    total = 0
    written = 0
    json_errors = 0
    missing_answer = 0
    missing_correct_answer = 0
    changed = 0
    list_to_string = 0
    none_to_empty = 0

    out_fh = None
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_fh = out_path.open("w", encoding="utf-8")
    try:
        for line_no, s in _read_nonempty_lines(in_path):
            total += 1
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                json_errors += 1
                continue

            ans = obj.get("answer", None)
            if not isinstance(ans, dict):
                missing_answer += 1
                if out_fh is not None:
                    out_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    written += 1
                continue

            if "correct_answer" not in ans:
                missing_correct_answer += 1
                if out_fh is not None:
                    out_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    written += 1
                continue

            old = ans.get("correct_answer")
            new, did_change = _normalize_correct_answer(old, join_sep=args.join_sep)
            if did_change:
                changed += 1
                if isinstance(old, list):
                    list_to_string += 1
                if old is None:
                    none_to_empty += 1
                ans["correct_answer"] = new
                obj["answer"] = ans

            if out_fh is not None:
                out_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
    finally:
        if out_fh is not None:
            out_fh.close()

    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "total_nonempty_lines": total,
        "written_lines": written,
        "json_errors": json_errors,
        "missing_answer": missing_answer,
        "missing_correct_answer": missing_correct_answer,
        "changed_lines": changed,
        "list_to_string": list_to_string,
        "none_to_empty": none_to_empty,
        "join_sep": args.join_sep,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    out_path.with_suffix(out_path.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
