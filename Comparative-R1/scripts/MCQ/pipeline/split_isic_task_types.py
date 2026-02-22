#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EIGHT_CLASS_LABELS = {
    "actinic keratosis",
    "basal cell carcinoma",
    "benign keratosis",
    "dermatofibroma",
    "melanocytic nevus",
    "melanoma",
    "squamous cell carcinoma",
    "vascular lesion",
}

BINARY_LABELS = {"benign", "malignant"}


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).casefold()


def _canon(text: Any) -> str:
    t = _norm(text)
    if "benign" in t and "malignant" not in t:
        return "benign"
    if "malignant" in t and "benign" not in t:
        return "malignant"
    return t


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_labels(answer: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    cand = answer.get("candidate_labels")
    if isinstance(cand, list):
        labels.extend(str(x).strip() for x in cand if str(x).strip())
    else:
        for k in ("option_A", "option_B", "option_C", "option_D"):
            v = answer.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                labels.append(s)
    return labels


def _primary_label(answer: dict[str, Any]) -> str:
    label = answer.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    correct = answer.get("correct_answer")
    if isinstance(correct, str) and correct.strip():
        return correct.strip()
    return ""


def _infer_task_type(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if not isinstance(answer, dict):
        return "unknown"

    labels = [_canon(x) for x in _extract_labels(answer)]
    label_set = {x for x in labels if x}
    if label_set:
        if label_set.issubset(BINARY_LABELS):
            return "binary"
        if label_set.issubset(EIGHT_CLASS_LABELS):
            return "8class"

    primary = _canon(_primary_label(answer))
    if primary in BINARY_LABELS:
        return "binary"
    if primary in EIGHT_CLASS_LABELS:
        return "8class"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Split ISIC JSONL into binary vs 8-class subsets.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-binary", type=Path, required=True)
    parser.add_argument("--out-8class", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rows = _read_jsonl(args.input)
    binary_rows: list[dict[str, Any]] = []
    class8_rows: list[dict[str, Any]] = []
    unknown_rows: list[dict[str, Any]] = []

    for row in rows:
        task_type = _infer_task_type(row)
        if task_type == "binary":
            binary_rows.append(row)
        elif task_type == "8class":
            class8_rows.append(row)
        else:
            unknown_rows.append(row)

    if args.strict and unknown_rows:
        raise RuntimeError(
            f"Found {len(unknown_rows)} unknown rows in {args.input}; "
            "run without --strict to ignore."
        )

    _write_jsonl(args.out_binary, binary_rows)
    _write_jsonl(args.out_8class, class8_rows)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "out_binary": str(args.out_binary),
                "out_8class": str(args.out_8class),
                "total": len(rows),
                "binary": len(binary_rows),
                "class8": len(class8_rows),
                "unknown": len(unknown_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

