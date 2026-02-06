#!/usr/bin/env python3
"""
Rewrite single-image MCQ rows into optionless label-text rows.

Input row (single-image MCQ):
  answer.task_type == "mcq_letter"
  prompt contains question/options
  answer contains option_A..option_D and label

Output row:
  - prompt rewritten to show a candidate label vocabulary
  - answer.correct_answer set to text label
  - answer.correct_label / answer.candidate_labels added
  - answer.task_type switched to "mcq_optionless_text"

Non-MCQ rows (e.g. B1-B7 comparative tasks) are passed through unchanged.

This transform intentionally mirrors:
  Comparative-R1/scripts/eval/eval_optionless_checkpoint_and_dump.py
to keep training and eval behavior consistent.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


_QUESTION_RE = re.compile(r"^\s*Question\s*:\s*(?P<q>.+?)\s*$", flags=re.IGNORECASE | re.MULTILINE)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_question(prompt: str) -> str:
    m = _QUESTION_RE.search(prompt)
    if not m:
        return ""
    return m.group("q").strip()


def _norm_label(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\.\s*$", "", s).strip()
    # common noise in OmniMed rows: "Benign image."
    s = re.sub(r"\bimage\b\.?$", "", s, flags=re.IGNORECASE).strip()
    return s


def _extract_options(ans: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k in ("option_A", "option_B", "option_C", "option_D"):
        v = ans.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() == "none":
            continue
        out.append(s)
    return out


def _is_binary_benign_malignant(options: list[str]) -> bool:
    if len(options) != 2:
        return False
    norm = {_norm_label(x).lower() for x in options if _norm_label(x)}
    norm = {re.sub(r"[^a-z]+", " ", x).strip() for x in norm}
    return norm == {"benign", "malignant"}


def _group_key(ans: dict[str, Any]) -> str:
    qtype = str(ans.get("question_type", "unknown")).strip() or "unknown"
    opts = _extract_options(ans)
    if _is_binary_benign_malignant(opts):
        return f"{qtype}|binary_benign_malignant"
    return f"{qtype}|option_count={len(opts)}"


def _build_prompt(question: str, candidate_labels: list[str]) -> str:
    labels = [x.strip() for x in candidate_labels if isinstance(x, str) and x.strip()]
    labels = sorted(dict.fromkeys(labels).keys())
    lines = [
        "You are a medical VQA assistant for dermoscopy images.",
        "",
        "Task: Predict the disease diagnosis label for the image.",
        "",
    ]
    if question:
        lines.extend([f"Question: {question}", ""])
    lines.append("Candidate labels (choose exactly one; copy the label text):")
    for x in labels:
        lines.append(f"- {x}")
    lines.extend(["", "Answer with ONLY the label text (no extra words).", "<answer></answer>"])
    return "\n".join(lines)


def transform_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels_by_group: dict[str, set[str]] = {}

    # Pass 1: collect label vocab per group (same strategy as eval_optionless).
    for r in rows:
        ans = r.get("answer") if isinstance(r.get("answer"), dict) else {}
        if str(ans.get("task_type", "")).strip() != "mcq_letter":
            continue
        gk = _group_key(ans)
        if gk.endswith("|binary_benign_malignant"):
            labels_by_group.setdefault(gk, set()).update({"Benign", "Malignant"})
        else:
            lbl = ans.get("label")
            if isinstance(lbl, str) and lbl.strip():
                labels_by_group.setdefault(gk, set()).add(lbl.strip())

    out_rows: list[dict[str, Any]] = []
    rewritten = 0
    skipped_invalid_prompt = 0

    for r in rows:
        ans = r.get("answer") if isinstance(r.get("answer"), dict) else {}
        task_type = str(ans.get("task_type", "")).strip()
        if task_type != "mcq_letter":
            out_rows.append(r)
            continue

        prompt = r.get("prompt")
        if not isinstance(prompt, str):
            skipped_invalid_prompt += 1
            out_rows.append(r)
            continue

        gk = _group_key(ans)
        candidate = sorted(labels_by_group.get(gk, set()))
        if gk.endswith("|binary_benign_malignant"):
            candidate = ["Benign", "Malignant"]

        question = _extract_question(prompt)
        new_prompt = _build_prompt(question, candidate)

        new_row = dict(r)
        new_row["prompt"] = new_prompt
        new_ans = dict(ans)

        label = _norm_label(new_ans.get("label"))
        if label:
            if gk.endswith("|binary_benign_malignant") and label.lower() in {"benign", "malignant"}:
                label = label.capitalize()
            new_ans["correct_label"] = label
            new_ans["correct_answer_mcq"] = new_ans.get("correct_answer")
            new_ans["correct_answer"] = label

        new_ans["candidate_labels"] = candidate
        new_ans["task_type"] = "mcq_optionless_text"
        new_ans["optionless_group_key"] = gk

        new_row["answer"] = new_ans
        out_rows.append(new_row)
        rewritten += 1

    info = {
        "rewritten_mcq": rewritten,
        "skipped_invalid_prompt": skipped_invalid_prompt,
        "num_groups": len(labels_by_group),
        "labels_by_group": {k: sorted(v) for k, v in sorted(labels_by_group.items())},
    }
    return out_rows, info


def main() -> None:
    ap = argparse.ArgumentParser(description="Rewrite mcq_letter rows to optionless label-text rows")
    ap.add_argument("--input", type=Path, required=True, help="Input JSONL path")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    ap.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional path for summary JSON. Default: <output>.summary.json",
    )
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Missing --input: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists, pass --overwrite: {args.output}")

    rows = _read_jsonl(args.input)
    out_rows, info = transform_rows(rows)
    _write_jsonl(args.output, out_rows)

    summary = args.summary or Path(str(args.output) + ".summary.json")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "total_input_rows": len(rows),
                "total_output_rows": len(out_rows),
                "info": info,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {args.output}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
