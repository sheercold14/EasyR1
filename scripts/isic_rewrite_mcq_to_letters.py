#!/usr/bin/env python3
"""
Rewrite OmniMedVQA-style MCQ data to use letter answers (A/B/C/D).

Why:
  - Enables logit-margin / contrastive rewards over a fixed small candidate set.
  - Keeps output short and fully verifiable (exact letter match).

Input JSONL row example:
  {
    "prompt": "Question: ...\\nOptions:\\nA. ...\\nB. ...\\nC. ...\\nD. ...\\nAnswer with the exact option text.",
    "images": ["..."],
    "answer": {"label": "Melanoma", "option_A": "...", ...}
  }

Output:
  - prompt instruction rewritten to request only the option letter
  - answer gains:
      - correct_answer: "A"|"B"|"C"|"D"
      - task_type: "mcq_letter"
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


OPTION_KEYS = ("option_A", "option_B", "option_C", "option_D")
LETTER_BY_KEY = {
    "option_A": "A",
    "option_B": "B",
    "option_C": "C",
    "option_D": "D",
}


@dataclass(frozen=True)
class RewriteStats:
    total: int = 0
    written: int = 0
    skipped_missing_options: int = 0
    skipped_no_match: int = 0
    skipped_multi_match: int = 0


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _infer_correct_letter(answer: dict) -> str | None:
    label = answer.get("label")
    if label is None:
        return None
    label_norm = _normalize_text(str(label))

    matches: list[str] = []
    for k in OPTION_KEYS:
        v = answer.get(k, None)
        if v is None:
            continue
        opt_norm = _normalize_text(str(v))
        if opt_norm == label_norm:
            matches.append(LETTER_BY_KEY[k])

    if len(matches) != 1:
        return None
    return matches[0]


def _rewrite_prompt(prompt: str) -> str:
    p = prompt.strip()
    # Replace the final instruction if present.
    p = re.sub(
        r"Answer with the exact option text\.?\s*$",
        "Answer with only the option letter (A, B, C, or D).\n<answer> A </answer>",
        p,
        flags=re.IGNORECASE,
    )
    # If not present, append a consistent instruction.
    if "<answer>" not in p.lower():
        p = (
            p
            + "\nAnswer with only the option letter (A, B, C, or D).\n<answer> A </answer>"
        )
    return p


def rewrite_file(src: Path, dst: Path) -> RewriteStats:
    stats = RewriteStats()
    out_rows: list[dict] = []

    for line_no, row in _read_jsonl(src):
        stats = RewriteStats(
            total=stats.total + 1,
            written=stats.written,
            skipped_missing_options=stats.skipped_missing_options,
            skipped_no_match=stats.skipped_no_match,
            skipped_multi_match=stats.skipped_multi_match,
        )

        answer = row.get("answer")
        if not isinstance(answer, dict):
            stats = RewriteStats(
                total=stats.total,
                written=stats.written,
                skipped_missing_options=stats.skipped_missing_options + 1,
                skipped_no_match=stats.skipped_no_match,
                skipped_multi_match=stats.skipped_multi_match,
            )
            continue

        if not all(k in answer for k in OPTION_KEYS):
            stats = RewriteStats(
                total=stats.total,
                written=stats.written,
                skipped_missing_options=stats.skipped_missing_options + 1,
                skipped_no_match=stats.skipped_no_match,
                skipped_multi_match=stats.skipped_multi_match,
            )
            continue

        label = answer.get("label")
        if label is None:
            stats = RewriteStats(
                total=stats.total,
                written=stats.written,
                skipped_missing_options=stats.skipped_missing_options + 1,
                skipped_no_match=stats.skipped_no_match,
                skipped_multi_match=stats.skipped_multi_match,
            )
            continue

        # Detect match vs multi-match for better diagnostics.
        label_norm = _normalize_text(str(label))
        matched_letters = []
        for k in OPTION_KEYS:
            v = answer.get(k, "")
            if _normalize_text(str(v)) == label_norm:
                matched_letters.append(LETTER_BY_KEY[k])

        if len(matched_letters) == 0:
            stats = RewriteStats(
                total=stats.total,
                written=stats.written,
                skipped_missing_options=stats.skipped_missing_options,
                skipped_no_match=stats.skipped_no_match + 1,
                skipped_multi_match=stats.skipped_multi_match,
            )
            continue
        if len(matched_letters) > 1:
            stats = RewriteStats(
                total=stats.total,
                written=stats.written,
                skipped_missing_options=stats.skipped_missing_options,
                skipped_no_match=stats.skipped_no_match,
                skipped_multi_match=stats.skipped_multi_match + 1,
            )
            continue

        correct_letter = matched_letters[0]

        prompt = row.get("prompt", "")
        if not isinstance(prompt, str):
            raise ValueError(f"Expected string 'prompt' at {src}:{line_no}")

        new_row = dict(row)
        new_row["prompt"] = _rewrite_prompt(prompt)
        new_answer = dict(answer)
        new_answer["correct_answer"] = correct_letter
        new_answer["task_type"] = "mcq_letter"
        new_row["answer"] = new_answer

        out_rows.append(new_row)
        stats = RewriteStats(
            total=stats.total,
            written=stats.written + 1,
            skipped_missing_options=stats.skipped_missing_options,
            skipped_no_match=stats.skipped_no_match,
            skipped_multi_match=stats.skipped_multi_match,
        )

    _write_jsonl(dst, out_rows)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite MCQ dataset to letter answers (A/B/C/D)")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stats = rewrite_file(args.input, args.output)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "total": stats.total,
                "written": stats.written,
                "skipped_missing_options": stats.skipped_missing_options,
                "skipped_no_match": stats.skipped_no_match,
                "skipped_multi_match": stats.skipped_multi_match,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

