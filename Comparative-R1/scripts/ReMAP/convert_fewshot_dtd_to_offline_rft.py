#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

"""
python3 Comparative-R1/scripts/ReMAP/convert_fewshot_dtd_to_offline_rft.py --input /data/shichao/EasyR1/data/CLS/ISIC/
            ISIC_fewshot_4.jsonl --output /data/shichao/EasyR1/data/offline_rft/isic/v1/train_cls_fewshot.jsonl

"""
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            payload = json.loads(s)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


_RE_CANDIDATES = re.compile(r"list\s*\[\s*(.*?)\s*\]", flags=re.IGNORECASE | re.DOTALL)
_RE_CHOOSE_ONE = re.compile(
    r"\s*Please\s+choose\s+one\s+from\s+list\s*\[\s*.*?\s*\]\s*\.?\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def _extract_candidate_labels(problem: str) -> list[str]:
    m = _RE_CANDIDATES.search(problem)
    if not m:
        return []
    inner = m.group(1).strip()
    if not inner:
        return []
    parts = [p.strip() for p in inner.split(",")]
    cleaned: list[str] = []
    for p in parts:
        p = p.strip().strip("\"'`")
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            cleaned.append(p)
    # stable unique
    return list(dict.fromkeys(cleaned).keys())


def _strip_choose_one_tail(problem: str) -> str:
    return _RE_CHOOSE_ONE.sub("", problem).strip()


def _build_prompt(problem: str, *, system: str, answer_instruction: str) -> str:
    lines: list[str] = []
    if system.strip():
        lines.append(system.strip())
    lines.append("<image>")
    lines.append(problem.strip())
    if answer_instruction.strip():
        lines.append(answer_instruction.strip())
    return "\n".join(lines).strip()


def _get_str(row: dict[str, Any], key: str) -> str:
    v = row.get(key, None)
    return v.strip() if isinstance(v, str) and v.strip() else ""


def main() -> None:
    p = argparse.ArgumentParser(description="Convert DTD-style few-shot JSONL to offline_rft JSONL.")
    p.add_argument("--input", required=True, type=Path, help="Input JSONL with keys: image,label,problem.")
    p.add_argument("--output", required=True, type=Path, help="Output offline_rft JSONL.")
    p.add_argument("--id-prefix", type=str, default="cls_", help="sample_id prefix.")
    p.add_argument(
        "--system",
        type=str,
        default="You are a medical VQA assistant for dermoscopy images.",
        help="System header line for prompt (stored in prompt text).",
    )
    p.add_argument(
        "--answer-instruction",
        type=str,
        default="Answer with ONLY the label text.\nUse format: <answer>...</answer>",
        help="Answer instruction appended to prompt.",
    )
    p.add_argument(
        "--strip-candidates-from-prompt",
        action="store_true",
        help="Remove trailing 'Please choose one from list [...]' from the prompt (still stores candidate_labels if found).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print summary and exit without writing output.")
    args = p.parse_args()

    rows = _read_jsonl(args.input)
    out_rows: list[dict[str, Any]] = []

    missing_image = 0
    missing_label = 0
    missing_problem = 0
    total_candidates = 0

    for i, r in enumerate(rows):
        line_idx = i + 1
        image = _get_str(r, "image") or _get_str(r, "images")
        label = _get_str(r, "label")
        problem = _get_str(r, "problem") or _get_str(r, "prompt") or _get_str(r, "question")
        split = _get_str(r, "split")

        if not image:
            missing_image += 1
        if not label:
            missing_label += 1
        if not problem:
            missing_problem += 1

        candidate_labels = _extract_candidate_labels(problem) if problem else []
        total_candidates += len(candidate_labels)

        prompt_problem = _strip_choose_one_tail(problem) if (args.strip_candidates_from_prompt and problem) else problem
        prompt = _build_prompt(prompt_problem or problem or "", system=args.system, answer_instruction=args.answer_instruction)

        sample_id = f"{args.id_prefix}{line_idx:06d}"
        out_rows.append(
            {
                "prompt": prompt,
                "images": [image] if image else [],
                "answer": {
                    "task_type": "cls",
                    "source_type": "cls",
                    "answer_type": "short_text",
                    "correct_answer": label,
                    "candidate_labels": candidate_labels,
                },
                "meta": {
                    "source_type": "cls",
                    "sample_id": sample_id,
                    "origin_input": str(args.input),
                    "origin_line": line_idx,
                    "origin_split": split or None,
                    "origin_image": image or None,
                    "prompt_version": "fewshot_dtd_to_offline_rft_v1",
                },
            }
        )

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "total_input_rows": len(rows),
        "total_output_rows": len(out_rows),
        "missing_image": missing_image,
        "missing_label": missing_label,
        "missing_problem": missing_problem,
        "avg_candidate_labels_per_row": (total_candidates / len(rows)) if rows else 0.0,
        "strip_candidates_from_prompt": bool(args.strip_candidates_from_prompt),
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    _write_jsonl(args.output, out_rows)
    summary_path = args.output.parent / f"{args.output.stem}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

