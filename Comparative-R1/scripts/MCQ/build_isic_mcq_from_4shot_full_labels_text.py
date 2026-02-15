#!/usr/bin/env python3
"""
Build EasyR1 ISIC text-answer JSONL from offline 4shot JSONL by image matching.

Compared with build_isic_mcq_from_4shot_full_labels.py:
- keep prompt style, but remove options block
- correct_answer is label text (not option letter)

python /data/shichao/EasyR1/Comparative-R1/scripts/MCQ/build_isic_mcq_from_4shot_full_labels_text.py \
    --input /data/shichao/EasyR1/data/offline_rft/isic/v1/test_4shot_nothinking.jsonl \
    --qa-open-access-dir /data/shichao/data/OmniMedVQA/QA_information/Open-access \
    --output /data/shichao/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.full_labels_text.jsonl \
    --strict
    
  python /data/shichao/EasyR1/Comparative-R1/scripts/MCQ/build_isic_mcq_from_4shot_full_labels_text.py \
    --input /data/shichao/EasyR1/data/offline_rft/isic/v1/test_4shot_nothinking.jsonl \
    --qa-open-access-dir /data/shichao/data/OmniMedVQA/QA_information/Open-access \
    --output /data/shichao/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.full_labels_text.jsonl \
    --strict
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x).strip()).casefold()


def _image_stem(image_path: str) -> str:
    return Path(image_path).stem


def _dedup_labels(labels: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in labels:
        s = str(x).strip()
        if not s:
            continue
        n = _norm_text(s)
        if n in seen:
            continue
        seen.add(n)
        out.append(s)
    return out


def _build_prompt(question: str) -> str:
    return (
        f"Question: {str(question).strip()}\n"
        "Answer with only the diagnosis text.\n"
        "<answer></answer>"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_isic_rows(qa_open_access_dir: Path) -> list[dict[str, Any]]:
    files = sorted(qa_open_access_dir.glob("ISIC*.json"))
    if not files:
        raise FileNotFoundError(f"No ISIC*.json found under: {qa_open_access_dir}")

    rows: list[dict[str, Any]] = []
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {fp}")
        for item in data:
            if not isinstance(item, dict):
                continue
            if str(item.get("question_type", "")).strip() != "Disease Diagnosis":
                continue
            rows.append(item)
    return rows


def _build_index(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_image: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        image_path = str(r.get("image_path", "")).strip()
        if not image_path:
            continue
        by_image.setdefault(image_path, []).append(r)
    return by_image


def _pick_candidate(cands: list[dict[str, Any]], input_correct_answer: Any) -> dict[str, Any] | None:
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]

    target = _norm_text(input_correct_answer)
    if target:
        matched = [c for c in cands if _norm_text(c.get("gt_answer", "")) == target]
        if len(matched) == 1:
            return matched[0]

        benign_like = "benign" in target
        malignant_like = "malignant" in target
        if benign_like ^ malignant_like:
            key = "benign" if benign_like else "malignant"
            fuzzy = [c for c in cands if key in _norm_text(c.get("gt_answer", ""))]
            if len(fuzzy) == 1:
                return fuzzy[0]

    return None


def _pick_label_text(input_correct: Any, candidate_labels: list[str], gt_answer: Any) -> str:
    if input_correct is not None:
        ic = _norm_text(input_correct)
        for x in candidate_labels:
            if _norm_text(x) == ic:
                return x
    return str(gt_answer).strip()


@dataclass
class Stats:
    total: int = 0
    written: int = 0
    skipped_no_image: int = 0
    skipped_not_found: int = 0
    skipped_ambiguous: int = 0
    skipped_no_labels: int = 0
    skipped_empty_label: int = 0


def convert(
    input_rows: list[dict[str, Any]], image_index: dict[str, list[dict[str, Any]]], strict: bool
) -> tuple[list[dict[str, Any]], Stats]:
    out: list[dict[str, Any]] = []
    stats = Stats()

    for row in input_rows:
        stats.total += 1

        images = row.get("images")
        image = ""
        if isinstance(images, list) and images:
            image = str(images[0]).strip()
        if not image:
            stats.skipped_no_image += 1
            continue

        cands = image_index.get(image, [])
        if not cands:
            stats.skipped_not_found += 1
            continue

        input_answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
        input_correct = input_answer.get("correct_answer") if isinstance(input_answer, dict) else None

        picked = _pick_candidate(cands, input_correct)
        if picked is None:
            stats.skipped_ambiguous += 1
            continue

        candidate_labels: list[str] = []
        if isinstance(input_answer, dict) and isinstance(input_answer.get("candidate_labels"), list):
            candidate_labels = _dedup_labels(input_answer["candidate_labels"])
        if not candidate_labels:
            stats.skipped_no_labels += 1
            continue

        label_text = _pick_label_text(input_correct, candidate_labels, picked.get("gt_answer"))
        if not label_text:
            stats.skipped_empty_label += 1
            continue

        image_path = str(picked.get("image_path", image)).strip()
        dataset = str(picked.get("dataset", "")).strip()
        image_stem = _image_stem(image_path)

        answer_payload = {
            "label": label_text,
            "candidate_labels": candidate_labels,
            "question_id": picked.get("question_id"),
            "dataset": dataset,
            "question_type": picked.get("question_type"),
            "modality_type": picked.get("modality_type"),
            "group_id": f"{dataset}::image={image_stem}",
            "group_id_source": "image_stem",
            "correct_answer": label_text,
            "task_type": "mcq_text",
        }

        out.append(
            {
                "prompt": _build_prompt(str(picked.get("question", "")).strip()),
                "images": [image_path],
                "answer": answer_payload,
            }
        )
        stats.written += 1

    if strict and stats.written != stats.total:
        raise RuntimeError(
            "Strict mode failed: "
            f"total={stats.total}, written={stats.written}, skipped_no_image={stats.skipped_no_image}, "
            f"skipped_not_found={stats.skipped_not_found}, skipped_ambiguous={stats.skipped_ambiguous}, "
            f"skipped_no_labels={stats.skipped_no_labels}, skipped_empty_label={stats.skipped_empty_label}"
        )

    return out, stats


def main() -> None:
    p = argparse.ArgumentParser(description="Build ISIC text-answer data from 4shot input (full labels).")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--qa-open-access-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    input_rows = _read_jsonl(args.input)
    qa_rows = _load_isic_rows(args.qa_open_access_dir)
    image_index = _build_index(qa_rows)
    out_rows, stats = convert(input_rows=input_rows, image_index=image_index, strict=args.strict)
    _write_jsonl(args.output, out_rows)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "qa_open_access_dir": str(args.qa_open_access_dir),
                "output": str(args.output),
                "strict": args.strict,
                "stats": stats.__dict__,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

