#!/usr/bin/env python3
"""
Build EasyR1 MCQ-letter ISIC JSONL from offline 4shot JSONL by image matching.

Input rows (example):
  {
    "images": ["Images/ISIC2019/train/ISIC_0069271.jpg"],
    "answer": {"correct_answer": "Squamous cell carcinoma", ...}
  }

Source QA files:
  /data/shichao/data/OmniMedVQA/QA_information/Open-access/ISIC*.json

python /data/shichao/EasyR1/Comparative-R1/scripts/MCQ/build_isic_mcq_from_4shot.py \
    --input /data/shichao/EasyR1/data/offline_rft/isic/v1/test_4shot_nothinking.jsonl \
    --qa-open-access-dir /data/shichao/data/OmniMedVQA/QA_information/Open-access \
    --output /data/shichao/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.jsonl \
    --strict
    
Output rows are aligned with:
  data/OminiMedExpert/isic_disease_diagnosis_v1_0.05/train_fewshot_0.5.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPTION_KEYS = ("option_A", "option_B", "option_C", "option_D")
LETTER_BY_OPTION_KEY = {
    "option_A": "A",
    "option_B": "B",
    "option_C": "C",
    "option_D": "D",
}


def _norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x).strip()).casefold()


def _image_stem(image_path: str) -> str:
    return Path(image_path).stem


def _letter_options(letters: list[str]) -> str:
    if len(letters) == 2:
        return f"{letters[0]} or {letters[1]}"
    if len(letters) == 1:
        return letters[0]
    if len(letters) <= 0:
        return "A"
    return ", ".join(letters[:-1]) + f", or {letters[-1]}"


def _build_prompt(question: str, options: dict[str, Any]) -> str:
    rendered: list[str] = []
    letters: list[str] = []
    for k in OPTION_KEYS:
        v = options.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        ltr = LETTER_BY_OPTION_KEY[k]
        letters.append(ltr)
        rendered.append(f"{ltr}. {s}")

    ltr_opts = _letter_options(letters)
    return (
        f"Question: {str(question).strip()}\n"
        f"Options:\n"
        + "\n".join(rendered)
        + f"\nAnswer with only the option letter ({ltr_opts}).\n<answer></answer>"
    )


def _infer_answer_id(gt_answer: Any, options: dict[str, Any]) -> str | None:
    gt = _norm_text(gt_answer)
    matches: list[str] = []
    for k in OPTION_KEYS:
        v = options.get(k)
        if v is None:
            continue
        if _norm_text(v) == gt:
            matches.append(LETTER_BY_OPTION_KEY[k])

    if len(matches) == 1:
        return matches[0]

    # Fallback for benign/malignant variants (e.g., "Benign image." vs "Benign").
    benign_like = "benign" in gt
    malignant_like = "malignant" in gt
    if benign_like ^ malignant_like:
        target = "benign" if benign_like else "malignant"
        fuzzy = []
        for k in OPTION_KEYS:
            v = options.get(k)
            if v is None:
                continue
            if target in _norm_text(v):
                fuzzy.append(LETTER_BY_OPTION_KEY[k])
        if len(fuzzy) == 1:
            return fuzzy[0]

    return None


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

    return None


@dataclass
class Stats:
    total: int = 0
    written: int = 0
    skipped_no_image: int = 0
    skipped_not_found: int = 0
    skipped_ambiguous: int = 0
    skipped_no_answer_id: int = 0


def convert(input_rows: list[dict[str, Any]], image_index: dict[str, list[dict[str, Any]]], strict: bool) -> tuple[list[dict[str, Any]], Stats]:
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

        options = {k: picked.get(k) for k in OPTION_KEYS}
        answer_id = _infer_answer_id(picked.get("gt_answer"), options)
        if answer_id is None:
            stats.skipped_no_answer_id += 1
            continue

        image_path = str(picked.get("image_path", image)).strip()
        dataset = str(picked.get("dataset", "")).strip()
        image_stem = _image_stem(image_path)

        answer_payload = {
            "label": picked.get("gt_answer"),
            "answer_id": answer_id,
            "question_id": picked.get("question_id"),
            "dataset": dataset,
            "question_type": picked.get("question_type"),
            "modality_type": picked.get("modality_type"),
            "group_id": f"{dataset}::image={image_stem}",
            "group_id_source": "image_stem",
            "option_A": picked.get("option_A"),
            "option_B": picked.get("option_B"),
            "option_C": picked.get("option_C"),
            "option_D": picked.get("option_D"),
            "correct_answer": answer_id,
            "task_type": "mcq_letter",
        }

        out.append(
            {
                "prompt": _build_prompt(str(picked.get("question", "")).strip(), options),
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
            f"skipped_no_answer_id={stats.skipped_no_answer_id}"
        )

    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ISIC MCQ-letter JSONL by matching source rows with image key")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("/data/shichao/EasyR1/data/offline_rft/isic/v1/4shot_nothinking.jsonl"),
        help="Input JSONL (source rows with images field)",
    )
    ap.add_argument(
        "--qa-open-access-dir",
        type=Path,
        default=Path("/data/shichao/data/OmniMedVQA/QA_information/Open-access"),
        help="Directory containing ISIC*.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("/data/shichao/EasyR1/data/OminiMedExpert/isic_disease_diagnosis_v1_0.05/train_fewshot_0.5.from_4shot_nothinking.jsonl"),
        help="Output JSONL path",
    )
    ap.add_argument("--strict", action="store_true", help="Fail if any input row cannot be converted")
    args = ap.parse_args()

    input_rows = _read_jsonl(args.input)
    isic_rows = _load_isic_rows(args.qa_open_access_dir)
    image_index = _build_index(isic_rows)

    out_rows, stats = convert(input_rows, image_index, strict=args.strict)
    _write_jsonl(args.output, out_rows)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "qa_open_access_dir": str(args.qa_open_access_dir),
                "output": str(args.output),
                "stats": {
                    "total": stats.total,
                    "written": stats.written,
                    "skipped_no_image": stats.skipped_no_image,
                    "skipped_not_found": stats.skipped_not_found,
                    "skipped_ambiguous": stats.skipped_ambiguous,
                    "skipped_no_answer_id": stats.skipped_no_answer_id,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
