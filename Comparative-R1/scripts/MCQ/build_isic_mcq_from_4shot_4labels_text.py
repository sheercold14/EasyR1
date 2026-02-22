#!/usr/bin/env python3
"""
Build EasyR1 ISIC text-answer JSONL from offline 4shot JSONL by image matching.

Target format:
- prompt has no options block, expects diagnosis text output
- answer.correct_answer is label text (not option letter)
- answer.candidate_labels comes from MCQ option_A~D (2-option rows keep A-B)

Example:
python EasyR1/Comparative-R1/scripts/MCQ/build_isic_mcq_from_4shot_4labels_text.py \
  --input EasyR1/data/offline_rft/isic/v1/test_4shot_nothinking.jsonl \
  --qa-open-access-dir /mnt/cache/wuruixiao/users/lsc/data/OmniMedVQA/QA_information/Open-access \
  --output EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.4labels_text.jsonl \
  --mcq-option-source EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.jsonl \
  --strict
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPTION_KEYS = ("option_A", "option_B", "option_C", "option_D")


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


def _build_prompt(question: str, candidate_labels: list[str]) -> str:
    label_block = ", ".join(str(x).strip() for x in candidate_labels if str(x).strip())
    return (
        f"{str(question).strip()}\n"
        f"Please choose one from list [{label_block}].\n\n"
        "Follow the exact output format:\n"
        "<answer> exact label text </answer>"
    )


def _extract_question_from_prompt(prompt: Any) -> str:
    text = str(prompt or "")
    m = re.search(r"Question:\s*(.*?)\n(?:Options:|Answer with only|$)", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


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


def _build_mcq_option_index(mcq_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_image: dict[str, list[dict[str, Any]]] = {}
    for row in mcq_rows:
        if not isinstance(row, dict):
            continue
        images = row.get("images")
        image = ""
        if isinstance(images, list) and images:
            image = str(images[0]).strip()
        if not image:
            continue
        by_image.setdefault(image, []).append(row)
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


def _labels_from_4options(input_answer: dict[str, Any], picked: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    # Prefer options already present in input row to guarantee alignment with current dataset.
    for k in OPTION_KEYS:
        v = input_answer.get(k)
        if v is None:
            v = picked.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            labels.append(s)
    return _dedup_labels(labels)


def _labels_from_mcq_row(mcq_row: dict[str, Any]) -> list[str]:
    ans = mcq_row.get("answer")
    if not isinstance(ans, dict):
        return []
    labels: list[str] = []
    for k in OPTION_KEYS:
        v = ans.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            labels.append(s)
    return _dedup_labels(labels)


def _pick_mcq_row(
    image: str, picked: dict[str, Any], mcq_option_index: dict[str, list[dict[str, Any]]]
) -> dict[str, Any] | None:
    cands = mcq_option_index.get(image, [])
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]

    picked_qid = str(picked.get("question_id", "")).strip()
    if picked_qid:
        matched = []
        for row in cands:
            ans = row.get("answer")
            qid = str(ans.get("question_id", "")).strip() if isinstance(ans, dict) else ""
            if qid == picked_qid:
                matched.append(row)
        if len(matched) == 1:
            return matched[0]

    picked_q = _norm_text(picked.get("question", ""))
    if picked_q:
        matched = []
        for row in cands:
            q = _norm_text(_extract_question_from_prompt(row.get("prompt", "")))
            if q == picked_q:
                matched.append(row)
        if len(matched) == 1:
            return matched[0]

    return None


def _pick_label_text(input_correct: Any, candidate_labels: list[str], gt_answer: Any) -> str:
    if input_correct is not None:
        ic = _norm_text(input_correct)
        for x in candidate_labels:
            if _norm_text(x) == ic:
                return x

    gt = _norm_text(gt_answer)
    for x in candidate_labels:
        if _norm_text(x) == gt:
            return x

    benign_like = "benign" in gt
    malignant_like = "malignant" in gt
    if benign_like ^ malignant_like:
        key = "benign" if benign_like else "malignant"
        fuzzy = [x for x in candidate_labels if key in _norm_text(x)]
        if len(fuzzy) == 1:
            return fuzzy[0]

    return str(gt_answer).strip()


def _resolve_label_in_pool(label_text: str, pool: list[str]) -> str:
    t = _norm_text(label_text)
    if not t:
        return ""
    for x in pool:
        if _norm_text(x) == t:
            return x
    return ""


def _constrain_candidates(
    candidates: list[str], allowed_labels: list[str], correct_label: str, target_n: int
) -> list[str]:
    allowed = _dedup_labels(allowed_labels)
    if not allowed:
        return _dedup_labels(candidates)

    allowed_norm = {_norm_text(x) for x in allowed}
    kept: list[str] = []
    seen: set[str] = set()
    for x in _dedup_labels(candidates):
        nx = _norm_text(x)
        if nx not in allowed_norm or nx in seen:
            continue
        seen.add(nx)
        kept.append(x)

    canonical_correct = _resolve_label_in_pool(correct_label, allowed)
    if canonical_correct:
        nc = _norm_text(canonical_correct)
        if nc not in seen:
            seen.add(nc)
            kept.insert(0, canonical_correct)

    for x in allowed:
        nx = _norm_text(x)
        if nx in seen:
            continue
        seen.add(nx)
        kept.append(x)
        if len(kept) >= target_n:
            break

    if target_n > 0 and len(kept) > target_n:
        if canonical_correct:
            out = [canonical_correct]
            out.extend([x for x in kept if _norm_text(x) != _norm_text(canonical_correct)])
            kept = out[:target_n]
        else:
            kept = kept[:target_n]

    return kept


@dataclass
class Stats:
    total: int = 0
    written: int = 0
    skipped_no_image: int = 0
    skipped_not_found: int = 0
    skipped_ambiguous: int = 0
    skipped_no_labels: int = 0
    skipped_empty_label: int = 0
    skipped_label_not_in_candidates: int = 0


def convert(
    input_rows: list[dict[str, Any]],
    image_index: dict[str, list[dict[str, Any]]],
    mcq_option_index: dict[str, list[dict[str, Any]]],
    strict: bool,
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
        mcq_row = _pick_mcq_row(image=image, picked=picked, mcq_option_index=mcq_option_index)
        if mcq_row is not None:
            candidate_labels = _labels_from_mcq_row(mcq_row)
        if not candidate_labels:
            candidate_labels = _labels_from_4options(input_answer=input_answer, picked=picked)
        if not candidate_labels:
            stats.skipped_no_labels += 1
            continue

        allowed_labels: list[str] = []
        if isinstance(input_answer, dict) and isinstance(input_answer.get("candidate_labels"), list):
            allowed_labels = _dedup_labels(input_answer.get("candidate_labels", []))
        if allowed_labels:
            target_n = 4 if len(allowed_labels) >= 4 else len(allowed_labels)
            candidate_labels = _constrain_candidates(
                candidates=candidate_labels,
                allowed_labels=allowed_labels,
                correct_label=str(input_correct if input_correct is not None else picked.get("gt_answer", "")),
                target_n=target_n,
            )

        label_text = _pick_label_text(input_correct, candidate_labels, picked.get("gt_answer"))
        if not label_text:
            stats.skipped_empty_label += 1
            continue

        if _norm_text(label_text) not in {_norm_text(x) for x in candidate_labels}:
            stats.skipped_label_not_in_candidates += 1
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
                "prompt": _build_prompt(str(picked.get("question", "")).strip(), candidate_labels),
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
            f"skipped_no_labels={stats.skipped_no_labels}, skipped_empty_label={stats.skipped_empty_label}, "
            f"skipped_label_not_in_candidates={stats.skipped_label_not_in_candidates}"
        )

    return out, stats


def main() -> None:
    p = argparse.ArgumentParser(description="Build ISIC text-answer data using only 4-option candidate labels.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--qa-open-access-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--mcq-option-source",
        type=Path,
        default=None,
        help="Optional MCQ-letter JSONL to source candidate options (defaults to sibling MCQ_<input_name>.jsonl if exists).",
    )
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    input_rows = _read_jsonl(args.input)
    qa_rows = _load_isic_rows(args.qa_open_access_dir)
    image_index = _build_index(qa_rows)

    mcq_option_source = args.mcq_option_source
    if mcq_option_source is None:
        guessed = args.input.with_name(f"MCQ_{args.input.name}")
        if guessed.exists():
            mcq_option_source = guessed
    mcq_option_rows: list[dict[str, Any]] = []
    if mcq_option_source is not None and mcq_option_source.exists():
        mcq_option_rows = _read_jsonl(mcq_option_source)
    mcq_option_index = _build_mcq_option_index(mcq_option_rows)

    out_rows, stats = convert(
        input_rows=input_rows,
        image_index=image_index,
        mcq_option_index=mcq_option_index,
        strict=args.strict,
    )
    _write_jsonl(args.output, out_rows)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "qa_open_access_dir": str(args.qa_open_access_dir),
                "output": str(args.output),
                "mcq_option_source": str(mcq_option_source) if mcq_option_source is not None else None,
                "strict": args.strict,
                "stats": stats.__dict__,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
