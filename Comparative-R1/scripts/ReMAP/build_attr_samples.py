#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from teacher_api import TeacherClient


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config" / "attr_config.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            content = line.strip()
            if not content:
                continue
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl_rows(file_obj: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
    file_obj.flush()


def _write_progress(
    file_obj: Any,
    *,
    base_index: int,
    total_base_rows: int,
    image_rel: str,
    status: str,
    generated_rows: int,
    filtered_rows: int,
    skipped_rows: int,
    total_generated_rows: int,
    message: str = "",
) -> None:
    payload = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "base_index": base_index,
        "total_base_rows": total_base_rows,
        "image_rel": image_rel,
        "status": status,
        "generated_rows": generated_rows,
        "filtered_rows": filtered_rows,
        "skipped_rows": skipped_rows,
        "total_generated_rows": total_generated_rows,
        "message": message,
    }
    file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")
    file_obj.flush()


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be object: {path}")
    return payload


def _normalize_answer_type(value: str) -> str:
    text = value.strip().lower()
    if text in {"bool", "boolean", "yesno", "yes_no"}:
        return "bool"
    if text in {"short_text", "text", "string"}:
        return "short_text"
    if text in {"list", "keywords"}:
        return "list"
    return "short_text"


def _extract_image_rel(row: dict[str, Any]) -> str:
    images = row.get("images")
    if isinstance(images, list) and images and isinstance(images[0], str):
        return images[0]
    image = row.get("image")
    if isinstance(image, str) and image.strip():
        return image.strip()
    raise ValueError("Missing image/images in input row")


def _extract_prompt_text(row: dict[str, Any]) -> str:
    for key in ("prompt", "problem", "question"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Classify this dermatology image."


def _extract_label(row: dict[str, Any]) -> str:
    value = row.get("label")
    if isinstance(value, str) and value.strip():
        return value.strip()
    answer = row.get("answer")
    if isinstance(answer, dict):
        label = answer.get("label")
        if isinstance(label, str) and label.strip():
            return label.strip()
    return ""


def _parse_list_from_prompt(prompt: str) -> list[str]:
    match = re.search(r"\[\s*(.*?)\s*\]", prompt, flags=re.DOTALL)
    if not match:
        return []
    raw = match.group(1)
    candidates = [item.strip() for item in raw.split(",")]
    return [item for item in candidates if item]


def _normalize_free_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.strip().lower()).strip()


def _contains_label_text(text: str, labels: list[str]) -> bool:
    normalized = f" {_normalize_free_text(text)} "
    for label in labels:
        label_norm = _normalize_free_text(label)
        if not label_norm:
            continue
        if f" {label_norm} " in normalized:
            return True
    return False


def _is_diagnostic_attr(
    *,
    question: str,
    correct_answer: str | list[str],
    candidate_answers: list[str],
    labels: list[str],
) -> bool:
    diagnostic_pattern = re.compile(
        r"\b(diagnos(?:is|e|tic)?|class(?:ification|ify|ified)?|category|label|disease|condition|benign|malignant)\b",
        flags=re.IGNORECASE,
    )
    answer_text = ", ".join(correct_answer) if isinstance(correct_answer, list) else correct_answer
    merged = "\n".join([question, answer_text, " ".join(candidate_answers)])
    if diagnostic_pattern.search(question):
        return True
    return _contains_label_text(merged, labels)


def _resolve_image_path(image_rel: str, image_root: str | None) -> Path:
    path = Path(image_rel)
    if path.is_absolute():
        return path
    if image_root:
        return Path(image_root) / image_rel.lstrip("/")
    return path


def _build_teacher_prompt(
    row: dict[str, Any],
    *,
    labels: list[str],
    max_questions: int,
    prompt_version: str,
) -> str:
    task_prompt = _extract_prompt_text(row)
    class_label = _extract_label(row)
    allowed = ", ".join(labels) if labels else "N/A"
    return f"""
You are generating image-grounded supervision for dermatology reasoning.

Prompt version: {prompt_version}
Primary task question (for context only):
{task_prompt}

Ground-truth diagnosis label (context only; DO NOT ask this directly):
{class_label}

Allowed diagnosis labels (context only; DO NOT use as question/answer options):
{allowed}

Return STRICT JSON object with this schema:
{{
  "attributes": [
    {{
      "attr_id": "short_snake_case_id",
      "question": "single clear visual question",
      "answer_type": "bool|short_text|list",
      "correct_answer": "yes/no OR short text OR list text",
      "candidate_answers": ["optional closed set"],
      "keywords": ["optional key terms used for matching"]
    }}
  ]
}}

Constraints:
- Generate at most {max_questions} attributes.
- Each question must be answerable from the given image and task.
- Ask ONLY visual attributes or visual-understanding details that support the task (e.g., color, border, asymmetry, texture, morphology, distribution, count, location, relative size).
- Do NOT ask diagnostic/classification/category questions.
- Do NOT ask for disease name, class label, benign/malignant judgement, or "which option from label list".
- Do NOT include diagnosis labels in question/candidate_answers/correct_answer.
- Keep answers deterministic and short.
- For bool, use only yes or no.
- For list, use 2-5 comma-separated items.
- Do not output extra keys outside the schema.
""".strip()


def _build_student_prompt(question: str, answer_type: str, candidate_answers: list[str]) -> str:
    lines = [
        "You are a dermatology visual assistant.",
        "<image>",
        f"Question: {question}",
    ]
    if answer_type == "bool":
        lines.append("Answer with exactly one token: yes or no.")
    elif answer_type == "list":
        lines.append("Answer with a short comma-separated list.")
    else:
        lines.append("Answer with a short text phrase.")
    if candidate_answers:
        lines.append("Candidate answers:")
        lines.extend([f"- {item}" for item in candidate_answers])
    lines.append("Use format: <answer>...</answer>")
    return "\n".join(lines)


def main() -> None:
    config_path = Path(os.getenv("REMAP_ATTR_CONFIG", str(DEFAULT_CONFIG_PATH)))
    config = _load_config(config_path)

    input_jsonl = Path(str(config["input_jsonl"]))
    output_jsonl = Path(str(config["output_jsonl"]))
    image_root = str(config.get("image_root", "")).strip() or None
    cache_dir = Path(str(config["cache_dir"])) if str(config.get("cache_dir", "")).strip() else (output_jsonl.parent / "cache_attr")
    max_questions = int(config.get("max_questions", 4))
    prompt_version = str(config.get("prompt_version", "attr_v1"))
    max_samples = int(config.get("max_samples", 0))
    temperature = float(config.get("temperature", 0.0))
    max_tokens = int(config.get("max_tokens", 1200))

    rows = _read_jsonl(input_jsonl)
    if max_samples > 0:
        rows = rows[: max_samples]

    labels = sorted({label for label in (_extract_label(row) for row in rows) if label})
    client = TeacherClient.from_config(config, cache_dir=cache_dir)
    progress_jsonl = Path(str(output_jsonl) + ".progress.jsonl")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("", encoding="utf-8")
    progress_jsonl.write_text("", encoding="utf-8")

    stats = {
        "total_base_rows": len(rows),
        "processed_base_rows": 0,
        "generated_attr_rows": 0,
        "skipped_rows": 0,
        "filtered_diagnostic_attrs": 0,
        "progress_jsonl": str(progress_jsonl),
    }
    missing_image_examples: list[str] = []
    with output_jsonl.open("a", encoding="utf-8") as output_file_obj, progress_jsonl.open("a", encoding="utf-8") as progress_file_obj:
        for index, row in enumerate(rows):
            stats["processed_base_rows"] += 1
            row_generated_rows: list[dict[str, Any]] = []
            row_filtered = 0
            image_rel = ""
            status = "ok"
            message = ""
            try:
                image_rel = _extract_image_rel(row)
                image_path = _resolve_image_path(image_rel, image_root)
                if not image_path.exists():
                    stats["skipped_rows"] += 1
                    if len(missing_image_examples) < 10:
                        missing_image_examples.append(str(image_path))
                    status = "missing_image"
                    message = f"image not found: {image_path}"
                    _write_progress(
                        progress_file_obj,
                        base_index=index,
                        total_base_rows=len(rows),
                        image_rel=image_rel,
                        status=status,
                        generated_rows=0,
                        filtered_rows=0,
                        skipped_rows=stats["skipped_rows"],
                        total_generated_rows=stats["generated_attr_rows"],
                        message=message,
                    )
                    print(
                        f"[{index + 1}/{len(rows)}] status={status} generated=0 "
                        f"filtered=0 total_generated={stats['generated_attr_rows']}"
                    )
                    continue

                row_labels = labels or _parse_list_from_prompt(_extract_prompt_text(row))
                task_prompt = _build_teacher_prompt(
                    row,
                    labels=row_labels,
                    max_questions=max_questions,
                    prompt_version=prompt_version,
                )
                result = client.chat_json(
                    user_prompt=task_prompt,
                    image_paths=[image_path],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                parsed = result["parsed"]
                attributes = parsed.get("attributes", [])
                if not isinstance(attributes, list):
                    stats["skipped_rows"] += 1
                    status = "invalid_teacher_schema"
                    message = "parsed.attributes is not a list"
                    _write_progress(
                        progress_file_obj,
                        base_index=index,
                        total_base_rows=len(rows),
                        image_rel=image_rel,
                        status=status,
                        generated_rows=0,
                        filtered_rows=0,
                        skipped_rows=stats["skipped_rows"],
                        total_generated_rows=stats["generated_attr_rows"],
                        message=message,
                    )
                    print(
                        f"[{index + 1}/{len(rows)}] image={image_rel} status={status} generated=0 "
                        f"filtered=0 total_generated={stats['generated_attr_rows']}"
                    )
                    continue

                for attr_index, attr in enumerate(attributes):
                    if not isinstance(attr, dict):
                        continue
                    question = str(attr.get("question", "")).strip()
                    correct_answer = attr.get("correct_answer", "")
                    answer_type = _normalize_answer_type(str(attr.get("answer_type", "short_text")))
                    if not question:
                        continue
                    if isinstance(correct_answer, list):
                        normalized_answer: str | list[str] = [str(item).strip() for item in correct_answer if str(item).strip()]
                    else:
                        normalized_answer = str(correct_answer).strip()
                    candidate_answers_raw = attr.get("candidate_answers", [])
                    if isinstance(candidate_answers_raw, list):
                        candidate_answers = [str(item).strip() for item in candidate_answers_raw if str(item).strip()]
                    else:
                        candidate_answers = []
                    keywords_raw = attr.get("keywords", [])
                    if isinstance(keywords_raw, list):
                        keywords = [str(item).strip().lower() for item in keywords_raw if str(item).strip()]
                    else:
                        keywords = []
                    if _is_diagnostic_attr(
                        question=question,
                        correct_answer=normalized_answer,
                        candidate_answers=candidate_answers,
                        labels=row_labels,
                    ):
                        row_filtered += 1
                        stats["filtered_diagnostic_attrs"] += 1
                        continue

                    sample_id = f"attr_{index:06d}_{attr_index:02d}"
                    row_generated_rows.append(
                        {
                            "prompt": _build_student_prompt(question, answer_type, candidate_answers),
                            "images": [image_rel],
                            "answer": {
                                "task_type": "attr",
                                "source_type": "attr",
                                "attr_id": str(attr.get("attr_id", sample_id)),
                                "answer_type": answer_type,
                                "correct_answer": normalized_answer,
                                "candidate_answers": candidate_answers,
                                "keywords": keywords,
                                "class_label": _extract_label(row),
                            },
                            "meta": {
                                "source_type": "attr",
                                "sample_id": sample_id,
                                "origin_input": str(input_jsonl),
                                "teacher_model": client.config.model,
                                "prompt_version": prompt_version,
                                "image_path": image_rel,
                            },
                        }
                    )

                if row_generated_rows:
                    _append_jsonl_rows(output_file_obj, row_generated_rows)
                stats["generated_attr_rows"] += len(row_generated_rows)
                _write_progress(
                    progress_file_obj,
                    base_index=index,
                    total_base_rows=len(rows),
                    image_rel=image_rel,
                    status=status,
                    generated_rows=len(row_generated_rows),
                    filtered_rows=row_filtered,
                    skipped_rows=stats["skipped_rows"],
                    total_generated_rows=stats["generated_attr_rows"],
                )
                print(
                    f"[{index + 1}/{len(rows)}] image={image_rel} status={status} generated={len(row_generated_rows)} "
                    f"filtered={row_filtered} total_generated={stats['generated_attr_rows']}"
                )
            except Exception as exc:
                stats["skipped_rows"] += 1
                status = "error"
                message = str(exc)
                _write_progress(
                    progress_file_obj,
                    base_index=index,
                    total_base_rows=len(rows),
                    image_rel=image_rel,
                    status=status,
                    generated_rows=0,
                    filtered_rows=row_filtered,
                    skipped_rows=stats["skipped_rows"],
                    total_generated_rows=stats["generated_attr_rows"],
                    message=message,
                )
                print(
                    f"[{index + 1}/{len(rows)}] image={image_rel or '<unknown>'} status={status} generated=0 "
                    f"filtered={row_filtered} total_generated={stats['generated_attr_rows']} error={message}"
                )
                continue
    stats["image_root"] = image_root or ""
    stats["config_path"] = str(config_path)
    stats["missing_image_examples"] = missing_image_examples

    summary_path = Path(str(output_jsonl) + ".summary.json")
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {stats['generated_attr_rows']} rows -> {output_jsonl}")
    print(f"Progress -> {progress_jsonl}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
