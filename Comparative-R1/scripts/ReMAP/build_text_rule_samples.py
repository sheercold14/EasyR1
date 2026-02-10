#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from teacher_api import TeacherClient


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config" / "text_rule_config.json"


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be object: {path}")
    return payload


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


def _normalize_ws(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_answer_type(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"bool", "boolean", "yesno", "yes_no"}:
        return "bool"
    if text in {"mcq4", "abcd", "a/b/c/d"}:
        return "mcq4"
    if text in {"binary", "a/b", "ab"}:
        return "binary"
    return "short_text"


def _token_count(text: str) -> int:
    return len([tok for tok in str(text).strip().split() if tok])


def _read_guideline_with_line_numbers(path: Path) -> tuple[str, list[str]]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    numbered_lines: list[str] = []
    for i, line in enumerate(raw_lines, start=1):
        numbered_lines.append(f"L{i:04d}: {line}")
    return "\n".join(numbered_lines).strip(), raw_lines


def _contains_direct_quote(text: str, evidence_lines: list[str]) -> bool:
    hay = _normalize_ws(text)
    if not hay:
        return False
    for ln in evidence_lines:
        src = _normalize_ws(str(ln))
        if len(src) >= 40 and src in hay:
            return True
    return False


def _build_teacher_prompt(
    *,
    labels: list[str],
    guideline_text_numbered: str,
    num_families: int,
    prompt_version: str,
) -> str:
    label_text = ", ".join(labels) if labels else "N/A"
    return f"""
You are creating STRICT domain-knowledge injection tasks.

IMPORTANT: Only create RULE EXECUTION questions. No explanation. No summarization. No open-ended QA.
The rule MUST be explicit in the question body (scope, trigger conditions, threshold, exceptions, conclusion).

Prompt version: {prompt_version}
Diagnosis labels:
{label_text}

Guideline/codebook text (with line numbers):
{guideline_text_numbered}

Return STRICT JSON object with this schema:
{{
  "families": [
    {{
      "rule_id": "short_snake_case_id",
      "scope": "explicit applicability scope",
      "conditions": ["explicit trigger conditions (checkable from case text)"],
      "threshold": "explicit threshold/decision boundary (if any)",
      "exceptions": ["explicit exceptions (optional)"],
      "conclusion": "explicit conclusion/decision outcome",
      "labels_involved": ["subset of provided labels"],
      "evidence": {{"line_start": 1, "line_end": 3}},
      "variants": [
        {{
          "variant": "affirm",
          "case": "explicit case facts; include all relevant conditions",
          "question": "rule execution question",
          "answer_type": "bool|mcq4|binary|short_text",
          "correct_answer": "yes|no OR A/B/C/D OR A/B OR <=20 tokens short fill"
        }},
        {{
          "variant": "negated_condition",
          "case": "change exactly one key condition so the rule does NOT apply (or flips outcome)",
          "question": "same semantics but different surface form",
          "answer_type": "bool|mcq4|binary|short_text",
          "correct_answer": "verifiable answer"
        }},
        {{
          "variant": "counterfactual",
          "case": "counterfactual: flip one condition or change threshold; keep it explicit",
          "question": "counterfactual rule execution question",
          "answer_type": "bool|mcq4|binary|short_text",
          "correct_answer": "verifiable answer"
        }}
      ]
    }}
  ]
}}

Constraints:
- Generate exactly {num_families} families.
- Each family MUST have exactly 3 variants: affirm, negated_condition, counterfactual.
- NO task-irrelevant questions. NO hallucinated medical facts not supported by evidence lines.
- Evidence must be a valid line range in the provided guideline (line_start <= line_end).
- DO NOT copy any guideline sentence into rule fields verbatim; paraphrase.
- Allowed answer types only: bool, mcq4, binary, short_text.
- For bool: correct_answer must be only yes or no.
- For mcq4: correct_answer must be one of A/B/C/D.
- For binary: correct_answer must be one of A/B.
- For short_text: <= 20 tokens, terminology/threshold/keyword only.
""".strip()


def _build_student_prompt(*, family: dict[str, Any], variant: dict[str, Any]) -> str:
    scope = str(family.get("scope", "")).strip()
    threshold = str(family.get("threshold", "")).strip()
    conclusion = str(family.get("conclusion", "")).strip()
    conditions = family.get("conditions", [])
    exceptions = family.get("exceptions", [])

    if not isinstance(conditions, list):
        conditions = []
    if not isinstance(exceptions, list):
        exceptions = []

    conditions_txt = "\n".join([f"- {str(c).strip()}" for c in conditions if str(c).strip()]) or "- (none)"
    exceptions_txt = "\n".join([f"- {str(e).strip()}" for e in exceptions if str(e).strip()]) or "- (none)"

    case_text = str(variant.get("case", "")).strip()
    question = str(variant.get("question", "")).strip()
    answer_type = _normalize_answer_type(variant.get("answer_type", "bool"))

    lines = [
        "You are a medical rule executor.",
        "",
        "Rule (execute exactly; no extra knowledge):",
        f"Scope: {scope}",
        "Trigger conditions:",
        conditions_txt,
        f"Threshold: {threshold}" if threshold else "Threshold: (none)",
        "Exceptions:",
        exceptions_txt,
        f"Conclusion: {conclusion}" if conclusion else "Conclusion: (none)",
        "",
        f"Case: {case_text}",
        "",
        f"Question: {question}",
    ]

    if answer_type == "bool":
        lines.append("Answer with exactly one token: yes or no.")
    elif answer_type == "mcq4":
        lines.append("Answer with exactly one letter: A, B, C, or D.")
    elif answer_type == "binary":
        lines.append("Answer with exactly one letter: A or B.")
    else:
        lines.append("Answer with <= 20 tokens (term/threshold/keyword only).")
    lines.append("Use format: <answer>...</answer>")
    return "\n".join(lines)


def _validate_family(
    family: dict[str, Any],
    *,
    labels: list[str],
    guideline_lines: list[str],
) -> None:
    rule_id = str(family.get("rule_id", "")).strip()
    if not rule_id:
        raise ValueError("family missing rule_id")

    evidence = family.get("evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError(f"{rule_id}: evidence must be object")
    line_start = int(evidence.get("line_start", 0))
    line_end = int(evidence.get("line_end", 0))
    if line_start <= 0 or line_end <= 0 or line_end < line_start:
        raise ValueError(f"{rule_id}: invalid evidence line range")
    if line_end > len(guideline_lines):
        raise ValueError(f"{rule_id}: evidence lines out of range (max={len(guideline_lines)})")

    evidence_lines = guideline_lines[line_start - 1 : line_end]

    for field in ("scope", "conclusion"):
        if not str(family.get(field, "")).strip():
            raise ValueError(f"{rule_id}: missing {field}")
    conditions = family.get("conditions", [])
    if not isinstance(conditions, list) or not any(str(c).strip() for c in conditions):
        raise ValueError(f"{rule_id}: missing conditions")

    # Avoid verbatim guideline text in the rule fields.
    for field in ("scope", "threshold", "conclusion"):
        if _contains_direct_quote(str(family.get(field, "")), evidence_lines):
            raise ValueError(f"{rule_id}: field '{field}' contains a direct quote from guideline")
    for c in conditions:
        if _contains_direct_quote(str(c), evidence_lines):
            raise ValueError(f"{rule_id}: a condition contains a direct quote from guideline")

    labels_involved = family.get("labels_involved", [])
    if not isinstance(labels_involved, list):
        raise ValueError(f"{rule_id}: labels_involved must be list")
    label_set = set(labels)
    for item in labels_involved:
        if str(item).strip() and str(item).strip() not in label_set:
            raise ValueError(f"{rule_id}: labels_involved contains unknown label: {item}")

    variants = family.get("variants", [])
    if not isinstance(variants, list) or len(variants) != 3:
        raise ValueError(f"{rule_id}: variants must have exactly 3 items")
    required = {"affirm", "negated_condition", "counterfactual"}
    got = {str(v.get("variant", "")).strip() for v in variants if isinstance(v, dict)}
    if got != required:
        raise ValueError(f"{rule_id}: variants must contain exactly {sorted(required)}")

    banned = ("summarize", "explain", "reason", "why", "analysis", "chain-of-thought", "cot")
    for v in variants:
        if not isinstance(v, dict):
            raise ValueError(f"{rule_id}: variant must be object")
        variant_name = str(v.get("variant", "")).strip()
        question = str(v.get("question", "")).strip()
        if not question:
            raise ValueError(f"{rule_id}:{variant_name}: missing question")
        ql = question.lower()
        if any(tok in ql for tok in banned):
            raise ValueError(f"{rule_id}:{variant_name}: open-ended wording detected in question")

        if not str(v.get("case", "")).strip():
            raise ValueError(f"{rule_id}:{variant_name}: missing case")

        answer_type = _normalize_answer_type(v.get("answer_type", "bool"))
        correct_answer = str(v.get("correct_answer", "")).strip()
        if answer_type == "bool":
            if correct_answer.lower() not in {"yes", "no"}:
                raise ValueError(f"{rule_id}:{variant_name}: bool correct_answer must be yes/no")
        elif answer_type == "mcq4":
            if correct_answer.upper() not in {"A", "B", "C", "D"}:
                raise ValueError(f"{rule_id}:{variant_name}: mcq4 correct_answer must be A/B/C/D")
        elif answer_type == "binary":
            if correct_answer.upper() not in {"A", "B"}:
                raise ValueError(f"{rule_id}:{variant_name}: binary correct_answer must be A/B")
        else:
            if _token_count(correct_answer) > 20:
                raise ValueError(f"{rule_id}:{variant_name}: short_text correct_answer > 20 tokens")


def main() -> None:
    config_path = Path(os.getenv("REMAP_TEXT_RULE_CONFIG", str(DEFAULT_CONFIG_PATH)))
    config = _load_config(config_path)

    cls_jsonl = Path(str(config["cls_jsonl"]))
    guideline_path = Path(str(config["guideline_path"]))
    output_jsonl = Path(str(config["output_jsonl"]))
    cache_dir = (
        Path(str(config["cache_dir"]))
        if str(config.get("cache_dir", "")).strip()
        else (output_jsonl.parent / "cache_text_rule")
    )
    num_families = int(config.get("num_families", config.get("num_questions", 5)))
    prompt_version = str(config.get("prompt_version", "text_rule_exec_family_v1"))
    temperature = float(config.get("temperature", 0.0))
    max_tokens = int(config.get("max_tokens", 2500))

    cls_rows = _read_jsonl(cls_jsonl)
    labels = sorted({label for label in (_extract_label(row) for row in cls_rows) if label})
    guideline_text_numbered, guideline_lines = _read_guideline_with_line_numbers(guideline_path)

    client = TeacherClient.from_config(config, cache_dir=cache_dir)
    prompt = _build_teacher_prompt(
        labels=labels,
        guideline_text_numbered=guideline_text_numbered,
        num_families=num_families,
        prompt_version=prompt_version,
    )
    result = client.chat_json(
        user_prompt=prompt,
        image_paths=[],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed = result["parsed"]
    families = parsed.get("families", [])
    if not isinstance(families, list):
        raise RuntimeError("Teacher output has no families list")

    out_rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        if not isinstance(family, dict):
            raise ValueError("family must be object")
        _validate_family(family, labels=labels, guideline_lines=guideline_lines)

        rule_id = str(family.get("rule_id", "")).strip()
        evidence = family.get("evidence", {})
        line_start = int(evidence.get("line_start", 0))
        line_end = int(evidence.get("line_end", 0))

        variants = family.get("variants", [])
        for variant in variants:
            variant_name = str(variant.get("variant", "")).strip()
            answer_type = _normalize_answer_type(variant.get("answer_type", "bool"))
            correct_answer = str(variant.get("correct_answer", "")).strip()
            if answer_type == "bool":
                correct_answer = correct_answer.lower()
            elif answer_type in {"mcq4", "binary"}:
                correct_answer = correct_answer.upper()

            tags = [f"family:{rule_id}", f"variant:{variant_name}"]
            sample_id = f"text_rule_{family_index:04d}_{variant_name}"
            out_rows.append(
                {
                    "prompt": _build_student_prompt(family=family, variant=variant),
                    "answer": {
                        "task_type": "text_rule",
                        "source_type": "text_rule",
                        "rule_id": rule_id,
                        "answer_type": answer_type,
                        "correct_answer": correct_answer,
                        "tags": tags,
                        "evidence": {
                            "path": str(guideline_path),
                            "line_start": line_start,
                            "line_end": line_end,
                        },
                    },
                    "meta": {
                        "source_type": "text_rule",
                        "sample_id": sample_id,
                        "family": rule_id,
                        "variant": variant_name,
                        "origin_cls_input": str(cls_jsonl),
                        "origin_guideline": str(guideline_path),
                        "teacher_model": client.config.model,
                        "prompt_version": prompt_version,
                        "config_path": str(config_path),
                    },
                }
            )

    _write_jsonl(output_jsonl, out_rows)
    summary = {
        "labels": labels,
        "num_families": num_families,
        "num_output_rows": len(out_rows),
        "teacher_model": client.config.model,
        "prompt_version": prompt_version,
        "config_path": str(config_path),
    }
    summary_path = Path(str(output_jsonl) + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out_rows)} rows -> {output_jsonl}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
