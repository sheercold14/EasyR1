#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
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
    return " ".join(str(text).strip().split())


def _sanitize_label_id(label: str) -> str:
    s = str(label).strip().upper()
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "UNK"


def _load_existing_output_state(output_jsonl: Path) -> tuple[int, dict[str, int], int]:
    if not output_jsonl.exists():
        return 0, {}, 0
    gen_i = 0
    per_label_i: dict[str, int] = {}
    row_count = 0
    re_gen = re.compile(r"^GEN_(\d+)$")
    re_lbl = re.compile(r"^LBL_([A-Z0-9_]+)_(\d+)$")

    with output_jsonl.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            content = line.strip()
            if not content:
                continue
            row_count += 1
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {output_jsonl}:{line_no}")
            answer = payload.get("answer", {})
            if not isinstance(answer, dict):
                continue
            group_id = str(answer.get("group_id", "")).strip()
            if not group_id:
                continue
            m = re_gen.match(group_id)
            if m:
                gen_i = max(gen_i, int(m.group(1)))
                continue
            m = re_lbl.match(group_id)
            if m:
                label_id = m.group(1)
                idx = int(m.group(2))
                per_label_i[label_id] = max(per_label_i.get(label_id, 0), idx)
    return gen_i, per_label_i, row_count


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        src = _normalize_ws(ln)
        if len(src) >= 40 and src in hay:
            return True
    return False


def _build_teacher_prompt(
    *,
    labels: list[str],
    guideline_text_numbered: str,
    num_groups: int,
    prompt_version: str,
    task_name: str,
    group_targets: list[dict[str, Any]] | None = None,
    focus_line_range: tuple[int, int] | None = None,
    focus_strict: bool = False,
) -> str:
    label_text = ", ".join(labels) if labels else "N/A"
    composition_block = ""
    if group_targets:
        lines: list[str] = []
        lines.append("Composition constraints (STRICT):")
        lines.append("You MUST generate qa_groups in EXACT order listed below.")
        lines.append("Each qa_group MUST match scope/label/answer_type exactly.")
        for idx, tgt in enumerate(group_targets, start=1):
            scope = str(tgt.get("scope", "")).strip()
            answer_type = str(tgt.get("answer_type", "")).strip()
            label = tgt.get("label", None)
            label_text_value = "null" if label is None else str(label)
            lines.append(f"{idx}) scope={scope}, label={label_text_value}, answer_type={answer_type}")
        lines.append("")
        lines.append("Do NOT add any extra groups beyond this list.")
        lines.append("")
        composition_block = "\n".join(lines)

    focus_block = ""
    if focus_line_range is not None:
        start, end = focus_line_range
        if start <= 0 or end < start:
            raise ValueError(f"Invalid focus_line_range: {focus_line_range}")
        if focus_strict:
            focus_block = (
                "Batch guideline focus (STRICT):\n"
                f"- Use evidence.lines ONLY from guideline line numbers in [{start}, {end}].\n"
                "- Do NOT mention guideline/line/citation in the question text; evidence goes only in evidence.\n"
            )
        else:
            focus_block = (
                "Batch guideline focus (IMPORTANT):\n"
                f"- Prefer choosing evidence.lines from guideline line numbers in [{start}, {end}] for this batch.\n"
                "- Avoid using evidence outside this range unless absolutely necessary.\n"
                "- Do NOT mention guideline/line/citation in the question text; evidence goes only in evidence.\n"
            )
    return f"""
You are creating domain-knowledge injection rule-execution questions for: {task_name}

Labels:
{label_text}

Guideline/codebook text (with line numbers):
{guideline_text_numbered}

Goal:
Convert the guideline into strictly verifiable Q/A groups.
Each Q/A group represents one atomic, non-ambiguous knowledge point.

{composition_block}

{focus_block}

Hard constraints:
1) Questions MUST be rule-execution questions (no summarization, no explanation, no open-ended QA).
   The rule trigger conditions / scope / thresholds / exceptions must be explicit IN THE QUESTION BODY.
2) Answers MUST be closed-form and auto-gradable:
   - boolean (yes/no)
	   - text (single term chosen from a candidate term list)
	   - short_list (multiple terms chosen from a candidate term list; <= 5 items; each item <= 20 tokens)
	3) Evidence must be line-number location only, and MUST ONLY appear in the evidence field.
	   The question text MUST NOT mention guideline/codebook/line numbers/citations
	4) Create BOTH:
	   - general groups (scope="general", label=null)
	   - per-label groups (scope="class", label=<one of labels>)
	5) Each group MUST contain 3 variants that are semantically equivalent (same answer),
	   but with different surface forms.
6) No task-irrelevant content. No hallucinated claims not supported by evidence.

Return STRICT JSON with this schema EXACTLY:

{{
  "task": {{"name": string, "labels": [string]}},
  "qa_groups": [
    {{
      "group_id": string,
      "scope": "general" | "class",
      "label": string | null,
      "evidence": [{{"source":"guideline"|"general_knowledge", "lines":[int]}}],
      "answer_type": "boolean" | "text" | "short_list",
      "answer_options": [string],
      "canonical_answer": string | [string],
      "variants": [
        {{"variant_id": string, "question": string, "answer": string | [string]}}
      ]
    }}
  ]
}}

Naming rules (IMPORTANT: use sanitized LABEL_ID):
- LABEL_ID = uppercase(label) with non-alphanumerics replaced by "_"
- group_id: "GEN_###" for general; "LBL_{{LABEL_ID}}_###" for class
- variant_id: "{{group_id}}_v1" / "{{group_id}}_v2" / "{{group_id}}_v3"

Generation rules:
- Generate exactly {num_groups} qa_groups.
- Each group MUST have exactly 3 variants.
- For boolean: canonical_answer must be "yes" or "no".
- For text: canonical_answer must be one string from answer_options.
- For short_list: canonical_answer must be an array of strings (<=5 items), each from answer_options.

Output ONLY valid JSON. No markdown. No extra text.

Prompt version: {prompt_version}
""".strip()


def _compute_focus_line_range(*, total_lines: int, num_blocks: int, block_index: int) -> tuple[int, int]:
    if total_lines <= 0:
        raise ValueError("total_lines must be > 0")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be > 0")
    if block_index < 0 or block_index >= num_blocks:
        raise ValueError("block_index out of range")
    block_size = (total_lines + num_blocks - 1) // num_blocks  # ceil
    start = block_index * block_size + 1
    end = min((block_index + 1) * block_size, total_lines)
    return start, end


def _group_uses_guideline_lines_outside_range(group: dict[str, Any], start: int, end: int) -> list[int]:
    outside: list[int] = []
    evidence = group.get("evidence", [])
    if not isinstance(evidence, list):
        return outside
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        if str(ev.get("source", "")).strip() != "guideline":
            continue
        lines = ev.get("lines", [])
        if not isinstance(lines, list):
            continue
        for ln in lines:
            if not isinstance(ln, int):
                continue
            if ln < start or ln > end:
                outside.append(ln)
    return sorted(set(outside))


def _quota_cycle(quota: dict[str, int], order: list[str]) -> list[str]:
    cycle: list[str] = []
    for key in order:
        cycle.extend([key] * int(quota.get(key, 0)))
    return [x for x in cycle if str(x).strip()]


def _build_batch_targets(
    *,
    labels: list[str],
    batch_size: int,
    scope_cycle: list[str],
    scope_offset: int,
    answer_type_cycle: list[str],
    answer_type_offset: int,
    label_cursor: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    if batch_size <= 0:
        return [], scope_offset, answer_type_offset, label_cursor
    if not scope_cycle:
        raise ValueError("scope_cycle is empty; check batch_scope_quota in config")
    if not answer_type_cycle:
        raise ValueError("answer_type_cycle is empty; check batch_answer_type_quota in config")

    scopes = [scope_cycle[(scope_offset + i) % len(scope_cycle)] for i in range(batch_size)]
    answer_types = [
        answer_type_cycle[(answer_type_offset + i) % len(answer_type_cycle)] for i in range(batch_size)
    ]
    # Rotate start offsets across batches to reduce positional bias.
    scope_offset += 1
    answer_type_offset += 1

    targets: list[dict[str, Any]] = []
    for scope, answer_type in zip(scopes, answer_types):
        scope = str(scope).strip()
        answer_type = str(answer_type).strip()
        if scope not in {"general", "class"}:
            raise ValueError(f"Invalid scheduled scope: {scope}")
        if answer_type not in {"boolean", "text", "short_list"}:
            raise ValueError(f"Invalid scheduled answer_type: {answer_type}")
        if scope == "class":
            if not labels:
                raise ValueError("No labels found in cls_jsonl; cannot schedule class-scope targets")
            label = labels[label_cursor % len(labels)]
            label_cursor += 1
        else:
            label = None
        targets.append({"scope": scope, "label": label, "answer_type": answer_type})
    return targets, scope_offset, answer_type_offset, label_cursor


def _normalize_answer_type(value: object) -> str:
    t = str(value).strip().lower()
    if t in {"boolean", "bool", "yesno", "yes_no"}:
        return "boolean"
    if t in {"select_one", "select_many", "single", "multi"}:
        raise ValueError("select_one/select_many are disabled; use text/short_list instead")
    if t in {"text", "short_text", "term"}:
        return "text"
    if t in {"short_list", "list", "multi_text"}:
        return "short_list"
    return "text"


def _normalize_bool(text: str) -> str | None:
    v = _normalize_ws(text).lower()
    if v in {"yes", "y", "true"}:
        return "yes"
    if v in {"no", "n", "false"}:
        return "no"
    return None


def _norm_key(text: str) -> str:
    return _normalize_ws(text).lower()


def _option_lookup(options: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for opt in options:
        key = _norm_key(opt)
        if key and key not in out:
            out[key] = opt
    return out


def _normalize_text_choice(answer: object, options: list[str]) -> str | None:
    if isinstance(answer, list):
        if len(answer) != 1:
            return None
        answer = answer[0]
    key = _norm_key(str(answer))
    if not key:
        return None
    return _option_lookup(options).get(key)


def _normalize_text_list_choice(answer: object, options: list[str]) -> list[str] | None:
    items = _normalize_short_list(answer)
    if not items:
        return None
    lookup = _option_lookup(options)
    mapped: list[str] = []
    for item in items:
        opt = lookup.get(_norm_key(item))
        if opt is None:
            return None
        mapped.append(opt)

    seen: set[str] = set()
    uniq: list[str] = []
    for opt in mapped:
        k = _norm_key(opt)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(opt)

    order = {_norm_key(opt): i for i, opt in enumerate(options)}
    uniq.sort(key=lambda x: order.get(_norm_key(x), 10**9))
    return uniq


def _normalize_short_list(answer: object) -> list[str]:
    if isinstance(answer, list):
        items = [str(x).strip() for x in answer if str(x).strip()]
    else:
        base = str(answer).replace("\n", ",")
        items = [p.strip() for p in base.split(",") if p.strip()]
    items = [_normalize_ws(x) for x in items if _normalize_ws(x)]
    # stable unique
    uniq: list[str] = []
    seen = set()
    for x in items:
        xl = x.lower()
        if xl in seen:
            continue
        seen.add(xl)
        uniq.append(x)
    return uniq


def _build_student_prompt(*, variant_question: str, answer_type: str, answer_options: list[str], task_name: str) -> str:
    q = variant_question.strip()
    lines = [
        f"As an expert in {task_name}",
        "",
        f"Question: {q}",
        "",
    ]
    if answer_type == "boolean":
        lines.append("Answer with exactly one token: yes or no.")
    else:
        opts = [o.strip() for o in answer_options if isinstance(o, str) and o.strip()]
        if answer_type in {"text", "short_list"} and not opts:
            raise ValueError(f"{answer_type} requires non-empty answer_options")
        if opts:
            lines.append("Candidate terms (copy exact term text; do not add extra words):")
            lines.extend([f"- {opt}" for opt in opts])
            lines.append("")
        if answer_type == "text":
            lines.append("Answer with ONLY one term from the candidate list.")
        else:
            lines.append("Answer with a comma-separated list of terms from the candidate list (<=5 items).")
            lines.append("Each item must be <=20 tokens.")
    lines.append("Use format: <answer>...</answer>")
    return "\n".join(lines)


def _validate_group(
    group: dict[str, Any],
    *,
    labels: list[str],
    guideline_lines: list[str],
) -> dict[str, Any]:
    scope = str(group.get("scope", "")).strip()
    if scope not in {"general", "class"}:
        raise ValueError(f"Invalid scope: {scope}")

    label = group.get("label", None)
    label_str = str(label).strip() if label is not None else ""
    if scope == "class":
        if not label_str:
            raise ValueError("class scope requires non-empty label")
        if label_str not in set(labels):
            raise ValueError(f"Unknown label in class scope: {label_str}")
    else:
        if label is not None and label_str:
            raise ValueError("general scope label must be null/empty")

    evidence = group.get("evidence", [])
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence must be non-empty list")
    evidence_lines: list[str] = []
    for ev in evidence:
        if not isinstance(ev, dict):
            raise ValueError("evidence item must be object")
        src = str(ev.get("source", "")).strip()
        lines = ev.get("lines", [])
        if src not in {"guideline", "general_knowledge"}:
            raise ValueError(f"invalid evidence.source: {src}")
        if not isinstance(lines, list):
            raise ValueError("evidence.lines must be list[int]")
        for ln in lines:
            if not isinstance(ln, int):
                raise ValueError("evidence.lines must be list[int]")
            if src == "guideline":
                if ln <= 0 or ln > len(guideline_lines):
                    raise ValueError(f"evidence line out of range: {ln}")
                evidence_lines.append(guideline_lines[ln - 1])

    answer_type = _normalize_answer_type(group.get("answer_type", "boolean"))
    answer_options = group.get("answer_options", [])
    if not isinstance(answer_options, list):
        raise ValueError("answer_options must be list[str]")
    answer_options = [str(x).strip() for x in answer_options if str(x).strip()]

    canonical = group.get("canonical_answer", None)
    if answer_type == "boolean":
        can = _normalize_bool(str(canonical))
        if can is None:
            raise ValueError("boolean canonical_answer must be yes/no")
        canonical_norm: Any = can
    elif answer_type == "text":
        if not answer_options:
            raise ValueError("text requires non-empty answer_options")
        can = _normalize_text_choice(canonical, answer_options)
        if can is None:
            raise ValueError("text canonical_answer must be one term from answer_options")
        if _token_count(can) > 20:
            raise ValueError("text canonical_answer > 20 tokens")
        canonical_norm = can
    else:
        if not answer_options:
            raise ValueError("short_list requires non-empty answer_options")
        can = _normalize_text_list_choice(canonical, answer_options)
        if not can or len(can) > 5:
            raise ValueError("short_list canonical_answer must be 1-5 terms from answer_options")
        for it in can:
            if _token_count(it) > 20:
                raise ValueError("short_list item > 20 tokens")
        canonical_norm = can

    variants = group.get("variants", [])
    if not isinstance(variants, list) or len(variants) != 3:
        raise ValueError("variants must be list of exactly 3 items")

    # Enforce semantic equivalence: all variant answers match canonical answer.
    for v in variants:
        if not isinstance(v, dict):
            raise ValueError("variant must be object")
        q = str(v.get("question", "")).strip()
        if not q:
            raise ValueError("variant missing question")
        if evidence_lines and _contains_direct_quote(q, evidence_lines):
            raise ValueError("question contains a direct guideline quote")

        ans = v.get("answer", None)
        if answer_type == "boolean":
            vn = _normalize_bool(str(ans))
            if vn is None or vn != canonical_norm:
                raise ValueError("boolean variant answer must match canonical_answer")
        elif answer_type == "text":
            vn = _normalize_text_choice(ans, answer_options)
            if vn is None or _norm_key(vn) != _norm_key(str(canonical_norm)):
                raise ValueError("text variant answer must match canonical_answer")
        else:
            vn = _normalize_text_list_choice(ans, answer_options)
            if vn is None:
                raise ValueError("short_list variant answer must be list of terms from answer_options")
            if [_norm_key(x) for x in vn] != [_norm_key(x) for x in canonical_norm]:
                raise ValueError("short_list variant answer must match canonical_answer")

    # Return normalized payload used by downstream output.
    normalized = dict(group)
    normalized["answer_type"] = answer_type
    normalized["answer_options"] = answer_options
    normalized["canonical_answer"] = canonical_norm
    return normalized


def main() -> None:
    config_path = Path(os.getenv("REMAP_TEXT_RULE_CONFIG", str(DEFAULT_CONFIG_PATH)))
    config = _load_config(config_path)

    cls_jsonl = Path(str(config["cls_jsonl"]))
    guideline_path = Path(str(config["guideline_path"]))
    output_jsonl = Path(str(config["output_jsonl"]))
    output_mode = str(config.get("output_mode", "overwrite")).strip().lower()
    append_output = output_mode in {"append", "a", "add"}
    cache_dir = (
        Path(str(config["cache_dir"]))
        if str(config.get("cache_dir", "")).strip()
        else (output_jsonl.parent / "cache_text_rule")
    )

    num_groups_total = int(
        config.get(
            "num_groups_total",
            config.get("num_groups", config.get("num_families", config.get("num_questions", 10))),
        )
    )
    batch_groups = int(config.get("batch_groups", 0))
    prompt_version = str(config.get("prompt_version", "text_rule_qa_groups_v1"))
    temperature = float(config.get("temperature", 0.0))
    max_tokens = int(config.get("max_tokens", 2500))
    task_name = str(config.get("task_name", "Medical Task")).strip() or "Medical Task"
    progress_every_batches = int(config.get("progress_every_batches", 1))
    focus_num_blocks = int(config.get("guideline_focus_num_blocks", 0))
    focus_strict = bool(config.get("guideline_focus_strict", False))

    cls_rows = _read_jsonl(cls_jsonl)
    labels = sorted({label for label in (_extract_label(row) for row in cls_rows) if label})
    guideline_text_numbered, guideline_lines = _read_guideline_with_line_numbers(guideline_path)

    client = TeacherClient.from_config(config, cache_dir=cache_dir)

    scope_counts: dict[str, int] = {"general": 0, "class": 0}
    answer_type_counts: dict[str, int] = {"boolean": 0, "text": 0, "short_list": 0}
    per_label_counts: dict[str, int] = {}
    rows_written = 0
    skipped_groups = 0
    skip_reason_counts: dict[str, int] = {}
    missing_targets_total = 0

    def _mark_skip(reason: str, *, count: int = 1) -> None:
        nonlocal skipped_groups
        skipped_groups += count
        skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + count

    existing_rows = 0
    if append_output:
        gen_i, per_label_i, existing_rows = _load_existing_output_state(output_jsonl)
    else:
        gen_i = 0
        per_label_i = {}
        # Streaming write mode: clear output at start for overwrite mode.
        _write_jsonl(output_jsonl, [], append=False)

    schedule_enabled = any(
        key in config
        for key in (
            "batch_groups",
            "batch_scope_quota",
            "batch_answer_type_quota",
        )
    )
    if not schedule_enabled:
        batch_groups = num_groups_total
    elif batch_groups <= 0:
        raise ValueError("batch_groups must be > 0 when batch scheduling is enabled")

    scope_quota = config.get("batch_scope_quota", None)
    answer_type_quota = config.get("batch_answer_type_quota", None)
    scope_cycle: list[str] = []
    answer_type_cycle: list[str] = []
    if schedule_enabled:
        if not isinstance(scope_quota, dict) or not scope_quota:
            raise ValueError("batch_scope_quota must be a non-empty object when scheduling is enabled")
        if not isinstance(answer_type_quota, dict) or not answer_type_quota:
            raise ValueError("batch_answer_type_quota must be a non-empty object when scheduling is enabled")
        scope_cycle = _quota_cycle({k: int(v) for k, v in scope_quota.items()}, ["general", "class"])
        answer_type_cycle = _quota_cycle(
            {k: int(v) for k, v in answer_type_quota.items()}, ["boolean", "text", "short_list"]
        )
        if len(scope_cycle) != batch_groups:
            raise ValueError("Sum(batch_scope_quota) must equal batch_groups")
        if len(answer_type_cycle) != batch_groups:
            raise ValueError("Sum(batch_answer_type_quota) must equal batch_groups")

    total_groups_done = 0
    batch_idx = 0
    scope_offset = 0
    answer_type_offset = 0
    label_cursor = 0

    while total_groups_done < num_groups_total:
        batch_idx += 1
        remaining = num_groups_total - total_groups_done
        this_batch_groups = min(batch_groups, remaining)

        if schedule_enabled:
            targets, scope_offset, answer_type_offset, label_cursor = _build_batch_targets(
                labels=labels,
                batch_size=this_batch_groups,
                scope_cycle=scope_cycle,
                scope_offset=scope_offset,
                answer_type_cycle=answer_type_cycle,
                answer_type_offset=answer_type_offset,
                label_cursor=label_cursor,
            )
        else:
            targets = []

        focus_range: tuple[int, int] | None = None
        focus_block_meta: dict[str, Any] | None = None
        if focus_num_blocks > 0:
            block_index = (batch_idx - 1) % focus_num_blocks
            start, end = _compute_focus_line_range(
                total_lines=len(guideline_lines),
                num_blocks=focus_num_blocks,
                block_index=block_index,
            )
            focus_range = (start, end)
            focus_block_meta = {
                "block": block_index + 1,
                "num_blocks": focus_num_blocks,
                "line_start": start,
                "line_end": end,
                "strict": focus_strict,
            }

        prompt = _build_teacher_prompt(
            labels=labels,
            guideline_text_numbered=guideline_text_numbered,
            num_groups=this_batch_groups,
            prompt_version=prompt_version,
            task_name=task_name,
            group_targets=targets if schedule_enabled else None,
            focus_line_range=focus_range,
            focus_strict=focus_strict,
        )
        result = client.chat_json(
            user_prompt=prompt,
            image_paths=[],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = result["parsed"]
        qa_groups = parsed.get("qa_groups", [])
        if not isinstance(qa_groups, list):
            print(
                json.dumps(
                    {
                        "batch": batch_idx,
                        "warning": "Teacher output has no qa_groups list; skip batch",
                        "qa_groups_type": type(qa_groups).__name__,
                    },
                    ensure_ascii=False,
                )
            )
            continue
        if len(qa_groups) != this_batch_groups:
            print(
                json.dumps(
                    {
                        "batch": batch_idx,
                        "warning": "qa_groups count mismatch; process available groups only",
                        "expected_groups": this_batch_groups,
                        "actual_groups": len(qa_groups),
                    },
                    ensure_ascii=False,
                )
            )
            if len(qa_groups) > this_batch_groups:
                qa_groups = qa_groups[:this_batch_groups]

        remaining_targets = list(targets) if schedule_enabled else []
        batch_rows: list[dict[str, Any]] = []
        valid_groups_this_batch = 0

        for group_index, group in enumerate(qa_groups, start=1):
            if not isinstance(group, dict):
                _mark_skip("qa_group_not_object")
                print(
                    json.dumps(
                        {
                            "batch": batch_idx,
                            "group_index": group_index,
                            "warning": "qa_group must be object; skipped",
                        },
                        ensure_ascii=False,
                    )
                )
                continue
            try:
                group = _validate_group(group, labels=labels, guideline_lines=guideline_lines)
            except Exception as exc:
                _mark_skip("group_validation_failed")
                print(
                    json.dumps(
                        {
                            "batch": batch_idx,
                            "group_index": group_index,
                            "warning": "qa_group validation failed; skipped",
                            "detail": str(exc),
                        },
                        ensure_ascii=False,
                    )
                )
                continue
            if focus_range is not None and focus_strict:
                start, end = focus_range
                outside = _group_uses_guideline_lines_outside_range(group, start, end)
                if outside:
                    _mark_skip("focus_range_violation")
                    print(
                        json.dumps(
                            {
                                "batch": batch_idx,
                                "group_index": group_index,
                                "warning": "qa_group evidence lines outside strict focus range; skipped",
                                "outside_lines": outside,
                            },
                            ensure_ascii=False,
                        )
                    )
                    continue

            scope = str(group["scope"]).strip()
            label = group.get("label", None)
            label_str = str(label).strip() if label is not None else ""
            answer_type = str(group["answer_type"]).strip()

            if schedule_enabled:
                match_index: int | None = None
                for i, expected in enumerate(remaining_targets):
                    exp_scope = str(expected["scope"]).strip()
                    exp_label = expected.get("label", None)
                    exp_label_str = str(exp_label).strip() if exp_label is not None else ""
                    exp_answer_type = str(expected["answer_type"]).strip()
                    if scope != exp_scope:
                        continue
                    if exp_scope == "general":
                        if label is not None and label_str:
                            continue
                    else:
                        if label_str != exp_label_str:
                            continue
                    if answer_type != exp_answer_type:
                        continue
                    match_index = i
                    break
                if match_index is None:
                    _mark_skip("batch_target_mismatch")
                    print(
                        json.dumps(
                            {
                                "batch": batch_idx,
                                "group_index": group_index,
                                "warning": "qa_group does not match any batch target; skipped",
                                "scope": scope,
                                "label": label_str,
                                "answer_type": answer_type,
                            },
                            ensure_ascii=False,
                        )
                    )
                    continue
                remaining_targets.pop(match_index)

            scope_counts[scope] = scope_counts.get(scope, 0) + 1
            if answer_type in answer_type_counts:
                answer_type_counts[answer_type] += 1
            else:
                answer_type_counts[answer_type] = 1
            if scope == "class":
                per_label_counts[label_str] = per_label_counts.get(label_str, 0) + 1

            if scope == "general":
                gen_i += 1
                group_id = f"GEN_{gen_i:03d}"
            else:
                label_id = _sanitize_label_id(label_str)
                per_label_i[label_id] = per_label_i.get(label_id, 0) + 1
                group_id = f"LBL_{label_id}_{per_label_i[label_id]:03d}"

            answer_options = group.get("answer_options", [])
            canonical_answer = group.get("canonical_answer")

            variants = group.get("variants", [])
            for idx, v in enumerate(variants, start=1):
                variant_id = f"{group_id}_v{idx}"
                question = str(v.get("question", "")).strip()
                prompt_text = _build_student_prompt(
                    variant_question=question,
                    answer_type=answer_type,
                    answer_options=answer_options,
                    task_name=task_name,
                )

                tags = [f"family:{group_id}", f"variant:{variant_id}"]
                batch_rows.append(
                    {
                        "prompt": prompt_text,
                        "answer": {
                            "task_type": "text_rule",
                            "source_type": "text_rule",
                            "group_id": group_id,
                            "variant_id": variant_id,
                            "scope": scope,
                            "label": label_str if scope == "class" else None,
                            "answer_type": answer_type,
                            "answer_options": answer_options,
                            "correct_answer": canonical_answer,
                            "tags": tags,
                            "evidence": group.get("evidence", []),
                            "evidence_path": str(guideline_path),
                        },
                        "meta": {
                            "source_type": "text_rule",
                            "sample_id": variant_id,
                            "family": group_id,
                            "origin_cls_input": str(cls_jsonl),
                            "origin_guideline": str(guideline_path),
                            "teacher_model": client.config.model,
                            "prompt_version": prompt_version,
                            "config_path": str(config_path),
                            "batch_index": batch_idx,
                        },
                    }
                )
            valid_groups_this_batch += 1

        if schedule_enabled and remaining_targets:
            missing = [
                {
                    "scope": str(t["scope"]).strip(),
                    "label": (None if t.get("label") is None else str(t.get("label")).strip()),
                    "answer_type": str(t["answer_type"]).strip(),
                }
                for t in remaining_targets
            ]
            missing_targets_total += len(missing)
            print(
                json.dumps(
                    {
                        "batch": batch_idx,
                        "warning": "Teacher output missing batch targets in this batch",
                        "missing_targets": missing,
                    },
                    ensure_ascii=False,
                )
            )

        if batch_rows:
            _write_jsonl(output_jsonl, batch_rows, append=True)
            rows_written += len(batch_rows)

        total_groups_done += valid_groups_this_batch
        if progress_every_batches > 0 and (batch_idx % progress_every_batches == 0):
            print(
                json.dumps(
                    {
                        "batch": batch_idx,
                        "batch_groups_requested": this_batch_groups,
                        "batch_groups_returned": len(qa_groups),
                        "batch_groups_valid": valid_groups_this_batch,
                        "batch_rows_written": len(batch_rows),
                        "groups_done": total_groups_done,
                        "groups_total": num_groups_total,
                        "rows_existing": existing_rows,
                        "rows_new": rows_written,
                        "rows_total_so_far": existing_rows + rows_written,
                        "skipped_groups_total": skipped_groups,
                        "skip_reason_counts": skip_reason_counts,
                        "missing_targets_total": missing_targets_total,
                        "guideline_focus": focus_block_meta,
                        "scope_counts": scope_counts,
                        "answer_type_counts": answer_type_counts,
                        "per_label_counts": per_label_counts,
                    },
                    ensure_ascii=False,
                )
            )

    summary = {
        "labels": labels,
        "num_groups_total": num_groups_total,
        "batch_groups": batch_groups,
        "schedule_enabled": schedule_enabled,
        "guideline_focus_num_blocks": focus_num_blocks,
        "guideline_focus_strict": focus_strict,
        "output_mode": output_mode,
        "rows_existing": existing_rows,
        "rows_written": rows_written,
        "rows_total_after": existing_rows + rows_written,
        "groups_written": total_groups_done,
        "skipped_groups": skipped_groups,
        "skip_reason_counts": skip_reason_counts,
        "missing_targets_total": missing_targets_total,
        "scope_counts": scope_counts,
        "answer_type_counts": answer_type_counts,
        "per_label_counts": per_label_counts,
        "num_output_rows": rows_written,
        "teacher_model": client.config.model,
        "prompt_version": prompt_version,
        "config_path": str(config_path),
    }
    summary_path = Path(str(output_jsonl) + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    action = "Appended" if append_output else "Wrote"
    print(f"{action} {rows_written} rows -> {output_jsonl}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
