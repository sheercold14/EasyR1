#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALLOWED_ANSWER_TYPES = {"boolean", "text", "short_list"}
ALLOWED_SCOPES = {"general", "class"}
ALLOWED_EVIDENCE_SOURCES = {"guideline", "general_knowledge"}
RE_GROUP_GEN = re.compile(r"^GEN_(\d+)$")
RE_GROUP_LBL = re.compile(r"^LBL_([A-Z0-9_]+)_(\d+)$")
RE_VARIANT = re.compile(r"^(.+)_v([1-9]\d*)$")


def _token_count(text: str) -> int:
    return len([tok for tok in str(text).strip().split() if tok])


def _normalize_text(text: Any) -> str:
    return " ".join(str(text).strip().split())


def _normalize_bool(value: Any) -> str | None:
    v = _normalize_text(value).lower()
    if v in {"yes", "y", "true"}:
        return "yes"
    if v in {"no", "n", "false"}:
        return "no"
    return None


def _normalize_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = [_normalize_text(x) for x in value if _normalize_text(x)]
    else:
        base = str(value).replace("\n", ",")
        items = [_normalize_text(x) for x in base.split(",") if _normalize_text(x)]
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _option_lookup(options: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for opt in options:
        key = _normalize_text(opt).lower()
        if key and key not in out:
            out[key] = opt
    return out


def _preview_detail(value: Any, limit: int = 200) -> str:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and validate text_rule JSONL data.")
    parser.add_argument(
        "--input",
        default="/data/shichao/EasyR1/data/offline_rft/isic/v1/train_text_rule.jsonl",
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--output",
        default="/data/shichao/EasyR1/data/offline_rft/isic/v1/train_text_rule.jsonl.summary.json",
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum stored examples per problem code.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    max_examples = max(1, int(args.max_examples))

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    problem_counts: Counter[str] = Counter()
    problem_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add_problem(code: str, *, line_no: int | None = None, detail: Any | None = None) -> None:
        problem_counts[code] += 1
        examples = problem_examples[code]
        if len(examples) >= max_examples:
            return
        item: dict[str, Any] = {}
        if line_no is not None:
            item["line"] = line_no
        if detail is not None:
            item["detail"] = detail
        examples.append(item)

    total_lines = 0
    blank_lines = 0
    parsed_rows = 0

    source_type_counts: Counter[str] = Counter()
    task_type_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    class_label_counts: Counter[str] = Counter()
    evidence_source_counts: Counter[str] = Counter()
    evidence_path_counts: Counter[str] = Counter()
    origin_guideline_counts: Counter[str] = Counter()
    prompt_version_counts: Counter[str] = Counter()
    teacher_model_counts: Counter[str] = Counter()
    batch_index_counts: Counter[str] = Counter()

    prompt_counts: Counter[str] = Counter()
    prompt_occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    variant_id_counts: Counter[str] = Counter()
    group_id_counts: Counter[str] = Counter()
    group_variant_ids: dict[str, set[str]] = defaultdict(set)
    group_scopes: dict[str, set[str]] = defaultdict(set)
    group_labels: dict[str, set[str]] = defaultdict(set)
    group_answer_types: dict[str, set[str]] = defaultdict(set)
    group_answers: dict[str, set[str]] = defaultdict(set)
    group_answer_options: dict[str, set[str]] = defaultdict(set)

    guideline_line_max_by_path: dict[str, int] = {}
    guideline_line_load_error: set[str] = set()

    with input_path.open("r", encoding="utf-8") as file_obj:
        for line_no, raw_line in enumerate(file_obj, start=1):
            total_lines += 1
            content = raw_line.strip()
            if not content:
                blank_lines += 1
                continue

            try:
                row = json.loads(content)
            except json.JSONDecodeError as exc:
                add_problem("invalid_json_line", line_no=line_no, detail=str(exc))
                continue

            if not isinstance(row, dict):
                add_problem("non_object_row", line_no=line_no, detail=type(row).__name__)
                continue

            parsed_rows += 1
            prompt = row.get("prompt")
            answer = row.get("answer")
            meta = row.get("meta")

            prompt_text = ""
            if not isinstance(prompt, str) or not prompt.strip():
                add_problem("missing_or_empty_prompt", line_no=line_no)
            else:
                prompt_text = prompt.strip()
                prompt_counts[prompt_text] += 1
                if "<answer>" not in prompt_text:
                    add_problem("prompt_missing_answer_tag_hint", line_no=line_no)

            if not isinstance(answer, dict):
                add_problem("missing_or_invalid_answer_obj", line_no=line_no)
                continue
            if not isinstance(meta, dict):
                add_problem("missing_or_invalid_meta_obj", line_no=line_no)
                meta = {}

            task_type = _normalize_text(answer.get("task_type", ""))
            source_type = _normalize_text(answer.get("source_type", ""))
            answer_type = _normalize_text(answer.get("answer_type", ""))
            scope = _normalize_text(answer.get("scope", ""))
            label = answer.get("label")
            label_text = _normalize_text(label) if label is not None else ""
            group_id = _normalize_text(answer.get("group_id", ""))
            variant_id = _normalize_text(answer.get("variant_id", ""))
            answer_options_raw = answer.get("answer_options", [])
            correct_answer = answer.get("correct_answer")
            tags = answer.get("tags", [])
            evidence = answer.get("evidence", [])
            evidence_path = _normalize_text(answer.get("evidence_path", ""))

            task_type_counts[task_type or "<empty>"] += 1
            source_type_counts[source_type or "<empty>"] += 1
            answer_type_counts[answer_type or "<empty>"] += 1
            scope_counts[scope or "<empty>"] += 1
            if scope == "class" and label_text:
                class_label_counts[label_text] += 1

            if task_type != "text_rule":
                add_problem("task_type_not_text_rule", line_no=line_no, detail=task_type)
            if source_type != "text_rule":
                add_problem("source_type_not_text_rule", line_no=line_no, detail=source_type)

            if scope not in ALLOWED_SCOPES:
                add_problem("invalid_scope", line_no=line_no, detail=scope)
            if scope == "general" and label_text:
                add_problem("general_scope_with_label", line_no=line_no, detail=label_text)
            if scope == "class" and not label_text:
                add_problem("class_scope_without_label", line_no=line_no)

            if answer_type not in ALLOWED_ANSWER_TYPES:
                add_problem("invalid_answer_type", line_no=line_no, detail=answer_type)

            if not group_id:
                add_problem("missing_group_id", line_no=line_no)
            if not variant_id:
                add_problem("missing_variant_id", line_no=line_no)

            if group_id:
                group_id_counts[group_id] += 1
                group_scopes[group_id].add(scope or "<empty>")
                group_labels[group_id].add(label_text or "<null>")
                group_answer_types[group_id].add(answer_type or "<empty>")

            if variant_id:
                variant_id_counts[variant_id] += 1

            if group_id and variant_id:
                group_variant_ids[group_id].add(variant_id)
                if not variant_id.startswith(group_id + "_v"):
                    add_problem(
                        "variant_id_group_prefix_mismatch",
                        line_no=line_no,
                        detail={"group_id": group_id, "variant_id": variant_id},
                    )
                m_variant = RE_VARIANT.match(variant_id)
                if m_variant is None or m_variant.group(1) != group_id:
                    add_problem(
                        "variant_id_pattern_invalid",
                        line_no=line_no,
                        detail={"group_id": group_id, "variant_id": variant_id},
                    )

            if prompt_text:
                prompt_occurrences[prompt_text].append(
                    {"line": line_no, "group_id": group_id, "variant_id": variant_id}
                )

            if group_id and not (RE_GROUP_GEN.match(group_id) or RE_GROUP_LBL.match(group_id)):
                add_problem("group_id_pattern_invalid", line_no=line_no, detail=group_id)

            if not isinstance(answer_options_raw, list):
                add_problem("answer_options_not_list", line_no=line_no, detail=type(answer_options_raw).__name__)
                answer_options_raw = []
            answer_options = [_normalize_text(x) for x in answer_options_raw if _normalize_text(x)]
            options_lookup = _option_lookup(answer_options)
            if group_id:
                group_answer_options[group_id].add(_preview_detail(answer_options))

            if answer_type == "boolean":
                can = _normalize_bool(correct_answer)
                if can is None:
                    add_problem(
                        "boolean_correct_answer_invalid",
                        line_no=line_no,
                        detail=_preview_detail(correct_answer),
                    )
                if answer_options:
                    if "yes" not in options_lookup or "no" not in options_lookup:
                        add_problem(
                            "boolean_answer_options_missing_yes_no",
                            line_no=line_no,
                            detail=answer_options,
                        )
                if group_id and can is not None:
                    group_answers[group_id].add(can)
            elif answer_type == "text":
                if not answer_options:
                    add_problem("text_answer_options_empty", line_no=line_no)
                if isinstance(correct_answer, list):
                    add_problem("text_correct_answer_is_list", line_no=line_no)
                    can_text = ""
                else:
                    can_text = _normalize_text(correct_answer)
                if not can_text:
                    add_problem("text_correct_answer_empty", line_no=line_no)
                elif _token_count(can_text) > 20:
                    add_problem("text_correct_answer_too_long", line_no=line_no, detail=can_text)
                if can_text and options_lookup and can_text.lower() not in options_lookup:
                    add_problem(
                        "text_correct_answer_not_in_options",
                        line_no=line_no,
                        detail={"correct_answer": can_text, "answer_options": answer_options},
                    )
                if group_id and can_text:
                    group_answers[group_id].add(can_text.lower())
            elif answer_type == "short_list":
                if not answer_options:
                    add_problem("short_list_answer_options_empty", line_no=line_no)
                items = _normalize_text_list(correct_answer)
                if not items:
                    add_problem("short_list_correct_answer_empty", line_no=line_no)
                if len(items) > 5:
                    add_problem("short_list_correct_answer_too_many_items", line_no=line_no, detail=items)
                for item in items:
                    if _token_count(item) > 20:
                        add_problem("short_list_item_too_long", line_no=line_no, detail=item)
                    if options_lookup and item.lower() not in options_lookup:
                        add_problem(
                            "short_list_item_not_in_options",
                            line_no=line_no,
                            detail={"item": item, "answer_options": answer_options},
                        )
                if group_id and items:
                    group_answers[group_id].add(",".join([x.lower() for x in items]))

            if not isinstance(tags, list):
                add_problem("tags_not_list", line_no=line_no, detail=type(tags).__name__)
                tags = []
            tag_set = {str(x).strip() for x in tags if str(x).strip()}
            if group_id and f"family:{group_id}" not in tag_set:
                add_problem(
                    "tags_missing_family",
                    line_no=line_no,
                    detail={"group_id": group_id, "tags": list(tag_set)},
                )
            if variant_id and f"variant:{variant_id}" not in tag_set:
                add_problem(
                    "tags_missing_variant",
                    line_no=line_no,
                    detail={"variant_id": variant_id, "tags": list(tag_set)},
                )

            if not isinstance(evidence, list) or not evidence:
                add_problem("evidence_missing_or_empty", line_no=line_no)
            else:
                for evidence_item in evidence:
                    if not isinstance(evidence_item, dict):
                        add_problem("evidence_item_not_object", line_no=line_no, detail=type(evidence_item).__name__)
                        continue
                    src = _normalize_text(evidence_item.get("source", ""))
                    lines = evidence_item.get("lines", [])
                    evidence_source_counts[src or "<empty>"] += 1
                    if src not in ALLOWED_EVIDENCE_SOURCES:
                        add_problem("invalid_evidence_source", line_no=line_no, detail=src)
                    if not isinstance(lines, list):
                        add_problem("evidence_lines_not_list", line_no=line_no, detail=type(lines).__name__)
                        continue
                    if not lines:
                        add_problem("evidence_lines_empty", line_no=line_no)
                    for ln in lines:
                        if isinstance(ln, bool) or not isinstance(ln, int):
                            add_problem("evidence_line_not_int", line_no=line_no, detail=ln)
                            continue
                        if ln <= 0:
                            add_problem("evidence_line_not_positive", line_no=line_no, detail=ln)
                        if src == "guideline":
                            if evidence_path:
                                if evidence_path not in guideline_line_max_by_path and evidence_path not in guideline_line_load_error:
                                    try:
                                        guideline_path = Path(evidence_path)
                                        guideline_line_max_by_path[evidence_path] = len(
                                            guideline_path.read_text(encoding="utf-8").splitlines()
                                        )
                                    except Exception:
                                        guideline_line_load_error.add(evidence_path)
                                max_line = guideline_line_max_by_path.get(evidence_path)
                                if max_line is not None and ln > max_line:
                                    add_problem(
                                        "evidence_line_out_of_guideline_range",
                                        line_no=line_no,
                                        detail={"line": ln, "max_line": max_line, "path": evidence_path},
                                    )

            if evidence_path:
                evidence_path_counts[evidence_path] += 1
            else:
                add_problem("evidence_path_missing", line_no=line_no)

            meta_source_type = _normalize_text(meta.get("source_type", ""))
            meta_sample_id = _normalize_text(meta.get("sample_id", ""))
            meta_family = _normalize_text(meta.get("family", ""))
            meta_origin_guideline = _normalize_text(meta.get("origin_guideline", ""))
            meta_prompt_version = _normalize_text(meta.get("prompt_version", ""))
            meta_teacher_model = _normalize_text(meta.get("teacher_model", ""))

            if meta_source_type and source_type and meta_source_type != source_type:
                add_problem(
                    "meta_source_type_mismatch",
                    line_no=line_no,
                    detail={"answer": source_type, "meta": meta_source_type},
                )
            if meta_sample_id and variant_id and meta_sample_id != variant_id:
                add_problem(
                    "meta_sample_id_mismatch",
                    line_no=line_no,
                    detail={"answer": variant_id, "meta": meta_sample_id},
                )
            if meta_family and group_id and meta_family != group_id:
                add_problem(
                    "meta_family_mismatch",
                    line_no=line_no,
                    detail={"answer": group_id, "meta": meta_family},
                )
            if meta_origin_guideline:
                origin_guideline_counts[meta_origin_guideline] += 1
                if evidence_path and meta_origin_guideline != evidence_path:
                    add_problem(
                        "origin_guideline_evidence_path_mismatch",
                        line_no=line_no,
                        detail={"origin_guideline": meta_origin_guideline, "evidence_path": evidence_path},
                    )
            if meta_prompt_version:
                prompt_version_counts[meta_prompt_version] += 1
            if meta_teacher_model:
                teacher_model_counts[meta_teacher_model] += 1

            if "batch_index" in meta:
                batch_index = meta.get("batch_index")
                if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index <= 0:
                    add_problem("invalid_batch_index", line_no=line_no, detail=batch_index)
                    batch_index_counts["invalid"] += 1
                else:
                    batch_index_counts[str(batch_index)] += 1
            else:
                add_problem("missing_batch_index", line_no=line_no)
                batch_index_counts["<missing>"] += 1

    duplicate_prompt_count = sum(1 for c in prompt_counts.values() if c > 1)
    duplicate_variant_id_count = sum(1 for c in variant_id_counts.values() if c > 1)
    duplicate_prompt_examples: list[dict[str, Any]] = []
    for p, c in prompt_counts.items():
        if c <= 1:
            continue
        duplicate_prompt_examples.append(
            {
                "count": c,
                "prompt_preview": (p[:220] + "...(truncated)") if len(p) > 220 else p,
                "occurrences": prompt_occurrences.get(p, [])[:max_examples],
            }
        )
    duplicate_prompt_examples.sort(key=lambda x: x["count"], reverse=True)

    rows_per_group = list(group_id_counts.values())
    groups_not_3_rows = {gid: n for gid, n in group_id_counts.items() if n != 3}
    groups_not_3_variants = {gid: len(vs) for gid, vs in group_variant_ids.items() if len(vs) != 3}
    groups_scope_conflict = {gid: sorted(list(vs)) for gid, vs in group_scopes.items() if len(vs) > 1}
    groups_label_conflict = {gid: sorted(list(vs)) for gid, vs in group_labels.items() if len(vs) > 1}
    groups_answer_type_conflict = {gid: sorted(list(vs)) for gid, vs in group_answer_types.items() if len(vs) > 1}
    groups_correct_answer_conflict = {gid: sorted(list(vs)) for gid, vs in group_answers.items() if len(vs) > 1}
    groups_answer_options_conflict = {gid: sorted(list(vs)) for gid, vs in group_answer_options.items() if len(vs) > 1}

    expected_variant_set = {"v1", "v2", "v3"}
    groups_missing_v123: dict[str, list[str]] = {}
    for gid, variants in group_variant_ids.items():
        suffixes = set()
        for vid in variants:
            m = RE_VARIANT.match(vid)
            if m is None:
                continue
            suffixes.add(f"v{m.group(2)}")
        if suffixes != expected_variant_set:
            groups_missing_v123[gid] = sorted(list(suffixes))

    issues_from_group_integrity: dict[str, int] = {
        "groups_not_3_rows": len(groups_not_3_rows),
        "groups_not_3_variants": len(groups_not_3_variants),
        "groups_scope_conflict": len(groups_scope_conflict),
        "groups_label_conflict": len(groups_label_conflict),
        "groups_answer_type_conflict": len(groups_answer_type_conflict),
        "groups_correct_answer_conflict": len(groups_correct_answer_conflict),
        "groups_answer_options_conflict": len(groups_answer_options_conflict),
        "groups_variant_suffix_not_exact_v1_v2_v3": len(groups_missing_v123),
        "duplicate_prompt_count": duplicate_prompt_count,
        "duplicate_variant_id_count": duplicate_variant_id_count,
    }
    for code, count in issues_from_group_integrity.items():
        if count > 0:
            problem_counts[code] += count

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": str(input_path),
        "total_lines": total_lines,
        "blank_lines": blank_lines,
        "parsed_rows": parsed_rows,
        "task_type_counts": dict(task_type_counts),
        "source_type_counts": dict(source_type_counts),
        "answer_type_counts": dict(answer_type_counts),
        "scope_counts": dict(scope_counts),
        "class_label_counts": dict(class_label_counts),
        "teacher_model_counts": dict(teacher_model_counts),
        "prompt_version_counts": dict(prompt_version_counts),
        "evidence_source_counts": dict(evidence_source_counts),
        "evidence_path_counts": dict(evidence_path_counts),
        "origin_guideline_counts": dict(origin_guideline_counts),
        "batch_index": {
            "missing_rows": batch_index_counts.get("<missing>", 0),
            "invalid_rows": batch_index_counts.get("invalid", 0),
            "min": min(
                [int(k) for k in batch_index_counts.keys() if k not in {"<missing>", "invalid"}],
                default=None,
            ),
            "max": max(
                [int(k) for k in batch_index_counts.keys() if k not in {"<missing>", "invalid"}],
                default=None,
            ),
            "distinct_count": len([k for k in batch_index_counts.keys() if k not in {"<missing>", "invalid"}]),
        },
        "group_stats": {
            "distinct_group_id_count": len(group_id_counts),
            "rows_per_group_min": min(rows_per_group) if rows_per_group else 0,
            "rows_per_group_max": max(rows_per_group) if rows_per_group else 0,
            "rows_per_group_avg": (sum(rows_per_group) / len(rows_per_group)) if rows_per_group else 0.0,
            "groups_not_3_rows_count": len(groups_not_3_rows),
            "groups_not_3_rows_examples": dict(list(groups_not_3_rows.items())[:max_examples]),
            "groups_not_3_variants_count": len(groups_not_3_variants),
            "groups_not_3_variants_examples": dict(list(groups_not_3_variants.items())[:max_examples]),
            "groups_scope_conflict_count": len(groups_scope_conflict),
            "groups_scope_conflict_examples": dict(list(groups_scope_conflict.items())[:max_examples]),
            "groups_label_conflict_count": len(groups_label_conflict),
            "groups_label_conflict_examples": dict(list(groups_label_conflict.items())[:max_examples]),
            "groups_answer_type_conflict_count": len(groups_answer_type_conflict),
            "groups_answer_type_conflict_examples": dict(list(groups_answer_type_conflict.items())[:max_examples]),
            "groups_correct_answer_conflict_count": len(groups_correct_answer_conflict),
            "groups_correct_answer_conflict_examples": dict(list(groups_correct_answer_conflict.items())[:max_examples]),
            "groups_answer_options_conflict_count": len(groups_answer_options_conflict),
            "groups_answer_options_conflict_examples": dict(
                list(groups_answer_options_conflict.items())[:max_examples]
            ),
            "groups_variant_suffix_not_exact_v1_v2_v3_count": len(groups_missing_v123),
            "groups_variant_suffix_not_exact_v1_v2_v3_examples": dict(
                list(groups_missing_v123.items())[:max_examples]
            ),
            "duplicate_prompt_count": duplicate_prompt_count,
            "duplicate_prompt_examples": duplicate_prompt_examples[:max_examples],
            "duplicate_variant_id_count": duplicate_variant_id_count,
        },
        "problem_counts": dict(problem_counts),
        "problem_examples": dict(problem_examples),
        "has_problems": bool(problem_counts),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote summary -> {output_path}")
    print(f"Parsed rows: {parsed_rows}/{total_lines}, has_problems={bool(problem_counts)}")


if __name__ == "__main__":
    main()
