"""
Dedicated reward for mixed ISIC training:
- Single-image optionless text-label task (mcq_optionless_text)
- Comparative multi-image tasks B1-B7 (letter / pair / same-different)

This file is standalone and does not modify existing reward files.
"""

from __future__ import annotations

import re
from typing import Any, List

REWARD_NAME = "omnimed_isic_optionless_mixed_v1"
REWARD_TYPE = "batch"

R_CORRECT = 2.0
R_WRONG = -1.5
R_UNPARSEABLE = -0.5

_KNOWN_TASK_TYPES: tuple[str, ...] = (
    "mcq_optionless_text",
    "mcq_letter",
    "B1_target_search",
    "B2_odd_one_out",
    "B3_label_corruption",
    "B4_exemplar_match",
    "B5_same_different",
    "B6_pair_finding",
    "B7_support_set_nway",
    "unknown",
)

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)


def _extract_answer_span(text: str) -> str:
    m = _ANSWER_TAG_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _normalize_token(text: str) -> str:
    return _normalize_ws(text).lower()


def _extract_single_letter(text: str) -> str | None:
    span_u = _extract_answer_span(text).upper()
    m = re.search(r"\b([A-Z])\b", span_u)
    if m:
        return m.group(1)
    m = re.search(r"answer\s*[:=]\s*([A-Z])\b", span_u, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"([A-Z])", span_u)
    if m:
        return m.group(1)
    return None


def _extract_letter_pair(text: str) -> str | None:
    span = _extract_answer_span(text).upper()
    letters = re.findall(r"\b([A-Z])\b", span)
    uniq = sorted(set(letters))
    if len(uniq) != 2:
        return None
    return f"{uniq[0]} {uniq[1]}"


def _extract_same_different(text: str) -> str | None:
    span = _normalize_token(_extract_answer_span(text))
    has_same = bool(re.search(r"\bsame\b", span))
    has_diff = bool(re.search(r"\bdifferent\b", span))
    if has_same and not has_diff:
        return "same"
    if has_diff and not has_same:
        return "different"
    return None


def _ground_truth_dict(gt: Any) -> dict:
    return gt if isinstance(gt, dict) else {"label": str(gt)}


def _expected_answer_kind(correct_answer: str) -> str:
    ca = correct_answer.strip()
    if re.fullmatch(r"[A-Za-z]", ca):
        return "letter"
    if re.fullmatch(r"[A-Za-z]\s+[A-Za-z]", ca):
        return "pair"
    if ca.lower() in {"same", "different"}:
        return "same_different"
    return "unknown"


def _task_type_from_gt(gt: dict[str, Any]) -> str:
    task_type = gt.get("task_type", None)
    if isinstance(task_type, str):
        tt = task_type.strip()
        if tt:
            return tt if tt in _KNOWN_TASK_TYPES else "unknown"
    return "unknown"


def _norm_text(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.endswith(" image"):
        s = s[: -len(" image")].strip()
    return s


_CANONICAL = {
    "Benign": {"benign", "non cancerous", "noncancerous", "non malignant", "nonmalignant", "not malignant"},
    "Malignant": {"malignant", "cancerous"},
    "Melanoma": {"melanoma", "malignant melanoma", "mel"},
    "Melanocytic nevus": {"melanocytic nevus", "nevus", "naevus", "melanocytic naevus", "nv"},
    "Basal cell carcinoma": {"basal cell carcinoma", "bcc"},
    "Actinic keratosis": {"actinic keratosis", "ak", "akiec"},
    "Benign keratosis": {"benign keratosis", "seborrheic keratosis", "bkl", "lplk"},
    "Dermatofibroma": {"dermatofibroma", "df"},
    "Vascular lesion": {"vascular lesion", "vasc"},
}

_SYN2CANON: dict[str, str] = {}
for canon, syns in _CANONICAL.items():
    _SYN2CANON[_norm_text(canon)] = canon
    for s in syns:
        _SYN2CANON[_norm_text(s)] = canon


def _to_canonical_label(text: str, *, candidate_labels: list[str] | None = None) -> str | None:
    raw = text.strip()
    if not raw:
        return None
    s = _norm_text(raw)
    if not s:
        return None

    if s in _SYN2CANON:
        return _SYN2CANON[s]

    if candidate_labels:
        cand = [c.strip() for c in candidate_labels if isinstance(c, str) and c.strip()]
        cand_norm = {_norm_text(c): c for c in cand}

        if s in cand_norm:
            c = cand_norm[s]
            return _SYN2CANON.get(_norm_text(c), c)

        hay = f" {s} "
        present: list[str] = []
        for c in cand:
            cn = _norm_text(c)
            if cn and f" {cn} " in hay:
                present.append(c)
        uniq = sorted(dict.fromkeys(present).keys())
        if len(uniq) == 1:
            c = uniq[0]
            return _SYN2CANON.get(_norm_text(c), c)

    hay = f" {s} "
    matches: list[str] = []
    for syn_norm, canon in _SYN2CANON.items():
        if syn_norm and f" {syn_norm} " in hay:
            matches.append(canon)
    uniq = sorted(set(matches))
    if len(uniq) == 1:
        return uniq[0]
    return None


def _is_optionless_row(gt: dict[str, Any], task_type: str) -> bool:
    if task_type == "mcq_optionless_text":
        return True
    if isinstance(gt.get("candidate_labels"), list):
        return True
    if isinstance(gt.get("correct_label"), str) and gt.get("correct_label", "").strip():
        return True
    return False


def compute_score(reward_inputs: List[dict[str, Any]]) -> List[dict[str, float]]:
    scores: list[dict[str, float]] = []
    per_task_total: dict[str, int] = {t: 0 for t in _KNOWN_TASK_TYPES}
    per_task_correct: dict[str, int] = {t: 0 for t in _KNOWN_TASK_TYPES}

    for item in reward_inputs:
        response = str(item.get("response", ""))
        gt = _ground_truth_dict(item.get("ground_truth"))
        tt = _task_type_from_gt(gt)

        # Optionless single-image branch.
        if _is_optionless_row(gt, tt):
            tt = "mcq_optionless_text"
            candidate_labels = gt.get("candidate_labels", None)
            if not isinstance(candidate_labels, list):
                candidate_labels = None

            correct_raw = gt.get("correct_label", None)
            if not isinstance(correct_raw, str) or not correct_raw.strip():
                correct_raw = gt.get("correct_answer", None)
            if not isinstance(correct_raw, str) or not correct_raw.strip():
                correct_raw = gt.get("label", None)

            if not isinstance(correct_raw, str) or not correct_raw.strip():
                per_task_total[tt] += 1
                scores.append({"overall": float(R_UNPARSEABLE), "acc": 0.0})
                continue

            pred_can = _to_canonical_label(_extract_answer_span(response), candidate_labels=candidate_labels)
            gt_can = _to_canonical_label(str(correct_raw), candidate_labels=candidate_labels)
            if gt_can is None:
                gt_can = _norm_text(correct_raw)

            if pred_can is None:
                per_task_total[tt] += 1
                scores.append({"overall": float(R_UNPARSEABLE), "acc": 0.0})
                continue

            is_correct = pred_can == gt_can
            per_task_total[tt] += 1
            if is_correct:
                per_task_correct[tt] += 1
                scores.append({"overall": float(R_CORRECT), "acc": 1.0})
            else:
                scores.append({"overall": float(R_WRONG), "acc": 0.0})
            continue

        # Legacy discrete-answer branch for mcq_letter + B tasks.
        correct_answer_raw = gt.get("correct_answer", None)
        if not isinstance(correct_answer_raw, str) or not correct_answer_raw.strip():
            per_task_total[tt] += 1
            scores.append({"overall": float(R_UNPARSEABLE), "acc": 0.0})
            continue

        correct_answer = _normalize_ws(correct_answer_raw)
        kind = _expected_answer_kind(correct_answer)

        if kind == "letter":
            pred_norm = _extract_single_letter(response)
            gt_norm = correct_answer.strip().upper()
        elif kind == "pair":
            pred_norm = _extract_letter_pair(response)
            gt_norm = " ".join(sorted(set(re.findall(r"[A-Za-z]", correct_answer.upper()))))
        elif kind == "same_different":
            pred_norm = _extract_same_different(response)
            gt_norm = correct_answer.strip().lower()
        else:
            pred_norm = None
            gt_norm = correct_answer

        if pred_norm is None:
            per_task_total[tt] += 1
            scores.append({"overall": float(R_UNPARSEABLE), "acc": 0.0})
            continue

        is_correct = pred_norm == gt_norm
        per_task_total[tt] += 1
        if is_correct:
            per_task_correct[tt] += 1
            scores.append({"overall": float(R_CORRECT), "acc": 1.0})
        else:
            scores.append({"overall": float(R_WRONG), "acc": 0.0})

    per_task_metrics: dict[str, float] = {}
    for t in _KNOWN_TASK_TYPES:
        total = per_task_total.get(t, 0)
        correct = per_task_correct.get(t, 0)
        per_task_metrics[f"task/n/{t}"] = float(total)
        per_task_metrics[f"task/correct/{t}"] = float(correct)
        if total > 0:
            per_task_metrics[f"task/acc/{t}"] = float(correct / total)

    for s in scores:
        s.update(per_task_metrics)

    return scores
