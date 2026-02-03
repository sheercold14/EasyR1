"""
Reward function for OminiMedExpert ISIC (mcq_letter + Comparative-RFT B1–B7).

The dataset uses short, verifiable answers:
  - Single-letter: "A".."Z"  (mcq_letter, B1/B2/B3/B4/B7)
  - Same/Different: "same" | "different" (B5)
  - Pair letters (order-free): "A B" (B6)
"""

from __future__ import annotations

import re
from typing import Any, List


REWARD_NAME = "omnimed_isic_benchmark_v1"
REWARD_TYPE = "batch"

# Simple accuracy-style reward.
R_CORRECT = 2.0
R_WRONG = -1.5
R_UNPARSEABLE = -0.5

_KNOWN_TASK_TYPES: tuple[str, ...] = (
    # Single-image MCQ.
    "mcq_letter",
    # Comparative suite B.
    "B1_target_search",
    "B2_odd_one_out",
    "B3_label_corruption",
    "B4_exemplar_match",
    "B5_same_different",
    "B6_pair_finding",
    "B7_support_set_nway",
    # Fallback.
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
    """
    Extract one A-Z letter from response (prefers <answer>...</answer> span).
    """
    span = _extract_answer_span(text)
    span_u = span.upper()

    # Prefer a single letter token.
    m = re.search(r"\b([A-Z])\b", span_u)
    if m:
        return m.group(1)

    # Fallback: allow "Answer: A" patterns.
    m = re.search(r"answer\s*[:=]\s*([A-Z])\b", span_u, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Last resort: pick the first A-Z seen.
    m = re.search(r"([A-Z])", span_u)
    if m:
        return m.group(1)
    return None


def _extract_letter_pair(text: str) -> str | None:
    """
    Extract exactly two distinct A-Z letters as an order-free canonical "A B".
    """
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


def compute_score(reward_inputs: List[dict[str, Any]]) -> List[dict[str, float]]:
    scores: list[dict[str, float]] = []
    per_task_total: dict[str, int] = {t: 0 for t in _KNOWN_TASK_TYPES}
    per_task_correct: dict[str, int] = {t: 0 for t in _KNOWN_TASK_TYPES}
    for item in reward_inputs:
        response = item["response"]
        gt = _ground_truth_dict(item["ground_truth"])

        tt = _task_type_from_gt(gt)

        correct_answer_raw = gt.get("correct_answer", None)
        if not isinstance(correct_answer_raw, str) or not correct_answer_raw.strip():
            per_task_total[tt] += 1
            scores.append({"overall": float(R_UNPARSEABLE), "acc": 0.0})
            continue

        correct_answer = _normalize_ws(correct_answer_raw)
        kind = _expected_answer_kind(correct_answer)

        predicted: str | None
        if kind == "letter":
            predicted = _extract_single_letter(response)
            pred_norm = predicted
            gt_norm = correct_answer.strip().upper()
        elif kind == "pair":
            predicted = _extract_letter_pair(response)
            pred_norm = predicted
            gt_norm = " ".join(sorted(set(re.findall(r"[A-Za-z]", correct_answer.upper()))))
        elif kind == "same_different":
            predicted = _extract_same_different(response)
            pred_norm = predicted
            gt_norm = correct_answer.strip().lower()
        else:
            predicted = None
            pred_norm = None
            gt_norm = correct_answer

        if pred_norm is None:
            per_task_total[tt] += 1
            scores.append({"overall": float(R_UNPARSEABLE), "acc": 0.0})
            continue

        is_correct = pred_norm == gt_norm
        if is_correct:
            per_task_total[tt] += 1
            per_task_correct[tt] += 1
            scores.append({"overall": float(R_CORRECT), "acc": 1.0})
        else:
            per_task_total[tt] += 1
            scores.append({"overall": float(R_WRONG), "acc": 0.0})

    # Log task-wise accuracy in the same reward dict so SwanLab can plot it.
    # These are per-batch metrics; counts help interpret steps where some tasks are absent.
    per_task_metrics: dict[str, float] = {}
    for t in _KNOWN_TASK_TYPES:
        total = per_task_total.get(t, 0)
        correct = per_task_correct.get(t, 0)
        acc = (correct / total) if total > 0 else 0.0
        per_task_metrics[f"task/acc/{t}"] = float(acc)
        per_task_metrics[f"task/n/{t}"] = float(total)
        per_task_metrics[f"task/correct/{t}"] = float(correct)

    for s in scores:
        s.update(per_task_metrics)

    return scores
