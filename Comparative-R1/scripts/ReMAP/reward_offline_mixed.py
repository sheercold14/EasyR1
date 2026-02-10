from __future__ import annotations

import re
from typing import Any


REWARD_NAME = "offline_rft_mixed_v1"
REWARD_TYPE = "batch"

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)


def _extract_answer_text(response: str) -> str:
    match = _ANSWER_RE.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()


def _to_ground_truth_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    return {"task_type": "unknown", "correct_answer": str(raw)}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_bool(text: str) -> str | None:
    value = _normalize_text(text)
    if value in {"yes", "y", "true"}:
        return "yes"
    if value in {"no", "n", "false"}:
        return "no"
    return None


def _normalize_list(text: str) -> list[str]:
    base = text.replace("\n", ",")
    parts = [part.strip().lower() for part in base.split(",")]
    parts = [part for part in parts if part]
    return sorted(dict.fromkeys(parts))


def _is_correct(predicted_raw: str, answer_type: str, correct_answer: Any) -> bool:
    if answer_type == "bool":
        pred = _normalize_bool(predicted_raw)
        gt = _normalize_bool(str(correct_answer))
        return pred is not None and gt is not None and pred == gt
    if answer_type == "list":
        pred_list = _normalize_list(predicted_raw)
        if isinstance(correct_answer, list):
            gt_list = sorted(dict.fromkeys([str(item).strip().lower() for item in correct_answer if str(item).strip()]))
        else:
            gt_list = _normalize_list(str(correct_answer))
        return pred_list == gt_list
    pred = _normalize_text(predicted_raw)
    gt = _normalize_text(str(correct_answer))
    return bool(pred) and bool(gt) and pred == gt


def _keyword_bonus(predicted_raw: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    text = _normalize_text(predicted_raw)
    matched = sum(1 for keyword in keywords if _normalize_text(str(keyword)) in text)
    if matched <= 0:
        return 0.0
    return min(0.5, 0.1 * matched)


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    outputs: list[dict[str, float]] = []
    counts = {"cls": 0, "attr": 0, "text_rule": 0, "unknown": 0}
    correct = {"cls": 0, "attr": 0, "text_rule": 0, "unknown": 0}

    for item in reward_inputs:
        response = str(item.get("response", ""))
        answer_text = _extract_answer_text(response)
        gt = _to_ground_truth_dict(item.get("ground_truth"))

        task_type = str(gt.get("task_type", "unknown")).strip() or "unknown"
        if task_type not in counts:
            task_type = "unknown"
        counts[task_type] += 1

        answer_type = str(gt.get("answer_type", "short_text")).strip().lower()
        is_ok = _is_correct(answer_text, answer_type, gt.get("correct_answer", ""))
        if is_ok:
            correct[task_type] += 1

        if task_type == "text_rule":
            base = 1.0 if is_ok else 0.0
            keywords = gt.get("keywords", [])
            keyword_bonus = _keyword_bonus(answer_text, keywords if isinstance(keywords, list) else [])
            overall = base + keyword_bonus
            score = {"overall": float(overall), "r_text": float(base), "r_text_bonus": float(keyword_bonus)}
        elif task_type == "attr":
            base = 1.0 if is_ok else 0.0
            score = {"overall": float(base), "r_attr": float(base)}
        elif task_type == "cls":
            base = 1.0 if is_ok else 0.0
            score = {"overall": float(base), "r_cls": float(base)}
        else:
            score = {"overall": 0.0}

        outputs.append(score)

    metrics: dict[str, float] = {}
    for task_type, total in counts.items():
        task_correct = correct[task_type]
        acc = (task_correct / total) if total > 0 else 0.0
        metrics[f"task/n/{task_type}"] = float(total)
        metrics[f"task/acc/{task_type}"] = float(acc)

    for out in outputs:
        out.update(metrics)

    return outputs
