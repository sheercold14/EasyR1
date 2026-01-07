import re
from typing import Any, List, Tuple

REWARD_NAME = "isic_answer_confidence_v2_3"
REWARD_TYPE = "batch"

AMBIGUOUS = [
    r"uncertain",
    r"unsure",
    r"cannot\s+determine",
    r"cannot\s+rule\s+out",
    r"indeterminate",
    r"not\s+sure",
]

ALPHA = 1.5
BETA = 0.5
LAMBDA = 3.0
ANSWER_COST = 0.0
FORMAT_WEIGHT = 0.1


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_answer(text: str) -> str:
    return re.sub(r"[\s\.,;:]+$", "", _normalize(text))


def _is_ambiguous(text: str) -> bool:
    t = text.lower()
    return any(re.search(pat, t) for pat in AMBIGUOUS)


def _extract_answer_text(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _has_option_prefix(text: str) -> bool:
    return bool(re.match(r"^\s*[abcd](?:\s*[\).:\-])?\s+", text.strip(), re.IGNORECASE))


def _format_score(text: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)
    if not re.fullmatch(pattern, text):
        return 0.0
    answer_text = _extract_answer_text(text)
    return 0.0 if _has_option_prefix(answer_text) else 1.0


def _strip_option_prefix(text: str) -> str:
    return re.sub(r"^\s*[abcd](?:\s*[\).:\-])?\s+", "", text.strip(), flags=re.IGNORECASE)


def _extract_confidence(text: str) -> Tuple[float, bool]:
    def _clamp(val: float) -> float:
        return max(0.0, min(1.0, val))

    t = text.lower()
    m = re.search(r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>", t)
    if not m:
        m = re.search(r"confidence\s*[:=]\s*([0-9]*\.?[0-9]+)%?", t)
    if not m:
        return 0.0, True
    try:
        val = float(m.group(1))
    except ValueError:
        return 0.0, True
    if val > 1.0:
        val /= 100.0
    return _clamp(val), False


def _ground_truth_text(ground_truth: Any) -> str:
    if isinstance(ground_truth, dict):
        return str(ground_truth.get("label", "")).strip()
    return str(ground_truth).strip()


def compute_score(reward_inputs: List[dict[str, Any]]) -> List[dict[str, float]]:
    interim: List[dict[str, Any]] = []
    total = len(reward_inputs)
    unsure_count = 0
    answered_count = 0
    correct_answered = 0
    for item in reward_inputs:
        resp = item["response"]
        gt = item["ground_truth"]
        gt_text = _ground_truth_text(gt)

        ambiguous = _is_ambiguous(resp)
        format_score = _format_score(resp)
        pred_text = _extract_answer_text(resp)
        pred_text_for_match = _strip_option_prefix(pred_text)
        pred_norm = _normalize_answer(pred_text_for_match)
        gt_norm = _normalize_answer(gt_text)
        is_unsure = ambiguous or not pred_norm or not gt_norm

        pred_conf, missing_conf = (0.0, True) if is_unsure else _extract_confidence(resp)

        if is_unsure:
            unsure_count += 1
            r_correct = 0.0
            acc = 0.0
        else:
            answered_count += 1
            if pred_norm == gt_norm:
                correct_answered += 1
                r_correct = ALPHA + BETA * pred_conf - ANSWER_COST
                acc = 1.0
            else:
                r_correct = -LAMBDA * pred_conf - ANSWER_COST
                acc = 0.0

        r_correct += FORMAT_WEIGHT * format_score

        interim.append(
            {
                "overall": float(r_correct),
                "correct": float(r_correct),
                "acc": float(acc),
                "ambiguous": float(1.0 if ambiguous else 0.0),
                "unsure": float(1.0 if is_unsure else 0.0),
                "confidence": float(pred_conf),
                "answered": float(0.0 if is_unsure else 1.0),
                "confidence_missing": float(1.0 if missing_conf else 0.0),
                "format": float(format_score),
            }
        )

    coverage = (answered_count / total) if total else 0.0
    unsure_rate = (unsure_count / total) if total else 0.0
    selective_acc = (correct_answered / answered_count) if answered_count else 0.0

    scores: List[dict[str, float]] = []
    for rec in interim:
        rec["coverage"] = float(coverage)
        rec["unsure_rate"] = float(unsure_rate)
        rec["selective_acc"] = float(selective_acc)
        scores.append(rec)

    return scores
