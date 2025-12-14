import re
from typing import Any, List

REWARD_NAME = "pathology_answer_confidence_v2_2"
REWARD_TYPE = "batch"

# Phrases that indicate the model is not committing to a label.
AMBIGUOUS = [
    r"uncertain",
    r"unsure",
    r"cannot\s+determine",
    r"cannot\s+rule\s+out",
    r"indeterminate",
    r"need\s+biopsy",
    r"need\s+histopath",
    r"not\s+sure",
    r"further\s+exam",
]

# Linear aggressive reward parameters.
ALPHA = 0.5
BETA = 1.0
LAMBDA = 2.0


def _is_ambiguous(text: str) -> bool:
    t = text.lower()
    return any(re.search(pat, t) for pat in AMBIGUOUS)


def _extract_label(text: str) -> str:
    t = text.lower()
    m = re.search(r"<answer>\s*(malignant|benign|unsure)\s*</answer>", t)
    if m:
        return m.group(1)
    m = re.search(r"answer\s*[:=]\s*(malignant|benign|unsure)", t)
    if m:
        return m.group(1)
    return "unknown"


def _extract_confidence(text: str) -> tuple[float, bool]:
    """Return (confidence, missing_flag). missing_flag=True when confidence tag absent or unparsable."""

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

    # Allow percent-style confidence values.
    if val > 1.0:
        val /= 100.0
    return _clamp(val), False


def compute_score(reward_inputs: List[dict[str, Any]]) -> List[dict[str, float]]:
    # First pass: compute per-sample reward components and aggregate stats.
    interim: list[dict[str, Any]] = []
    total = len(reward_inputs)
    unsure_count = 0
    answered_count = 0
    correct_answered = 0

    for item in reward_inputs:
        resp = item["response"]
        gt = item["ground_truth"]
        label = gt["label"] if isinstance(gt, dict) else str(gt)
        label = str(label).lower()

        ambiguous = _is_ambiguous(resp)
        pred_label = _extract_label(resp)
        is_unsure = ambiguous or pred_label in {"unsure", "unknown"}
        pred_conf, missing_conf = (0.0, True) if is_unsure else _extract_confidence(resp)

        if is_unsure:
            unsure_count += 1
            r_correct = 0.0
            acc = 0.0
        else:
            answered_count += 1
            if pred_label == label:
                correct_answered += 1
                r_correct = ALPHA + BETA * pred_conf
                acc = 1.0
            else:
                r_correct = -LAMBDA * pred_conf
                acc = 0.0

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
            }
        )

    coverage = (answered_count / total) if total else 0.0
    unsure_rate = (unsure_count / total) if total else 0.0
    selective_acc = (correct_answered / answered_count) if answered_count else 0.0

    # Second pass: attach aggregate metrics to each sample record.
    scores: List[dict[str, float]] = []
    for rec in interim:
        rec["coverage"] = float(coverage)
        rec["unsure_rate"] = float(unsure_rate)
        rec["selective_acc"] = float(selective_acc)
        scores.append(rec)

    return scores
