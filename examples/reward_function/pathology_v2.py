import re
from typing import Any, List

REWARD_NAME = "pathology_answer_only"
REWARD_TYPE = "batch"

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


def _is_ambiguous(text: str) -> bool:
    t = text.lower()
    return any(re.search(pat, t) for pat in AMBIGUOUS)


def _extract_label(text: str) -> str:
    t = text.lower()
    m = re.search(r"<answer>\s*(malignant|benign|unsure)\s*</answer>", t)
    if m:
        return m.group(1)
    m = re.search(r"answer\s*:\s*(malignant|benign|unsure)", t)
    if m:
        return m.group(1)
    return "unknown"


def _extract_confidence(text: str) -> float:
    def _clamp(val: float) -> float:
        return max(0.0, min(1.0, val))

    t = text.lower()
    m = re.search(r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>", t)
    if not m:
        m = re.search(r"confidence\s*[:=]\s*([0-9]*\.?[0-9]+)%?", t)
    if not m:
        return 0.5

    try:
        val = float(m.group(1))
    except ValueError:
        return 0.5

    # Allow percent style confidence.
    if val > 1.0:
        val /= 100.0
    return _clamp(val)


def compute_score(reward_inputs: List[dict[str, Any]]) -> List[dict[str, float]]:
    scores = []
    for item in reward_inputs:
        resp = item["response"]
        gt = item["ground_truth"]
        label = gt["label"] if isinstance(gt, dict) else str(gt)
        label = str(label).lower()
        ambiguous = _is_ambiguous(resp)
        pred_label = _extract_label(resp)
        pred_conf = 0.0 if pred_label == "unsure" else _extract_confidence(resp)

        if pred_label in {"unsure", "unknown"} or ambiguous:
            r_correct = 0.0  # no penalty for explicitly unsure answers
            acc = 0.0
        else:
            if pred_label == label:
                # Higher reward for correct high-confidence answers.
                r_correct = 0.5 + 1.5 * pred_conf
                acc = 1.0
            else:
                # Larger penalty when the model is confidently wrong.
                r_correct = -1.5 * pred_conf
                acc = 0.0

        scores.append(
            {
                "overall": float(r_correct),
                "correct": float(r_correct),
                "acc": float(acc),
                "ambiguous": float(1.0 if (ambiguous or pred_label == "unsure") else 0.0),
            }
        )

    return scores
