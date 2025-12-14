import re
from typing import Any, List

REWARD_NAME = "pathology_answer_only"
REWARD_TYPE = "batch"

AMBIGUOUS = [
    r"uncertain",
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
    m = re.search(r"<answer>\s*(malignant|benign)\s*</answer>", t)
    if m:
        return m.group(1)
    m = re.search(r"answer\s*:\s*(malignant|benign)", t)
    if m:
        return m.group(1)
    return "unknown"


def compute_score(reward_inputs: List[dict[str, Any]]) -> List[dict[str, float]]:
    scores = []
    for item in reward_inputs:
        resp = item["response"]
        gt = item["ground_truth"]
        label = gt["label"] if isinstance(gt, dict) else str(gt)
        ambiguous = _is_ambiguous(resp)
        pred_label = _extract_label(resp)

        if ambiguous or pred_label == "unknown":
            r_correct = -1.5  # strong penalty for non-committal answers
        elif pred_label == label:
            r_correct = 1.5
        else:
            r_correct = -1.0

        acc = 1.0 if pred_label == label else 0.0
        scores.append(
            {
                "overall": float(r_correct),
                "correct": float(r_correct),
                "acc": float(acc),
                "ambiguous": float(1.0 if ambiguous else 0.0),
            }
        )

    return scores
