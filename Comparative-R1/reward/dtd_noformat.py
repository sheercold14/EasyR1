import re
from typing import Any, List

REWARD_NAME = "dtd_b2n_answer_v1"
REWARD_TYPE = "batch"

def _normalize(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^[\"'`]+|[\"'`]+$", "", t)
    t = re.sub(r"[\s\.,;:]+$", "", t)
    return t


def _extract_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"answer\s*[:=]\s*(.*)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def compute_score(reward_inputs: List[dict[str, Any]]) -> List[dict[str, float]]:
    scores: List[dict[str, float]] = []
    for item in reward_inputs:
        resp = item["response"]
        gt = item["ground_truth"]
        gt_text = gt.get("label", "") if isinstance(gt, dict) else str(gt)

        pred = _normalize(_extract_answer(resp))
        gold = _normalize(gt_text)
        acc = 1.0 if pred == gold and pred != "" else 0.0
        # fmt = 1.0 if FORMAT_PATTERN.fullmatch(resp) else 0.0

        scores.append(
            {
                "overall": float(acc),
                "accuracy": float(acc),
            }
        )
    return scores

