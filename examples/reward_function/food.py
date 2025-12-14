import re
from typing import Any, List

REWARD_NAME = "food101_clsrl"
REWARD_TYPE = "batch"

# pattern for <think>...</think><answer>...</answer>
FORMAT_PATTERN = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)


def _extract_answer(text: str) -> str:
    """Extract answer inside <answer> tags; fallback to full text."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip().lower()
    # fallback: Answer: xxx
    m = re.search(r"answer\s*:\s*(.*)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    return text.strip().lower()


def compute_score(reward_inputs: List[dict[str, Any]]) -> List[dict[str, float]]:
    scores = []
    for item in reward_inputs:
        resp = item["response"]
        gt = item["ground_truth"]
        label = gt["label"].lower() if isinstance(gt, dict) else str(gt).lower()

        student = _extract_answer(resp)
        acc = 1.0 if student == label else 0.0
        fmt = 1.0 if FORMAT_PATTERN.fullmatch(resp) else 0.0

        scores.append({
            "overall": float(acc + fmt),
            "accuracy": float(acc),
            "format": float(fmt),
        })
    return scores
