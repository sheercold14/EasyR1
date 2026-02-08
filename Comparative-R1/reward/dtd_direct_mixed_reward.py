import re
from typing import Any, List

REWARD_NAME = "dtd_direct_mixed_v1"
REWARD_TYPE = "batch"

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _extract_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"answer\s*[:=]\s*(.*)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def _normalize_text(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^[\"'`]+|[\"'`]+$", "", t)
    t = re.sub(r"[\s\.,;:]+$", "", t)
    return t


def _letter_tokens(text: str) -> list[str]:
    toks = [t.upper() for t in _TOKEN_RE.findall(text)]
    return [t for t in toks if len(t) == 1 and "A" <= t <= "Z"]


def _is_same_different(text: str) -> bool:
    return _normalize_text(text) in {"same", "different"}


def _match_pred_to_gold(pred_raw: str, gold_raw: str) -> bool:
    pred = pred_raw.strip()
    gold = gold_raw.strip()
    if not pred or not gold:
        return False

    # B5
    if _is_same_different(gold):
        return _normalize_text(pred) == _normalize_text(gold)

    # B1/B2/B3/B4/B7 (single letter)
    gold_letters = _letter_tokens(gold)
    if len(gold_letters) == 1:
        pred_letters = _letter_tokens(pred)
        return len(pred_letters) >= 1 and pred_letters[0] == gold_letters[0]

    # B6 (pair letters, order-free)
    if len(gold_letters) == 2:
        pred_letters = _letter_tokens(pred)
        return sorted(pred_letters[:2]) == sorted(gold_letters)

    # Single-image DTD label
    return _normalize_text(pred) == _normalize_text(gold)


def _get_gold_answer(gt: Any) -> str:
    if isinstance(gt, dict):
        if "correct_answer" in gt and gt.get("correct_answer") is not None:
            return str(gt.get("correct_answer"))
        if "label" in gt and gt.get("label") is not None:
            return str(gt.get("label"))
        return ""
    return str(gt) if gt is not None else ""


def compute_score(reward_inputs: List[dict[str, Any]]) -> List[dict[str, float]]:
    scores: List[dict[str, float]] = []
    for item in reward_inputs:
        resp = str(item.get("response", ""))
        gt = item.get("ground_truth", "")

        pred = _extract_answer(resp)
        gold = _get_gold_answer(gt)
        acc = 1.0 if _match_pred_to_gold(pred, gold) else 0.0

        scores.append(
            {
                "overall": float(acc),
                "accuracy": float(acc),
            }
        )
    return scores

