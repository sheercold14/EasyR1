import re
from typing import Any, List

REWARD_NAME = "pathology_no_ct"
REWARD_TYPE = "batch"

AMBIGUOUS = [
    r"uncertain", r"cannot\s+determine", r"cannot\s+rule\s+out",
    r"indeterminate", r"need\s+biopsy", r"need\s+histopath",
    r"not\s+sure", r"further\s+exam"
]
KEYWORDS = {
    "hemorrhage": ["hemorrhage", "bleeding"],
    "cystic": ["cystic"],
    "necrosis": ["necrosis", "necrotic"],
    "margin_capsule": ["margin", "border", "capsule"],
    "solid": ["solid"],
    "fibrosis": ["fibrosis"],
    "calcification": ["calcification", "calcified"],
    "vascular": ["vascular", "congestion"],
}

def _has(text: str, kws) -> bool:
    t = text.lower()
    return any(kw in t for kw in kws)

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

def _structure_score(text: str) -> float:
    t = text.lower()
    s = 0.0
    if "<think>" in t and "</think>" in t:
        s += 0.1
    if re.search(r"<answer>\s*(malignant|benign)\s*</answer>", t):
        s += 0.2
    elif "<answer>" in t and "</answer>" in t:
        s += 0.1
    if "answer:" in t:
        s += 0.1
    return s

def _reasoning_score(text: str) -> float:
    t = text.lower()
    score = 0.0
    # each hit +0.2 up to 0.5
    for group in KEYWORDS.values():
        if _has(t, group):
            score += 0.2
    return min(score, 0.5)

def compute_score(reward_inputs: List[dict[str, Any]]) -> List[dict[str, float]]:
    out = []
    for item in reward_inputs:
        resp = item["response"]
        gt = item["ground_truth"]
        label = gt["label"] if isinstance(gt, dict) else str(gt)
        ambiguous = _is_ambiguous(resp)
        pred = _extract_label(resp)

        if ambiguous or pred == "unknown":
            r_correct = -1.5
        elif pred == label:
            r_correct = 1.5
        else:
            r_correct = -1.0

        r_struct = _structure_score(resp)
        r_reason = _reasoning_score(resp)
        acc = 1.0 if pred == label else 0.0
        out.append({
            "overall": float(r_correct + r_struct + r_reason),
            "correct": float(r_correct),
            "structure": float(r_struct),
            "reasoning": float(r_reason),
            "ambiguous": float(1.0 if ambiguous else 0.0),
            "acc": float(acc)
        })
    return out
