import re
from typing import Any, List

REWARD_NAME = "pathology"
REWARD_TYPE = "batch"


def _has_keyword(text: str, keywords) -> bool:
    t = text.lower()
    return any(kw in t for kw in keywords)


def _extract_answer_label(text: str) -> str:
    t = text.lower()
    # Prefer <answer>...</answer>
    m = re.search(r"<answer>\s*([a-z]+)\s*</answer>", t)
    if m:
        cand = m.group(1)
        if "malignant" in cand:
            return "malignant"
        if "benign" in cand:
            return "benign"
    # Fallback: Answer: xxx
    m = re.search(r"answer\s*:\s*([a-z]+)", t)
    if m:
        cand = m.group(1)
        if "malignant" in cand:
            return "malignant"
        if "benign" in cand:
            return "benign"
    # Keywords fallback
    if "malignant" in t and "benign" not in t:
        return "malignant"
    if "benign" in t and "malignant" not in t:
        return "benign"
    return "unknown"


def _structure_score(text: str) -> float:
    t = text.lower()
    score = 0.0
    if "<think>" in t and "</think>" in t:
        score += 0.2
    if "<answer>" in t and "</answer>" in t:
        score += 0.2
    if "answer:" in t:
        score += 0.2
    return score


def _extract_gross_features(case_text: str) -> dict:
    t = case_text.lower()
    return {
        "solid": _has_keyword(t, ["solid"]),
        "cystic": _has_keyword(t, ["cystic", "cystic change", "cystic component"]),
        "hemorrhage": _has_keyword(t, ["hemorrhage", "haemorrhage", "bleeding"]),
        "necrosis": _has_keyword(t, ["necrosis", "necrotic"]),
        "irregular_margin": _has_keyword(t, ["irregular", "ill-defined", "poorly defined", "infiltrative"]),
        "well_defined_margin": _has_keyword(t, ["well-defined", "well defined", "clear margin", "sharp margin"]),
        "capsule_present": _has_keyword(t, ["complete capsule", "intact capsule", "well-formed capsule"]),
        "capsule_absent": _has_keyword(t, ["no capsule", "lack of capsule", "capsule not identified", "without capsule"]),
    }


def _reasoning_alignment_score(reasoning_text: str, case_text: str, gold_label: str) -> float:
    r = reasoning_text.lower()
    feats = _extract_gross_features(case_text)
    score = 0.0

    # Mentioning key concepts
    if _has_keyword(r, ["margin", "border"]):
        score += 0.1
    if _has_keyword(r, ["capsule"]):
        score += 0.1
    if _has_keyword(r, ["hemorrhage", "haemorrhage", "bleeding"]):
        score += 0.1
    if _has_keyword(r, ["necrosis", "necrotic"]):
        score += 0.1
    if _has_keyword(r, ["solid", "cystic"]):
        score += 0.1

    # Consistency with case_text
    if feats["hemorrhage"] and _has_keyword(r, ["hemorrhage", "haemorrhage", "bleeding"]):
        score += 0.2
    if feats["necrosis"] and _has_keyword(r, ["necrosis", "necrotic"]):
        score += 0.2
    if feats["irregular_margin"] and _has_keyword(r, ["irregular", "ill-defined", "poorly defined"]):
        score += 0.2
    if feats["well_defined_margin"] and _has_keyword(r, ["well-defined", "well defined", "clear margin"]):
        score += 0.2
    if feats["capsule_present"] and _has_keyword(r, ["intact capsule", "complete capsule", "well-formed capsule"]):
        score += 0.2
    if feats["capsule_absent"] and _has_keyword(r, ["no capsule", "lack of capsule", "no intact capsule"]):
        score += 0.2

    # Alignment with malignant/benign priors
    malignant_cues = (
        _has_keyword(r, ["infiltrative", "invasion", "invasive"])
        or _has_keyword(r, ["irregular", "ill-defined", "poorly defined"])
        or _has_keyword(r, ["no capsule", "lack of capsule"])
        or _has_keyword(r, ["necrosis", "necrotic"])
    )
    benign_cues = (
        _has_keyword(r, ["well-defined", "well defined", "clear margin", "smooth margin"])
        or _has_keyword(r, ["intact capsule", "complete capsule", "well-formed capsule"])
        or _has_keyword(r, ["homogeneous", "uniform"])
    )
    if gold_label == "malignant":
        if malignant_cues:
            score += 0.3
        if benign_cues:
            score -= 0.3
    elif gold_label == "benign":
        if benign_cues:
            score += 0.3
        if malignant_cues:
            score -= 0.3

    return max(-0.8, min(1.2, score))


def compute_score(reward_inputs: List[dict[str, Any]]) -> List[dict[str, float]]:
    scores = []
    for item in reward_inputs:
        response = item["response"]
        ground_truth = item["ground_truth"]
        if isinstance(ground_truth, dict):
            label = ground_truth.get("label", "unknown")
            case_text = ground_truth.get("case_text", "")
        else:
            label = str(ground_truth)
            case_text = ""

        lower_resp = response.lower()
        m = re.search(r"<think>(.*)</think>", lower_resp, flags=re.DOTALL)
        reasoning_part = response[m.start(1):m.end(1)] if m else response

        pred_label = _extract_answer_label(response)
        if pred_label == "unknown":
            r_correct = -0.3
        elif pred_label == label:
            r_correct = 1.5
        else:
            r_correct = -1.0

        r_struct = _structure_score(response)
        r_reason = _reasoning_alignment_score(reasoning_part, case_text, label)

        scores.append(
            {
                "overall": float(r_correct + r_struct + r_reason),
                "correct": float(r_correct),
                "structure": float(r_struct),
                "reasoning": float(r_reason),
            }
        )

    return scores
