"""
Reward function for optionless (text-label) evaluation on OminiMedExpert ISIC.

Goal:
- Evaluate models trained under MCQ letter supervision in an "optionless" setting where the model must
  output a disease label as text (still typically from a closed candidate set shown in the prompt).

Why a canonical label table:
- Free-form text outputs are sensitive to surface forms (punctuation/case/synonyms).
- We canonicalize predictions and ground truth with conservative normalization + explicit synonyms.

Output:
- `overall`: +2.0 correct, -1.5 wrong, -0.5 unparseable
- `acc`: 1.0/0.0
- `parseable`: 1.0 if we mapped to a canonical label else 0.0
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

REWARD_NAME = "omnimed_isic_optionless_v1"
REWARD_TYPE = "batch"

R_CORRECT = 2.0
R_WRONG = -1.5
R_UNPARSEABLE = -0.5

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)


def _extract_answer_span(text: str) -> str:
    m = _ANSWER_TAG_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _norm_text(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Common benign/malignant label noise: "Benign image." / "Malignant image".
    if s.endswith(" image"):
        s = s[: -len(" image")].strip()
    return s


# Canonical label table.
# Keep this intentionally conservative to avoid inflating scores via overly-broad string matches.
# - Prefer exact matches to the provided `candidate_labels`.
# - Allow a small number of standard abbreviations/synonyms.
_CANONICAL = {
    # Binary labels (present in some subsets, e.g. benign vs malignant questions).
    "Benign": {"benign", "non cancerous", "noncancerous", "non malignant", "nonmalignant", "not malignant"},
    "Malignant": {"malignant", "cancerous"},
    # ISIC-7 style disease names.
    "Melanoma": {"melanoma", "malignant melanoma", "mel"},
    "Melanocytic nevus": {"melanocytic nevus", "nevus", "naevus", "melanocytic naevus", "nv"},
    "Basal cell carcinoma": {"basal cell carcinoma", "bcc"},
    "Actinic keratosis": {"actinic keratosis", "ak", "akiec"},
    "Benign keratosis": {"benign keratosis", "seborrheic keratosis", "bkl", "lplk"},
    "Dermatofibroma": {"dermatofibroma", "df"},
    "Vascular lesion": {"vascular lesion", "vasc"},
}

_SYN2CANON: dict[str, str] = {}
for canon, syns in _CANONICAL.items():
    _SYN2CANON[_norm_text(canon)] = canon
    for s in syns:
        _SYN2CANON[_norm_text(s)] = canon


def _to_canonical_label(text: str, *, candidate_labels: Optional[list[str]] = None) -> Optional[str]:
    """
    Map free-form text to a canonical label.

    Strategy (conservative):
    1) Exact normalized match against synonyms table.
    2) If candidate_labels provided: exact normalized match to any candidate label.
    3) Unique substring match against canonical names/synonyms/candidates (word-boundary-ish).
    """
    raw = text.strip()
    if not raw:
        return None

    s = _norm_text(raw)
    if not s:
        return None

    if candidate_labels:
        cand = [c.strip() for c in candidate_labels if isinstance(c, str) and c.strip()]
        cand_norm = {_norm_text(c): c for c in cand}
        allowed = {_SYN2CANON.get(_norm_text(c), c) for c in cand}

        # 1) Exact match to a candidate surface form.
        if s in cand_norm:
            c = cand_norm[s]
            return _SYN2CANON.get(_norm_text(c), c)

        # 2) Unique candidate substring match (allows "this is melanoma" style responses).
        hay = f" {s} "
        present: list[str] = []
        for c in cand:
            cn = _norm_text(c)
            if not cn:
                continue
            if f" {cn} " in hay:
                present.append(c)
        present_uniq = sorted(dict.fromkeys(present).keys())
        if len(present_uniq) == 1:
            c = present_uniq[0]
            return _SYN2CANON.get(_norm_text(c), c)

        # 3) Exact synonym match but restricted to allowed candidates.
        if s in _SYN2CANON and _SYN2CANON[s] in allowed:
            return _SYN2CANON[s]

        # 4) Unique synonym substring match restricted to allowed candidates.
        matches: list[str] = []
        for syn_norm, canon in _SYN2CANON.items():
            if canon not in allowed:
                continue
            if syn_norm and f" {syn_norm} " in hay:
                matches.append(canon)
        uniq = sorted(set(matches))
        if len(uniq) == 1:
            return uniq[0]
        return None

    # No candidates provided: fall back to global canonicalization.
    if s in _SYN2CANON:
        return _SYN2CANON[s]

    hay = f" {s} "
    matches: list[str] = []
    for syn_norm, canon in _SYN2CANON.items():
        if syn_norm and f" {syn_norm} " in hay:
            matches.append(canon)
    uniq = sorted(set(matches))
    if len(uniq) == 1:
        return uniq[0]
    return None


def _ground_truth_dict(gt: Any) -> dict:
    return gt if isinstance(gt, dict) else {"label": str(gt)}


def compute_score(reward_inputs: List[dict[str, Any]]) -> List[dict[str, float]]:
    scores: list[dict[str, float]] = []

    for item in reward_inputs:
        response = item["response"]
        gt = _ground_truth_dict(item["ground_truth"])

        # Prefer explicit correct_label, then correct_answer, then label.
        correct_raw = gt.get("correct_label", None)
        if not isinstance(correct_raw, str) or not correct_raw.strip():
            correct_raw = gt.get("correct_answer", None)
        if not isinstance(correct_raw, str) or not correct_raw.strip():
            correct_raw = gt.get("label", None)

        candidate_labels = gt.get("candidate_labels", None)
        if not isinstance(candidate_labels, list):
            candidate_labels = None

        if not isinstance(correct_raw, str) or not correct_raw.strip():
            scores.append({"overall": float(R_UNPARSEABLE), "acc": 0.0, "parseable": 0.0})
            continue

        pred_span = _extract_answer_span(str(response))
        pred_can = _to_canonical_label(pred_span, candidate_labels=candidate_labels)
        gt_can = _to_canonical_label(str(correct_raw), candidate_labels=candidate_labels)
        if gt_can is None:
            # If GT is outside our canonical table, fall back to normalized surface-form equality.
            gt_can = _norm_text(correct_raw)

        if pred_can is None:
            scores.append({"overall": float(R_UNPARSEABLE), "acc": 0.0, "parseable": 0.0})
            continue

        is_correct = pred_can == gt_can
        if is_correct:
            scores.append({"overall": float(R_CORRECT), "acc": 1.0, "parseable": 1.0})
        else:
            scores.append({"overall": float(R_WRONG), "acc": 0.0, "parseable": 1.0})

    return scores
