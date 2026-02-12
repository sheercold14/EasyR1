from __future__ import annotations

import re
from typing import Any, Iterable, List

REWARD_NAME = "offline_rft_mixed_v1"
REWARD_TYPE = "batch"

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)


def _extract_answer_span(text: str) -> str:
    m = _ANSWER_TAG_RE.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _strip_quotes(text: str) -> str:
    return re.sub(r"^[\"'`]+|[\"'`]+$", "", text.strip())


def _norm_key(text: Any) -> str:
    s = _normalize_ws(_strip_quotes(str(text)))
    s = re.sub(r"[\s\.,;:]+$", "", s)
    return s.lower()


def _normalize_bool(text: Any) -> str | None:
    v = _norm_key(text)
    if v in {"yes", "y", "true"}:
        return "yes"
    if v in {"no", "n", "false"}:
        return "no"
    return None


def _unify_answer_type(answer_type: Any) -> str:
    t = str(answer_type).strip().lower()
    if t in {"bool", "boolean"}:
        return "bool"
    if t in {"short_text", "text"}:
        return "text"
    if t in {"list", "short_list"}:
        return "list"
    return t or "text"


def _iter_nonempty(items: Iterable[Any]) -> Iterable[str]:
    for x in items:
        s = _normalize_ws(str(x))
        if s:
            yield s


def _parse_list_items(answer_text: str) -> list[str]:
    # Accept comma/semicolon/newline separated lists.
    parts = re.split(r"[,\n;]+", answer_text)
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        s = _normalize_ws(_strip_quotes(p))
        if not s:
            continue
        k = _norm_key(s)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _to_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return list(_iter_nonempty(value))
    s = _normalize_ws(str(value))
    return [s] if s else []


def _list_score(pred: list[str], gold: list[str], mode: str) -> float:
    mode = str(mode or "f1").strip().lower()
    if not gold:
        return 0.0
    if mode == "exact":
        return 1.0 if [_norm_key(x) for x in pred] == [_norm_key(x) for x in gold] else 0.0

    pred_set = set(_norm_key(x) for x in pred if _norm_key(x))
    gold_set = set(_norm_key(x) for x in gold if _norm_key(x))
    if not pred_set or not gold_set:
        return 0.0
    inter = len(pred_set & gold_set)
    if mode == "recall":
        return float(inter / max(1, len(gold_set)))
    if mode == "precision":
        return float(inter / max(1, len(pred_set)))
    if mode == "jaccard":
        return float(inter / max(1, len(pred_set | gold_set)))

    # default: f1
    prec = inter / max(1, len(pred_set))
    rec = inter / max(1, len(gold_set))
    if (prec + rec) <= 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def _ground_truth_dict(gt: Any) -> dict[str, Any]:
    if isinstance(gt, dict):
        return gt
    return {"task_type": "cls", "answer_type": "text", "correct_answer": str(gt)}


def _norm_text_for_label(s: Any) -> str:
    text = "" if s is None else (s if isinstance(s, str) else str(s))
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.endswith(" image"):
        text = text[: -len(" image")].strip()
    return text


_CANONICAL = {
    "Benign": {"benign", "non cancerous", "noncancerous", "non malignant", "nonmalignant", "not malignant"},
    "Malignant": {"malignant", "cancerous"},
    "Melanoma": {"melanoma", "malignant melanoma", "mel"},
    "Melanocytic nevus": {"melanocytic nevus", "nevus", "naevus", "melanocytic naevus", "nv"},
    "Basal cell carcinoma": {"basal cell carcinoma", "bcc"},
    "Actinic keratosis": {"actinic keratosis", "ak", "akiec"},
    "Benign keratosis": {"benign keratosis", "seborrheic keratosis", "bkl", "lplk"},
    "Dermatofibroma": {"dermatofibroma", "df"},
    "Vascular lesion": {"vascular lesion", "vasc"},
    "Squamous cell carcinoma": {"squamous cell carcinoma", "scc"},
}

_SYN2CANON: dict[str, str] = {}
for canon, syns in _CANONICAL.items():
    _SYN2CANON[_norm_text_for_label(canon)] = canon
    for s in syns:
        _SYN2CANON[_norm_text_for_label(s)] = canon


def _to_canonical_label(text: str, *, candidate_labels: list[str] | None = None) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    s = _norm_text_for_label(raw)
    if not s:
        return None

    if s in _SYN2CANON:
        return _SYN2CANON[s]

    if candidate_labels:
        cand = [c.strip() for c in candidate_labels if isinstance(c, str) and c.strip()]
        cand_norm = {_norm_text_for_label(c): c for c in cand}

        if s in cand_norm:
            c = cand_norm[s]
            return _SYN2CANON.get(_norm_text_for_label(c), c)

        hay = f" {s} "
        present: list[str] = []
        for c in cand:
            cn = _norm_text_for_label(c)
            if cn and f" {cn} " in hay:
                present.append(c)
        uniq = list(dict.fromkeys(present).keys())
        if len(uniq) == 1:
            c = uniq[0]
            return _SYN2CANON.get(_norm_text_for_label(c), c)

    hay = f" {s} "
    matches: list[str] = []
    for syn_norm, canon in _SYN2CANON.items():
        if syn_norm and f" {syn_norm} " in hay:
            matches.append(canon)
    uniq = sorted(set(matches))
    if len(uniq) == 1:
        return uniq[0]
    return None


def _candidate_list_from_gt(gt: dict[str, Any]) -> list[str] | None:
    for key in ("answer_options", "candidate_answers", "candidate_labels"):
        v = gt.get(key, None)
        if isinstance(v, list):
            out = [str(x).strip() for x in v if str(x).strip()]
            return out if out else None
    return None


def compute_score(
    reward_inputs: List[dict[str, Any]],
    *,
    list_score_mode: str = "f1",
) -> List[dict[str, float]]:
    outputs: list[dict[str, float]] = []
    counts: dict[str, int] = {"cls": 0, "attr": 0, "text_rule": 0, "unknown": 0}
    correct: dict[str, int] = {"cls": 0, "attr": 0, "text_rule": 0, "unknown": 0}

    for item in reward_inputs:
        response = str(item.get("response", ""))
        answer_text = _extract_answer_span(response)
        gt = _ground_truth_dict(item.get("ground_truth"))

        task_type = str(gt.get("task_type", "unknown")).strip() or "unknown"
        # Treat optionless single-image classification as cls for metrics/scoring.
        if task_type == "mcq_optionless_text":
            task_type = "cls"
        if task_type not in counts:
            task_type = "unknown"
        counts[task_type] += 1

        answer_type_raw = gt.get("answer_type", "text")
        answer_type = _unify_answer_type(answer_type_raw)
        gold = gt.get("correct_answer", "")

        # --- scoring ---
        acc = 0.0
        if answer_type == "bool":
            pred_b = _normalize_bool(answer_text)
            gt_b = _normalize_bool(gold)
            acc = 1.0 if (pred_b is not None and gt_b is not None and pred_b == gt_b) else 0.0
        elif answer_type == "list":
            pred_items = _parse_list_items(answer_text)
            gold_items = _to_list(gold)
            # For list-like tasks with candidate options, normalize to candidate surface forms if possible.
            options = _candidate_list_from_gt(gt) or []
            if options:
                lookup = {_norm_key(o): o for o in options if _norm_key(o)}

                mapped_pred: list[str] = []
                for it in pred_items:
                    mapped_pred.append(lookup.get(_norm_key(it), it))
                pred_items = mapped_pred

                mapped_gold: list[str] = []
                for it in gold_items:
                    mapped_gold.append(lookup.get(_norm_key(it), it))
                gold_items = mapped_gold

            acc = _list_score(pred_items, gold_items, list_score_mode)
        else:
            # text
            if task_type == "cls":
                candidate_labels = gt.get("candidate_labels", None)
                if not isinstance(candidate_labels, list):
                    candidate_labels = None
                pred_can = _to_canonical_label(answer_text, candidate_labels=candidate_labels) or _norm_text_for_label(
                    answer_text
                )
                gt_can = _to_canonical_label(str(gold), candidate_labels=candidate_labels) or _norm_text_for_label(gold)
                acc = 1.0 if (pred_can is not None and pred_can == gt_can) else 0.0
            else:
                options = _candidate_list_from_gt(gt) or []
                if options:
                    lookup = {_norm_key(o): o for o in options if _norm_key(o)}
                    pred = lookup.get(_norm_key(answer_text), answer_text)
                    acc = 1.0 if _norm_key(pred) == _norm_key(gold) and _norm_key(pred) else 0.0
                else:
                    acc = 1.0 if _norm_key(answer_text) == _norm_key(gold) and _norm_key(answer_text) else 0.0

        if acc >= 0.999:
            correct[task_type] += 1

        outputs.append(
            {
                "overall": float(acc),
                "acc": float(acc),
            }
        )

    metrics: dict[str, float] = {}
    for tt, total in counts.items():
        c = correct.get(tt, 0)
        metrics[f"task/n/{tt}"] = float(total)
        metrics[f"task/acc/{tt}"] = float(c / total) if total > 0 else 0.0

    for out in outputs:
        out.update(metrics)
    return outputs
