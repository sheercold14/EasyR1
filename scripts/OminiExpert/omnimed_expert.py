#!/usr/bin/env python3
"""
OmniMedVQA -> EasyR1 data construction engine + Comparative-RFT (B1–B7).

This script supports:
  1) Loading / merging OmniMedVQA JSON files (QA_information/*/*.json)
  2) Filtering by question_type (and other optional fields)
  3) Train/val/test splitting with leak-avoidance via grouping keys
  4) Few-shot sampling from train with per-label balancing
  5) Writing EasyR1-compatible JSONL with root-relative image paths
  6) Expanding train data into multi-image comparative tasks (B1–B7)

Default paths assume this repo layout:
  <repo_root>/data/OmniMedVQA
  <repo_root>/data/OminiMedExpert
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


OPTION_KEYS: tuple[str, ...] = ("option_A", "option_B", "option_C", "option_D")
LETTER_BY_OPTION_KEY = {
    "option_A": "A",
    "option_B": "B",
    "option_C": "C",
    "option_D": "D",
}
PATIENT_LIKE_KEYS: tuple[str, ...] = (
    "patient_id",
    "patient",
    "subject_id",
    "subject",
    "case_id",
    "case",
    "study_id",
    "study",
    "series_id",
    "exam_id",
    "lesion_id",
    "lesion",
)


def _repo_root() -> Path:
    # EasyR1/scripts/OminiExpert/omnimed_expert.py -> parents[3] == <repo_root>
    return Path(__file__).resolve().parents[3]


def _default_omni_root() -> Path:
    return _repo_root() / "data" / "OmniMedVQA"


def _default_out_root() -> Path:
    return _repo_root() / "data" / "OminiMedExpert"


def _norm_text(val: object) -> str:
    return re.sub(r"\s+", " ", str(val).strip())


def _is_empty_option(val: object) -> bool:
    s = _norm_text(val)
    return (not s) or (s.lower() == "none")


def _read_json(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected a JSON list at {path}")
    out: list[dict] = []
    for i, it in enumerate(obj):
        if not isinstance(it, dict):
            raise ValueError(f"Expected dict items at {path}[{i}]")
        out.append(it)
    return out


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield obj


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def discover_omnimedvqa_datasets(omni_root: Path) -> dict[str, Path]:
    qa_root = omni_root / "QA_information"
    out: dict[str, Path] = {}
    # Prefer Open-access when duplicates exist.
    for subdir in ("Open-access", "Restricted-access"):
        d = qa_root / subdir
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            name = p.stem
            if name not in out:
                out[name] = p
    return out


def build_prompt(question: str, options: dict[str, str | None]) -> str:
    q = _norm_text(question)
    rendered: list[str] = []
    letters: list[str] = []
    for k in OPTION_KEYS:
        v = options.get(k)
        if v is None:
            continue
        if _is_empty_option(v):
            continue
        ltr = LETTER_BY_OPTION_KEY[k]
        letters.append(ltr)
        rendered.append(f"{ltr}. {_norm_text(v)}")
    if rendered:
        ltr_opts = _letter_options(letters)
        # Keep the prompt self-contained and verifiable (short discrete output).
        return (
            f"Question: {q}\nOptions:\n"
            + "\n".join(rendered)
            + f"\nAnswer with only the option letter ({ltr_opts}).\n<answer></answer>"
        )
    return f"Question: {q}\nAnswer succinctly."


def infer_answer_id(gt_answer: str, options: dict[str, str | None]) -> str | None:
    gt = _norm_text(gt_answer)
    matches: list[str] = []
    for k in OPTION_KEYS:
        v = options.get(k)
        if v is None:
            continue
        if _is_empty_option(v):
            continue
        if _norm_text(v) == gt:
            matches.append(LETTER_BY_OPTION_KEY[k])
    if len(matches) != 1:
        return None
    return matches[0]


def to_root_relative_image_path(image_path: str) -> str:
    # Store as root-relative (leading "/") to match the user's desired convention.
    p = str(image_path).strip()
    if not p:
        return ""
    return "/" + p.lstrip("/")


def resolve_image_path(omni_root: Path, root_rel_path: str) -> Path:
    p = str(root_rel_path).strip()
    if p.startswith("/"):
        p = p[1:]
    return omni_root / p


_SLICE_RE = re.compile(r"^(?P<prefix>.+)_(?P<axis>[xyz])_(?P<idx>\\d+)$", flags=re.IGNORECASE)


def default_group_id(dataset: str, root_rel_image_path: str) -> str:
    """
    Group key for leak-robust splitting.

    - Groups all QA items for the same image together.
    - Additionally groups 3D-slice images by original case prefix:
        {ori name}_{axis}_{slice}.png -> group by {ori name}
      (OmniMedVQA README: split-from-3D naming convention)
    """
    p = root_rel_image_path.lstrip("/")
    stem = Path(p).stem
    m = _SLICE_RE.match(stem)
    if m:
        stem = m.group("prefix")
    # Include dataset to avoid accidental collisions.
    return f"{dataset}::{stem}"


def infer_group_id(raw_item: dict, *, dataset: str, root_rel_image_path: str) -> tuple[str, str]:
    """
    Infer a grouping key for leakage-robust splitting.

    Preference order:
      1) Patient/subject/case-like fields if present in raw JSON
      2) 3D-slice prefix heuristic (README naming convention)
      3) Image stem
    """
    for k in PATIENT_LIKE_KEYS:
        if k not in raw_item:
            continue
        v = _norm_text(raw_item.get(k, ""))
        if not v:
            continue
        return f"{dataset}::{k}={v}", f"field:{k}"

    p = root_rel_image_path.lstrip("/")
    stem = Path(p).stem
    m = _SLICE_RE.match(stem)
    if m:
        return f"{dataset}::case={m.group('prefix')}", "3d_slice_prefix"
    return f"{dataset}::image={stem}", "image_stem"


def _parse_ratio3(raw: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("--split must be like '0.8,0.1,0.1'")
    a, b, c = (float(x) for x in parts)
    if a < 0 or b < 0 or c < 0:
        raise ValueError("--split ratios must be >= 0")
    s = a + b + c
    if s <= 0:
        raise ValueError("--split ratios must sum to > 0")
    return a / s, b / s, c / s


def split_by_group(
    rows: list[dict],
    *,
    seed: int,
    ratios: tuple[float, float, float],
    group_id_fn,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    group_source_counts = Counter()
    for r in rows:
        imgs = r.get("images") or []
        if not isinstance(imgs, list) or not imgs or not isinstance(imgs[0], str):
            continue
        ans = r.get("answer") or {}
        if not isinstance(ans, dict):
            continue
        gid = ans.get("group_id")
        if isinstance(gid, str) and gid.strip():
            gid = gid.strip()
            group_source_counts[str(ans.get("group_id_source", "provided")).strip() or "provided"] += 1
        else:
            gid = group_id_fn(str(ans.get("dataset", "")).strip(), imgs[0])
            group_source_counts["fallback:image"] += 1
        groups[gid].append(r)

    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n = len(group_keys)
    r_train, r_val, r_test = ratios
    n_train = int(n * r_train)
    n_val = int(n * r_val)
    # Ensure we cover all groups.
    n_test = n - n_train - n_val

    train_keys = set(group_keys[:n_train])
    val_keys = set(group_keys[n_train : n_train + n_val])
    test_keys = set(group_keys[n_train + n_val :])

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for gid, items in groups.items():
        if gid in train_keys:
            train.extend(items)
        elif gid in val_keys:
            val.extend(items)
        else:
            test.extend(items)

    info = {
        "num_groups": n,
        "num_groups_train": len(train_keys),
        "num_groups_val": len(val_keys),
        "num_groups_test": len(test_keys),
        "group_id_sources_top20": dict(group_source_counts.most_common(20)),
        "ratios": {"train": r_train, "val": r_val, "test": r_test},
        "seed": seed,
    }
    return train, val, test, info


def _summarize_rows(rows: list[dict]) -> dict:
    labels = Counter()
    datasets = Counter()
    qtypes = Counter()
    for r in rows:
        a = r.get("answer") or {}
        if not isinstance(a, dict):
            continue
        labels[_norm_text(a.get("label", ""))] += 1
        datasets[_norm_text(a.get("dataset", ""))] += 1
        qtypes[_norm_text(a.get("question_type", ""))] += 1
    return {
        "total": len(rows),
        "labels_top50": dict(labels.most_common(50)),
        "datasets": dict(datasets.most_common()),
        "question_types": dict(qtypes.most_common()),
    }


def fewshot_balanced_by_label(rows: list[dict], *, ratio: float, seed: int) -> list[dict]:
    if ratio <= 0:
        return []
    if ratio >= 1:
        return list(rows)

    rng = random.Random(seed)

    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        a = r.get("answer") or {}
        if not isinstance(a, dict):
            continue
        lbl = _norm_text(a.get("label", ""))
        if not lbl:
            continue
        by_label[lbl].append(r)

    labels = list(by_label.keys())
    if not labels:
        return []

    target_total = max(1, int(round(len(rows) * ratio)))
    rng.shuffle(labels)
    per_label = target_total // len(labels)
    remainder = target_total - per_label * len(labels)

    out: list[dict] = []
    for i, lbl in enumerate(labels):
        pool = by_label[lbl]
        want = per_label + (1 if i < remainder else 0)
        if want <= 0:
            continue
        if len(pool) <= want:
            out.extend(pool)
            continue
        out.extend(rng.sample(pool, want))

    rng.shuffle(out)
    return out


def build_base_rows(
    raw_items: list[dict],
    *,
    omni_root: Path,
    allowed_question_types: set[str] | None,
    allowed_datasets: set[str] | None,
    allowed_modality_types: set[str] | None,
    min_option_count: int | None,
    max_option_count: int | None,
    skip_missing_images: bool,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    missing_images = 0
    missing_answer_id = 0
    skipped_filter = 0
    invalid_items = 0

    for it in raw_items:
        dataset = _norm_text(it.get("dataset", ""))
        qtype = _norm_text(it.get("question_type", ""))
        modality = _norm_text(it.get("modality_type", ""))

        if allowed_datasets is not None and dataset not in allowed_datasets:
            skipped_filter += 1
            continue
        if allowed_question_types is not None and qtype not in allowed_question_types:
            skipped_filter += 1
            continue
        if allowed_modality_types is not None and modality not in allowed_modality_types:
            skipped_filter += 1
            continue

        question_id = _norm_text(it.get("question_id", ""))
        question = _norm_text(it.get("question", ""))
        gt_answer = _norm_text(it.get("gt_answer", ""))
        image_path = _norm_text(it.get("image_path", ""))

        if not dataset or not question_id or not question or not gt_answer or not image_path:
            invalid_items += 1
            continue

        options: dict[str, str | None] = {k: it.get(k) for k in OPTION_KEYS}
        opt_count = _option_count(options)
        if min_option_count is not None and opt_count < min_option_count:
            skipped_filter += 1
            continue
        if max_option_count is not None and opt_count > max_option_count:
            skipped_filter += 1
            continue
        prompt = build_prompt(question, options)
        root_rel_img = to_root_relative_image_path(image_path)
        group_id, group_id_source = infer_group_id(it, dataset=dataset, root_rel_image_path=root_rel_img)

        abs_img = resolve_image_path(omni_root, root_rel_img)
        if not abs_img.exists():
            missing_images += 1
            if skip_missing_images:
                continue

        answer_id = infer_answer_id(gt_answer, options)
        if answer_id is None:
            missing_answer_id += 1

        answer_payload = {
            "label": gt_answer,
            "answer_id": answer_id,
            "question_id": question_id,
            "dataset": dataset,
            "question_type": qtype,
            "modality_type": modality,
            "group_id": group_id,
            "group_id_source": group_id_source,
            **{k: options.get(k) for k in OPTION_KEYS},
        }
        if opt_count > 0 and answer_id is not None:
            answer_payload["correct_answer"] = answer_id
            answer_payload["task_type"] = "mcq_letter"

        rows.append(
            {
                "prompt": prompt,
                "images": [root_rel_img],
                "answer": answer_payload,
            }
        )

    info = {
        "total_raw_items": len(raw_items),
        "written_rows": len(rows),
        "skipped_filter": skipped_filter,
        "invalid_items": invalid_items,
        "missing_images": missing_images,
        "missing_answer_id": missing_answer_id,
        "omni_root": str(omni_root),
    }
    return rows, info


@dataclass(frozen=True)
class VqaItem:
    image: str
    label: str
    question_id: str
    dataset: str
    question_type: str
    modality_type: str
    options: dict[str, str | None]


def _as_vqa_item(row: dict) -> VqaItem | None:
    imgs = row.get("images") or []
    if not isinstance(imgs, list) or not imgs or not isinstance(imgs[0], str):
        return None
    a = row.get("answer") or {}
    if not isinstance(a, dict):
        return None
    label = a.get("label")
    if label is None:
        return None
    return VqaItem(
        image=str(imgs[0]),
        label=_norm_text(label),
        question_id=_norm_text(a.get("question_id", "")),
        dataset=_norm_text(a.get("dataset", "")),
        question_type=_norm_text(a.get("question_type", "")),
        modality_type=_norm_text(a.get("modality_type", "")),
        options={k: (a.get(k) if a.get(k) is not None else None) for k in OPTION_KEYS},
    )


LabelSpaceBy = Literal[
    "question_type",
    "question_type+modality",
    "question_type+optioncount",
    "question_type+modality+optioncount",
]


def _option_count(options: dict[str, str | None]) -> int:
    return sum(1 for k in OPTION_KEYS if options.get(k) is not None and not _is_empty_option(options[k]))


def label_space_key(item: VqaItem, by: LabelSpaceBy) -> str:
    parts: list[str] = []
    if by in {"question_type", "question_type+modality", "question_type+optioncount", "question_type+modality+optioncount"}:
        parts.append(f"question_type={item.question_type}")
    if by in {"question_type+modality", "question_type+modality+optioncount"}:
        parts.append(f"modality_type={item.modality_type}")
    if by in {"question_type+optioncount", "question_type+modality+optioncount"}:
        parts.append(f"option_count={_option_count(item.options)}")
    return "|".join(parts) if parts else "all"


def _letters(n: int) -> list[str]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if n > 26:
        raise ValueError("n too large for A-Z")
    return [chr(ord("A") + i) for i in range(n)]


def _letter_options(letters: list[str]) -> str:
    if len(letters) == 1:
        return letters[0]
    if len(letters) == 2:
        return f"{letters[0]} or {letters[1]}"
    return ", ".join(letters[:-1]) + f", or {letters[-1]}"


def _render_images(letters: list[str]) -> str:
    return "\n".join([f"({l}) Image {l}: <image>" for l in letters])


def _render_labeled_images(letters: list[str], shown_labels: list[str]) -> str:
    if len(letters) != len(shown_labels):
        raise ValueError("letters and shown_labels length mismatch")
    lines: list[str] = []
    for ltr, lbl in zip(letters, shown_labels, strict=True):
        lines.append(f"({ltr}) Claimed label: {lbl}\n({ltr}) Image {ltr}: <image>")
    return "\n".join(lines)


def _sample_distinct(rng: random.Random, items: list[VqaItem], k: int) -> list[VqaItem]:
    if k <= 0:
        return []
    if k == 1:
        return [rng.choice(items)]
    if len(items) >= k:
        return rng.sample(items, k)
    # fallback to sampling with replacement
    return [rng.choice(items) for _ in range(k)]


def gen_b1_target_search(rng: random.Random, *, by_label: dict[str, list[VqaItem]], k: int) -> dict:
    labels = sorted(by_label.keys())
    if len(labels) < 2:
        raise ValueError("Need at least 2 labels for B1")

    target_label = rng.choice(labels)
    target_item = rng.choice(by_label[target_label])

    other_labels = [l for l in labels if l != target_label]
    distractor_labels = (
        rng.sample(other_labels, k - 1)
        if (k - 1) <= len(other_labels)
        else [rng.choice(other_labels) for _ in range(k - 1)]
    )
    distractors = [rng.choice(by_label[l]) for l in distractor_labels]

    chosen = [target_item, *distractors]
    rng.shuffle(chosen)

    letters = _letters(k)
    correct_idx = next(i for i, it in enumerate(chosen) if it.image == target_item.image)
    correct = letters[correct_idx]

    prompt = (
        "You are a medical VQA assistant.\n\n"
        f"Below are {k} images, labeled {_letter_options(letters)}.\n\n"
        f"{_render_images(letters)}\n\n"
        f"Task (B1 Target-search): Exactly one image shows **{target_label}**. Which image is it?\n\n"
        "Answer with only one letter.\n"
        f"<answer> {_letter_options(letters)} </answer>"
    )

    return {
        "prompt": prompt,
        "images": [it.image for it in chosen],
        "answer": {
            "task_suite": "B",
            "task_type": "B1_target_search",
            "target_label": target_label,
            "correct_answer": correct,
            "labels": [it.label for it in chosen],
            "question_ids": [it.question_id for it in chosen],
            "datasets": [it.dataset for it in chosen],
        },
    }


def gen_b2_odd_one_out(rng: random.Random, *, by_label: dict[str, list[VqaItem]], k: int) -> dict:
    labels = sorted(by_label.keys())
    if len(labels) < 2:
        raise ValueError("Need at least 2 labels for B2")

    majority_label = rng.choice(labels)
    odd_label = rng.choice([l for l in labels if l != majority_label])

    majority_items = _sample_distinct(rng, by_label[majority_label], k - 1)
    odd_item = rng.choice(by_label[odd_label])
    chosen = [*majority_items, odd_item]
    rng.shuffle(chosen)

    letters = _letters(k)
    correct_idx = next(i for i, it in enumerate(chosen) if it.image == odd_item.image)
    correct = letters[correct_idx]

    prompt = (
        "You are a medical VQA assistant.\n\n"
        f"Below are {k} images, labeled {_letter_options(letters)}.\n\n"
        f"{_render_images(letters)}\n\n"
        f"Task (B2 Odd-one-out): Exactly {k-1} images depict the same condition and 1 image depicts a different condition.\n"
        "Which image is the odd one out?\n\n"
        "Answer with only one letter.\n"
        f"<answer> {_letter_options(letters)} </answer>"
    )

    return {
        "prompt": prompt,
        "images": [it.image for it in chosen],
        "answer": {
            "task_suite": "B",
            "task_type": "B2_odd_one_out",
            "correct_answer": correct,
            "majority_label": majority_label,
            "odd_label": odd_label,
            "labels": [it.label for it in chosen],
            "question_ids": [it.question_id for it in chosen],
            "datasets": [it.dataset for it in chosen],
        },
    }


def gen_b3_label_corruption(rng: random.Random, *, by_label: dict[str, list[VqaItem]], k: int) -> dict:
    labels = sorted(by_label.keys())
    if k < 2:
        raise ValueError("k must be >= 2 for B3")
    if len(labels) < 2:
        raise ValueError("Need at least 2 labels for B3")
    if k >= len(labels):
        # Prefer distinct true labels to avoid text-only shortcut via duplicates.
        raise ValueError(f"B3 with distinct true labels requires k <= {len(labels)-1} (got k={k}).")

    true_labels = rng.sample(labels, k)
    chosen = [rng.choice(by_label[lbl]) for lbl in true_labels]

    corrupt_idx = rng.randrange(k)
    disallowed = set(true_labels)
    disallowed.remove(true_labels[corrupt_idx])
    candidate_corrupt_labels = [l for l in labels if l != true_labels[corrupt_idx] and l not in disallowed]
    if not candidate_corrupt_labels:
        candidate_corrupt_labels = [l for l in labels if l != true_labels[corrupt_idx]]

    corrupt_to = rng.choice(candidate_corrupt_labels)
    shown_labels = list(true_labels)
    shown_labels[corrupt_idx] = corrupt_to

    letters = _letters(k)
    correct = letters[corrupt_idx]

    prompt = (
        "You are a medical VQA assistant.\n\n"
        "Below are image-label pairs. Exactly one claimed label is incorrect.\n"
        "Identify which position has the corrupted (wrong) label.\n\n"
        f"Pairs are labeled {_letter_options(letters)}.\n\n"
        f"{_render_labeled_images(letters, shown_labels)}\n\n"
        "Answer with only one letter.\n"
        f"<answer> {_letter_options(letters)} </answer>"
    )

    return {
        "prompt": prompt,
        "images": [it.image for it in chosen],
        "answer": {
            "task_suite": "B",
            "task_type": "B3_label_corruption",
            "correct_answer": correct,
            "true_labels": true_labels,
            "shown_labels": shown_labels,
            "corrupt_from": true_labels[corrupt_idx],
            "corrupt_to": corrupt_to,
            "question_ids": [it.question_id for it in chosen],
            "datasets": [it.dataset for it in chosen],
        },
    }


def gen_b4_exemplar_match(rng: random.Random, *, by_label: dict[str, list[VqaItem]], num_candidates: int) -> dict:
    if num_candidates < 1:
        raise ValueError("num_candidates must be >= 1 for B4")

    labels = sorted(by_label.keys())
    if len(labels) < 2:
        raise ValueError("Need at least 2 labels for B4")

    anchor_candidates = [l for l in labels if len(by_label[l]) >= 2]
    if not anchor_candidates:
        raise ValueError("Need at least one label with >=2 items for B4")
    anchor_label = rng.choice(anchor_candidates)
    anchor_pool = by_label[anchor_label]

    ref_item, pos_item = _sample_distinct(rng, anchor_pool, 2)

    negative_labels = [l for l in labels if l != anchor_label]
    neg_labels = (
        rng.sample(negative_labels, num_candidates - 1)
        if (num_candidates - 1) <= len(negative_labels)
        else [rng.choice(negative_labels) for _ in range(num_candidates - 1)]
    )
    neg_items = [rng.choice(by_label[lbl]) for lbl in neg_labels]

    candidates = [pos_item, *neg_items]
    rng.shuffle(candidates)

    letters = _letters(1 + num_candidates)
    ref_letter = letters[0]
    cand_letters = letters[1:]

    correct_idx = next(i for i, it in enumerate(candidates) if it.image == pos_item.image)
    correct = cand_letters[correct_idx]

    prompt = (
        "You are a medical VQA assistant.\n\n"
        f"Reference image ({ref_letter}): <image>\n\n"
        f"Candidate images, labeled {_letter_options(cand_letters)}:\n\n"
        + "\n".join([f"({ltr}) Candidate {ltr}: <image>" for ltr in cand_letters])
        + "\n\n"
        f"Task (B4 Exemplar-match): Exactly one candidate depicts the same condition as the reference image {ref_letter}.\n"
        "Which candidate is it?\n\n"
        "Answer with only one letter.\n"
        f"<answer> {_letter_options(cand_letters)} </answer>"
    )

    return {
        "prompt": prompt,
        "images": [ref_item.image, *[it.image for it in candidates]],
        "answer": {
            "task_suite": "B",
            "task_type": "B4_exemplar_match",
            "correct_answer": correct,
            "anchor_label": anchor_label,
            "labels": [ref_item.label, *[it.label for it in candidates]],
            "question_ids": [ref_item.question_id, *[it.question_id for it in candidates]],
            "datasets": [ref_item.dataset, *[it.dataset for it in candidates]],
        },
    }


def gen_b5_same_different(rng: random.Random, *, by_label: dict[str, list[VqaItem]], same_prob: float) -> dict:
    labels = sorted(by_label.keys())
    if len(labels) < 2:
        raise ValueError("Need at least 2 labels for B5")

    want_same = rng.random() < same_prob
    same_labels = [l for l in labels if len(by_label[l]) >= 2]
    if want_same and same_labels:
        lbl = rng.choice(same_labels)
        pool = by_label[lbl]
        a, b = _sample_distinct(rng, pool, 2)
        correct = "same"
        used = [a, b]
    else:
        a_lbl, b_lbl = rng.sample(labels, 2)
        a = rng.choice(by_label[a_lbl])
        b = rng.choice(by_label[b_lbl])
        correct = "different" if a_lbl != b_lbl else "same"
        used = [a, b]

    prompt = (
        "You are a medical VQA assistant.\n\n"
        "(A) Image A: <image>\n"
        "(B) Image B: <image>\n\n"
        "Task (B5 Same/Different): Do these two images depict the same condition?\n\n"
        "Answer with exactly one token: 'same' or 'different'.\n"
        "<answer> same or different </answer>"
    )

    return {
        "prompt": prompt,
        "images": [used[0].image, used[1].image],
        "answer": {
            "task_suite": "B",
            "task_type": "B5_same_different",
            "correct_answer": correct,
            "labels": [used[0].label, used[1].label],
            "question_ids": [used[0].question_id, used[1].question_id],
            "datasets": [used[0].dataset, used[1].dataset],
        },
    }


def gen_b6_pair_finding(rng: random.Random, *, by_label: dict[str, list[VqaItem]], k: int) -> dict:
    if k < 2:
        raise ValueError("k must be >= 2 for B6")

    labels = [l for l, items in by_label.items() if len(items) >= 2]
    if not labels:
        raise ValueError("Need at least one label with >=2 items for B6")

    pair_label = rng.choice(sorted(labels))
    a, b = _sample_distinct(rng, by_label[pair_label], 2)

    other_labels = [l for l in sorted(by_label.keys()) if l != pair_label]
    if len(other_labels) < (k - 2):
        raise ValueError(f"Need at least {k-2} other labels (got {len(other_labels)}) for B6 with k={k}")

    distractor_labels = rng.sample(other_labels, k - 2)
    distractors = [rng.choice(by_label[l]) for l in distractor_labels]

    chosen = [a, b, *distractors]
    rng.shuffle(chosen)

    letters = _letters(k)
    idxs = [i for i, it in enumerate(chosen) if it.label == pair_label]
    if len(idxs) != 2:
        raise RuntimeError("Internal error: expected exactly two items of pair_label in chosen")
    pair_letters = sorted([letters[i] for i in idxs])
    correct = " ".join(pair_letters)

    prompt = (
        "You are a medical VQA assistant.\n\n"
        f"Below are {k} images, labeled {_letter_options(letters)}.\n\n"
        f"{_render_images(letters)}\n\n"
        "Task (B6 Pair-finding): Exactly two images depict the same condition; all other images depict different conditions.\n"
        "Identify the two matching images.\n\n"
        "Answer with exactly two letters separated by a space (order does not matter).\n"
        "<answer> A B </answer>"
    )

    return {
        "prompt": prompt,
        "images": [it.image for it in chosen],
        "answer": {
            "task_suite": "B",
            "task_type": "B6_pair_finding",
            "correct_answer": correct,
            "pair_label": pair_label,
            "labels": [it.label for it in chosen],
            "question_ids": [it.question_id for it in chosen],
            "datasets": [it.dataset for it in chosen],
        },
    }


def gen_b7_support_set_nway(rng: random.Random, *, by_label: dict[str, list[VqaItem]], n_way: int) -> dict:
    if n_way < 2:
        raise ValueError("n_way must be >= 2 for B7")

    labels = sorted(by_label.keys())
    if len(labels) < n_way:
        raise ValueError(f"Need at least {n_way} labels for B7 (got {len(labels)})")

    queryable = [l for l in labels if len(by_label[l]) >= 2]
    if not queryable:
        raise ValueError("Need at least one label with >=2 items to sample a distinct query for B7")
    target_label = rng.choice(queryable)

    other_support = rng.sample([l for l in labels if l != target_label], n_way - 1)
    support_labels = [target_label, *other_support]
    rng.shuffle(support_labels)
    supports = [rng.choice(by_label[lbl]) for lbl in support_labels]

    support_idx = support_labels.index(target_label)
    support_item = supports[support_idx]
    # sample query different from support_item when possible
    pool = [it for it in by_label[target_label] if it.image != support_item.image]
    query_item = rng.choice(pool) if pool else rng.choice(by_label[target_label])

    letters = _letters(n_way)
    correct = letters[support_idx]

    support_block = "\n".join(
        [f"({ltr}) Support label: {lbl}\n({ltr}) Support image {ltr}: <image>" for ltr, lbl in zip(letters, support_labels, strict=True)]
    )
    prompt = (
        "You are a medical VQA assistant.\n\n"
        f"Support set (B7 Support-set N-way): There are {n_way} labeled support examples:\n\n"
        f"{support_block}\n\n"
        "(Q) Query image: <image>\n\n"
        "Task: Which support label matches the query image?\n\n"
        "Answer with only one letter corresponding to the matching support example.\n"
        f"<answer> {_letter_options(letters)} </answer>"
    )

    return {
        "prompt": prompt,
        "images": [* [s.image for s in supports], query_item.image],
        "answer": {
            "task_suite": "B",
            "task_type": "B7_support_set_nway",
            "correct_answer": correct,
            "support_labels": support_labels,
            "target_label": target_label,
            # Keep types consistent across tasks for HF/pyarrow JSON loading:
            # use flat lists aligned with `images` (supports then query).
            "question_ids": [* [s.question_id for s in supports], query_item.question_id],
            "datasets": [* [s.dataset for s in supports], query_item.dataset],
        },
    }


TaskName = Literal["B1", "B2", "B3", "B4", "B5", "B6", "B7"]


def _parse_task_specs(raw_specs: list[str]) -> dict[TaskName, int]:
    out: dict[TaskName, int] = {}
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"Invalid --task {raw!r}. Expected like B1=1000.")
        name, num = raw.split("=", 1)
        name = name.strip().upper()
        num = num.strip()
        if name not in {"B1", "B2", "B3", "B4", "B5", "B6", "B7"}:
            raise ValueError(f"Unknown task {name!r}. Expected one of B1..B7.")
        count = int(num)
        if count < 0:
            raise ValueError(f"Task count must be >= 0 (got {count})")
        out[name] = count  # type: ignore[assignment]
    return out


def build_label_spaces(rows: Iterable[dict], by: LabelSpaceBy) -> dict[str, list[VqaItem]]:
    spaces: dict[str, list[VqaItem]] = defaultdict(list)
    for r in rows:
        it = _as_vqa_item(r)
        if it is None:
            continue
        key = label_space_key(it, by)
        spaces[key].append(it)
    return spaces


def _by_label(items: list[VqaItem]) -> dict[str, list[VqaItem]]:
    out: dict[str, list[VqaItem]] = defaultdict(list)
    for it in items:
        out[it.label].append(it)
    return out


def generate_b_tasks(
    rows: list[dict],
    *,
    seed: int,
    label_space_by: LabelSpaceBy,
    task_counts: dict[TaskName, int],
    k: int,
    b4_candidates: int,
    b5_same_prob: float,
    b7_nway: int,
    shuffle: bool,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    spaces = build_label_spaces(rows, label_space_by)

    # Precompute eligible spaces per task (avoid infinite retries).
    eligible: dict[TaskName, list[str]] = {t: [] for t in task_counts}
    for key, items in spaces.items():
        by_lbl = _by_label(items)
        labels = sorted(by_lbl.keys())

        def has_label_with_at_least(n: int) -> bool:
            return any(len(v) >= n for v in by_lbl.values())

        if len(labels) >= 2:
            eligible.setdefault("B1", []).append(key)
            eligible.setdefault("B2", []).append(key)
            eligible.setdefault("B5", []).append(key)
        if len(labels) >= (k + 1):
            eligible.setdefault("B3", []).append(key)
        if len(labels) >= 2 and has_label_with_at_least(2):
            eligible.setdefault("B4", []).append(key)
        if len(labels) >= (k - 1) and has_label_with_at_least(2):
            eligible.setdefault("B6", []).append(key)
        if len(labels) >= b7_nway and has_label_with_at_least(2):
            eligible.setdefault("B7", []).append(key)

    out_rows: list[dict] = []
    task_failures: dict[str, int] = Counter()

    def pick_space(task: TaskName) -> str:
        keys = eligible.get(task) or []
        if not keys:
            raise ValueError(f"No eligible label spaces for task {task} under label_space_by={label_space_by!r}")
        return rng.choice(keys)

    for task, count in task_counts.items():
        if count <= 0:
            continue
        for _ in range(count):
            row: dict | None = None
            for _attempt in range(50):
                space_key = pick_space(task)
                items = spaces[space_key]
                by_lbl = _by_label(items)
                try:
                    if task == "B1":
                        row = gen_b1_target_search(rng, by_label=by_lbl, k=k)
                    elif task == "B2":
                        row = gen_b2_odd_one_out(rng, by_label=by_lbl, k=k)
                    elif task == "B3":
                        row = gen_b3_label_corruption(rng, by_label=by_lbl, k=k)
                    elif task == "B4":
                        row = gen_b4_exemplar_match(rng, by_label=by_lbl, num_candidates=b4_candidates)
                    elif task == "B5":
                        row = gen_b5_same_different(rng, by_label=by_lbl, same_prob=b5_same_prob)
                    elif task == "B6":
                        row = gen_b6_pair_finding(rng, by_label=by_lbl, k=k)
                    elif task == "B7":
                        row = gen_b7_support_set_nway(rng, by_label=by_lbl, n_way=b7_nway)
                    else:
                        raise ValueError(f"Unhandled task {task}")
                    # Add label-space metadata for tracking.
                    ans = row.get("answer")
                    if isinstance(ans, dict):
                        ans["label_space_key"] = space_key
                        ans["label_space_by"] = label_space_by
                    break
                except Exception:
                    task_failures[task] += 1
                    row = None
                    continue
            if row is None:
                continue

            out_rows.append(row)

    if shuffle:
        rng.shuffle(out_rows)

    info = {
        "seed": seed,
        "label_space_by": label_space_by,
        "k": k,
        "b4_candidates": b4_candidates,
        "b5_same_prob": b5_same_prob,
        "b7_nway": b7_nway,
        "requested": dict(task_counts),
        "generated": len(out_rows),
        "task_failures": dict(task_failures),
        "num_label_spaces": len(spaces),
    }
    return out_rows, info


def cmd_list(args: argparse.Namespace) -> None:
    ds = discover_omnimedvqa_datasets(args.omni_root)
    for name in sorted(ds.keys()):
        print(name)


def cmd_inspect(args: argparse.Namespace) -> None:
    ds_map = discover_omnimedvqa_datasets(args.omni_root)
    chosen = args.datasets
    missing = [d for d in chosen if d not in ds_map]
    if missing:
        raise SystemExit(f"Unknown datasets: {missing}. Use 'list' to see available names.")

    all_items: list[dict] = []
    for d in chosen:
        all_items.extend(_read_json(ds_map[d]))

    qtypes = Counter(_norm_text(it.get("question_type", "")) for it in all_items)
    modalities = Counter(_norm_text(it.get("modality_type", "")) for it in all_items)
    print(json.dumps({"total": len(all_items), "question_types": qtypes.most_common(), "modality_types": modalities.most_common()}, ensure_ascii=False, indent=2))


def cmd_build_base(args: argparse.Namespace) -> None:
    ds_map = discover_omnimedvqa_datasets(args.omni_root)
    if args.dataset_regex:
        pat = re.compile(args.dataset_regex)
        datasets = [name for name in sorted(ds_map.keys()) if pat.search(name)]
    else:
        datasets = args.datasets

    if not datasets:
        raise SystemExit("No datasets selected. Provide --datasets ... or --dataset-regex ...")

    missing = [d for d in datasets if d not in ds_map]
    if missing:
        raise SystemExit(f"Unknown datasets: {missing}. Use 'list' to see available names.")

    raw_items: list[dict] = []
    for d in datasets:
        raw_items.extend(_read_json(ds_map[d]))

    allowed_qtypes = set(args.question_type) if args.question_type else None
    allowed_ds = set(datasets)
    allowed_mod = set(args.modality_type) if args.modality_type else None

    rows, build_info = build_base_rows(
        raw_items,
        omni_root=args.omni_root,
        allowed_question_types=allowed_qtypes,
        allowed_datasets=allowed_ds,
        allowed_modality_types=allowed_mod,
        min_option_count=args.min_option_count,
        max_option_count=args.max_option_count,
        skip_missing_images=args.skip_missing_images,
    )

    ratios = _parse_ratio3(args.split)
    train, val, test, split_info = split_by_group(
        rows,
        seed=args.seed,
        ratios=ratios,
        group_id_fn=default_group_id,
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "val.jsonl", val)
    _write_jsonl(out_dir / "test.jsonl", test)

    fewshot_rows: list[dict] | None = None
    if args.fewshot_ratio is not None:
        fewshot_rows = fewshot_balanced_by_label(train, ratio=args.fewshot_ratio, seed=args.seed)
        _write_jsonl(out_dir / f"train_fewshot_{args.fewshot_ratio}.jsonl", fewshot_rows)

    summary = {
        "datasets": datasets,
        "filters": {
            "question_type": args.question_type,
            "modality_type": args.modality_type,
            "min_option_count": args.min_option_count,
            "max_option_count": args.max_option_count,
        },
        "build": build_info,
        "split": split_info,
        "train": _summarize_rows(train),
        "val": _summarize_rows(val),
        "test": _summarize_rows(test),
    }
    if fewshot_rows is not None:
        summary["fewshot"] = {
            "ratio": args.fewshot_ratio,
            "train_fewshot": _summarize_rows(fewshot_rows),
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_build_comparative(args: argparse.Namespace) -> None:
    rows = list(_read_jsonl(args.input))
    task_counts = _parse_task_specs(args.task)
    if not task_counts:
        raise SystemExit("No tasks requested. Provide at least one --task like B1=1000 (repeatable).")
    if args.k < 2:
        raise SystemExit("--k must be >= 2")
    if args.k > 26:
        raise SystemExit("--k must be <= 26 (A-Z labeling)")
    if (1 + args.b4_candidates) > 26:
        raise SystemExit("--b4-candidates too large (needs 1+N <= 26 for A-Z labeling)")
    if args.b7_nway > 26:
        raise SystemExit("--b7-nway must be <= 26 (A-Z labeling)")
    if not (0.0 <= args.b5_same_prob <= 1.0):
        raise SystemExit("--b5-same-prob must be in [0, 1]")

    out_rows, info = generate_b_tasks(
        rows,
        seed=args.seed,
        label_space_by=args.label_space_by,
        task_counts=task_counts,
        k=args.k,
        b4_candidates=args.b4_candidates,
        b5_same_prob=args.b5_same_prob,
        b7_nway=args.b7_nway,
        shuffle=not args.no_shuffle,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output, out_rows)
    (args.output.parent / (args.output.stem + ".summary.json")).write_text(
        json.dumps({"output": str(args.output), "info": info, "tasks": Counter(r.get("answer", {}).get("task_type", "unknown") for r in out_rows).most_common()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "generated": len(out_rows), "info": info}, ensure_ascii=False, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OmniMedVQA benchmark builder + Comparative-RFT tasks (B1–B7)")
    p.add_argument("--omni_root", type=Path, default=_default_omni_root(), help="Path to OmniMedVQA root dir")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List available dataset JSON names")
    p_list.set_defaults(func=cmd_list)

    p_ins = sub.add_parser("inspect", help="Inspect question_type / modality_type stats for selected datasets")
    p_ins.add_argument("--datasets", nargs="+", required=True, help="Dataset names (match QA_information/*.json stem)")
    p_ins.set_defaults(func=cmd_inspect)

    p_base = sub.add_parser("build-base", help="Build base EasyR1 JSONL (prompt/images/answer) + split + optional fewshot")
    p_base.add_argument("--datasets", nargs="*", default=[], help="Dataset names (match QA_information/*.json stem)")
    p_base.add_argument("--dataset-regex", dest="dataset_regex", default=None, help="Regex to select datasets by name")
    p_base.add_argument("--question-type", action="append", default=[], help="Filter by question_type (repeatable)")
    p_base.add_argument("--modality-type", action="append", default=[], help="Filter by modality_type (repeatable)")
    p_base.add_argument("--min-option-count", type=int, default=None, help="Filter by minimum non-empty option count")
    p_base.add_argument("--max-option-count", type=int, default=None, help="Filter by maximum non-empty option count")
    p_base.add_argument("--split", type=str, default="0.8,0.1,0.1", help="train,val,test ratios (comma-separated)")
    p_base.add_argument("--seed", type=int, default=42)
    p_base.add_argument("--skip-missing-images", action="store_true", help="Skip rows whose image file is missing")
    p_base.add_argument("--fewshot-ratio", type=float, default=None, help="If set, write balanced fewshot train subset")
    p_base.add_argument("--out-dir", type=Path, required=True, help="Output directory (writes train/val/test.jsonl)")
    p_base.set_defaults(func=cmd_build_base)

    p_comp = sub.add_parser("build-comparative", help="Generate comparative tasks (B1–B7) from a base train.jsonl")
    p_comp.add_argument("--input", type=Path, required=True, help="Input JSONL (usually train.jsonl)")
    p_comp.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    p_comp.add_argument("--seed", type=int, default=42)
    p_comp.add_argument(
        "--label-space-by",
        type=str,
        default="question_type+optioncount",
        choices=["question_type", "question_type+modality", "question_type+optioncount", "question_type+modality+optioncount"],
        help="How to define 'label space' groups for B1–B7 sampling",
    )
    p_comp.add_argument("--task", action="append", default=[], help="Task spec like B1=1000 (repeatable)")
    p_comp.add_argument("--k", type=int, default=4, help="K images for B1/B2/B3/B6")
    p_comp.add_argument("--b4-candidates", type=int, default=3, help="Num candidates (excluding reference) for B4")
    p_comp.add_argument("--b5-same-prob", type=float, default=0.5, help="P(same) for B5")
    p_comp.add_argument("--b7-nway", type=int, default=5, help="N-way for B7 support-set")
    p_comp.add_argument("--no-shuffle", action="store_true", help="Do not shuffle the final dataset")
    p_comp.set_defaults(func=cmd_build_comparative)

    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
