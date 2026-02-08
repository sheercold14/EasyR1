#!/usr/bin/env python3
"""
OmniMedVQA -> EasyR1 data construction engine + Comparative-RFT (B1–B7).

This script supports:
  1) Loading / merging OmniMedVQA JSON files (QA_information/*/*.json)
  2) Filtering by question_type (and other optional fields)
  3) Train/val/test splitting with leak-avoidance via grouping keys
  4) Few-shot sampling from train with per-label balancing
  5) Writing EasyR1-compatible JSONL with `Images/...` relative image paths
  6) Rewriting single-image MCQ rows to optionless label-text rows
  7) Expanding train data into multi-image comparative tasks (B1–B7)
  8) One-shot mixed-train build via config (optionless + B tasks + ratio/shuffle/summary)
  9) Export DTD-style few-shot JSONL (image/label/split/problem) for OmniMedVQA

Default paths assume this repo layout:
  <repo_root>/data/OmniMedVQA
  <repo_root>/data/OminiMedExpert

Usage (few-shot DTD export, matches data/datasets_fewshot/*.jsonl format):
  python3 Comparative-R1/scripts/ominimed_expertv2.py \
    --omni_root /path/to/OmniMedVQA \
    build-fewshot-dtd \
    --dataset-regex "ISIC" \
    --question-type "Disease Diagnosis" \
    --min-option-count 2 \
    --max-option-count 4 \
    --shots 4 \
    --seed 42 \
    --label-pool-size 30 \
    --skip-missing-images \
    --out-dir /mnt/cache/wuruixiao/users/lsc/EasyR1/data/CLS/ISIC
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
    # .../EasyR1/scripts/OminiExpert/omnimed_expert.py -> parents[2] == .../EasyR1
    return Path(__file__).resolve().parents[2]


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


def _load_structured_config(path: Path) -> dict:
    """
    Load JSON/YAML config for one-shot data build pipelines.
    """
    txt = path.read_text(encoding="utf-8")
    suf = path.suffix.lower()
    if suf == ".json":
        obj = json.loads(txt)
    else:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("YAML config requires PyYAML. Install it or use .json config.") from exc
        obj = yaml.safe_load(txt)
    if not isinstance(obj, dict):
        raise ValueError(f"Config root must be an object/dict: {path}")
    return obj


def _cfg_path(value: object, *, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected non-empty string path in config")
    p = Path(value.strip()).expanduser()
    if not p.is_absolute():
        # Prefer repo-root relative paths for portability across invocation dirs.
        # Fallback to CWD / config-dir when the relative input exists there.
        repo_candidate = (_repo_root() / p).resolve()
        cwd_candidate = (Path.cwd() / p).resolve()
        cfg_candidate = (base_dir / p).resolve()

        if repo_candidate.exists():
            return repo_candidate
        if cwd_candidate.exists():
            return cwd_candidate
        if cfg_candidate.exists():
            return cfg_candidate

        # For outputs that do not exist yet, default to repo-root relative.
        return repo_candidate
    return p


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
    # Store as OmniMedVQA-root relative path without leading slash, e.g. "Images/...".
    p = str(image_path).strip()
    if not p:
        return ""
    return p.lstrip("/")


def resolve_image_path(omni_root: Path, root_rel_path: str) -> Path:
    p = str(root_rel_path).strip()
    if p.startswith("/"):
        p = p[1:]
    return omni_root / p


_SLICE_RE = re.compile(r"^(?P<prefix>.+)_(?P<axis>[xyz])_(?P<idx>\\d+)$", flags=re.IGNORECASE)
_OPTIONLESS_QUESTION_RE = re.compile(r"^\s*Question\s*:\s*(?P<q>.+?)\s*$", flags=re.IGNORECASE | re.MULTILINE)


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


def fewshot_kshot_by_label(rows: list[dict], *, shots: int, seed: int) -> list[dict]:
    if shots <= 0:
        return []

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

    out: list[dict] = []
    missing: list[str] = []
    for lbl in sorted(by_label.keys()):
        pool = by_label[lbl]
        if len(pool) < shots:
            missing.append(f"{lbl} ({len(pool)}/{shots})")
            continue
        out.extend(rng.sample(pool, shots))
    if missing:
        raise ValueError("Not enough samples for k-shot labels: " + ", ".join(missing))
    rng.shuffle(out)
    return out


def fewshot_kshot_with_rest_as_test(
    rows: list[dict], *, shots: int, seed: int
) -> tuple[list[dict], list[dict], dict[str, object]]:
    if shots <= 0:
        raise ValueError("shots must be > 0")

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

    train: list[dict] = []
    test: list[dict] = []
    missing: list[str] = []
    for lbl in sorted(by_label.keys()):
        pool = list(by_label[lbl])
        if len(pool) < shots:
            missing.append(f"{lbl} ({len(pool)}/{shots})")
            continue
        rng.shuffle(pool)
        train.extend(pool[:shots])
        test.extend(pool[shots:])

    if missing:
        raise ValueError("Not enough samples for k-shot labels: " + ", ".join(missing))

    rng.shuffle(train)
    rng.shuffle(test)
    info: dict[str, object] = {
        "mode": "fewshot_train_rest_test",
        "shots": shots,
        "seed": seed,
        "train_rows": len(train),
        "test_rows": len(test),
        "labels": len(by_label),
    }
    return train, test, info


def _extract_question_from_prompt(prompt: str) -> str:
    m = _OPTIONLESS_QUESTION_RE.search(prompt)
    if not m:
        return ""
    return m.group("q").strip()


def _norm_label_for_optionless(s: object) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return ""
    s2 = re.sub(r"\s+", " ", s)
    s2 = re.sub(r"\.\s*$", "", s2).strip()
    # Common noise in OmniMed: "Benign image."
    s2 = re.sub(r"\bimage\b\.?$", "", s2, flags=re.IGNORECASE).strip()
    return s2


def _normalize_label_value(val: object, *, normalize: bool) -> str:
    if val is None:
        return ""
    if normalize:
        return _norm_label_for_optionless(val)
    return _norm_text(val)


def _normalize_rows_labels(rows: list[dict]) -> tuple[list[dict], dict]:
    """
    Normalize answer labels in rows and drop rows that become empty.
    """
    out: list[dict] = []
    dropped = 0
    changed = 0
    for r in rows:
        ans = r.get("answer") if isinstance(r.get("answer"), dict) else {}
        raw_label = ans.get("label")
        new_label = _norm_label_for_optionless(raw_label)
        if not new_label:
            dropped += 1
            continue
        if isinstance(raw_label, str) and raw_label.strip() != new_label:
            changed += 1
        new_ans = dict(ans)
        new_ans["label"] = new_label
        new_row = dict(r)
        new_row["answer"] = new_ans
        out.append(new_row)
    info = {
        "normalized": True,
        "rows_in": len(rows),
        "rows_out": len(out),
        "labels_changed": changed,
        "rows_dropped": dropped,
    }
    return out, info


def build_fewshot_problem(question: str, candidate_labels: list[str], *, default_question: str) -> str:
    q = question.strip() if isinstance(question, str) else ""
    if not q:
        q = default_question
    labels = [l.strip() for l in candidate_labels if isinstance(l, str) and l.strip()]
    return f"{q}\nPlease choose one from list [{', '.join(labels)}]."


def _sample_candidate_labels(
    rng: random.Random, *, label_pool: list[str], correct_label: str, k: int
) -> list[str]:
    if not correct_label:
        return []
    pool = list(label_pool)
    if correct_label not in pool:
        pool.append(correct_label)
    if k <= 0:
        return [correct_label]
    k = min(k, len(pool))
    if k == 1:
        return [correct_label]
    distractors = [l for l in pool if l != correct_label]
    if len(distractors) >= (k - 1):
        distractors = rng.sample(distractors, k - 1)
    candidates = [correct_label, *distractors]
    rng.shuffle(candidates)
    return candidates


def _label_pool_key_from_answer(ans: dict[str, object], by: str) -> str:
    if by == "all":
        return "all"
    qtype = _norm_text(ans.get("question_type", ""))
    modality = _norm_text(ans.get("modality_type", ""))
    question = _norm_text(ans.get("question", ""))
    opt_count = _option_count({k: ans.get(k) for k in OPTION_KEYS})
    if by == "question_type+modality":
        return f"question_type={qtype}|modality_type={modality}"
    if by == "question":
        return f"question={question}"
    if by == "option_count":
        return f"option_count={opt_count}"
    if by == "question_type+option_count":
        return f"question_type={qtype}|option_count={opt_count}"
    return f"question_type={qtype}"


def _sample_from_pool_with_label(
    *,
    rng: random.Random,
    pool: list[str],
    correct_label: str,
    k: int,
) -> list[str]:
    if k <= 0 or k >= len(pool):
        return list(pool)
    pool_set = list(dict.fromkeys(pool).keys())
    if correct_label and correct_label not in pool_set:
        pool_set.append(correct_label)
    if k >= len(pool_set):
        return pool_set
    if k == 1:
        return [correct_label] if correct_label else [rng.choice(pool_set)]
    distractors = [l for l in pool_set if l != correct_label]
    if len(distractors) >= (k - 1):
        distractors = rng.sample(distractors, k - 1)
    candidates = [correct_label, *distractors] if correct_label else rng.sample(pool_set, k)
    rng.shuffle(candidates)
    return candidates


def _build_fewshot_dtd_rows(
    rows: list[dict],
    *,
    split: str,
    rng: random.Random,
    label_pools: dict[str, list[str]],
    label_pool_size: int,
    label_pool_by: str,
    candidate_source: str,
    default_question: str,
    normalize_labels: bool,
) -> tuple[list[dict], dict]:
    out: list[dict] = []
    skipped = 0
    candidate_source_counts = Counter()
    for r in rows:
        imgs = r.get("images") or []
        if not isinstance(imgs, list) or not imgs or not isinstance(imgs[0], str):
            skipped += 1
            continue
        ans = r.get("answer") if isinstance(r.get("answer"), dict) else {}
        label = _normalize_label_value(ans.get("label", ""), normalize=normalize_labels)
        if not label:
            skipped += 1
            continue
        prompt = r.get("prompt")
        question = _extract_question_from_prompt(prompt) if isinstance(prompt, str) else ""
        candidates: list[str] = []
        if candidate_source == "original_options":
            candidate_labels: list[str] | None = None
            if isinstance(ans.get("candidate_labels"), list):
                candidate_labels = [str(x) for x in ans.get("candidate_labels") if x is not None]
                candidate_source_counts["candidate_labels"] += 1
            else:
                options = _extract_options_for_optionless(ans)
                if options:
                    candidate_labels = options
                    candidate_source_counts["options"] += 1

            if candidate_labels:
                seen: set[str] = set()
                for c in candidate_labels:
                    lbl = _normalize_label_value(c, normalize=normalize_labels)
                    if not lbl or lbl in seen:
                        continue
                    seen.add(lbl)
                    candidates.append(lbl)
                if label and label not in seen:
                    candidates.append(label)
            else:
                candidate_source_counts["label_pool_fallback"] += 1
                pool_key = _label_pool_key_from_answer(ans, label_pool_by)
                pool = label_pools.get(pool_key) or label_pools.get("all") or []
                candidates = _sample_from_pool_with_label(
                    rng=rng,
                    pool=pool,
                    correct_label=label,
                    k=label_pool_size,
                )
        else:
            candidate_source_counts["label_pool"] += 1
            pool_key = _label_pool_key_from_answer(ans, label_pool_by)
            pool = label_pools.get(pool_key) or label_pools.get("all") or []
            candidates = _sample_from_pool_with_label(
                rng=rng,
                pool=pool,
                correct_label=label,
                k=label_pool_size,
            )
        if label and label not in candidates:
            candidates.append(label)
        problem = build_fewshot_problem(
            question,
            candidates,
            default_question=default_question,
        )
        out.append(
            {
                "image": imgs[0],
                "label": label,
                "split": split,
                "problem": problem,
            }
        )
    info = {
        "split": split,
        "rows_in": len(rows),
        "rows_out": len(out),
        "rows_skipped": skipped,
        "candidate_source_counts": dict(candidate_source_counts),
    }
    return out, info


def _extract_options_for_optionless(ans: dict[str, object]) -> list[str]:
    opts: list[str] = []
    for k in OPTION_KEYS:
        v = ans.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() == "none":
            continue
        opts.append(s)
    return opts


def _is_binary_benign_malignant_options(options: list[str]) -> bool:
    if len(options) != 2:
        return False
    norm = {_norm_label_for_optionless(o).lower() for o in options if _norm_label_for_optionless(o)}
    norm = {re.sub(r"[^a-z]+", " ", x).strip() for x in norm}
    return norm == {"benign", "malignant"}


def _optionless_group_key(ans: dict[str, object]) -> str:
    qtype = str(ans.get("question_type", "unknown")).strip() or "unknown"
    opts = _extract_options_for_optionless(ans)
    if _is_binary_benign_malignant_options(opts):
        return f"{qtype}|binary_benign_malignant"
    return f"{qtype}|option_count={len(opts)}"


def build_optionless_prompt(*, question: str, candidate_labels: list[str], style: str = "omnimed") -> str:
    question = question.strip()
    labels = [l.strip() for l in candidate_labels if isinstance(l, str) and l.strip()]
    labels = sorted(dict.fromkeys(labels).keys())

    if style == "dtd":
        q = question or "What is the diagnosis shown in the image?"
        return (
            f"{q}\n"
            f"Please choose one from list [ {', '.join(labels)} ]."
        )

    if _is_binary_benign_malignant_options(labels):
        return (
            "You are a medical VQA assistant for dermoscopy images. "
            "Does the anomaly in this image indicate benign or malignant characteristics?<image>\n"
            " Please choose one from list [Benign, Malignant]. "
            "Answer with ONLY the label text.\n<answer></answer>"
        )

    label_list = ", ".join(labels)
    return (
        "You are a medical VQA assistant for dermoscopy images. "
        "What is the specific skin condition depicted in this image? <image>\n"
        f"Please choose one from list [{label_list}]. "
        "Answer with ONLY the label text.\n<answer></answer>"
    )


def transform_rows_to_optionless(rows: list[dict], *, prompt_style: str = "omnimed") -> tuple[list[dict], dict]:
    """
    Rewrite only single-image MCQ rows to optionless text-label format.

    Design goal: keep train-time transform behavior aligned with
    `Comparative-R1/scripts/eval/eval_optionless_checkpoint_and_dump.py`.
    """
    labels_by_group: dict[str, set[str]] = {}
    for r in rows:
        ans = r.get("answer") if isinstance(r.get("answer"), dict) else {}
        task_type = str(ans.get("task_type", "")).strip()
        if task_type != "mcq_letter":
            continue

        gk = _optionless_group_key(ans)
        if gk.endswith("|binary_benign_malignant"):
            labels_by_group.setdefault(gk, set()).update({"Benign", "Malignant"})
        else:
            lbl = ans.get("label")
            if isinstance(lbl, str) and lbl.strip():
                labels_by_group.setdefault(gk, set()).add(lbl.strip())

    out_rows: list[dict] = []
    rewritten = 0
    skipped_invalid_prompt = 0
    for r in rows:
        prompt = r.get("prompt")
        if not isinstance(prompt, str):
            skipped_invalid_prompt += 1
            out_rows.append(r)
            continue

        ans = r.get("answer") if isinstance(r.get("answer"), dict) else {}
        task_type = str(ans.get("task_type", "")).strip()
        if task_type != "mcq_letter":
            out_rows.append(r)
            continue

        gk = _optionless_group_key(ans)
        candidate = sorted(labels_by_group.get(gk, set()))
        if gk.endswith("|binary_benign_malignant"):
            candidate = ["Benign", "Malignant"]

        question = _extract_question_from_prompt(prompt)
        new_prompt = build_optionless_prompt(question=question, candidate_labels=candidate, style=prompt_style)

        new_row = dict(r)
        new_row["prompt"] = new_prompt

        new_ans = dict(ans)
        if isinstance(new_ans.get("label"), str) and new_ans["label"].strip():
            correct_label = new_ans["label"].strip()
            if gk.endswith("|binary_benign_malignant"):
                correct_label = _norm_label_for_optionless(correct_label)
                if correct_label.lower() in {"benign", "malignant"}:
                    correct_label = correct_label.capitalize()
            new_ans["correct_label"] = correct_label
            new_ans["correct_answer_mcq"] = new_ans.get("correct_answer")
            new_ans["correct_answer"] = new_ans["correct_label"]

        new_ans["candidate_labels"] = candidate
        new_ans["task_type"] = "mcq_optionless_text"
        new_ans["optionless_group_key"] = gk
        new_row["answer"] = new_ans

        out_rows.append(new_row)
        rewritten += 1

    info = {
        "rewritten_mcq": rewritten,
        "skipped_invalid_prompt": skipped_invalid_prompt,
        "num_groups": len(labels_by_group),
        "prompt_style": prompt_style,
        "labels_by_group": {k: sorted(v) for k, v in sorted(labels_by_group.items())},
    }
    return out_rows, info


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
            "question": question,
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


def _render_stepwise_answer_format(letters: list[str], final_placeholder: str) -> str:
    """
    Render a stepwise answer format that first lists per-image labels,
    then provides a final answer line.
    """
    lines = [f"{ltr}: {{label}}" for ltr in letters]
    lines.append(f"Final: {final_placeholder}")
    return "Answer in the following format:\n<answer>\n" + "\n".join(lines) + "\n</answer>"


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
        "First, identify the condition shown in each image.\n"
        f"Final must be one letter: {_letter_options(letters)}.\n"
        + _render_stepwise_answer_format(letters, "{one letter}")
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
        "First, identify the condition shown in each image.\n"
        f"Final must be one letter: {_letter_options(letters)}.\n"
        + _render_stepwise_answer_format(letters, "{one letter}")
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
        "First, identify the condition shown in each image (including the reference).\n"
        f"Final must be one letter: {_letter_options(cand_letters)}.\n"
        + _render_stepwise_answer_format(letters, "{one letter}")
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

    letters = ["A", "B"]
    prompt = (
        "You are a medical VQA assistant.\n\n"
        "(A) Image A: <image>\n"
        "(B) Image B: <image>\n\n"
        "Task (B5 Same/Different): Do these two images depict the same condition?\n\n"
        "First, identify the condition shown in each image.\n"
        "Final must be exactly one token: same or different.\n"
        + _render_stepwise_answer_format(letters, "{same|different}")
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
        "First, identify the condition shown in each image.\n"
        "Final must be exactly two letters separated by a space (order does not matter).\n"
        + _render_stepwise_answer_format(letters, "{two letters}")
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


def _task_counts_from_dict(obj: dict) -> dict[TaskName, int]:
    """
    Parse task counts from config style:
      task_counts:
        B1: 800
        B2: 800
    """
    out: dict[TaskName, int] = {}
    for k, v in obj.items():
        name = str(k).strip().upper()
        if name not in {"B1", "B2", "B3", "B4", "B5", "B6", "B7"}:
            raise ValueError(f"Unknown task name in config: {k!r}")
        try:
            count = int(v)
        except Exception as exc:
            raise ValueError(f"Invalid task count for {name}: {v!r}") from exc
        if count < 0:
            raise ValueError(f"Task count must be >= 0 for {name}, got {count}")
        out[name] = count  # type: ignore[assignment]
    return out


def _sample_rows_with_ratio(
    *,
    single_rows: list[dict],
    btask_rows: list[dict],
    single_ratio: float,
    btask_ratio: float,
    total: int,
    seed: int,
) -> tuple[list[dict], dict]:
    if total <= 0:
        raise ValueError("mix.total must be > 0")
    if single_ratio < 0 or btask_ratio < 0:
        raise ValueError("mix.ratio.* must be >= 0")
    if single_ratio + btask_ratio <= 0:
        raise ValueError("mix.ratio.single + mix.ratio.btask must be > 0")

    s = single_ratio / (single_ratio + btask_ratio)
    b = btask_ratio / (single_ratio + btask_ratio)
    n_single = int(round(total * s))
    n_single = max(0, min(total, n_single))
    n_btask = total - n_single

    if n_single > 0 and len(single_rows) == 0:
        raise ValueError("Requested single rows in mix, but single_rows is empty")
    if n_btask > 0 and len(btask_rows) == 0:
        raise ValueError("Requested btask rows in mix, but btask_rows is empty")

    rng = random.Random(seed)

    oversample_single = n_single > len(single_rows)
    oversample_btask = n_btask > len(btask_rows)

    if oversample_single:
        picked_single = [rng.choice(single_rows) for _ in range(n_single)] if single_rows else []
    else:
        picked_single = rng.sample(single_rows, n_single) if n_single > 0 else []

    if oversample_btask:
        picked_btask = [rng.choice(btask_rows) for _ in range(n_btask)] if btask_rows else []
    else:
        picked_btask = rng.sample(btask_rows, n_btask) if n_btask > 0 else []

    out = [*picked_single, *picked_btask]
    info = {
        "mode": "ratio+total",
        "requested_total": total,
        "requested_ratio": {"single": single_ratio, "btask": btask_ratio},
        "normalized_ratio": {"single": s, "btask": b},
        "picked": {"single": len(picked_single), "btask": len(picked_btask), "total": len(out)},
        "oversample": {"single": oversample_single, "btask": oversample_btask},
    }
    return out, info


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
    if args.fewshot_shots is not None and args.fewshot_shots <= 0:
        raise SystemExit("--fewshot-shots must be > 0")
    if args.fewshot_as_train_rest_as_test and args.fewshot_shots is None:
        raise SystemExit("--fewshot-as-train-rest-as-test requires --fewshot-shots")

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

    if args.fewshot_as_train_rest_as_test:
        train, test, split_info = fewshot_kshot_with_rest_as_test(rows, shots=args.fewshot_shots, seed=args.seed)
        val: list[dict] = []
    else:
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
    if args.fewshot_shots is not None and not args.fewshot_as_train_rest_as_test:
        kshot_rows = fewshot_kshot_by_label(train, shots=args.fewshot_shots, seed=args.seed)
        _write_jsonl(out_dir / f"train_fewshot_{args.fewshot_shots}shot.jsonl", kshot_rows)

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
    if args.fewshot_shots is not None and not args.fewshot_as_train_rest_as_test:
        summary["fewshot_kshot"] = {
            "shots": args.fewshot_shots,
            "train_fewshot": _summarize_rows(kshot_rows),
        }
    if args.fewshot_as_train_rest_as_test:
        summary["fewshot_kshot"] = {
            "shots": args.fewshot_shots,
            "mode": "train_is_kshot_per_label_test_is_rest",
            "train_fewshot": _summarize_rows(train),
            "test_rest": _summarize_rows(test),
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_build_fewshot_dtd(args: argparse.Namespace) -> None:
    if args.shots <= 0:
        raise SystemExit("--shots must be > 0")
    if args.label_pool_size < 0:
        raise SystemExit("--label-pool-size must be >= 0 (0 means use all labels)")

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

    norm_info = {"normalized": False, "rows_in": len(rows), "rows_out": len(rows)}
    if args.normalize_labels:
        rows, norm_info = _normalize_rows_labels(rows)

    train_rows, test_rows, split_info = fewshot_kshot_with_rest_as_test(
        rows, shots=args.shots, seed=args.seed
    )

    label_pool_by_key: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if not isinstance(r.get("answer"), dict):
            continue
        ans = r.get("answer", {})
        raw_label = ans.get("label", "")
        lbl = _normalize_label_value(raw_label, normalize=args.normalize_labels)
        if not lbl:
            continue
        key = _label_pool_key_from_answer(ans, args.label_pool_by)
        label_pool_by_key[key].add(lbl)
    if not label_pool_by_key:
        raise SystemExit("No labels found after filtering.")

    label_pools = {k: sorted(v) for k, v in label_pool_by_key.items()}
    if "all" not in label_pools:
        all_labels = sorted({l for v in label_pools.values() for l in v})
        label_pools["all"] = all_labels

    rng = random.Random(args.seed)

    train_out, train_info = _build_fewshot_dtd_rows(
        train_rows,
        split="train",
        rng=rng,
        label_pools=label_pools,
        label_pool_size=args.label_pool_size,
        label_pool_by=args.label_pool_by,
        candidate_source=args.candidate_source,
        default_question=args.default_question,
        normalize_labels=args.normalize_labels,
    )
    test_out, test_info = _build_fewshot_dtd_rows(
        test_rows,
        split="test",
        rng=rng,
        label_pools=label_pools,
        label_pool_size=args.label_pool_size,
        label_pool_by=args.label_pool_by,
        candidate_source=args.candidate_source,
        default_question=args.default_question,
        normalize_labels=args.normalize_labels,
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def _slugify(name: str) -> str:
        s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        return s or "omnimedvqa"

    if args.out_stem:
        stem = _slugify(args.out_stem)
    elif len(datasets) == 1:
        stem = _slugify(datasets[0])
    elif args.dataset_regex:
        stem = _slugify(args.dataset_regex)
    else:
        stem = "omnimedvqa"

    train_path = out_dir / f"{stem}_fewshot_{args.shots}.jsonl"
    test_path = out_dir / f"{stem}_fewshot_test.jsonl"
    _write_jsonl(train_path, train_out)
    _write_jsonl(test_path, test_out)

    summary = {
        "datasets": datasets,
        "filters": {
            "question_type": args.question_type,
            "modality_type": args.modality_type,
            "min_option_count": args.min_option_count,
            "max_option_count": args.max_option_count,
        },
        "build": build_info,
        "label_normalization": norm_info,
        "fewshot_split": split_info,
        "export": {
            "shots": args.shots,
            "label_pool_size_requested": args.label_pool_size,
            "label_pool_by": args.label_pool_by,
            "candidate_source": args.candidate_source,
            "label_pool_total": len(label_pools.get("all", [])),
            "default_question": args.default_question,
            "train_output": str(train_path),
            "test_output": str(test_path),
            "train_info": train_info,
            "test_info": test_info,
        },
    }
    summary_path = out_dir / f"{stem}_fewshot_{args.shots}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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


def cmd_build_optionless(args: argparse.Namespace) -> None:
    rows = list(_read_jsonl(args.input))
    out_rows, info = transform_rows_to_optionless(rows, prompt_style=args.prompt_style)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output, out_rows)

    summary_path = args.summary
    if summary_path is None:
        summary_path = args.output.parent / f"{args.output.stem}.summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "total_input_rows": len(rows),
        "total_output_rows": len(out_rows),
        "info": info,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_build_train_mix(args: argparse.Namespace) -> None:
    """
    One-shot pipeline:
      1) read single-image source rows
      2) convert single rows to optionless text-label format
      3) read/generate B tasks
      4) mix with ratio + optional total sampling
      5) optional shuffle, write output + summary
    """
    cfg_path = args.config.resolve()
    if not cfg_path.exists():
        raise SystemExit(f"Missing --config: {cfg_path}")
    cfg = _load_structured_config(cfg_path)
    base_dir = cfg_path.parent

    # -------- required paths --------
    single_input = _cfg_path(cfg.get("single_input"), base_dir=base_dir)
    output_train = _cfg_path(cfg.get("output_train"), base_dir=base_dir)
    output_train.parent.mkdir(parents=True, exist_ok=True)
    output_summary = cfg.get("output_summary", None)
    if output_summary is None:
        summary_path = output_train.parent / f"{output_train.stem}.summary.json"
    else:
        summary_path = _cfg_path(output_summary, base_dir=base_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if not single_input.exists():
        raise SystemExit(f"single_input not found: {single_input}")

    # -------- single rows -> optionless --------
    single_rows_raw = list(_read_jsonl(single_input))
    single_rows_opt, optionless_info = transform_rows_to_optionless(
        single_rows_raw, prompt_style=str(cfg.get("optionless_prompt_style", "omnimed"))
    )

    # Optional intermediate dumps.
    output_single_optionless = cfg.get("output_single_optionless", None)
    if output_single_optionless is not None:
        p = _cfg_path(output_single_optionless, base_dir=base_dir)
        _write_jsonl(p, single_rows_opt)

    # -------- B tasks source (read existing or generate) --------
    b_rows: list[dict] = []
    b_source: str = "none"
    if cfg.get("btasks_input", None) is not None:
        b_input = _cfg_path(cfg.get("btasks_input"), base_dir=base_dir)
        if not b_input.exists():
            raise SystemExit(f"btasks_input not found: {b_input}")
        b_rows = list(_read_jsonl(b_input))
        b_source = f"input:{b_input}"
        b_info: dict = {"mode": "input", "rows": len(b_rows)}
    else:
        gen_b = bool(cfg.get("generate_btasks", False))
        if not gen_b:
            b_rows = []
            b_source = "disabled"
            b_info = {"mode": "disabled", "rows": 0}
        else:
            bcfg = cfg.get("btasks", {})
            if not isinstance(bcfg, dict):
                raise SystemExit("`btasks` must be a dict in config when generate_btasks=true")

            task_counts_raw = bcfg.get("task_counts", {})
            if not isinstance(task_counts_raw, dict):
                raise SystemExit("`btasks.task_counts` must be a dict")
            task_counts = _task_counts_from_dict(task_counts_raw)
            if not task_counts:
                raise SystemExit("`btasks.task_counts` is empty")

            label_space_by = str(bcfg.get("label_space_by", "question_type+optioncount"))
            seed = int(bcfg.get("seed", 42))
            k = int(bcfg.get("k", 4))
            b4_candidates = int(bcfg.get("b4_candidates", 3))
            b5_same_prob = float(bcfg.get("b5_same_prob", 0.5))
            b7_nway = int(bcfg.get("b7_nway", 5))
            no_shuffle = bool(bcfg.get("no_shuffle", False))

            b_rows, b_info = generate_b_tasks(
                single_rows_raw,
                seed=seed,
                label_space_by=label_space_by,  # type: ignore[arg-type]
                task_counts=task_counts,
                k=k,
                b4_candidates=b4_candidates,
                b5_same_prob=b5_same_prob,
                b7_nway=b7_nway,
                shuffle=not no_shuffle,
            )
            b_source = "generated"

    output_btasks = cfg.get("output_btasks", None)
    if output_btasks is not None:
        p = _cfg_path(output_btasks, base_dir=base_dir)
        _write_jsonl(p, b_rows)

    # -------- mixing --------
    mix_cfg = cfg.get("mix", {})
    if not isinstance(mix_cfg, dict):
        raise SystemExit("`mix` must be a dict in config")

    mix_seed = int(mix_cfg.get("seed", 42))
    mix_shuffle = bool(mix_cfg.get("shuffle", True))
    total_obj = mix_cfg.get("total", None)

    if total_obj is None:
        mixed_rows = [*single_rows_opt, *b_rows]
        mix_info = {
            "mode": "concat_all",
            "picked": {"single": len(single_rows_opt), "btask": len(b_rows), "total": len(mixed_rows)},
            "note": "ratio ignored when mix.total is null",
        }
    else:
        total = int(total_obj)
        ratio_cfg = mix_cfg.get("ratio", {})
        if not isinstance(ratio_cfg, dict):
            raise SystemExit("`mix.ratio` must be a dict")
        single_ratio = float(ratio_cfg.get("single", 0.5))
        btask_ratio = float(ratio_cfg.get("btask", 0.5))
        mixed_rows, mix_info = _sample_rows_with_ratio(
            single_rows=single_rows_opt,
            btask_rows=b_rows,
            single_ratio=single_ratio,
            btask_ratio=btask_ratio,
            total=total,
            seed=mix_seed,
        )

    if mix_shuffle:
        rng = random.Random(mix_seed)
        rng.shuffle(mixed_rows)

    _write_jsonl(output_train, mixed_rows)

    task_counter = Counter()
    for r in mixed_rows:
        ans = r.get("answer")
        if isinstance(ans, dict):
            task_counter[str(ans.get("task_type", "unknown"))] += 1
        else:
            task_counter["unknown"] += 1

    summary = {
        "config": str(cfg_path),
        "single_input": str(single_input),
        "single_rows_input": len(single_rows_raw),
        "single_rows_optionless": len(single_rows_opt),
        "optionless_info": optionless_info,
        "btask_source": b_source,
        "btask_rows": len(b_rows),
        "btask_info": b_info,
        "mix": mix_info,
        "shuffle": {"enabled": mix_shuffle, "seed": mix_seed},
        "output_train": str(output_train),
        "output_rows": len(mixed_rows),
        "task_type_counts": dict(task_counter),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


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
    p_base.add_argument("--fewshot-shots", type=int, default=None, help="If set, write balanced k-shot per-label train subset")
    p_base.add_argument(
        "--fewshot-as-train-rest-as-test",
        action="store_true",
        help="Use k-shot per label as train and put the remaining samples into test (val will be empty)",
    )
    p_base.add_argument("--out-dir", type=Path, required=True, help="Output directory (writes train/val/test.jsonl)")
    p_base.set_defaults(func=cmd_build_base)

    p_few = sub.add_parser(
        "build-fewshot-dtd",
        help="Export K-shot train + rest-as-test in DTD-style JSONL (image/label/split/problem)",
    )
    p_few.add_argument("--datasets", nargs="*", default=[], help="Dataset names (match QA_information/*.json stem)")
    p_few.add_argument("--dataset-regex", dest="dataset_regex", default=None, help="Regex to select datasets by name")
    p_few.add_argument("--question-type", action="append", default=[], help="Filter by question_type (repeatable)")
    p_few.add_argument("--modality-type", action="append", default=[], help="Filter by modality_type (repeatable)")
    p_few.add_argument("--min-option-count", type=int, default=None, help="Filter by minimum non-empty option count")
    p_few.add_argument("--max-option-count", type=int, default=None, help="Filter by maximum non-empty option count")
    p_few.add_argument("--shots", type=int, required=True, help="K-shot per label for train split")
    p_few.add_argument(
        "--label-pool-size",
        type=int,
        default=0,
        help="Number of labels to include in prompt list (0 means use all labels)",
    )
    p_few.add_argument(
        "--label-pool-by",
        choices=["all", "question", "question_type", "question_type+modality", "option_count", "question_type+option_count"],
        default="question",
        help="How to build the label pool for each question (used when candidate source is label_pool)",
    )
    p_few.add_argument(
        "--candidate-source",
        choices=["label_pool", "original_options"],
        default="label_pool",
        help="Use full label pool or the original options/candidate_labels from the MCQ item",
    )
    p_few.add_argument("--seed", type=int, default=42)
    p_few.add_argument("--skip-missing-images", action="store_true", help="Skip rows whose image file is missing")
    p_few.set_defaults(normalize_labels=True)
    p_few.add_argument(
        "--normalize-labels",
        action="store_true",
        help="Normalize labels (strip trailing '.', remove 'image') before few-shot sampling (default)",
    )
    p_few.add_argument(
        "--no-normalize-labels",
        dest="normalize_labels",
        action="store_false",
        help="Disable label normalization",
    )
    p_few.add_argument(
        "--default-question",
        type=str,
        default="What is the diagnosis shown in the image?",
        help="Fallback question when prompt parsing fails",
    )
    p_few.add_argument("--out-dir", type=Path, required=True, help="Output directory for few-shot JSONL files")
    p_few.add_argument("--out-stem", type=str, default=None, help="Output filename stem (default: dataset name/regex)")
    p_few.set_defaults(func=cmd_build_fewshot_dtd)

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

    p_opt = sub.add_parser(
        "build-optionless",
        help="Rewrite single-image mcq_letter rows to optionless label-text rows (non-MCQ rows are passed through)",
    )
    p_opt.add_argument("--input", type=Path, required=True, help="Input JSONL path")
    p_opt.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    p_opt.add_argument("--prompt-style", choices=["omnimed", "dtd"], default="omnimed", help="Prompt style for rewritten rows")
    p_opt.add_argument("--summary", type=Path, default=None, help="Optional summary JSON path")
    p_opt.set_defaults(func=cmd_build_optionless)

    p_mix = sub.add_parser(
        "build-train-mix",
        help="One-shot build: single->optionless + B tasks (input/generate) + ratio mix + shuffle + summary",
    )
    p_mix.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML/JSON config path for the unified pipeline",
    )
    p_mix.set_defaults(func=cmd_build_train_mix)

    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
