#!/usr/bin/env python3
"""
Sanity-check for OminiMedExpert ISIC training pipeline pieces:
  1) Reward function parsing + correctness
  2) Patched dataset path-resolution (root-relative "/Images/..." under OmniMedVQA)

This does NOT run a full training loop. It is intended to quickly catch:
  - wrong PYTHONPATH overlay (patch not active)
  - reward parse failures
  - image path join issues for "/Images/..." style paths

Run (recommended from repo root):
  python EasyR1/Comparative-R1/scripts/test_omnimed_isic_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read_first_jsonl(path: Path, n: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OminiMedExpert ISIC reward + patched dataset resolution")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/OminiMedExpert/isic2018_2019_2020_disease_diagnosis"),
        help="OminiMedExpert dataset directory",
    )
    parser.add_argument(
        "--omni_root",
        type=Path,
        default=Path("data/OmniMedVQA"),
        help="OmniMedVQA root (contains Images/)",
    )
    parser.add_argument("--n", type=int, default=16, help="How many rows to sample from each file")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    easyr1_root = script_path.parents[2]  # EasyR1/
    repo_root = easyr1_root.parent
    patches_root = easyr1_root / "Comparative-R1" / "verl_patches"

    # Mimic training launcher behavior: ensure patched verl is imported.
    sys.path.insert(0, str(patches_root))
    sys.path.insert(1, str(easyr1_root))
    sys.path.insert(2, str(repo_root))

    import verl  # noqa: E402
    from verl.utils import dataset as dataset_mod  # noqa: E402

    print("verl:", Path(verl.__file__).as_posix())
    print("dataset:", Path(dataset_mod.__file__).as_posix())

    if "Comparative-R1/verl_patches" not in str(dataset_mod.__file__):
        raise SystemExit(
            "Patched dataset is NOT active. Make sure PYTHONPATH includes "
            "`EasyR1/Comparative-R1/verl_patches` before `EasyR1`."
        )

    # Test path resolution logic without constructing a full dataset (no tokenizer/processor needed).
    ds = dataset_mod.RLHFDataset.__new__(dataset_mod.RLHFDataset)
    ds.image_dir = str(args.omni_root.resolve())
    resolved = ds._resolve_media_paths(["/Images/ISIC2018/val/ISIC_0034343.jpg", "Images/ACRIMA/Im002_ACRIMA.png"])
    assert resolved[0].endswith("Images/ISIC2018/val/ISIC_0034343.jpg"), resolved[0]
    assert resolved[1].endswith("Images/ACRIMA/Im002_ACRIMA.png"), resolved[1]
    print("dataset path resolution: OK")

    # Reward tests (load by filepath to avoid PYTHONPATH/package assumptions)
    import importlib.util  # noqa: E402

    reward_path = easyr1_root / "Comparative-R1" / "reward" / "omnimed_isic_reward.py"
    if not reward_path.exists():
        raise SystemExit(f"Missing reward file: {reward_path}")
    spec = importlib.util.spec_from_file_location("omnimed_isic_reward", reward_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to load reward module spec: {reward_path}")
    omnimed_isic_reward = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(omnimed_isic_reward)

    base_train = args.data_dir / "train.jsonl"
    b_tasks = args.data_dir / "comparative" / "train_b_tasks_fewshot0.05.jsonl"
    if not base_train.exists():
        raise SystemExit(f"Missing file: {base_train}")
    if not b_tasks.exists():
        raise SystemExit(f"Missing file: {b_tasks}")

    base_rows = _read_first_jsonl(base_train, args.n)
    b_rows = _read_first_jsonl(b_tasks, args.n)

    def mk_item(row: dict, *, correct: bool) -> dict:
        gt = row.get("answer", {})
        if not isinstance(gt, dict):
            gt = {"correct_answer": "A"}
        ca = str(gt.get("correct_answer", "A")).strip()
        if not ca:
            ca = "A"
        if correct:
            resp = f"<answer> {ca} </answer>"
        else:
            # pick a deterministic wrong answer
            resp = "<answer> Z </answer>" if ca.strip().upper() != "Z" else "<answer> Y </answer>"
        return {"response": resp, "ground_truth": gt}

    reward_inputs = [mk_item(r, correct=True) for r in base_rows] + [mk_item(r, correct=True) for r in b_rows]
    scores = omnimed_isic_reward.compute_score(reward_inputs)
    if not all(s.get("acc", 0.0) == 1.0 for s in scores):
        bad = [(i, scores[i], reward_inputs[i]["ground_truth"].get("correct_answer")) for i in range(len(scores)) if scores[i].get("acc", 0.0) != 1.0]
        raise SystemExit(f"Reward correctness failed for some samples: {bad[:3]}")

    reward_inputs_wrong = [mk_item(r, correct=False) for r in base_rows[:4]] + [mk_item(r, correct=False) for r in b_rows[:4]]
    scores_wrong = omnimed_isic_reward.compute_score(reward_inputs_wrong)
    if not all(s.get("acc", 1.0) == 0.0 for s in scores_wrong):
        raise SystemExit("Reward wrong-answer test failed (expected acc=0.0).")

    print(f"reward parsing/correctness: OK (tested {len(scores)} correct + {len(scores_wrong)} wrong)")
    print("ALL OK")


if __name__ == "__main__":
    main()
