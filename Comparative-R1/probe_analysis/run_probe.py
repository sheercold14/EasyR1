#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe_analysis.datasets import Sample, load_jsonl
from probe_analysis.extractors import build_extractor
from probe_analysis.probes import run_knn_probe, run_linear_probe, run_mlp_probe
from probe_analysis.stats import bootstrap_ci, compute_metrics, json_ready


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe analysis for visual feature separability.")
    p.add_argument("--data", type=str, required=True, help="JSONL dataset path")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--image-root", type=str, default="")
    p.add_argument("--image-key", type=str, default="images.0")
    p.add_argument("--label-key", type=str, default="answer.correct_answer")
    p.add_argument("--sample-id-key", type=str, default="")

    p.add_argument("--extractor", type=str, default="mean_rgb", help="mean_rgb | npz | module:factory")
    p.add_argument("--features-npz", type=str, default="")
    p.add_argument("--features-key", type=str, default="features")
    p.add_argument("--ids-key", type=str, default="ids")
    p.add_argument("--plugin-kwargs", type=str, default="{}", help="JSON string for custom extractor kwargs")

    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--probes", type=str, default="linear,knn,mlp", help="comma list: linear,knn,mlp")
    p.add_argument("--knn-k", type=int, default=7)
    p.add_argument("--mlp-hidden", type=int, default=256)

    p.add_argument("--bootstrap", type=int, default=1000, help="bootstrap rounds for 95% CI")
    p.add_argument("--save-features", action="store_true")
    p.add_argument(
        "--group-by-candidate-labels",
        action="store_true",
        help="Run probes separately for each candidate label schema (from --candidate-labels-key).",
    )
    p.add_argument("--candidate-labels-key", type=str, default="answer.candidate_labels")
    p.add_argument("--min-group-size", type=int, default=20)
    return p.parse_args()


def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(labels)
    cls_to_idx = {c: i for i, c in enumerate(classes.tolist())}
    y = np.asarray([cls_to_idx[x] for x in labels.tolist()], dtype=np.int64)
    return y, classes


def split_samples(samples: list[Sample], y: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    _ = samples
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for i, c in enumerate(y.tolist()):
        by_class.setdefault(int(c), []).append(i)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for _, idxs in by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n_test = max(1, int(round(len(idxs) * test_size)))
        if len(idxs) >= 2:
            n_test = min(max(1, n_test), len(idxs) - 1)
        test_idx.extend(idxs[:n_test])
        train_idx.extend(idxs[n_test:])

    return np.asarray(sorted(train_idx), dtype=np.int64), np.asarray(sorted(test_idx), dtype=np.int64)


def candidate_schema(row: dict, candidate_labels_key: str) -> str:
    from probe_analysis.utils import deep_get

    vals = deep_get(row, candidate_labels_key, None)
    if not isinstance(vals, list):
        return "__missing__"
    labels = [str(x).strip() for x in vals if str(x).strip()]
    if not labels:
        return "__empty__"
    return " | ".join(labels)


def run_single_eval(
    *,
    x_all: np.ndarray,
    y_all: np.ndarray,
    classes: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    train_idx, test_idx = split_samples([], y_all, args.test_size, args.seed)
    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_test, y_test = x_all[test_idx], y_all[test_idx]

    probe_names = {x.strip() for x in args.probes.split(",") if x.strip()}
    outputs = []
    if "linear" in probe_names:
        outputs.append(run_linear_probe(x_train, y_train, x_test, seed=args.seed))
    if "knn" in probe_names:
        outputs.append(run_knn_probe(x_train, y_train, x_test, k=args.knn_k))
    if "mlp" in probe_names:
        outputs.append(run_mlp_probe(x_train, y_train, x_test, seed=args.seed, hidden_dim=args.mlp_hidden))
    if not outputs:
        raise ValueError("No valid probes selected")

    results = []
    for out in outputs:
        m = compute_metrics(y_test, out.y_pred)
        ci = bootstrap_ci(y_test, out.y_pred, n_bootstrap=args.bootstrap, seed=args.seed)
        results.append(
            {
                "probe": out.name,
                "metrics": m,
                "bootstrap_ci": ci,
            }
        )

    return {
        "n_classes": int(len(classes)),
        "classes": classes.tolist(),
        "split": {
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "test_ratio": args.test_size,
            "seed": args.seed,
        },
        "results": results,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(
        args.data,
        image_key=args.image_key,
        label_key=args.label_key,
        sample_id_key=args.sample_id_key,
    )
    if not samples:
        raise SystemExit("No valid samples loaded from JSONL")

    labels = np.asarray([s.label for s in samples], dtype=object)
    y_all, classes = encode_labels(labels)

    plugin_kwargs = json.loads(args.plugin_kwargs)
    extractor = build_extractor(
        args.extractor,
        image_root=args.image_root or None,
        features_npz=args.features_npz or None,
        features_key=args.features_key,
        ids_key=args.ids_key,
        plugin_kwargs=plugin_kwargs,
    )

    x_all = np.asarray(extractor.extract(samples), dtype=np.float32)
    if x_all.ndim != 2:
        raise ValueError(f"Expected 2D features [N, D], got {x_all.shape}")
    if x_all.shape[0] != len(samples):
        raise ValueError(f"Feature/sample size mismatch: features={x_all.shape[0]} samples={len(samples)}")

    summary = {
        "data": args.data,
        "n_samples": len(samples),
        "feature_shape": list(x_all.shape),
        "extractor": args.extractor,
        "group_by_candidate_labels": bool(args.group_by_candidate_labels),
    }

    if not args.group_by_candidate_labels:
        summary.update(run_single_eval(x_all=x_all, y_all=y_all, classes=classes, args=args))
    else:
        schemas = [candidate_schema(s.raw, args.candidate_labels_key) for s in samples]
        group_to_idx: dict[str, list[int]] = {}
        for i, g in enumerate(schemas):
            group_to_idx.setdefault(g, []).append(i)

        group_summaries = []
        skipped = []
        for g, idxs in sorted(group_to_idx.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            if len(idxs) < args.min_group_size:
                skipped.append({"group": g, "size": len(idxs), "reason": "below_min_group_size"})
                continue
            sub_labels = labels[idxs]
            sub_y, sub_classes = encode_labels(sub_labels)
            if len(sub_classes) < 2:
                skipped.append({"group": g, "size": len(idxs), "reason": "single_class"})
                continue
            sub_x = x_all[idxs]
            res = run_single_eval(x_all=sub_x, y_all=sub_y, classes=sub_classes, args=args)
            res["group"] = g
            res["n_samples"] = len(idxs)
            group_summaries.append(res)

        summary["groups"] = group_summaries
        summary["group_skipped"] = skipped

    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    if args.save_features:
        np.savez_compressed(
            out_dir / "features_dump.npz",
            features=x_all,
            labels=labels,
            ids=np.asarray([s.sample_id for s in samples], dtype=str),
            train_idx=train_idx,
            test_idx=test_idx,
        )

    print(json.dumps({"output_dir": str(out_dir), "summary": str(out_dir / 'summary.json')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
