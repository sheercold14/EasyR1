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

    train_idx, test_idx = split_samples(samples, y_all, args.test_size, args.seed)
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

    summary = {
        "data": args.data,
        "n_samples": len(samples),
        "n_classes": int(len(classes)),
        "classes": classes.tolist(),
        "feature_shape": list(x_all.shape),
        "extractor": args.extractor,
        "split": {
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "test_ratio": args.test_size,
            "seed": args.seed,
        },
        "results": results,
    }

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
