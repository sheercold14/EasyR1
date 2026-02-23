#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe_analysis.datasets import load_jsonl
from probe_analysis.hf_extractors import Qwen2VLMultiTapExtractor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract visual features from HF checkpoint into npz")
    p.add_argument("--data", required=True, help="EasyR1 JSONL")
    p.add_argument("--output-npz", required=True)
    p.add_argument("--image-root", default="")
    p.add_argument("--image-key", default="images.0")
    p.add_argument("--label-key", default="answer.correct_answer")
    p.add_argument("--sample-id-key", default="")

    p.add_argument("--checkpoint", required=True, help="HF checkpoint path (e.g., .../actor/huggingface)")
    p.add_argument("--processor-path", default="", help="Optional processor path; default uses checkpoint")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default="auto", help="transformers device_map, e.g. auto/cuda:0/cpu")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--prompt-text", default="Describe the image briefly.")
    p.add_argument(
        "--prompt-key",
        default="",
        help="Optional JSONL key for per-sample prompt text, e.g. 'prompt'. If empty, uses --prompt-text.",
    )
    p.add_argument(
        "--prompt-fallback",
        default="",
        help="Fallback prompt when --prompt-key is set but missing/empty for a sample.",
    )
    p.add_argument(
        "--tap",
        default="vision_mean",
        choices=["vision_mean", "hidden_mean"],
        help="(legacy) Single tap mode. Prefer --taps for multi-tap extraction.",
    )
    p.add_argument("--layer", type=int, default=0, help="(legacy) hidden_mean only: layer index (0=embed, -1=last).")
    p.add_argument("--token-scope", default="image", choices=["image", "text", "all"], help="(legacy) hidden_mean only.")
    p.add_argument(
        "--taps",
        action="append",
        default=[],
        help="Multi-tap extraction. Repeatable. Examples: vision_mean ; hs:0:image ; hs:16:image ; hs:-1:text",
    )
    p.add_argument("--progress-every", type=int, default=50, help="Print progress every N samples (0 to disable).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(
        args.data,
        image_key=args.image_key,
        label_key=args.label_key,
        sample_id_key=args.sample_id_key,
    )
    if not samples:
        raise SystemExit("No valid samples loaded")

    extractor = Qwen2VLMultiTapExtractor(
        checkpoint=args.checkpoint,
        processor_path=args.processor_path or None,
        image_root=args.image_root or None,
        dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        prompt_text=args.prompt_text,
    )

    if args.taps:
        tap_specs = args.taps
    else:
        # legacy single-tap compatibility
        if args.tap == "vision_mean":
            tap_specs = ["vision_mean"]
        else:
            tap_specs = [f"hs:{args.layer}:{args.token_scope}"]

    feats_by_tap = extractor.extract_taps(
        samples,
        tap_specs=tap_specs,
        progress_every=args.progress_every,
        verbose=args.verbose,
        prompt_key=args.prompt_key,
        prompt_fallback=args.prompt_fallback,
    )

    ids = np.asarray([s.sample_id for s in samples], dtype=str)
    labels = np.asarray([s.label for s in samples], dtype=str)
    image_paths = np.asarray([s.image_path for s in samples], dtype=str)

    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Keep a default `features` for compatibility with older scripts (points to the first tap).
    first_key = next(iter(feats_by_tap.keys()))
    pack = {
        "features": feats_by_tap[first_key],
        "ids": ids,
        "labels": labels,
        "image_paths": image_paths,
        "features_key_default": np.asarray(first_key, dtype=str),
    }
    for k, v in feats_by_tap.items():
        pack[f"features_{k}"] = v
    np.savez_compressed(out, **pack)

    print(
        json.dumps(
            {
                "output_npz": str(out),
                "num_samples": int(len(samples)),
                "feature_dims": {k: int(v.shape[1]) for k, v in feats_by_tap.items()},
                "features_keys": sorted(list(feats_by_tap.keys())),
                "features_key_default": first_key,
                "checkpoint": args.checkpoint,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
