#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from probe_analysis.datasets import load_jsonl
from probe_analysis.hf_extractors import build_qwen2vl_visual_mean_extractor


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

    extractor = build_qwen2vl_visual_mean_extractor(
        image_root=args.image_root or None,
        checkpoint=args.checkpoint,
        processor_path=args.processor_path or None,
        dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        prompt_text=args.prompt_text,
    )
    feats = extractor.extract(samples)

    ids = np.asarray([s.sample_id for s in samples], dtype=str)
    labels = np.asarray([s.label for s in samples], dtype=str)
    image_paths = np.asarray([s.image_path for s in samples], dtype=str)

    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, features=feats, ids=ids, labels=labels, image_paths=image_paths)

    print(
        json.dumps(
            {
                "output_npz": str(out),
                "num_samples": int(len(samples)),
                "feature_dim": int(feats.shape[1]),
                "checkpoint": args.checkpoint,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
