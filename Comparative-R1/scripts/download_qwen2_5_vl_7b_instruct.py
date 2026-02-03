#!/usr/bin/env python3
"""
Download Qwen/Qwen2.5-VL-7B-Instruct from Hugging Face to a local directory.

Examples:
  # Download to a local folder (recommended)
  python EasyR1/Comparative-R1/scripts/download_qwen2_5_vl_7b_instruct.py \
    --out_dir /mnt/cache/wuruixiao/users/lsc/models/Qwen2.5-VL-7B-Instruct

  # If you need a specific revision
  python EasyR1/Comparative-R1/scripts/download_qwen2_5_vl_7b_instruct.py \
    --revision main

Notes:
  - Requires: `pip install -U huggingface_hub`
  - If the repo is gated or you hit rate limits, set `HF_TOKEN` env var or run `huggingface-cli login`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen/Qwen2.5-VL-7B-Instruct via huggingface_hub")
    parser.add_argument("--repo_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--out_dir", type=Path, default=Path("models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--revision", type=str, default=None, help="Branch/tag/commit (default: repo default)")
    parser.add_argument(
        "--allow_patterns",
        type=str,
        default=None,
        help="Comma-separated glob patterns to include (default: all). Example: '*.json,*.safetensors,tokenizer.*'",
    )
    parser.add_argument(
        "--ignore_patterns",
        type=str,
        default=None,
        help="Comma-separated glob patterns to exclude. Example: '*.bin,*.pth'",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # noqa: BLE001
        raise SystemExit("Missing dependency: huggingface_hub. Install with `pip install -U huggingface_hub`.") from e

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = None
    if args.allow_patterns:
        allow_patterns = [p.strip() for p in args.allow_patterns.split(",") if p.strip()]

    ignore_patterns = None
    if args.ignore_patterns:
        ignore_patterns = [p.strip() for p in args.ignore_patterns.split(",") if p.strip()]

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        resume_download=True,
    )

    print("Downloaded to:")
    print(snapshot_path)


if __name__ == "__main__":
    main()

