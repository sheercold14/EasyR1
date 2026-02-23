#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
"""
  python3 -m probe_analysis.run_all_feature_keys_prepost \
    --data /mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.full_labels_text.jsonl \
    --output-dir /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/4label_list_all_taps \
    --pre-npz /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/isic_4labellist_pre_multitap.npz \
    --post-npz /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/isic_4labellist_post_multitap.npz \
    --probe-label-key answer.label \
    --probe-sample-id-key images.0 \
    --ids-key image_paths \
    --group-by-candidate-labels \
    --candidate-labels-key answer.candidate_labels \
    --min-group-size 20 \
    --verbose


"""

def _safe_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)


def _load_feature_keys(npz_path: Path) -> list[str]:
    z = np.load(npz_path, allow_pickle=True)
    keys = [k for k in z.files if k.startswith("features_")]
    keys.sort()
    return keys


def _run(cmd: list[str], *, cwd: Path | None = None) -> dict:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n\n{proc.stdout}")
    out = proc.stdout.strip().splitlines()
    # run_probe/extract_features prints a JSON object on the last line
    try:
        return json.loads(out[-1]) if out else {}
    except Exception:
        raise RuntimeError(f"Expected JSON on last line, got:\n{proc.stdout}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run probes for all feature keys in pre/post npz.")
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--pre-npz", required=True)
    p.add_argument("--post-npz", required=True)

    # Optional extraction (if you want to regenerate npz from checkpoints)
    p.add_argument("--pre-checkpoint", default="")
    p.add_argument("--post-checkpoint", default="")
    p.add_argument("--image-root", default="")
    p.add_argument("--image-key", default="images.0")
    p.add_argument("--label-key", default="answer.correct_answer")
    p.add_argument("--sample-id-key", default="")
    p.add_argument("--prompt-key", default="")
    p.add_argument("--prompt-fallback", default="")
    p.add_argument("--taps", action="append", default=[])
    p.add_argument("--extract", action="store_true", help="If set, (re)extract pre/post npz from checkpoints.")
    p.add_argument("--extract-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--extract-device", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--extract-verbose", action="store_true")

    # Probe args
    p.add_argument("--probe-label-key", default="answer.label")
    p.add_argument("--probe-sample-id-key", default="")
    p.add_argument("--ids-key", default="ids")
    p.add_argument("--probes", default="linear,knn,mlp")
    p.add_argument("--knn-k", type=int, default=7)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=1000)

    p.add_argument("--group-by-candidate-labels", action="store_true")
    p.add_argument("--candidate-labels-key", default="answer.candidate_labels")
    p.add_argument("--min-group-size", type=int, default=20)

    p.add_argument("--key-mode", choices=["intersection", "union"], default="intersection")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent  # .../Comparative-R1
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pre_npz = Path(args.pre_npz)
    post_npz = Path(args.post_npz)

    py = sys.executable

    if args.extract:
        if not args.pre_checkpoint or not args.post_checkpoint:
            raise SystemExit("--extract requires --pre-checkpoint and --post-checkpoint")
        if not args.image_root:
            raise SystemExit("--extract requires --image-root")

        taps = args.taps
        if not taps:
            # Reasonable default multi-tap set.
            taps = ["vision_mean", "hs:0:image", "hs:16:image", "hs:-1:image", "last:-1"]

        def extract_one(ckpt: str, out_npz: Path, tag: str) -> None:
            cmd = [
                py,
                "-m",
                "probe_analysis.extract_features_from_checkpoint",
                "--data",
                args.data,
                "--output-npz",
                str(out_npz),
                "--image-root",
                args.image_root,
                "--image-key",
                args.image_key,
                "--label-key",
                args.label_key,
                "--sample-id-key",
                args.sample_id_key,
                "--checkpoint",
                ckpt,
                "--dtype",
                args.extract_dtype,
                "--device",
                args.extract_device,
                "--progress-every",
                str(args.progress_every),
            ]
            if args.trust_remote_code:
                cmd.append("--trust-remote-code")
            if args.extract_verbose:
                cmd.append("--verbose")
            if args.prompt_key:
                cmd += ["--prompt-key", args.prompt_key]
            if args.prompt_fallback:
                cmd += ["--prompt-fallback", args.prompt_fallback]
            for t in taps:
                cmd += ["--taps", t]

            if args.verbose:
                print(f"[extract:{tag}] {' '.join(cmd)}")
            _run(cmd, cwd=repo_root)

        extract_one(args.pre_checkpoint, pre_npz, "pre")
        extract_one(args.post_checkpoint, post_npz, "post")

    if not pre_npz.exists():
        raise SystemExit(f"Missing --pre-npz: {pre_npz}")
    if not post_npz.exists():
        raise SystemExit(f"Missing --post-npz: {post_npz}")

    pre_keys = _load_feature_keys(pre_npz)
    post_keys = _load_feature_keys(post_npz)

    if args.key_mode == "intersection":
        keys = sorted(set(pre_keys) & set(post_keys))
    else:
        keys = sorted(set(pre_keys) | set(post_keys))

    if not keys:
        raise SystemExit("No features_* keys found in npz")

    out_pre = out_root / "pre"
    out_post = out_root / "post"
    out_cmp = out_root / "compare"
    out_pre.mkdir(parents=True, exist_ok=True)
    out_post.mkdir(parents=True, exist_ok=True)
    out_cmp.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"[keys] pre={len(pre_keys)} post={len(post_keys)} run={len(keys)} mode={args.key_mode}")

    results = []
    for fk in keys:
        safe_fk = _safe_name(fk)
        cmp_path = out_cmp / f"compare.{safe_fk}.json"
        if args.skip_existing and cmp_path.exists():
            if args.verbose:
                print(f"[skip] {cmp_path}")
            continue

        def probe_one(tag: str, npz_path: Path, out_dir: Path) -> Path:
            cmd = [
                py,
                "-m",
                "probe_analysis.run_probe",
                "--data",
                args.data,
                "--output-dir",
                str(out_dir),
                "--image-key",
                args.image_key,
                "--label-key",
                args.probe_label_key,
                "--sample-id-key",
                args.probe_sample_id_key,
                "--extractor",
                "npz",
                "--features-npz",
                str(npz_path),
                "--features-key",
                fk,
                "--ids-key",
                args.ids_key,
                "--probes",
                args.probes,
                "--knn-k",
                str(args.knn_k),
                "--mlp-hidden",
                str(args.mlp_hidden),
                "--test-size",
                str(args.test_size),
                "--seed",
                str(args.seed),
                "--bootstrap",
                str(args.bootstrap),
                "--summary-filename",
                f"summary.{tag}.json",
                "--auto-summary-suffix",
            ]
            if args.group_by_candidate_labels:
                cmd += [
                    "--group-by-candidate-labels",
                    "--candidate-labels-key",
                    args.candidate_labels_key,
                    "--min-group-size",
                    str(args.min_group_size),
                ]
            if args.verbose:
                cmd.append("--verbose")

            meta = _run(cmd, cwd=repo_root)
            return Path(meta["summary"])

        t0 = time.perf_counter()
        pre_summary = probe_one("pre", pre_npz, out_pre)
        post_summary = probe_one("post", post_npz, out_post)

        cmp_cmd = [
            py,
            "-m",
            "probe_analysis.compare_runs",
            "--pre",
            str(pre_summary),
            "--post",
            str(post_summary),
        ]
        cmp = subprocess.run(cmp_cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if cmp.returncode != 0:
            raise RuntimeError(f"compare failed: {' '.join(cmp_cmd)}\n\n{cmp.stdout}")
        cmp_path.write_text(cmp.stdout, encoding="utf-8")

        dt = time.perf_counter() - t0
        results.append(
            {
                "features_key": fk,
                "pre_summary": str(pre_summary),
                "post_summary": str(post_summary),
                "compare": str(cmp_path),
                "seconds": dt,
            }
        )
        if args.verbose:
            print(f"[done] {fk} ({dt:.2f}s)")

    (out_root / "index.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_root), "num_runs": len(results), "index": str(out_root / 'index.json')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
