#!/usr/bin/env python3
"""
Offline evaluate a checkpoint on a given JSONL (single-image or B-tasks) and dump per-sample results.

Design goals:
- Reuse EasyR1/veRL's own validation pipeline so FSDP checkpoints load correctly.
- Log *all* validation generations (not just a small subset).
- Produce a structured JSONL with prompt/output/ground_truth/score for downstream analysis.

This script runs:
  python -m verl.trainer.main ... trainer.val_only=true ...

and then parses the resulting generations log.

Example (mcq single-image file, keep all):
  python EasyR1/scripts/OminiExpert/eval_checkpoint_and_dump.py \
    --config EasyR1/Comparative-R1/configs/omnimed_isic_gspo_taskaware.yaml \
    --checkpoint EasyR1/checkpoints/comparative_r1/omnimed_isic_btasks_n4_t0.7_taskaware/global_step_355 \
    --val EasyR1/data/OminiMedExpert/isic_disease_diagnosis_v0_0.05/val.jsonl \
    --out EasyR1/checkpoints/eval_runs/isic_taskaware_step500_val_mcq \
    --override worker.actor.model.model_path=/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b

python3 EasyR1/scripts/OminiExpert/eval_checkpoint_and_dump.py \
  --config EasyR1/Comparative-R1/configs/omnimed_isic_gspo_taskaware.yaml \
  --val EasyR1/data/OminiMedExpert/isic_disease_diagnosis_v1_0.05/test.jsonl \
  --out EasyR1/checkpoints/eval_runs/isic_pretrain_test \
  --override worker.actor.model.model_path=/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b

  python EasyR1/scripts/OminiExpert/eval_checkpoint_and_dump.py \
    --config EasyR1/Comparative-R1/configs/omnimed_isic_gspo_taskaware.yaml \
    --checkpoint EasyR1/checkpoints/comparative_r1/omnimed_isic_v1_single_n4_t0.7/global_step_285 \
    --val EasyR1/data/OminiMedExpert/isic_disease_diagnosis_v1_0.05/comparative/test_b_tasks_7_100.jsonl \
    --out EasyR1/checkpoints/eval_runs/isic_single_v1_val_7_100_btasks \
    --override worker.actor.model.model_path=/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b


"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


_PROMPT_MARK = "[prompt] "
_OUTPUT_MARK = "[output] "
_GT_MARK = "[ground_truth] "
_SCORE_MARK = "[score] "


@dataclass
class GenSample:
    prompt: str
    output: str
    ground_truth_raw: str
    ground_truth: Any
    score: float


def _parse_ground_truth(raw: str) -> Any:
    s = raw.strip()
    if not s:
        return s
    # Try JSON first (rare), then Python-literal dict/list (what FileGenerationLogger writes).
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def parse_generations_log(path: Path) -> list[GenSample]:
    txt = path.read_text(encoding="utf-8", errors="replace")
    out: list[GenSample] = []
    i = 0
    n = len(txt)
    while True:
        p0 = txt.find(_PROMPT_MARK, i)
        if p0 < 0:
            break
        o0 = txt.find(_OUTPUT_MARK, p0)
        g0 = txt.find(_GT_MARK, o0)
        s0 = txt.find(_SCORE_MARK, g0)
        if o0 < 0 or g0 < 0 or s0 < 0:
            break

        prompt = txt[p0 + len(_PROMPT_MARK) : o0]
        output = txt[o0 + len(_OUTPUT_MARK) : g0]
        gt_raw = txt[g0 + len(_GT_MARK) : s0]

        # score ends at next sample start or EOF; prefer double-newline but don't require it.
        next_p = txt.find(_PROMPT_MARK, s0 + len(_SCORE_MARK))
        score_blob = txt[s0 + len(_SCORE_MARK) : (next_p if next_p >= 0 else n)].strip()
        # score_blob may include extra newlines; take first token parseable as float.
        score_token = score_blob.split()[0] if score_blob.split() else "nan"
        try:
            score = float(score_token)
        except Exception:
            score = float("nan")

        out.append(
            GenSample(
                prompt=prompt.rstrip("\n"),
                output=output.rstrip("\n"),
                ground_truth_raw=gt_raw.rstrip("\n"),
                ground_truth=_parse_ground_truth(gt_raw),
                score=score,
            )
        )
        i = s0 + len(_SCORE_MARK)
    return out


def run_val_only(
    *,
    config: Path,
    checkpoint: Optional[Path],
    val_jsonl: Path,
    out_dir: Path,
    experiment_name: str,
    extra_overrides: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Always log to file so generations.log exists.
    overrides = [
        f"config={config}",
        "trainer.logger=[file]",
        "trainer.val_only=true",
        "trainer.val_before_train=true",
        # log all generations (trainer will cap at <= total samples)
        "trainer.val_generations_to_log=1000000000",
        # ensure we evaluate on the provided val file
        f"data.val_files={val_jsonl}",
        # write outputs into a separate folder
        f"trainer.save_checkpoint_path={out_dir}",
        f"trainer.experiment_name={experiment_name}",
        # always avoid auto-finding a checkpoint in out_dir
        "trainer.find_last_checkpoint=false",
    ]
    if checkpoint is not None:
        overrides.append(f"trainer.load_checkpoint_path={checkpoint}")
    overrides.extend(extra_overrides)

    cmd = ["python3", "-m", "verl.trainer.main", *overrides]
    subprocess.run(cmd, check=True, cwd=str(config.parent.parent.parent))  # <repo_root>/EasyR1/...


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="A training config YAML (used for model/rollout/reward).")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint dir ending with global_step_*. If omitted, evaluates the base model from config/overrides.",
    )
    ap.add_argument("--val", type=Path, required=True, help="Eval JSONL (single-image or B tasks).")
    ap.add_argument("--out", type=Path, required=True, help="Output dir for eval logs and dumps.")
    ap.add_argument("--name", type=str, default=None, help="Optional experiment name override.")
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra OmegaConf overrides, e.g. worker.rollout.val_override_config.n=1",
    )
    args = ap.parse_args()

    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    val_jsonl = args.val.resolve()
    out_dir = args.out.resolve()

    if not config.exists():
        raise SystemExit(f"Missing --config: {config}")
    if checkpoint is not None and not checkpoint.exists():
        raise SystemExit(f"Missing --checkpoint: {checkpoint}")
    if not val_jsonl.exists():
        raise SystemExit(f"Missing --val: {val_jsonl}")

    ckpt_name = checkpoint.name if checkpoint is not None else "pretrained"
    exp_name = args.name or f"eval_{ckpt_name}_{val_jsonl.stem}"

    run_val_only(
        config=config,
        checkpoint=checkpoint,
        val_jsonl=val_jsonl,
        out_dir=out_dir,
        experiment_name=exp_name,
        extra_overrides=args.override,
    )

    gen_log = out_dir / "generations.log"
    if not gen_log.exists():
        raise SystemExit(f"Expected generations.log at {gen_log} but it does not exist.")

    samples = parse_generations_log(gen_log)
    dump_path = out_dir / "predictions.jsonl"
    with dump_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(
                json.dumps(
                    {
                        "prompt": s.prompt,
                        "output": s.output,
                        "ground_truth": s.ground_truth,
                        "score": s.score,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Quick aggregate stats to help you sanity-check.
    stats_path = out_dir / "predictions.summary.json"
    stats = {
        "config": str(config),
        "checkpoint": str(checkpoint),
        "val": str(val_jsonl),
        "out": str(out_dir),
        "num_samples": len(samples),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
