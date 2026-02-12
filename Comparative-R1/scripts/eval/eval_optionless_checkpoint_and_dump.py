#!/usr/bin/env python3
"""
Optionless (text-label) evaluation runner.

What it does:
1) Reads an input JSONL (single-image MCQ rows).
2) For each "question_type" group, collects the set of labels appearing in the eval file.
3) Rewrites each prompt into an optionless prompt that shows *all labels for that group* and asks the
   model to answer with the label text.
4) Runs veRL val_only and uses an optionless reward function to score open-text outputs.
5) Dumps generations and per-sample records like `eval_checkpoint_and_dump.py`.

Usage:
# Pretrain model
# B tasks
python3 ./Comparative-R1/scripts/eval/eval_optionless_checkpoint_and_dump.py \
        --val /mnt/cache/wuruixiao/users/lsc/data/datasets_b2n/DescribableTextures_b2n_base_test.jsonl \
    --out ./checkpoints/Eval-CLS/dtd_abase_pretrain_eval \
    --config Comparative-R1/configs/dtd_config.yaml \
    --mode passthrough \
    --prompt-key prompt \
    --answer-key answer \
    --image-key images \
    --image-dir /mnt/cache/wuruixiao/users/lsc/data/dtd/images/ \
    --reward-function Comparative-R1/reward/dtd_direct_mixed_reward.py:compute_score \
    --format-prompt null \
    --override worker.actor.model.model_path=/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b
# single-image eval
python3 ./Comparative-R1/scripts/eval/eval_optionless_checkpoint_and_dump.py \
        --val /mnt/cache/wuruixiao/users/lsc/data/datasets_b2n/DescribableTextures_b2n_base_test.jsonl \
    --out ./checkpoints/Eval-CLS/dtd_abase_pretrain_eval \
    --config Comparative-R1/configs/dtd_config.yaml \
    --mode passthrough \
    --prompt-key problem \
    --answer-key label \
    --image-key image \
    --image-dir /mnt/cache/wuruixiao/users/lsc/data/dtd/images/ \
    --reward-function Comparative-R1/reward/dtd_direct_mixed_reward.py:compute_score \
    --format-prompt /mnt/cache/wuruixiao/users/lsc/EasyR1/Comparative-R1/prompts/dtd_nothinking.jinja \
    --override worker.actor.model.model_path=/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b
# RFT models
python3 ./Comparative-R1/scripts/eval/eval_optionless_checkpoint_and_dump.py \
    --val ./data/CLS/DTD/B-tasks/DescribableTextures_b2n_B700.jsonl \
    --out ./checkpoints/Eval-CLS/dtd_b700_thinking_fewshot4_eval \
    --override [worker.actor.model.model_path=/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b,trainer.reward_function=Comparative-R1/reward/dtd_direct_mixed_reward.py:compute_score]

python3 EasyR1/Comparative-R1/scripts/eval/eval_optionless_checkpoint_and_dump.py \
    --config EasyR1/Comparative-R1/configs/omnimed_isic_gspo_taskaware.yaml \
    --val /mnt/cache/wuruixiao/users/lsc/EasyR1/data/OminiMedExpert/isic_disease_diagnosis_v1_0.05/test_optionless.jsonl \
    --out EasyR1/checkpoints/eval_runs/isic_pretrain_optionless_test_v2 \
    --override worker.actor.model.model_path=/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b 

python3 ./Comparative-R1/scripts/eval/eval_optionless_checkpoint_and_dump.py \
    --config Comparative-R1/configs/dtd_config.yaml \
    --checkpoint /mnt/cache/wuruixiao/users/lsc/EasyR1/checkpoints/CLS-RL/comparative_r1/qwen2_5_7b_dtd_b2n_gspo_thinking/global_step_155 \
    --mode passthrough \
    --prompt-key prompt \
    --answer-key answer \
    --image-key images \
    --image-dir /mnt/cache/wuruixiao/users/lsc/data/dtd/images/ \
    --reward-function Comparative-R1/reward/dtd_direct_mixed_reward.py:compute_score \
    --format-prompt null \
    --override worker.actor.model.model_path=/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b

Notes:
- This is meant for *single-image* disease diagnosis style tasks (task_type=mcq_letter).
- Multi-image B tasks are passed through unchanged.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


_PROMPT_MARK = "[prompt] "
_OUTPUT_MARK = "[output] "
_GT_MARK = "[ground_truth] "
_SCORE_MARK = "[score] "

_QUESTION_RE = re.compile(r"^\s*Question\s*:\s*(?P<q>.+?)\s*$", flags=re.IGNORECASE | re.MULTILINE)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON in {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise SystemExit(f"Expected JSON object in {path}:{line_no}, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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

        next_p = txt.find(_PROMPT_MARK, s0 + len(_SCORE_MARK))
        score_blob = txt[s0 + len(_SCORE_MARK) : (next_p if next_p >= 0 else n)].strip()
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
    reward_function: str,
    prompt_key: str,
    answer_key: str,
    image_key: str,
    image_dir: Optional[str],
    format_prompt: Optional[str],
    extra_overrides: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"config={config}",
        "trainer.logger=[file]",
        "trainer.val_only=true",
        "trainer.val_before_train=true",
        "trainer.val_generations_to_log=1000000000",
        # val_only still builds train dataset in current pipeline,
        # so keep train/val schema aligned.
        f"data.train_files={val_jsonl}",
        f"data.val_files={val_jsonl}",
        f"data.prompt_key={prompt_key}",
        f"data.answer_key={answer_key}",
        f"data.image_key={image_key}",
        "data.shuffle=false",
        f"trainer.save_checkpoint_path={out_dir}",
        f"trainer.experiment_name={experiment_name}",
        "trainer.find_last_checkpoint=false",
        f"worker.reward.reward_function={reward_function}",
        f"data.image_dir={image_dir}"
    ]
    if format_prompt is None:
        overrides.append("data.format_prompt=null")
    else:
        overrides.append(f"data.format_prompt={format_prompt}")
    if checkpoint is not None:
        overrides.append(f"trainer.load_checkpoint_path={checkpoint}")
    overrides.extend(extra_overrides)

    cmd = ["python3", "-m", "verl.trainer.main", *overrides]
    subprocess.run(cmd, check=True, cwd=str(config.parent.parent.parent))  # <repo_root>/EasyR1/...


def _extract_question(prompt: str) -> str:
    m = _QUESTION_RE.search(prompt)
    if not m:
        return ""
    return m.group("q").strip()


def build_optionless_prompt(*, question: str, candidate_labels: list[str]) -> str:
    labels = [l.strip() for l in candidate_labels if isinstance(l, str) and l.strip()]
    # Keep order stable for reproducibility.
    labels = sorted(dict.fromkeys(labels).keys())
    lines = [
        "You are a medical VQA assistant for dermoscopy images.",
        "",
        "Task: Predict the disease diagnosis label for the image.",
        "",
    ]
    if question:
        lines.extend([f"Question: {question}", ""])
    lines.append("Candidate labels (choose exactly one; copy the label text):")
    for l in labels:
        lines.append(f"- {l}")
    lines.extend(
        [
            "",
            "Answer with ONLY the label text (no extra words).",
            "<answer></answer>",
        ]
    )
    return "\n".join(lines)


def _norm_label_for_group(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return ""
    s2 = re.sub(r"\s+", " ", s)
    s2 = re.sub(r"\.\s*$", "", s2).strip()
    # common noise: "Benign image."
    s2 = re.sub(r"\bimage\b\.?$", "", s2, flags=re.IGNORECASE).strip()
    return s2


def _extract_options(ans: dict[str, Any]) -> list[str]:
    opts = []
    for k in ("option_A", "option_B", "option_C", "option_D"):
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

    norm = {_norm_label_for_group(o).lower() for o in options if _norm_label_for_group(o)}
    # accept small variants
    norm = {re.sub(r"[^a-z]+", " ", x).strip() for x in norm}
    return norm == {"benign", "malignant"}


def _group_key_for_row(ans: dict[str, Any]) -> str:
    qtype = str(ans.get("question_type", "unknown")).strip() or "unknown"
    opts = _extract_options(ans)
    if _is_binary_benign_malignant_options(opts):
        return f"{qtype}|binary_benign_malignant"
    return f"{qtype}|option_count={len(opts)}"


def transform_val_jsonl_optionless(in_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Group by a "problem type" key to avoid mixing benign/malignant binary questions with disease-name questions
    # under the same question_type (e.g., both are "Disease Diagnosis").
    labels_by_group: dict[str, set[str]] = {}
    for r in in_rows:
        ans = r.get("answer") if isinstance(r.get("answer"), dict) else {}
        task_type = str(ans.get("task_type", "")).strip()
        if task_type != "mcq_letter":
            continue
        gk = _group_key_for_row(ans)
        if gk.endswith("|binary_benign_malignant"):
            labels_by_group.setdefault(gk, set()).update({"Benign", "Malignant"})
        else:
            lbl = ans.get("label")
            if isinstance(lbl, str) and lbl.strip():
                labels_by_group.setdefault(gk, set()).add(lbl.strip())

    out_rows: list[dict[str, Any]] = []
    rewritten = 0
    skipped = 0

    for r in in_rows:
        prompt = r.get("prompt")
        if not isinstance(prompt, str):
            skipped += 1
            out_rows.append(r)
            continue

        ans = r.get("answer") if isinstance(r.get("answer"), dict) else {}
        task_type = str(ans.get("task_type", "")).strip()
        if task_type != "mcq_letter":
            # pass-through non-mcq rows
            out_rows.append(r)
            continue

        gk = _group_key_for_row(ans)
        candidate = sorted(labels_by_group.get(gk, set()))
        # For binary benign/malignant group, enforce clean candidates (no "image" suffix).
        if gk.endswith("|binary_benign_malignant"):
            candidate = ["Benign", "Malignant"]
        question = _extract_question(prompt)
        new_prompt = build_optionless_prompt(question=question, candidate_labels=candidate)

        new_row = dict(r)
        new_row["prompt"] = new_prompt

        new_ans = dict(ans)
        # Keep label as-is, but set correct_answer to text label for easier downstream tooling.
        if isinstance(new_ans.get("label"), str) and new_ans["label"].strip():
            correct_label = new_ans["label"].strip()
            if gk.endswith("|binary_benign_malignant"):
                # normalize benign/malignant labels like "Benign image."
                correct_label = _norm_label_for_group(correct_label)
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
        "skipped_invalid_prompt": skipped,
        "num_groups": len(labels_by_group),
        "labels_by_group": {k: sorted(v) for k, v in sorted(labels_by_group.items())},
    }
    return out_rows, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--analyze-only", action="store_true", help="Skip running veRL; only parse existing generations.log.")
    ap.add_argument("--keep-transformed", action="store_true", help="Keep the transformed val jsonl under <out>/_transformed/.")
    ap.add_argument(
        "--mode",
        choices=["optionless", "passthrough"],
        default="optionless",
        help="optionless: rewrite mcq_letter rows; passthrough: keep val file unchanged",
    )
    ap.add_argument("--prompt-key", type=str, default="prompt", help="data.prompt_key override for veRL")
    ap.add_argument("--answer-key", type=str, default="answer", help="data.answer_key override for veRL")
    ap.add_argument("--image-key", type=str, default="images", help="data.image_key override for veRL")
    ap.add_argument("--image-dir", type=str, default="")
    
    ap.add_argument(
        "--reward-function",
        type=str,
        default="Comparative-R1/reward/omnimed_isic_optionless_reward_v1.py:compute_score",
        help="worker.reward.reward_function override",
    )
    ap.add_argument(
        "--format-prompt",
        type=str,
        default=None,
        help="Optional data.format_prompt path; default null (disabled)",
    )
    ap.add_argument("--override", action="append", default=[])
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

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "optionless":
        in_rows = _read_jsonl(val_jsonl)
        out_rows, info = transform_val_jsonl_optionless(in_rows)
        # Use a dedicated subdir to avoid clobbering user datasets when `--out` points to a data directory.
        transformed_dir = out_dir / "_transformed"
        transformed_dir.mkdir(parents=True, exist_ok=True)
        transformed_val = transformed_dir / f"{val_jsonl.stem}.optionless.jsonl"
        _write_jsonl(transformed_val, out_rows)
        (out_dir / f"{val_jsonl.stem}.optionless.summary.json").write_text(
            json.dumps({"input": str(val_jsonl), "output": str(transformed_val), "info": info}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        transformed_val = val_jsonl
        info = {"mode": "passthrough", "rewritten_mcq": 0}

    ckpt_name = checkpoint.name if checkpoint is not None else "pretrained"
    exp_name = args.name or f"eval_optionless_{ckpt_name}_{val_jsonl.stem}"

    if not args.analyze_only:
        run_val_only(
            config=config,
            checkpoint=checkpoint,
            val_jsonl=transformed_val,
            out_dir=out_dir,
            experiment_name=exp_name,
            reward_function=args.reward_function,
            prompt_key=args.prompt_key,
            answer_key=args.answer_key,
            image_key=args.image_key,
            image_dir=args.image_dir,
            format_prompt=args.format_prompt,
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
                    {"prompt": s.prompt, "output": s.output, "ground_truth": s.ground_truth, "score": s.score},
                    ensure_ascii=False,
                )
                + "\n"
            )

    stats_path = out_dir / "predictions.summary.json"
    stats = {
        "config": str(config),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "val": str(val_jsonl),
        "val_optionless": str(transformed_val),
        "out": str(out_dir),
        "num_samples": len(samples),
        "experiment_name": exp_name,
        "mode": args.mode,
        "prompt_key": args.prompt_key,
        "answer_key": args.answer_key,
        "image_key": args.image_key,
        "reward_function": args.reward_function,
        "format_prompt": args.format_prompt,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if (not args.keep_transformed) and args.mode == "optionless":
        # Keep the summary, but remove the big transformed file by default.
        # IMPORTANT: never delete the user-provided `--val` file.
        try:
            if transformed_val.resolve() == val_jsonl.resolve():
                raise RuntimeError(f"Refusing to delete --val file: {transformed_val}")
            transformed_val.unlink()
        except Exception:
            pass
        try:
            transformed_dir.rmdir()  # type: ignore[name-defined]
        except Exception:
            pass


if __name__ == "__main__":
    main()
