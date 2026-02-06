#!/usr/bin/env python3
"""
Evaluate a checkpoint with a two-stage output constraint on B-tasks:
  (1) model predicts a label per image (intermediate)
  (2) model outputs the final task answer (letter(s) / same|different)

This script mirrors `eval_checkpoint_and_dump.py` but:
  - builds a transformed val JSONL where each prompt is prefixed with a two-stage instruction
  - runs veRL validation (val_only) to generate outputs
  - parses generations.log and computes:
      * per-task intermediate single-image accuracy
      * per-task final accuracy
      * conditional final accuracy vs intermediate correctness (e.g., all-images-correct)

Example:
  python3 EasyR1/scripts/OminiExpert/eval_two_stage_checkpoint_and_dump.py \
    --config EasyR1/Comparative-R1/configs/omnimed_isic_gspo_taskaware.yaml \
    --checkpoint EasyR1/checkpoints/comparative_r1/omnimed_isic_v1_single_n4_t0.7/global_step_285 \
    --val EasyR1/data/OminiMedExpert/isic_disease_diagnosis_v1_0.05/comparative/test_b_tasks_7_100.jsonl \
    --out EasyR1/checkpoints/eval_runs/isic_two_stage_val_7_100_btasks \
    --override worker.actor.model.model_path=/mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


_PROMPT_MARK = "[prompt] "
_OUTPUT_MARK = "[output] "
_GT_MARK = "[ground_truth] "
_SCORE_MARK = "[score] "

_LABELS_RE = re.compile(r"<labels_json>(?P<body>.*?)</labels_json>", flags=re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(?P<body>.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)
_EVALID_RE = re.compile(r"^EvalID\s*:\s*(?P<id>\d+)\s*$", flags=re.IGNORECASE | re.MULTILINE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_ORIG_ANSWER_TAG_RE = re.compile(r"<answer>.*?</answer>", flags=re.IGNORECASE | re.DOTALL)


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


@dataclass(frozen=True)
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
    extra_overrides: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = [
        f"config={config}",
        "trainer.logger=[file]",
        "trainer.val_only=true",
        "trainer.val_before_train=true",
        "trainer.val_generations_to_log=1000000000",
        f"data.val_files={val_jsonl}",
        f"trainer.save_checkpoint_path={out_dir}",
        f"trainer.experiment_name={experiment_name}",
        "trainer.find_last_checkpoint=false",
    ]
    if checkpoint is not None:
        overrides.append(f"trainer.load_checkpoint_path={checkpoint}")
    overrides.extend(extra_overrides)

    cmd = ["python3", "-m", "verl.trainer.main", *overrides]
    subprocess.run(cmd, check=True, cwd=str(config.parent.parent.parent))  # <repo_root>/EasyR1/...


def _normalize_label(s: Any) -> Optional[str]:
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    if not s:
        return None
    # make matching robust to punctuation differences
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s or None


def _extract_eval_id(prompt: str) -> Optional[int]:
    m = _EVALID_RE.search(prompt)
    if not m:
        return None
    try:
        return int(m.group("id"))
    except Exception:
        return None


def _extract_answer_text(output: str) -> str:
    m = _ANSWER_RE.search(output)
    if m:
        return m.group("body").strip()
    return output.strip()


def _extract_labels_mapping(output: str) -> dict[str, str]:
    m = _LABELS_RE.search(output)
    if not m:
        return {}
    body = m.group("body").strip()
    if not body:
        return {}

    # Try JSON, then Python literal, then a simple "A: xxx" fallback.
    obj: Any = None
    try:
        obj = json.loads(body)
    except Exception:
        try:
            obj = ast.literal_eval(body)
        except Exception:
            obj = None

    mapping: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k is None:
                continue
            key = str(k).strip().upper()
            if not key:
                continue
            mapping[key] = str(v).strip() if v is not None else ""
        return mapping

    if isinstance(obj, list):
        # Accept a list of {"key":"A","label":"..."} or ["label_for_A", ...] (positional)
        for it in obj:
            if isinstance(it, dict):
                k = it.get("key") or it.get("image") or it.get("id")
                v = it.get("label") or it.get("pred") or it.get("value")
                if k is None or v is None:
                    continue
                mapping[str(k).strip().upper()] = str(v).strip()
        if mapping:
            return mapping

    # Fallback parse: lines like "A: xxx"
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().upper().strip("()")
        if len(k) == 1 and (k.isalpha() or k == "Q"):
            mapping[k] = v.strip()
    return mapping


def _tokenize_letters(s: str) -> list[str]:
    toks = [t.upper() for t in _TOKEN_RE.findall(s)]
    letters = [t for t in toks if len(t) == 1 and ("A" <= t <= "Z" or t == "Q")]
    return letters


def _final_is_correct(*, pred_text: str, gt: Any) -> Optional[bool]:
    if gt is None:
        return None
    gt_s = str(gt).strip()
    if not gt_s:
        return None

    pred_s = str(pred_text).strip()
    if not pred_s:
        return False

    gt_norm = gt_s.strip().lower()
    pred_norm = pred_s.strip().lower()
    if gt_norm in {"same", "different"}:
        # pick first token-like
        ptoks = [t.lower() for t in _TOKEN_RE.findall(pred_norm)]
        if not ptoks:
            return False
        return ptoks[0] == gt_norm

    gt_letters = _tokenize_letters(gt_s)
    pred_letters = _tokenize_letters(pred_s)
    if not gt_letters:
        # non-letter answers: strict normalize
        return _normalize_label(pred_s) == _normalize_label(gt_s)

    # If gt expects 2 letters (B6), compare as unordered set. Else, compare first letter.
    if len(gt_letters) >= 2:
        return set(pred_letters) == set(gt_letters)
    return (pred_letters[0] if pred_letters else None) == gt_letters[0]


def _infer_image_keys_and_gt_labels(row: dict[str, Any]) -> tuple[list[str], dict[str, str], str]:
    ans = row.get("answer") if isinstance(row.get("answer"), dict) else {}
    task_type = str(ans.get("task_type", "unknown"))
    images = row.get("images") if isinstance(row.get("images"), list) else []
    n = len(images)

    def letters(k: int) -> list[str]:
        return [chr(ord("A") + i) for i in range(k)]

    gt: dict[str, str] = {}
    keys: list[str] = []

    if task_type == "B7_support_set_nway":
        support_labels = ans.get("support_labels") if isinstance(ans.get("support_labels"), list) else []
        target_label = str(ans.get("target_label", "")).strip()
        n_way = len(support_labels)
        keys = [*letters(n_way), "Q"]
        for i, k in enumerate(letters(n_way)):
            gt[k] = str(support_labels[i]).strip() if i < len(support_labels) else ""
        gt["Q"] = target_label
        return keys, gt, task_type

    keys = letters(n)

    if task_type == "B3_label_corruption":
        true_labels = ans.get("true_labels") if isinstance(ans.get("true_labels"), list) else []
        for i, k in enumerate(keys):
            gt[k] = str(true_labels[i]).strip() if i < len(true_labels) else ""
        return keys, gt, task_type

    labels = ans.get("labels") if isinstance(ans.get("labels"), list) else []
    for i, k in enumerate(keys):
        gt[k] = str(labels[i]).strip() if i < len(labels) else ""
    return keys, gt, task_type


def build_two_stage_prompt(*, original_prompt: str, eval_id: int, keys: list[str], prefix_template: str) -> str:
    keys_csv = ", ".join(keys)
    example = {k: "..." for k in keys}
    prefix = prefix_template.replace("{KEYS}", keys_csv).replace("{EXAMPLE_JSON}", json.dumps(example, ensure_ascii=False))
    # Remove original <answer>...</answer> placeholder(s) to avoid biasing the model to emit <answer> immediately.
    cleaned = _ORIG_ANSWER_TAG_RE.sub("", original_prompt).strip()
    return f"{prefix.strip()}\n\nEvalID: {eval_id}\n\n{cleaned}"


def compute_two_stage_metrics(
    *,
    val_rows_by_id: dict[int, dict[str, Any]],
    samples: list[GenSample],
    out_dir: Path,
) -> dict[str, Any]:
    per_task = defaultdict(lambda: {"num_samples": 0, "final_correct": 0, "img_correct": 0, "img_total": 0, "all_imgs_correct": 0})
    # buckets: task -> (k, correct_count) -> [final_correct_count, total]
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    per_sample_path = out_dir / "two_stage_per_sample.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as f:
        for s in samples:
            eval_id: Optional[int] = None
            if isinstance(s.ground_truth, dict) and "eval_id" in s.ground_truth:
                try:
                    eval_id = int(s.ground_truth["eval_id"])
                except Exception:
                    eval_id = None
            if eval_id is None:
                eval_id = _extract_eval_id(s.prompt)
            row = val_rows_by_id.get(eval_id) if eval_id is not None else None
            if row is None:
                continue

            keys, gt_labels, task_type = _infer_image_keys_and_gt_labels(row)
            pred_labels = _extract_labels_mapping(s.output)
            final_pred_text = _extract_answer_text(s.output)
            gt_final = None
            ans = row.get("answer") if isinstance(row.get("answer"), dict) else {}
            gt_final = ans.get("correct_answer")

            per_image = []
            correct_count = 0
            for k in keys:
                gt = _normalize_label(gt_labels.get(k, ""))
                pred = _normalize_label(pred_labels.get(k)) if k in pred_labels else None
                ok = (pred is not None) and (gt is not None) and (pred == gt)
                per_image.append({"key": k, "gt": gt_labels.get(k, ""), "pred": pred_labels.get(k, None), "is_correct": bool(ok)})
                correct_count += 1 if ok else 0

            all_correct = correct_count == len(keys) and len(keys) > 0
            final_ok = _final_is_correct(pred_text=final_pred_text, gt=gt_final)
            final_ok_bool = bool(final_ok) if final_ok is not None else False

            per_task[task_type]["num_samples"] += 1
            per_task[task_type]["final_correct"] += 1 if final_ok_bool else 0
            per_task[task_type]["img_correct"] += correct_count
            per_task[task_type]["img_total"] += len(keys)
            per_task[task_type]["all_imgs_correct"] += 1 if all_correct else 0

            bucket_key = f"k={len(keys)}|correct={correct_count}"
            buckets[task_type][bucket_key][0] += 1 if final_ok_bool else 0
            buckets[task_type][bucket_key][1] += 1

            f.write(
                json.dumps(
                    {
                        "eval_id": eval_id,
                        "task_type": task_type,
                        "num_images": len(keys),
                        "intermediate": {
                            "correct_count": correct_count,
                            "total": len(keys),
                            "acc": (correct_count / len(keys)) if keys else None,
                            "all_correct": all_correct,
                            "per_image": per_image,
                        },
                        "final": {
                            "pred_answer": final_pred_text,
                            "gt_answer": gt_final,
                            "is_correct": final_ok,
                        },
                        "raw_output": s.output,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = {"per_task": [], "meta": {"num_matched_samples": sum(v["num_samples"] for v in per_task.values())}}
    for task_type, m in sorted(per_task.items(), key=lambda kv: (-kv[1]["num_samples"], kv[0])):
        n = m["num_samples"]
        img_total = m["img_total"]
        img_acc = (m["img_correct"] / img_total) if img_total else float("nan")
        final_acc = (m["final_correct"] / n) if n else float("nan")
        all_img_rate = (m["all_imgs_correct"] / n) if n else float("nan")
        metrics["per_task"].append(
            {
                "task_type": task_type,
                "num_samples": n,
                "intermediate_image_acc": img_acc,
                "intermediate_all_images_correct_rate": all_img_rate,
                "final_acc": final_acc,
                "counts": {
                    "final_correct": m["final_correct"],
                    "final_total": n,
                    "img_correct": m["img_correct"],
                    "img_total": img_total,
                    "all_imgs_correct": m["all_imgs_correct"],
                },
                "final_acc_by_intermediate_bucket": {
                    b: (c[0] / c[1]) if c[1] else None for b, c in sorted(buckets[task_type].items())
                },
            }
        )
    out_path = out_dir / "two_stage_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="A training config YAML (used for model/rollout/reward).")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint dir ending with global_step_*. If omitted, evaluates the base model from config/overrides.",
    )
    ap.add_argument("--val", type=Path, required=True, help="Eval JSONL (B tasks recommended).")
    ap.add_argument("--out", type=Path, required=True, help="Output dir for eval logs and dumps.")
    ap.add_argument("--name", type=str, default=None, help="Optional experiment name override.")
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra OmegaConf overrides, e.g. worker.rollout.val_override_config.n=1",
    )
    ap.add_argument(
        "--two-stage-prefix",
        type=Path,
        default=Path("EasyR1/Comparative-R1/prompts/omnimed_isic_two_stage_prefix.txt"),
        help="Prefix template file to prepend to each prompt.",
    )
    ap.add_argument("--analyze-only", action="store_true", help="Skip running veRL; only analyze existing generations.log.")
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

    prefix_path = args.two_stage_prefix
    if not prefix_path.exists():
        raise SystemExit(f"Missing --two-stage-prefix: {prefix_path}")
    prefix_template = prefix_path.read_text(encoding="utf-8")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build transformed val JSONL (prompt prefix + EvalID).
    in_rows = _read_jsonl(val_jsonl)
    transformed_rows: list[dict[str, Any]] = []
    val_rows_by_id: dict[int, dict[str, Any]] = {}
    for i, row in enumerate(in_rows):
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            continue
        keys, _, _ = _infer_image_keys_and_gt_labels(row)
        new_prompt = build_two_stage_prompt(
            original_prompt=prompt,
            eval_id=i,
            keys=keys,
            prefix_template=prefix_template,
        )
        new_row = dict(row)
        new_row["prompt"] = new_prompt
        # Add a stable id for downstream joins (should be ignored by training code).
        ans = new_row.get("answer") if isinstance(new_row.get("answer"), dict) else {}
        ans = dict(ans)
        ans["eval_id"] = i
        new_row["answer"] = ans
        transformed_rows.append(new_row)
        val_rows_by_id[i] = new_row

    transformed_val = out_dir / f"{val_jsonl.stem}.two_stage.jsonl"
    _write_jsonl(transformed_val, transformed_rows)

    ckpt_name = checkpoint.name if checkpoint is not None else "pretrained"
    exp_name = args.name or f"eval_two_stage_{ckpt_name}_{val_jsonl.stem}"

    if not args.analyze_only:
        run_val_only(
            config=config,
            checkpoint=checkpoint,
            val_jsonl=transformed_val,
            out_dir=out_dir,
            experiment_name=exp_name,
            extra_overrides=args.override,
        )

    gen_log = out_dir / "generations.log"
    if not gen_log.exists():
        raise SystemExit(f"Expected generations.log at {gen_log} but it does not exist.")

    samples = parse_generations_log(gen_log)

    # Dump raw predictions (for compatibility with existing tooling).
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

    stats_path = out_dir / "predictions.summary.json"
    stats = {
        "config": str(config),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "val": str(val_jsonl),
        "val_two_stage": str(transformed_val),
        "out": str(out_dir),
        "num_samples": len(samples),
        "experiment_name": exp_name,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Two-stage metrics.
    compute_two_stage_metrics(val_rows_by_id=val_rows_by_id, samples=samples, out_dir=out_dir)


if __name__ == "__main__":
    main()
