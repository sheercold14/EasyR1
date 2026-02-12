#!/usr/bin/env python3
"""
Compute per-task accuracy (acc) from an evaluation run directory.

This repo's eval runs typically contain:
  - predictions.jsonl  (one JSON per sample)
  - generations.log    (human-readable log)

We compute accuracy per `ground_truth.task_type` and overall accuracy.

Correctness is determined in the following priority:
  1) record["is_correct"] if present (bool)
  2) record["score"] if present (numeric): `score > 0` => correct
  3) fallback: compare record["output"] vs ground_truth["correct_answer"]

Example:
  python ./Comparative-R1/scripts/eval/task_acc_by_type.py \
    --run-dir /mnt/cache/wuruixiao/users/lsc/EasyR1/checkpoints/Eval-CLS/dtd_abase_pretrain_eval"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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
            yield obj


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return None if math.isnan(v) else v
    if isinstance(x, str):
        try:
            v = float(x.strip())
            return None if math.isnan(v) else v
        except Exception:
            return None
    return None


def _normalize_answer(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    return str(x).strip() or None


def _tokenize_answer(s: str) -> list[str]:
    # Accept outputs like: "C D", "B,C", "same", "different"
    toks = [t.lower() for t in _TOKEN_RE.findall(s)]
    # Special-case: if it contains letters A-D, keep those only
    letter_toks = [t for t in toks if t in {"a", "b", "c", "d"}]
    if letter_toks:
        return sorted(letter_toks)
    return toks


def _fallback_match(output: Any, correct_answer: Any) -> Optional[bool]:
    out = _normalize_answer(output)
    gt = _normalize_answer(correct_answer)
    if out is None or gt is None:
        return None
    return _tokenize_answer(out) == _tokenize_answer(gt)


@dataclass(frozen=True)
class Acc:
    correct: int = 0
    total: int = 0

    def add(self, is_correct: bool) -> "Acc":
        return Acc(self.correct + (1 if is_correct else 0), self.total + 1)

    @property
    def acc(self) -> float:
        return (self.correct / self.total) if self.total else float("nan")


def _fmt_pct(x: float) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{100.0 * x:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="Eval run directory containing predictions.jsonl")
    ap.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional explicit predictions.jsonl path (defaults to <run-dir>/predictions.jsonl).",
    )
    ap.add_argument("--task-key", type=str, default="ground_truth.task_type", help="Task field (dot path).")
    ap.add_argument("--save-json", type=Path, default=None, help="Optional output JSON path for metrics.")
    ap.add_argument("--quiet", action="store_true", help="Only print the final JSON path if --save-json is set.")
    args = ap.parse_args()

    pred_path = args.predictions if args.predictions is not None else (args.run_dir / "predictions.jsonl")
    if not pred_path.exists():
        raise SystemExit(f"Missing predictions file: {pred_path}")

    # Dot-path extractor (supports dict-only)
    key_parts = [p for p in args.task_key.split(".") if p]
    if not key_parts:
        raise SystemExit("--task-key cannot be empty")

    per_task: dict[str, Acc] = defaultdict(Acc)
    overall = Acc()
    used_rule = Counter()
    mismatch_rule = 0
    missing_task = 0

    for rec in _read_jsonl(pred_path):
        # task type
        cur: Any = rec
        for part in key_parts:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(part)
        task = str(cur) if cur is not None else "<missing_task>"
        if cur is None:
            missing_task += 1

        # correctness
        is_correct: Optional[bool] = None
        if isinstance(rec.get("is_correct"), bool):
            is_correct = rec["is_correct"]
            used_rule["is_correct"] += 1
        else:
            score = _as_float(rec.get("score"))
            if score is not None:
                is_correct = score > 0
                used_rule["score>0"] += 1
            else:
                gt = rec.get("ground_truth") if isinstance(rec.get("ground_truth"), dict) else {}
                is_correct = _fallback_match(rec.get("output"), gt.get("correct_answer"))
                if is_correct is None:
                    # Can't decide; count as incorrect but track.
                    is_correct = False
                    used_rule["fallback_unusable"] += 1
                else:
                    used_rule["fallback_match"] += 1

        # optional mismatch check between score rule and fallback (debugging only)
        score = _as_float(rec.get("score"))
        if score is not None:
            gt = rec.get("ground_truth") if isinstance(rec.get("ground_truth"), dict) else {}
            fb = _fallback_match(rec.get("output"), gt.get("correct_answer"))
            if fb is not None and (fb != (score > 0)):
                mismatch_rule += 1

        per_task[task] = per_task[task].add(bool(is_correct))
        overall = overall.add(bool(is_correct))

    rows = []
    for task, acc in sorted(per_task.items(), key=lambda kv: (-kv[1].acc, -kv[1].total, kv[0])):
        rows.append(
            {
                "task_type": task,
                "correct": acc.correct,
                "total": acc.total,
                "acc": acc.acc,
                "acc_pct": _fmt_pct(acc.acc),
            }
        )

    result = {
        "run_dir": str(args.run_dir),
        "predictions": str(pred_path),
        "overall": {
            "correct": overall.correct,
            "total": overall.total,
            "acc": overall.acc,
            "acc_pct": _fmt_pct(overall.acc),
        },
        "per_task": rows,
        "meta": {
            "task_key": args.task_key,
            "used_rule_counts": dict(used_rule),
            "missing_task": missing_task,
            "score_vs_fallback_mismatches": mismatch_rule,
        },
    }

    save_path = args.save_json
    if save_path is None:
        save_path = args.run_dir / "task_acc_by_type.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.quiet:
        print(str(save_path))
        return

    print("Overall:", f'{overall.correct}/{overall.total}', _fmt_pct(overall.acc))
    print("Per task:")
    for r in rows:
        print(f"  {r['task_type']}: {r['correct']}/{r['total']} ({r['acc_pct']})")
    if mismatch_rule:
        print(f"Note: score>0 disagrees with output-vs-correct_answer for {mismatch_rule} rows (pair order etc.).")
    print("Saved:", save_path)


if __name__ == "__main__":
    main()
