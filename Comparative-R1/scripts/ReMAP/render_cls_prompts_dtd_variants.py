#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

"""
Render (pre-wrap) classification prompts into two variants that match:
  - Comparative-R1/prompts/dtd_nothinking.jinja
  - Comparative-R1/prompts/dtd.jinja

This removes the need to pass `data.format_prompt` at training time.

Example:
  python3 Comparative-R1/scripts/ReMAP/render_cls_prompts_dtd_variants.py \
    --input data/CLS/ISIC/4shot/ISIC_fewshot_test.jsonl \
    --output-dir data/CLS/ISIC/4shot
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _wrap_with_dtd(content: str, *, include_think: bool) -> str:
    """
    Produce the same string as dtd(.jinja) would render when `content` is the raw prompt.

    dtd_nothinking.jinja:
      <image>
      {{ content | trim }}
      ...

    dtd.jinja:
      <image>
      {{ content | trim }}
      ...
      <think> brief reasoning </think>
      <answer> exact label text </answer>
    """
    c = (content or "").strip()
    if "<image>" not in c:
        c = "<image>\n" + c

    suffix_lines = ["Follow the exact output format:"]
    if include_think:
        suffix_lines.append("<think> brief reasoning </think>")
    suffix_lines.append("<answer> exact label text </answer>")

    return (c.rstrip() + "\n\n" + "\n".join(suffix_lines)).strip()


def main() -> None:
    p = argparse.ArgumentParser(description="Render cls prompts into DTD template variants (no Jinja dependency).")
    p.add_argument("--input", required=True, type=Path, help="Input JSONL (e.g., ISIC_fewshot_4.jsonl).")
    p.add_argument(
        "--prompt-key",
        type=str,
        default="problem",
        help="Which key contains the raw prompt text (default: problem).",
    )
    p.add_argument(
        "--keep-raw",
        action="store_true",
        help="If set, preserve the original prompt under `<prompt_key>_raw` in the output files.",
    )

    out = p.add_mutually_exclusive_group(required=True)
    out.add_argument("--output-dir", type=Path, help="Directory to write both outputs.")
    out.add_argument("--output-prefix", type=Path, help="Prefix path for outputs; adds suffixes automatically.")

    p.add_argument("--output-nothinking", type=Path, default=None, help="Override output path for nothinking variant.")
    p.add_argument("--output-thinking", type=Path, default=None, help="Override output path for thinking variant.")
    args = p.parse_args()

    rows = _read_jsonl(args.input)

    if args.output_dir is not None:
        base = args.output_dir / args.input.stem
    else:
        base = args.output_prefix

    out_nothinking = args.output_nothinking or Path(str(base) + ".dtd_nothinking.jsonl")
    out_thinking = args.output_thinking or Path(str(base) + ".dtd_thinking.jsonl")

    key = args.prompt_key
    raw_key = f"{key}_raw"

    missing_prompt = 0
    out_rows_nt: list[dict[str, Any]] = []
    out_rows_t: list[dict[str, Any]] = []
    for i, r in enumerate(rows, start=1):
        if key not in r or not isinstance(r.get(key), str) or not r.get(key, "").strip():
            missing_prompt += 1
        raw_prompt = r.get(key, "")

        r_nt = dict(r)
        r_t = dict(r)
        if args.keep_raw:
            r_nt[raw_key] = raw_prompt
            r_t[raw_key] = raw_prompt

        r_nt[key] = _wrap_with_dtd(raw_prompt, include_think=False)
        r_t[key] = _wrap_with_dtd(raw_prompt, include_think=True)

        out_rows_nt.append(r_nt)
        out_rows_t.append(r_t)

    _write_jsonl(out_nothinking, out_rows_nt)
    _write_jsonl(out_thinking, out_rows_t)

    summary = {
        "input": str(args.input),
        "prompt_key": key,
        "keep_raw": bool(args.keep_raw),
        "output_nothinking": str(out_nothinking),
        "output_thinking": str(out_thinking),
        "total_rows": len(rows),
        "missing_prompt": missing_prompt,
    }
    # Write paired summaries next to outputs.
    # out_nothinking.with_suffix(".summary.json").write_text(
    #     json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    # )
    # out_thinking.with_suffix(".summary.json").write_text(
    #     json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    # )
    # print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

