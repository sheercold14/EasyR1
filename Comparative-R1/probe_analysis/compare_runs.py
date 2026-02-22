#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _probe_map(summary: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for x in summary.get("results", []):
        name = str(x.get("probe", ""))
        if name:
            out[name] = x
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two probe summary.json files")
    p.add_argument("--pre", required=True, help="summary.json before RFT")
    p.add_argument("--post", required=True, help="summary.json after RFT")
    args = p.parse_args()

    pre = _load(args.pre)
    post = _load(args.post)
    pre_map = _probe_map(pre)
    post_map = _probe_map(post)

    shared = sorted(set(pre_map.keys()) & set(post_map.keys()))
    if not shared:
        raise SystemExit("No shared probe names between pre and post summaries")

    rows = []
    for name in shared:
        pr = pre_map[name].get("metrics", {})
        po = post_map[name].get("metrics", {})
        row = {
            "probe": name,
            "pre_accuracy": pr.get("accuracy"),
            "post_accuracy": po.get("accuracy"),
            "delta_accuracy": (po.get("accuracy", 0.0) - pr.get("accuracy", 0.0)),
            "pre_balanced_accuracy": pr.get("balanced_accuracy"),
            "post_balanced_accuracy": po.get("balanced_accuracy"),
            "delta_balanced_accuracy": (po.get("balanced_accuracy", 0.0) - pr.get("balanced_accuracy", 0.0)),
            "pre_macro_f1": pr.get("macro_f1"),
            "post_macro_f1": po.get("macro_f1"),
            "delta_macro_f1": (po.get("macro_f1", 0.0) - pr.get("macro_f1", 0.0)),
        }
        rows.append(row)

    print(json.dumps({"pre": args.pre, "post": args.post, "comparison": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
