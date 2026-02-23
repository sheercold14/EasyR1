#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _group_probe_maps(summary: dict) -> dict[str, dict[str, dict]]:
    """
    Normalize summary into {group_name: {probe_name: result_dict}}.
    - Non-grouped summaries use group_name="__all__" and read summary["results"].
    - Grouped summaries read summary["groups"][*]["results"].
    """
    groups = summary.get("groups", None)
    if isinstance(groups, list) and groups:
        out: dict[str, dict[str, dict]] = {}
        for g in groups:
            if not isinstance(g, dict):
                continue
            gname = str(g.get("group", "")).strip() or "__unnamed_group__"
            pmap: dict[str, dict] = {}
            for x in g.get("results", []) or []:
                if not isinstance(x, dict):
                    continue
                name = str(x.get("probe", "")).strip()
                if name:
                    pmap[name] = x
            out[gname] = pmap
        return out

    # Fallback: non-grouped
    out = {"__all__": {}}
    for x in summary.get("results", []) or []:
        if not isinstance(x, dict):
            continue
        name = str(x.get("probe", "")).strip()
        if name:
            out["__all__"][name] = x
    return out


def _group_meta(summary: dict) -> dict[str, dict]:
    groups = summary.get("groups", None)
    if isinstance(groups, list) and groups:
        out: dict[str, dict] = {}
        for g in groups:
            if not isinstance(g, dict):
                continue
            gname = str(g.get("group", "")).strip() or "__unnamed_group__"
            out[gname] = {
                "n_samples": g.get("n_samples", None),
                "n_classes": g.get("n_classes", None),
                "classes": g.get("classes", None),
            }
        return out
    return {"__all__": {"n_samples": summary.get("n_samples", None), "n_classes": summary.get("n_classes", None)}}


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two probe summary.json files")
    p.add_argument("--pre", required=True, help="summary.json before RFT")
    p.add_argument("--post", required=True, help="summary.json after RFT")
    args = p.parse_args()

    pre = _load(args.pre)
    post = _load(args.post)

    pre_groups = _group_probe_maps(pre)
    post_groups = _group_probe_maps(post)
    pre_meta = _group_meta(pre)
    post_meta = _group_meta(post)

    shared_groups = sorted(set(pre_groups.keys()) & set(post_groups.keys()))
    if not shared_groups:
        raise SystemExit("No shared group names between pre and post summaries")

    rows = []
    for gname in shared_groups:
        pre_map = pre_groups[gname]
        post_map = post_groups[gname]
        shared_probes = sorted(set(pre_map.keys()) & set(post_map.keys()))
        for name in shared_probes:
            pr = pre_map[name].get("metrics", {})
            po = post_map[name].get("metrics", {})
            row = {
                "group": None if gname == "__all__" else gname,
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

    if not rows:
        raise SystemExit("No shared probe names between pre and post summaries (after grouping)")

    print(
        json.dumps(
            {
                "pre": args.pre,
                "post": args.post,
                "pre_group_meta": pre_meta,
                "post_group_meta": post_meta,
                "comparison": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
