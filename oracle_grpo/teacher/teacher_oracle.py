"""Simple oracle teacher that pushes prompts to Redis."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any


def _get_redis_client():
    redis_url = os.getenv("ORACLE_REDIS_URL")
    if not redis_url:
        raise RuntimeError("ORACLE_REDIS_URL is required")
    try:
        import redis  # type: ignore
    except ImportError as exc:
        raise RuntimeError("redis package is required for online mode") from exc

    timeout = float(os.getenv("ORACLE_REDIS_TIMEOUT", "2"))
    return redis.Redis.from_url(
        redis_url,
        socket_timeout=timeout,
        socket_connect_timeout=timeout,
        decode_responses=True,
    )


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _get_weakness(client) -> list[str]:
    key = os.getenv("ORACLE_WEAKNESS_KEY", "student_weakness")
    try:
        payload = client.get(key)
    except Exception:
        return []
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    weak = data.get("weak_classes", [])
    if isinstance(weak, list):
        return [str(item) for item in weak]
    return []


def _build_payload(sample: dict[str, Any], weak_class: str | None) -> dict[str, Any]:
    prompt = sample.get("prompt") or sample.get("question") or "Please answer the visual question."
    images = sample.get("images") or sample.get("image") or sample.get("image_path")
    if isinstance(images, str):
        images = [images]

    meta = {
        "answer": sample.get("answer", ""),
    }
    if weak_class:
        meta["class_label"] = weak_class
        meta["logic_constraints"] = f"Focus on evidence for class {weak_class}."

    return {
        "prompt": prompt,
        "images": images,
        "meta": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-jsonl", required=True, help="Seed jsonl to sample base examples.")
    parser.add_argument("--push-per-cycle", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    seed_rows = _read_jsonl(args.seed_jsonl)
    if not seed_rows:
        raise RuntimeError("Seed jsonl is empty")

    client = _get_redis_client()
    queue_key = os.getenv("ORACLE_QUEUE_KEY", "oracle_queue")

    rng = random.Random(1234)

    while True:
        weak_classes = _get_weakness(client)
        for _ in range(args.push_per_cycle):
            sample = rng.choice(seed_rows)
            weak_class = rng.choice(weak_classes) if weak_classes else None
            payload = _build_payload(sample, weak_class)
            client.rpush(queue_key, json.dumps(payload, ensure_ascii=True))

        if args.once:
            break
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
