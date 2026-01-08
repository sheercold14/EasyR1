"""Logic-based reward with weakness feedback."""

from __future__ import annotations

import json
import os
import re
from typing import Any


REWARD_NAME = "visual_logic_reward"
REWARD_TYPE = "batch"


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _to_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).lower() for v in value]
    return [str(value).lower()]


def _parse_ground_truth(raw) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"answer": text}
        return {"answer": text}
    return {"answer": str(raw)}


def _extract_think(response: str) -> str:
    match = _THINK_RE.search(response)
    if not match:
        return ""
    return match.group(1).lower()


def _get_redis_client():
    redis_url = os.getenv("ORACLE_REDIS_URL")
    if not redis_url:
        return None
    try:
        import redis  # type: ignore
    except ImportError:
        return None
    timeout = float(os.getenv("ORACLE_REDIS_TIMEOUT", "2"))
    return redis.Redis.from_url(
        redis_url,
        socket_timeout=timeout,
        socket_connect_timeout=timeout,
        decode_responses=True,
    )


def _update_weakness(weak_classes: list[str]) -> None:
    if not weak_classes:
        return
    client = _get_redis_client()
    if client is None:
        return

    weakness_key = os.getenv("ORACLE_WEAKNESS_KEY", "student_weakness")
    payload = json.dumps({"weak_classes": weak_classes}, ensure_ascii=True)
    try:
        client.set(weakness_key, payload)
    except Exception:
        return


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores = []
    scored_labels: list[tuple[float, str]] = []

    for reward_input in reward_inputs:
        response = str(reward_input.get("response", ""))
        response_lc = response.lower()
        think_lc = _extract_think(response)
        meta = _parse_ground_truth(reward_input.get("ground_truth"))

        must_include = _to_list(meta.get("must_include"))
        must_not_include = _to_list(meta.get("must_not_include"))
        avoid_in_think = _to_list(meta.get("avoid_in_think"))
        answer = str(meta.get("answer", "")).lower()

        include_ok = all(term in response_lc for term in must_include) if must_include else False
        hallucinated = any(term in response_lc for term in must_not_include)
        think_violate = any(term in think_lc for term in avoid_in_think)

        if must_include:
            score = 1.0 if include_ok else -1.0
        elif answer:
            score = 1.0 if answer in response_lc else 0.0
        else:
            score = 0.0

        if hallucinated:
            score -= 1.5
        if think_violate:
            score -= 0.5

        class_label = meta.get("class_label")
        if class_label:
            scored_labels.append((score, str(class_label)))

        scores.append({"overall": score, "accuracy": float(include_ok)})

    if scored_labels:
        threshold = float(os.getenv("ORACLE_WEAKNESS_THRESHOLD", "-0.1"))
        topk = int(os.getenv("ORACLE_WEAKNESS_TOPK", "3"))
        scored_labels.sort(key=lambda x: x[0])
        weak_classes = [label for score, label in scored_labels if score <= threshold]
        if not weak_classes:
            weak_classes = [label for _, label in scored_labels[:topk]]
        _update_weakness(weak_classes)

    return scores
