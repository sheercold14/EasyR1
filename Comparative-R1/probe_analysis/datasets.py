from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import deep_get


@dataclass
class Sample:
    sample_id: str
    image_path: str
    label: str
    raw: dict[str, Any]


def load_jsonl(
    path: str | Path,
    *,
    image_key: str = "images.0",
    label_key: str = "answer.correct_answer",
    sample_id_key: str = "",
) -> list[Sample]:
    p = Path(path)
    rows: list[Sample] = []
    with p.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image = deep_get(row, image_key, "")
            label = deep_get(row, label_key, "")
            if image is None or label is None:
                continue
            image_str = str(image).strip()
            label_str = str(label).strip()
            if not image_str or not label_str:
                continue
            if sample_id_key:
                sid = str(deep_get(row, sample_id_key, "")).strip()
            else:
                sid = ""
            if not sid:
                sid = f"{p.name}:{idx}:{image_str}"
            rows.append(Sample(sample_id=sid, image_path=image_str, label=label_str, raw=row))
    return rows
