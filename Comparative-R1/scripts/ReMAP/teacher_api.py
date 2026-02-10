#!/usr/bin/env python3

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_api_key(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"API key file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for field in ("qwen_api_key", "key", "api_key"):
                value = payload.get(field)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        raise RuntimeError(f"No key field found in JSON file: {path}")

    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if isinstance(row, dict):
                for field in ("qwen_api_key", "key", "api_key"):
                    value = row.get(field)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
    raise RuntimeError(f"No non-empty key field found in {path}")


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("Teacher response does not contain a JSON object")


def _to_data_uri(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


@dataclass
class TeacherConfig:
    base_url: str
    model: str
    api_key: str = ""
    timeout_seconds: int = 90
    max_retries: int = 3
    retry_sleep: float = 1.0


class TeacherClient:
    def __init__(self, config: TeacherConfig, cache_dir: Optional[Path] = None):
        self.config = config
        self.cache_dir = cache_dir
        self.base_url = config.base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.client = OpenAI(api_key=self.config.api_key, base_url=self.base_url, timeout=self.config.timeout_seconds)

    @classmethod
    def from_env(cls, cache_dir: Optional[Path] = None, api_key_jsonl: Optional[Path] = None) -> "TeacherClient":
        base_url = os.getenv("REMAP_API_BASE", "").strip()
        model = os.getenv("REMAP_API_MODEL", "").strip() or "qwen-vl-max-latest"
        if not base_url:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"


        api_key = os.getenv("REMAP_API_KEY", "").strip()
        env_key_jsonl = os.getenv("REMAP_API_KEY_JSONL", "").strip()
        key_jsonl_path = api_key_jsonl or (Path(env_key_jsonl) if env_key_jsonl else None)
        if not api_key and key_jsonl_path is not None:
            api_key = _read_api_key(key_jsonl_path)
        if not api_key:
            raise RuntimeError("Missing API key. Set REMAP_API_KEY or REMAP_API_KEY_JSONL/--api-key-jsonl")

        timeout_seconds = int(os.getenv("REMAP_API_TIMEOUT", "90"))
        max_retries = int(os.getenv("REMAP_API_MAX_RETRIES", "3"))
        retry_sleep = float(os.getenv("REMAP_API_RETRY_SLEEP", "1.0"))
        return cls(
            TeacherConfig(
                base_url=base_url,
                model=model,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
            ),
            cache_dir=cache_dir,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any], cache_dir: Optional[Path] = None) -> "TeacherClient":
        base_url = str(config.get("api_base", "")).strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = str(config.get("api_model", "")).strip() or "qwen-vl-max-latest"
        api_key = str(config.get("api_key", "")).strip()
        api_key_jsonl = str(config.get("api_key_jsonl", "")).strip()
        if not api_key and api_key_jsonl:
            api_key = _read_api_key(Path(api_key_jsonl))
        if not api_key:
            raise RuntimeError("Missing api_key/api_key_jsonl in config")

        timeout_seconds = int(config.get("api_timeout", 90))
        return cls(
            TeacherConfig(
                base_url=base_url,
                model=model,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                max_retries=1,
                retry_sleep=0.0,
            ),
            cache_dir=cache_dir,
        )

    def _cache_path(self, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{key}.json"

    def _build_key(self, payload: dict[str, Any]) -> str:
        normalized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return _sha256_text(normalized)

    def chat_json(
        self,
        user_prompt: str,
        image_paths: Optional[list[Path]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        image_paths = image_paths or []
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image_path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": _to_data_uri(image_path)}})

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        cache_key = self._build_key(payload)
        cache_path = self._cache_path(cache_key)
        if cache_path is not None:
            cached = _read_json(cache_path)
            if cached is not None:
                return cached

        completion = self.client.chat.completions.create(**payload)
        raw_text = completion.choices[0].message.content or ""
        try:
            parsed = _extract_json_object(raw_text)
        except Exception:
            if cache_path is not None:
                _write_text(cache_path.with_suffix(".raw.txt"), raw_text)
            raise
        result = {
            "raw": raw_text,
            "parsed": parsed,
            "model": self.config.model,
            "cache_key": cache_key,
        }
        if cache_path is not None:
            _write_json(cache_path, result)
        return result
