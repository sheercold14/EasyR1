from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from probe_analysis.datasets import Sample


def _to_torch_dtype(name: str) -> torch.dtype:
    x = name.strip().lower()
    if x in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if x in {"fp16", "float16", "half"}:
        return torch.float16
    if x in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


class Qwen2VLVisualMeanExtractor:
    """Extract per-image feature by mean-pooling Qwen2/2.5-VL visual tokens."""

    def __init__(
        self,
        *,
        checkpoint: str,
        processor_path: str | None = None,
        image_root: str | None = None,
        dtype: str = "bf16",
        device: str = "auto",
        trust_remote_code: bool = True,
        prompt_text: str = "Describe the image briefly.",
    ):
        self.image_root = Path(image_root) if image_root else None
        self.prompt_text = prompt_text

        model_path = checkpoint
        proc_path = processor_path or checkpoint

        self.processor = AutoProcessor.from_pretrained(proc_path, trust_remote_code=trust_remote_code)
        torch_dtype = _to_torch_dtype(dtype)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map=device,
        ).eval()

        if hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
            self.visual = self.model.model.visual
        elif hasattr(self.model, "visual"):
            self.visual = self.model.visual
        else:
            raise AttributeError("Cannot find vision tower on model (expected .model.visual or .visual)")

        self.device = next(self.model.parameters()).device

    def _resolve(self, image_path: str) -> Path:
        p = Path(image_path)
        if p.is_absolute() or self.image_root is None:
            return p
        return self.image_root / p

    def _extract_single(self, image_path: str) -> np.ndarray:
        p = self._resolve(image_path)
        with Image.open(p) as img:
            image = img.convert("RGB")

        # Build a standard multi-modal prompt so processor emits image tokens/grids.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt_text},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")

        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        if pixel_values is None:
            raise ValueError("Processor did not return pixel_values")

        pixel_values = pixel_values.to(self.device)
        if hasattr(self.visual, "dtype"):
            pixel_values = pixel_values.to(self.visual.dtype)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)

        with torch.no_grad():
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

        if isinstance(image_embeds, tuple):
            image_embeds = image_embeds[0]
        if not isinstance(image_embeds, torch.Tensor):
            raise TypeError(f"Unexpected visual output type: {type(image_embeds)}")

        # Normalize to [N, D] then mean-pool to [D].
        x = image_embeds
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected visual embedding shape: {tuple(x.shape)}")

        feat = x.float().mean(dim=0)
        return feat.detach().cpu().numpy().astype(np.float32)

    def extract(self, samples: list[Sample]) -> np.ndarray:
        feats: list[np.ndarray] = []
        for s in samples:
            feats.append(self._extract_single(s.image_path))
        return np.stack(feats, axis=0)


def build_qwen2vl_visual_mean_extractor(
    image_root: str | None = None,
    features_npz: str | None = None,
    **kwargs: Any,
) -> Qwen2VLVisualMeanExtractor:
    _ = features_npz
    return Qwen2VLVisualMeanExtractor(image_root=image_root, **kwargs)
