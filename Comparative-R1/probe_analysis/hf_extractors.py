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

class Qwen2VLHiddenStateMeanExtractor:
    """
    Extract per-image feature by mean-pooling selected token hidden states.

    Typical use:
    - token_scope=image: probe vision->LLM fusion / decision pathways
    - layer=0: embedding output (pre-transformer blocks)
    - layer=k: mid/late transformer block output
    """

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
        layer: int = 0,
        token_scope: str = "image",  # image|text|all
    ):
        self.image_root = Path(image_root) if image_root else None
        self.prompt_text = prompt_text
        self.layer = int(layer)
        self.token_scope = token_scope.strip().lower()
        if self.token_scope not in {"image", "text", "all"}:
            raise ValueError(f"Unsupported token_scope: {token_scope}")

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
        self.device = next(self.model.parameters()).device

        # Try best-effort image_token_id; Qwen2/2.5-VL should provide it.
        self.image_token_id = getattr(getattr(self.model, "config", None), "image_token_id", None)

    def _resolve(self, image_path: str) -> Path:
        p = Path(image_path)
        if p.is_absolute() or self.image_root is None:
            return p
        return self.image_root / p

    def _build_inputs(self, image_path: str) -> dict[str, torch.Tensor]:
        p = self._resolve(image_path)
        with Image.open(p) as img:
            image = img.convert("RGB")

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
        # Move all tensors to model device.
        return {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    def _select_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        if self.token_scope == "all":
            return attention_mask.to(torch.bool)

        if self.image_token_id is None:
            raise ValueError(
                "model.config.image_token_id is missing; cannot select image/text tokens reliably for this model."
            )

        is_image = input_ids == int(self.image_token_id)
        if self.token_scope == "image":
            return is_image & attention_mask.to(torch.bool)
        # text
        return (~is_image) & attention_mask.to(torch.bool)

    def _extract_single(self, image_path: str) -> np.ndarray:
        inputs = self._build_inputs(image_path)

        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            raise ValueError("Processor did not return input_ids")
        attn = inputs.get("attention_mask", None)

        with torch.no_grad():
            out = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Model did not return hidden_states; cannot extract layer features.")

        # HF convention: len = n_layers + 1, hidden_states[0] is embedding output.
        num = len(hidden_states)
        layer = self.layer
        if layer < 0:
            layer = num + layer
        if layer < 0 or layer >= num:
            raise ValueError(f"Invalid layer={self.layer} for hidden_states len={num}")

        hs = hidden_states[layer]  # [B, T, D]
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            raise ValueError(f"Unexpected hidden state shape at layer {layer}: {getattr(hs, 'shape', None)}")

        mask = self._select_mask(input_ids=input_ids, attention_mask=attn)  # [B, T]
        if mask.ndim != 2 or mask.shape[0] != hs.shape[0] or mask.shape[1] != hs.shape[1]:
            raise ValueError(f"Mask/hidden mismatch: mask={tuple(mask.shape)} hs={tuple(hs.shape)}")

        sel = hs[mask]  # [N_tok, D]
        if sel.numel() == 0:
            raise ValueError(f"No tokens selected for token_scope={self.token_scope}")

        feat = sel.float().mean(dim=0)
        return feat.detach().cpu().numpy().astype(np.float32)

    def extract(self, samples: list[Sample]) -> np.ndarray:
        feats: list[np.ndarray] = []
        for s in samples:
            feats.append(self._extract_single(s.image_path))
        return np.stack(feats, axis=0)

class Qwen2VLMultiTapExtractor:
    """
    Compute multiple tap features per image, with one LLM forward pass shared across hidden-state taps.

    Tap spec strings:
    - `vision_mean`
    - `hs:<layer>:<scope>` where scope in {image,text,all}, layer can be negative (e.g. -1)
    - `last:<layer>` where layer can be negative (e.g. -1). Uses the last non-pad token position.
    """

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
        self.device = next(self.model.parameters()).device

        if hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
            self.visual = self.model.model.visual
        elif hasattr(self.model, "visual"):
            self.visual = self.model.visual
        else:
            self.visual = None

        self.image_token_id = getattr(getattr(self.model, "config", None), "image_token_id", None)

    def _resolve(self, image_path: str) -> Path:
        p = Path(image_path)
        if p.is_absolute() or self.image_root is None:
            return p
        return self.image_root / p

    def _build_inputs(self, image_path: str) -> dict[str, torch.Tensor]:
        p = self._resolve(image_path)
        with Image.open(p) as img:
            image = img.convert("RGB")

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
        return {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    @staticmethod
    def _parse_tap(spec: str) -> tuple[str, tuple]:
        s = spec.strip()
        if s == "vision_mean":
            return ("vision_mean", ())
        if s.startswith("last:"):
            parts = s.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid tap spec: {spec}")
            layer = int(parts[1])
            return (f"last{layer}", (layer,))
        if s.startswith("hs:"):
            parts = s.split(":")
            if len(parts) == 2:
                layer = int(parts[1])
                scope = "image"
            elif len(parts) == 3:
                layer = int(parts[1])
                scope = parts[2].strip().lower()
            else:
                raise ValueError(f"Invalid tap spec: {spec}")
            if scope not in {"image", "text", "all"}:
                raise ValueError(f"Invalid token scope in tap spec: {spec}")
            return (f"hs{layer}_{scope}", (layer, scope))
        raise ValueError(f"Unknown tap spec: {spec}")

    def _mask_for_scope(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, scope: str) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        if scope == "all":
            return attention_mask.to(torch.bool)
        if self.image_token_id is None:
            raise ValueError("model.config.image_token_id missing; cannot select image/text tokens.")
        is_image = input_ids == int(self.image_token_id)
        if scope == "image":
            return is_image & attention_mask.to(torch.bool)
        return (~is_image) & attention_mask.to(torch.bool)

    def extract_taps(
        self,
        samples: list[Sample],
        *,
        tap_specs: list[str],
        progress_every: int = 50,
        verbose: bool = False,
    ) -> dict[str, np.ndarray]:
        parsed = [self._parse_tap(s) for s in tap_specs]
        want_vision = any(name == "vision_mean" for name, _ in parsed)
        want_hs = any((name.startswith("hs") or name.startswith("last")) for name, _ in parsed)

        feats: dict[str, list[np.ndarray]] = {name: [] for name, _ in parsed}

        for i, s in enumerate(samples, 1):
            inputs = self._build_inputs(s.image_path)
            input_ids = inputs.get("input_ids", None)
            if input_ids is None:
                raise ValueError("Processor did not return input_ids")
            attn = inputs.get("attention_mask", None)
            if attn is not None:
                last_pos = int(attn[0].sum().item()) - 1
            else:
                last_pos = int(input_ids.shape[1]) - 1

            hs = None
            if want_hs:
                with torch.no_grad():
                    out = self.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
                hs = getattr(out, "hidden_states", None)
                if hs is None:
                    raise ValueError("Model did not return hidden_states")

            vis_tokens = None
            if want_vision:
                if self.visual is None:
                    raise ValueError("Cannot find vision tower on model for vision_mean tap")
                pixel_values = inputs.get("pixel_values", None)
                grid_thw = inputs.get("image_grid_thw", None)
                if pixel_values is None:
                    raise ValueError("Processor did not return pixel_values")
                pv = pixel_values
                if hasattr(self.visual, "dtype"):
                    pv = pv.to(self.visual.dtype)
                if grid_thw is not None:
                    grid_thw = grid_thw.to(self.device)
                with torch.no_grad():
                    vis_tokens = self.visual(pv, grid_thw=grid_thw)
                if isinstance(vis_tokens, tuple):
                    vis_tokens = vis_tokens[0]
                if not isinstance(vis_tokens, torch.Tensor):
                    raise TypeError(f"Unexpected visual output type: {type(vis_tokens)}")

            for name, cfg in parsed:
                if name == "vision_mean":
                    x = vis_tokens
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    feat = x.float().mean(dim=0).detach().cpu().numpy().astype(np.float32)
                    feats[name].append(feat)
                    continue

                num = len(hs)
                if name.startswith("last"):
                    (layer,) = cfg
                    layer_idx = layer if layer >= 0 else num + layer
                    if layer_idx < 0 or layer_idx >= num:
                        raise ValueError(f"Invalid layer={layer} for hidden_states len={num}")
                    h = hs[layer_idx]  # [B,T,D]
                    if last_pos < 0 or last_pos >= h.shape[1]:
                        raise ValueError(f"Invalid last_pos={last_pos} for seq_len={h.shape[1]}")
                    feat = h[0, last_pos].float().detach().cpu().numpy().astype(np.float32)
                    feats[name].append(feat)
                else:
                    layer, scope = cfg
                    layer_idx = layer if layer >= 0 else num + layer
                    if layer_idx < 0 or layer_idx >= num:
                        raise ValueError(f"Invalid layer={layer} for hidden_states len={num}")
                    h = hs[layer_idx]
                    mask = self._mask_for_scope(input_ids, attn, scope)
                    sel = h[mask]
                    if sel.numel() == 0:
                        raise ValueError(f"No tokens selected for tap={name}")
                    feat = sel.float().mean(dim=0).detach().cpu().numpy().astype(np.float32)
                    feats[name].append(feat)

            if verbose and progress_every > 0 and (i == 1 or i % progress_every == 0 or i == len(samples)):
                print(f"[extract] {i}/{len(samples)}")

        return {k: np.stack(v, axis=0) for k, v in feats.items()}


def build_qwen2vl_visual_mean_extractor(
    image_root: str | None = None,
    features_npz: str | None = None,
    **kwargs: Any,
) -> Qwen2VLVisualMeanExtractor:
    _ = features_npz
    return Qwen2VLVisualMeanExtractor(image_root=image_root, **kwargs)


def build_qwen2vl_hidden_state_mean_extractor(
    image_root: str | None = None,
    features_npz: str | None = None,
    **kwargs: Any,
) -> Qwen2VLHiddenStateMeanExtractor:
    _ = features_npz
    return Qwen2VLHiddenStateMeanExtractor(image_root=image_root, **kwargs)
