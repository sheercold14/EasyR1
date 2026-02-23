from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .datasets import Sample
from .utils import load_symbol


class BaseExtractor:
    def extract(self, samples: list[Sample]) -> np.ndarray:
        raise NotImplementedError


class MeanRGBExtractor(BaseExtractor):
    """Simple sanity-check baseline: 3-dim mean RGB feature."""

    def __init__(self, image_root: str | None = None):
        self.image_root = Path(image_root) if image_root else None

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute() or self.image_root is None:
            return p
        return self.image_root / p

    def extract(self, samples: list[Sample]) -> np.ndarray:
        feats: list[np.ndarray] = []
        for s in samples:
            p = self._resolve(s.image_path)
            with Image.open(p) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.float32)
            mean_rgb = arr.reshape(-1, 3).mean(axis=0)
            feats.append(mean_rgb)
        return np.asarray(feats, dtype=np.float32)


class NpzExtractor(BaseExtractor):
    """
    Load precomputed features from .npz.

    Supported layout:
    - aligned: features=[N, D], same order as input samples
    - id-map: features=[N, D], ids=[N] and sample_id matching
    """

    def __init__(self, npz_path: str, feature_key: str = "features", id_key: str = "ids"):
        pack = np.load(npz_path, allow_pickle=False)
        if feature_key not in pack:
            keys = ", ".join(pack.files)
            raise KeyError(f"Missing feature_key '{feature_key}' in npz. Available keys: {keys}")
        self.features = np.asarray(pack[feature_key])
        self.ids = np.asarray(pack[id_key]).astype(str) if id_key in pack else None

    def extract(self, samples: list[Sample]) -> np.ndarray:
        if self.features.ndim != 2:
            raise ValueError(
                f"Selected feature_key does not point to a 2D feature matrix: shape={getattr(self.features, 'shape', None)}"
            )
        if self.ids is None:
            if len(samples) != len(self.features):
                raise ValueError(
                    f"Aligned features mismatch: samples={len(samples)}, features={len(self.features)}"
                )
            return self.features

        id_to_idx = {sid: i for i, sid in enumerate(self.ids.tolist())}
        rows: list[np.ndarray] = []
        missing: list[str] = []
        for s in samples:
            idx = id_to_idx.get(s.sample_id)
            if idx is None:
                missing.append(s.sample_id)
                continue
            rows.append(self.features[idx])
        if missing:
            raise KeyError(f"Missing {len(missing)} sample_ids in feature npz, first: {missing[0]}")
        return np.asarray(rows)


def build_extractor(
    extractor: str,
    *,
    image_root: str | None,
    features_npz: str | None,
    features_key: str,
    ids_key: str,
    plugin_kwargs: dict[str, Any] | None = None,
) -> BaseExtractor:
    plugin_kwargs = plugin_kwargs or {}
    if extractor == "mean_rgb":
        return MeanRGBExtractor(image_root=image_root)
    if extractor == "npz":
        if not features_npz:
            raise ValueError("--features-npz is required when --extractor=npz")
        return NpzExtractor(npz_path=features_npz, feature_key=features_key, id_key=ids_key)

    # Custom extractor hook: module:function that returns BaseExtractor-like object.
    factory = load_symbol(extractor)
    obj = factory(image_root=image_root, features_npz=features_npz, **plugin_kwargs)
    if not hasattr(obj, "extract"):
        raise TypeError(f"Extractor '{extractor}' must return object with .extract(samples)")
    return obj
