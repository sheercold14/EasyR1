from __future__ import annotations

from typing import Any

import numpy as np

from probe_analysis.datasets import Sample


class MyExtractor:
    def __init__(self, image_root: str | None = None, features_npz: str | None = None, **kwargs: Any):
        self.image_root = image_root
        self.features_npz = features_npz
        self.kwargs = kwargs

    def extract(self, samples: list[Sample]) -> np.ndarray:
        # Replace with your model forward pass.
        # Must return shape [N, D] float32/float64.
        n = len(samples)
        d = int(self.kwargs.get("dim", 8))
        return np.zeros((n, d), dtype=np.float32)


def build_extractor(image_root: str | None = None, features_npz: str | None = None, **kwargs: Any) -> MyExtractor:
    return MyExtractor(image_root=image_root, features_npz=features_npz, **kwargs)
