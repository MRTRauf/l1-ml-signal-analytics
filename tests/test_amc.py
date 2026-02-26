from __future__ import annotations

import numpy as np

from l1sa.amc import normalize_frames


def test_normalize_frames_zero_mean_unit_std() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(loc=3.0, scale=2.5, size=(8, 2, 128)).astype(np.float32)
    xn = normalize_frames(x)
    means = np.mean(xn, axis=2)
    stds = np.std(xn, axis=2)
    assert np.all(np.abs(means) < 1e-4)
    assert np.all(np.abs(stds - 1.0) < 1e-3)
