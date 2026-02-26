from __future__ import annotations

import numpy as np
import pytest

from l1sa.spectral import get_window, mean_psd, one_sided_psd


def _synthetic_frame(n: int = 256, fs: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(1234)
    t = np.arange(n, dtype=float) / fs
    tone = np.exp(1j * 2.0 * np.pi * 0.125 * t)
    noise = 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return tone + noise


def test_one_sided_psd_properties() -> None:
    frame = _synthetic_frame()
    freq, pxx, _ = one_sided_psd(frame, fs=1.0, window="hann")
    assert freq.shape == pxx.shape
    assert freq[0] == pytest.approx(0.0)
    assert np.all(np.diff(freq) >= 0.0)
    assert np.all(pxx >= 0.0)


def test_mean_psd_properties() -> None:
    frames = np.stack([_synthetic_frame() for _ in range(4)], axis=0)
    freq, pxx, _ = mean_psd(frames, fs=1.0, window="hamming")
    assert freq.shape == pxx.shape
    assert freq[0] == pytest.approx(0.0)
    assert np.all(np.diff(freq) >= 0.0)
    assert np.all(pxx >= 0.0)


@pytest.mark.parametrize("window", ["rect", "hann", "hamming", "blackman"])
def test_window_metrics(window: str) -> None:
    win = get_window(window, n=256, fs=1.0)
    assert np.isfinite(win.enbw_bins)
    assert np.isfinite(win.enbw_hz)
    assert np.isfinite(win.coherent_gain)
    assert win.enbw_bins > 0.0
    assert win.enbw_hz > 0.0
    if window == "rect":
        assert win.coherent_gain == pytest.approx(1.0)
    else:
        assert 0.0 < win.coherent_gain <= 1.0
