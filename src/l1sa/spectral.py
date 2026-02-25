"""Spectral analysis utilities for IQ data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-15
WINDOWS = ("rect", "hann", "hamming", "blackman")


@dataclass(frozen=True)
class WindowInfo:
    """Window properties useful for spectral diagnostics."""

    name: str
    values: np.ndarray
    coherent_gain: float
    enbw_bins: float
    enbw_hz: float


def get_window(name: str, n: int, fs: float = 1.0) -> WindowInfo:
    """Build window and compute coherent gain + ENBW."""
    lname = name.lower()
    if lname == "rect":
        values = np.ones(n, dtype=float)
    elif lname == "hann":
        values = np.hanning(n)
    elif lname == "hamming":
        values = np.hamming(n)
    elif lname == "blackman":
        values = np.blackman(n)
    else:
        raise ValueError(f"Unsupported window {name!r}. Valid: {WINDOWS}")

    coherent_gain = float(np.mean(values))
    enbw_bins = float(n * np.sum(values**2) / ((np.sum(values) ** 2) + EPS))
    enbw_hz = float(enbw_bins * fs / n)
    return WindowInfo(
        name=lname,
        values=values,
        coherent_gain=coherent_gain,
        enbw_bins=enbw_bins,
        enbw_hz=enbw_hz,
    )


def one_sided_psd(
    x: np.ndarray, fs: float = 1.0, window: str = "hann", nfft: int | None = None
) -> tuple[np.ndarray, np.ndarray, WindowInfo]:
    """Single-sided PSD for complex IQ using FFT and 2x interior scaling."""
    frame = np.asarray(x, dtype=np.complex128).ravel()
    if frame.ndim != 1:
        raise ValueError("Input must be 1D.")
    n = frame.size
    nfft = int(n if nfft is None else nfft)
    if nfft <= 0:
        raise ValueError("nfft must be positive.")

    win = get_window(window, n=n, fs=fs)
    # Coherent-gain correction keeps sinusoid amplitude unbiased across windows.
    w_corr = win.values / (win.coherent_gain + EPS)

    xw = frame * w_corr
    xfft = np.fft.fft(xw, n=nfft)
    scale = fs * np.sum(w_corr**2)
    psd_two_sided = (np.abs(xfft) ** 2) / (scale + EPS)

    if nfft % 2 == 0:
        pos_bins = np.arange(0, (nfft // 2) + 1, dtype=int)
    else:
        pos_bins = np.arange(0, (nfft + 1) // 2, dtype=int)

    freqs = pos_bins * (fs / nfft)
    psd_one_sided = psd_two_sided[pos_bins].real.copy()

    if nfft % 2 == 0:
        if psd_one_sided.size > 2:
            psd_one_sided[1:-1] *= 2.0
    elif psd_one_sided.size > 1:
        psd_one_sided[1:] *= 2.0

    return freqs, psd_one_sided, win


def mean_psd(
    frames: np.ndarray, fs: float = 1.0, window: str = "hann", nfft: int | None = None
) -> tuple[np.ndarray, np.ndarray, WindowInfo]:
    """Mean single-sided PSD over a batch of frames."""
    x = np.asarray(frames, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError(f"Expected shape (N, L), got {x.shape!r}")
    n_frames, n = x.shape
    nfft = int(n if nfft is None else nfft)
    if nfft <= 0:
        raise ValueError("nfft must be positive.")

    win = get_window(window, n=n, fs=fs)
    w_corr = win.values / (win.coherent_gain + EPS)

    xw = x * w_corr[None, :]
    xfft = np.fft.fft(xw, n=nfft, axis=1)
    scale = fs * np.sum(w_corr**2)
    psd_two_sided = (np.abs(xfft) ** 2) / (scale + EPS)

    if nfft % 2 == 0:
        pos_bins = np.arange(0, (nfft // 2) + 1, dtype=int)
    else:
        pos_bins = np.arange(0, (nfft + 1) // 2, dtype=int)

    freqs = pos_bins * (fs / nfft)
    psd_one_sided = psd_two_sided[:, pos_bins].real

    if nfft % 2 == 0:
        if psd_one_sided.shape[1] > 2:
            psd_one_sided[:, 1:-1] *= 2.0
    elif psd_one_sided.shape[1] > 1:
        psd_one_sided[:, 1:] *= 2.0

    mean_spec = np.mean(psd_one_sided, axis=0)
    return freqs, mean_spec, win


def to_db(x: np.ndarray) -> np.ndarray:
    """Convert linear power values to dB."""
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), EPS))

