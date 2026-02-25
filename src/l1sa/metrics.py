"""Signal quality metrics for IQ frames."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .io import DatasetDict, available_snrs, get_frames, iq_to_complex
from .spectral import one_sided_psd

EPS = 1e-15
SUPPORTED_EVM_MODS = ("QPSK", "16QAM", "64QAM")
EVM_MOD_ALIASES = {
    "QPSK": "QPSK",
    "16QAM": "16QAM",
    "QAM16": "16QAM",
    "64QAM": "64QAM",
    "QAM64": "64QAM",
}


def canonical_evm_mod(mod: str) -> str:
    """Normalize modulation names used by EVM-related methods."""
    key = str(mod).upper()
    if key not in EVM_MOD_ALIASES:
        raise ValueError(f"EVM constellation not implemented for {mod!r}")
    return EVM_MOD_ALIASES[key]


def is_supported_evm_mod(mod: str) -> bool:
    """Whether modulation has EVM decision-directed support."""
    return str(mod).upper() in EVM_MOD_ALIASES


def sample_power(complex_frames: np.ndarray) -> np.ndarray:
    """Per-sample power |x|^2 for frames shaped (N, L)."""
    x = np.asarray(complex_frames, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError(f"Expected shape (N, L), got {x.shape!r}")
    return np.abs(x) ** 2


def frame_power(complex_frames: np.ndarray) -> np.ndarray:
    """Per-frame average power."""
    return np.mean(sample_power(complex_frames), axis=1)


def get_constellation(mod: str) -> np.ndarray:
    """Return normalized ideal constellation points for EVM/SNR decisions."""
    m = canonical_evm_mod(mod)
    if m == "QPSK":
        points = np.array(
            [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
            dtype=np.complex128,
        )
    elif m == "16QAM":
        axis = np.array([-3, -1, 1, 3], dtype=float)
        points = np.array([i + 1j * q for i in axis for q in axis], dtype=np.complex128)
    elif m == "64QAM":
        axis = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=float)
        points = np.array([i + 1j * q for i in axis for q in axis], dtype=np.complex128)

    rms = np.sqrt(np.mean(np.abs(points) ** 2))
    return points / (rms + EPS)


def nearest_reference(samples: np.ndarray, constellation: np.ndarray) -> np.ndarray:
    """Nearest-point hard decision for complex samples."""
    x = np.asarray(samples, dtype=np.complex128).ravel()
    distances = np.abs(x[:, None] - constellation[None, :])
    return constellation[np.argmin(distances, axis=1)]


def evm_rms_frame(frame: np.ndarray, mod: str) -> float:
    """
    EVM (RMS) for one frame using nearest-point decisions.

    Limitations:
    - This assumes samples are symbol-synchronous.
    - Uses only hard nearest-point decisions (no equalizer/timing recovery).
    """
    x = np.asarray(frame, dtype=np.complex128).ravel()
    constellation = get_constellation(mod)

    x_norm = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + EPS)
    ref_symbols = nearest_reference(x_norm, constellation)

    # Remove common complex gain/phase via least-squares fit.
    gain = np.vdot(ref_symbols, x_norm) / (np.vdot(ref_symbols, ref_symbols) + EPS)
    ref_aligned = gain * ref_symbols
    err = x_norm - ref_aligned
    evm = np.sqrt(np.mean(np.abs(err) ** 2) / (np.mean(np.abs(ref_aligned) ** 2) + EPS))
    return float(evm)


def evm_rms_per_frame(complex_frames: np.ndarray, mod: str) -> np.ndarray:
    """Vectorized EVM over frames for supported modulations."""
    x = np.asarray(complex_frames, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError(f"Expected shape (N, L), got {x.shape!r}")
    return np.array([evm_rms_frame(frame, mod) for frame in x], dtype=float)


def _decision_directed_snr_db(frame: np.ndarray, mod: str) -> float:
    """Decision-directed SNR estimate for supported EVM constellations."""
    x = np.asarray(frame, dtype=np.complex128).ravel()
    x_norm = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + EPS)

    constellation = get_constellation(mod)
    ref = nearest_reference(x_norm, constellation)
    gain = np.vdot(ref, x_norm) / (np.vdot(ref, ref) + EPS)
    signal = gain * ref
    noise = x_norm - signal

    sig_pow = np.mean(np.abs(signal) ** 2)
    noise_pow = np.mean(np.abs(noise) ** 2)
    return float(10.0 * np.log10((sig_pow + EPS) / (noise_pow + EPS)))


def _spectral_floor_snr_db(frame: np.ndarray, fs: float = 1.0, q: float = 0.2) -> float:
    """
    Blind SNR estimate from PSD floor.

    Estimator:
    - Compute one-sided PSD (Hann window).
    - Estimate noise floor from lower-q quantile of bins.
    - Convert floor to total noise power over 0..fs/2.
    - SNR = (total_power - noise_power) / noise_power.
    """
    x = np.asarray(frame, dtype=np.complex128).ravel()
    _, pxx, _ = one_sided_psd(x, fs=fs, window="hann")
    noise_floor = float(np.quantile(pxx, q))

    noise_power = noise_floor * (fs / 2.0)
    total_power = float(np.mean(np.abs(x) ** 2))
    signal_power = max(total_power - noise_power, EPS)
    return float(10.0 * np.log10(signal_power / max(noise_power, EPS)))


def estimate_snr_per_frame(complex_frames: np.ndarray, mod: str, fs: float = 1.0) -> np.ndarray:
    """Empirical SNR estimate per frame."""
    x = np.asarray(complex_frames, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError(f"Expected shape (N, L), got {x.shape!r}")

    if is_supported_evm_mod(mod):
        estimator: Callable[[np.ndarray], float] = lambda frame: _decision_directed_snr_db(
            frame, mod=mod
        )
    else:
        estimator = lambda frame: _spectral_floor_snr_db(frame, fs=fs)

    return np.array([estimator(frame) for frame in x], dtype=float)


def metric_curve_vs_snr(
    dataset: DatasetDict,
    mod: str,
    metric_name: str,
    fs: float = 1.0,
) -> pd.DataFrame:
    """Aggregate metric vs SNR label for one modulation."""
    rows: list[dict[str, float | int | str]] = []
    for snr in available_snrs(dataset):
        frames = iq_to_complex(get_frames(dataset, mod=mod, snr=snr))
        if metric_name == "power":
            values = frame_power(frames)
        elif metric_name == "snr_est":
            values = estimate_snr_per_frame(frames, mod=mod, fs=fs)
        elif metric_name == "evm":
            if not is_supported_evm_mod(mod):
                raise ValueError(f"EVM is only supported for: {SUPPORTED_EVM_MODS}")
            values = evm_rms_per_frame(frames, mod=mod)
        else:
            raise ValueError(f"Unknown metric_name={metric_name!r}")

        rows.append(
            {
                "modulation": mod,
                "snr_label_db": int(snr),
                "metric_mean": float(np.mean(values)),
                "metric_std": float(np.std(values)),
                "n_frames": int(values.size),
            }
        )
    return pd.DataFrame(rows).sort_values("snr_label_db").reset_index(drop=True)


def evm_curves(dataset: DatasetDict, mods: list[str]) -> pd.DataFrame:
    """Concatenate EVM-vs-SNR curves for multiple modulations."""
    parts = []
    for mod in mods:
        if not is_supported_evm_mod(mod):
            continue
        part = metric_curve_vs_snr(dataset, mod=mod, metric_name="evm")
        part = part.rename(columns={"metric_mean": "evm_rms_mean", "metric_std": "evm_rms_std"})
        parts.append(part)
    if not parts:
        return pd.DataFrame(
            columns=["modulation", "snr_label_db", "evm_rms_mean", "evm_rms_std", "n_frames"]
        )
    return pd.concat(parts, ignore_index=True)
