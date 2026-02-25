"""Matplotlib visualizations for L1 signal analytics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .eda import class_distribution, class_snr_matrix
from .io import DatasetDict, ensure_output_dir, get_frames, iq_to_complex
from .metrics import metric_curve_vs_snr
from .spectral import mean_psd, one_sided_psd, to_db


def _save(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_class_snr_distribution(dataset: DatasetDict, out_dir: str | Path) -> Path:
    """Save class and (class, snr) distribution figure."""
    out = ensure_output_dir(out_dir)
    out_path = out / "class_snr_distribution.png"

    class_counts = class_distribution(dataset)
    matrix = class_snr_matrix(dataset)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    axes[0].bar(class_counts.index.tolist(), class_counts.values, color="#2A9D8F")
    axes[0].set_title("Frame Count per Modulation")
    axes[0].set_xlabel("Modulation")
    axes[0].set_ylabel("Frames")
    axes[0].tick_params(axis="x", rotation=45)

    im = axes[1].imshow(matrix.values, aspect="auto", cmap="viridis")
    axes[1].set_title("Frame Count Heatmap (Modulation x SNR)")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Modulation")
    axes[1].set_xticks(np.arange(matrix.shape[1]), matrix.columns.astype(str).tolist())
    axes[1].set_yticks(np.arange(matrix.shape[0]), matrix.index.tolist())
    plt.colorbar(im, ax=axes[1], label="Frames")

    return _save(fig, out_path)


def plot_constellation(
    complex_frames: np.ndarray,
    mod: str,
    snr: int,
    n_points: int,
    out_dir: str | Path,
) -> Path:
    """Save single constellation plot."""
    out = ensure_output_dir(out_dir)
    out_path = out / f"constellation_{mod}_{snr}.png"

    x = np.asarray(complex_frames, dtype=np.complex128).ravel()
    n_keep = min(int(n_points), x.size)
    pts = x[:n_keep]

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(pts.real, pts.imag, s=6, alpha=0.35, c="#264653", edgecolors="none")
    ax.set_title(f"Constellation: {mod} @ {snr} dB (n={n_keep})")
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", "box")
    return _save(fig, out_path)


def plot_constellation_sweep(
    dataset: DatasetDict,
    mod: str,
    snrs: list[int],
    n_points_per_snr: int,
    out_dir: str | Path,
) -> Path:
    """Save constellation-vs-SNR grid for one modulation."""
    out = ensure_output_dir(out_dir)
    out_path = out / f"constellation_sweep_{mod}.png"

    n = len(snrs)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows), constrained_layout=True)
    axs = np.atleast_1d(axes).ravel()

    for idx, snr in enumerate(snrs):
        frames = iq_to_complex(get_frames(dataset, mod=mod, snr=snr))
        pts = frames.ravel()
        n_keep = min(int(n_points_per_snr), pts.size)
        pts = pts[:n_keep]
        ax = axs[idx]
        ax.scatter(pts.real, pts.imag, s=4, alpha=0.3, c="#1D3557", edgecolors="none")
        ax.set_title(f"SNR {snr} dB")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")

    for idx in range(len(snrs), len(axs)):
        axs[idx].axis("off")

    fig.suptitle(f"Constellation Across SNR: {mod}", fontsize=13)
    return _save(fig, out_path)


def plot_psd(
    complex_frames: np.ndarray,
    mod: str,
    snr: int,
    window: str,
    out_dir: str | Path,
    fs: float = 1.0,
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Save mean PSD for one modulation/SNR group."""
    out = ensure_output_dir(out_dir)
    out_path = out / f"psd_{mod}_{snr}_{window.lower()}.png"

    freq, pxx, _ = mean_psd(complex_frames, fs=fs, window=window)
    pxx_db = to_db(pxx)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(freq, pxx_db, color="#E76F51", linewidth=1.5)
    ax.set_title(f"Mean PSD: {mod} @ {snr} dB [{window.lower()}]")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("PSD (dB)")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path), freq, pxx


def plot_psd_sweep(
    dataset: DatasetDict,
    mod: str,
    snrs: list[int],
    window: str,
    out_dir: str | Path,
    fs: float = 1.0,
    curves: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> Path:
    """Save PSD-vs-SNR overlay for one modulation."""
    out = ensure_output_dir(out_dir)
    out_path = out / f"psd_sweep_{mod}_{window.lower()}.png"

    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    for snr in snrs:
        if curves is not None and int(snr) in curves:
            freq, pxx = curves[int(snr)]
        else:
            frames = iq_to_complex(get_frames(dataset, mod=mod, snr=snr))
            freq, pxx, _ = mean_psd(frames, fs=fs, window=window)
        ax.plot(freq, to_db(pxx), linewidth=1.2, label=f"{snr} dB")

    ax.set_title(f"PSD Across SNR: {mod} [{window.lower()}]")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("PSD (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    return _save(fig, out_path)


def plot_window_demo(
    one_frame: np.ndarray,
    mod: str,
    snr: int,
    out_dir: str | Path,
    fs: float = 1.0,
) -> Path:
    """Leakage demo: same signal with rect/hann/hamming/blackman windows."""
    out = ensure_output_dir(out_dir)
    out_path = out / f"window_demo_{mod}_{snr}.png"

    windows = ("rect", "hann", "hamming", "blackman")
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in windows:
        freq, pxx, _ = one_sided_psd(one_frame, fs=fs, window=name)
        db = to_db(pxx)
        db = db - np.max(db)
        curves[name] = (freq, db)

    rect_freq, rect_db = curves["rect"]
    peak_idx = int(np.argmax(rect_db))
    i0 = max(0, peak_idx - 20)
    i1 = min(rect_db.size, peak_idx + 21)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    for name, (freq, db) in curves.items():
        axes[0].plot(freq, db, linewidth=1.2, label=name)
        axes[1].plot(freq[i0:i1], db[i0:i1], linewidth=1.2, label=name)

    axes[0].set_title("Window Leakage Comparison (Full Band)")
    axes[0].set_xlabel("Frequency")
    axes[0].set_ylabel("Relative PSD (dB)")
    axes[0].set_ylim(-120, 3)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Main-Lobe Zoom")
    axes[1].set_xlabel("Frequency")
    axes[1].set_ylabel("Relative PSD (dB)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Window Demo: {mod} @ {snr} dB", fontsize=13)
    return _save(fig, out_path)


def plot_metric_vs_snr(
    metric_df: pd.DataFrame,
    mod: str,
    metric_label: str,
    out_path: Path,
) -> Path:
    """Generic mean+-std curve plot."""
    fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    x = metric_df["snr_label_db"].to_numpy(dtype=float)
    y = metric_df["metric_mean"].to_numpy(dtype=float)
    y_std = metric_df["metric_std"].to_numpy(dtype=float)

    ax.plot(x, y, marker="o", linewidth=1.4, color="#2A9D8F")
    ax.fill_between(x, y - y_std, y + y_std, color="#2A9D8F", alpha=0.2)
    ax.set_title(f"{metric_label} vs SNR: {mod}")
    ax.set_xlabel("SNR label (dB)")
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


def plot_quality_curves(dataset: DatasetDict, mod: str, out_dir: str | Path, fs: float = 1.0) -> list[Path]:
    """Save power-vs-SNR and SNR_est-vs-SNR plots."""
    out = ensure_output_dir(out_dir)
    power_df = metric_curve_vs_snr(dataset, mod=mod, metric_name="power", fs=fs)
    snr_df = metric_curve_vs_snr(dataset, mod=mod, metric_name="snr_est", fs=fs)

    p1 = plot_metric_vs_snr(
        power_df,
        mod=mod,
        metric_label="Frame Power",
        out_path=out / f"power_vs_snr_{mod}.png",
    )
    p2 = plot_metric_vs_snr(
        snr_df,
        mod=mod,
        metric_label="Empirical SNR Estimate (dB)",
        out_path=out / f"snr_est_vs_snr_{mod}.png",
    )
    return [p1, p2]


def plot_evm_vs_snr(evm_df: pd.DataFrame, out_dir: str | Path) -> Path:
    """Save EVM-vs-SNR curves for one or more modulations."""
    out = ensure_output_dir(out_dir)
    mods = sorted(evm_df["modulation"].unique().tolist())
    mod_tag = "-".join(mods) if mods else "none"
    out_path = out / f"evm_vs_snr_{mod_tag}.png"

    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    for mod in mods:
        part = evm_df[evm_df["modulation"] == mod].sort_values("snr_label_db")
        x = part["snr_label_db"].to_numpy(dtype=float)
        y = part["evm_rms_mean"].to_numpy(dtype=float)
        y_std = part["evm_rms_std"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.4, label=mod)
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)

    ax.set_title("EVM vs SNR")
    ax.set_xlabel("SNR label (dB)")
    ax.set_ylabel("EVM (RMS)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, out_path)


def _safe_tag(value: str) -> str:
    return str(value).replace(" ", "")


def plot_inspect_frame(
    one_frame: np.ndarray,
    mod: str,
    snr: int,
    idx: int,
    window: str,
    out_dir: str | Path,
    fs: float = 1.0,
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Save compact single-frame diagnostic figure and return PSD data."""
    out = ensure_output_dir(out_dir)
    out_path = out / f"inspect_{_safe_tag(mod)}_{snr}_idx{idx}_{window.lower()}.png"

    frame = np.asarray(one_frame, dtype=np.complex128).ravel()
    n = frame.size
    sample_idx = np.arange(n, dtype=int)
    freq, pxx, _ = one_sided_psd(frame, fs=fs, window=window)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    axes[0].plot(sample_idx, frame.real, linewidth=1.1, label="I")
    axes[0].plot(sample_idx, frame.imag, linewidth=1.1, label="Q")
    axes[0].set_title("Time Waveform")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(frame.real, frame.imag, s=10, alpha=0.45, c="#1D3557", edgecolors="none")
    axes[1].set_title("Constellation")
    axes[1].set_xlabel("I")
    axes[1].set_ylabel("Q")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect("equal", "box")

    axes[2].semilogy(freq, np.maximum(pxx, 1e-15), linewidth=1.2, color="#E76F51")
    axes[2].set_title(f"Single-Sided PSD [{window.lower()}]")
    axes[2].set_xlabel("Frequency")
    axes[2].set_ylabel("PSD")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Inspect: {mod} @ {snr} dB (idx={idx})", fontsize=12)
    return _save(fig, out_path), freq, pxx