"""I/O helpers for RadioML 2016.10A dictionary datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import numpy as np
import pandas as pd

DatasetKey = tuple[str, int]
DatasetDict = dict[DatasetKey, np.ndarray]


def load_radioml_dict(pkl_path: str | Path) -> DatasetDict:
    """Load and validate a RadioML dict dataset from a pickle file."""
    path = Path(pkl_path)
    with path.open("rb") as handle:
        raw: Any = pickle.load(handle, encoding="latin1")

    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict in {path}, got {type(raw)!r}")

    dataset: DatasetDict = {}
    for key, value in raw.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"Invalid key format: {key!r}")

        mod, snr = key
        if not isinstance(mod, str):
            raise ValueError(f"Invalid modulation key: {mod!r}")
        if not isinstance(snr, (int, np.integer)):
            raise ValueError(f"Invalid SNR key: {snr!r}")

        arr = np.asarray(value)
        if arr.ndim != 3 or arr.shape[1] != 2:
            raise ValueError(
                f"Expected value shape (N, 2, L) for key {key!r}, got {arr.shape!r}"
            )
        dataset[(mod, int(snr))] = arr

    return dataset


def available_mods(dataset: DatasetDict) -> list[str]:
    """Return sorted modulation names present in dataset."""
    return sorted({mod for mod, _ in dataset})


def available_snrs(dataset: DatasetDict) -> list[int]:
    """Return sorted SNR labels present in dataset."""
    return sorted({snr for _, snr in dataset})


def get_frames(dataset: DatasetDict, mod: str, snr: int) -> np.ndarray:
    """Fetch IQ frames for a specific modulation/SNR pair."""
    key = (mod, int(snr))
    if key not in dataset:
        mods = ", ".join(available_mods(dataset))
        snrs = ", ".join(str(x) for x in available_snrs(dataset))
        raise KeyError(f"Key {key!r} not found. mods=[{mods}] snrs=[{snrs}]")
    return dataset[key]


def iq_to_complex(frames_iq: np.ndarray) -> np.ndarray:
    """Convert (N, 2, L) IQ frames into complex array (N, L)."""
    frames = np.asarray(frames_iq)
    if frames.ndim != 3 or frames.shape[1] != 2:
        raise ValueError(f"Expected shape (N, 2, L), got {frames.shape!r}")
    return frames[:, 0, :] + 1j * frames[:, 1, :]


def summary_table(dataset: DatasetDict) -> pd.DataFrame:
    """Return one row per (mod, snr) with counts and frame shape."""
    rows: list[dict[str, int | str]] = []
    for (mod, snr), frames in sorted(dataset.items(), key=lambda x: (x[0][0], x[0][1])):
        rows.append(
            {
                "modulation": mod,
                "snr_db": int(snr),
                "num_frames": int(frames.shape[0]),
                "channels": int(frames.shape[1]),
                "frame_len": int(frames.shape[2]),
            }
        )
    return pd.DataFrame(rows)


def ensure_output_dir(path: str | Path) -> Path:
    """Create output directory if needed and return it."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out

