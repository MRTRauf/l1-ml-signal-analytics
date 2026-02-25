"""Dataset-level exploratory analysis helpers."""

from __future__ import annotations

import pandas as pd

from .io import DatasetDict, available_mods, available_snrs, summary_table


def class_distribution(dataset: DatasetDict) -> pd.Series:
    """Total frame count per modulation."""
    table = summary_table(dataset)
    return table.groupby("modulation", as_index=True)["num_frames"].sum().sort_index()


def snr_distribution(dataset: DatasetDict) -> pd.Series:
    """Total frame count per SNR label."""
    table = summary_table(dataset)
    return table.groupby("snr_db", as_index=True)["num_frames"].sum().sort_index()


def class_snr_matrix(dataset: DatasetDict) -> pd.DataFrame:
    """Matrix of frame counts with modulation rows and SNR columns."""
    table = summary_table(dataset)
    pivot = (
        table.pivot_table(
            index="modulation",
            columns="snr_db",
            values="num_frames",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return pivot


def dataset_summary_text(dataset: DatasetDict) -> str:
    """Human-readable summary for CLI output."""
    table = summary_table(dataset)
    mods = available_mods(dataset)
    snrs = available_snrs(dataset)
    total_frames = int(table["num_frames"].sum())
    unique_shapes = (
        table[["channels", "frame_len"]].drop_duplicates().astype(int).to_dict("records")
    )

    lines = [
        "RadioML Dataset Summary",
        f"- Modulations ({len(mods)}): {', '.join(mods)}",
        f"- SNR labels ({len(snrs)}): {', '.join(str(x) for x in snrs)}",
        f"- Total (mod, snr) groups: {len(table)}",
        f"- Total frames: {total_frames}",
        f"- Unique (channels, frame_len): {unique_shapes}",
    ]
    return "\n".join(lines)

