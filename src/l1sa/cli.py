"""Command line interface for l1sa."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .eda import dataset_summary_text
from .io import (
    available_mods,
    available_snrs,
    ensure_output_dir,
    get_frames,
    iq_to_complex,
    load_radioml_dict,
    summary_table,
)
from .metrics import SUPPORTED_EVM_MODS, evm_curves, is_supported_evm_mod, metric_curve_vs_snr
from .spectral import get_window, mean_psd
from .viz import (
    plot_class_snr_distribution,
    plot_constellation,
    plot_constellation_sweep,
    plot_evm_vs_snr,
    plot_inspect_frame,
    plot_metric_vs_snr,
    plot_psd,
    plot_psd_sweep,
    plot_window_demo,
)


DEFAULT_PKL = "data/RML2016.10a_dict.pkl"
DEFAULT_OUTDIR = "outputs"


def _safe_tag(value: str) -> str:
    return str(value).replace(" ", "")


def _load_dataset(pkl_path: str):
    return load_radioml_dict(pkl_path)


def _resolve_snrs(requested: list[int] | None, dataset) -> list[int]:
    if requested:
        return [int(x) for x in requested]
    return available_snrs(dataset)


def _resolve_mod(dataset, mod: str) -> str:
    lookup = {name.upper(): name for name in available_mods(dataset)}
    key = mod.upper()
    if key not in lookup:
        raise KeyError(f"Unknown modulation {mod!r}. Available: {', '.join(sorted(lookup.values()))}")
    return lookup[key]


def _resolve_evm_mod(dataset, mod: str) -> str:
    key = str(mod).upper()
    candidates = [mod]
    if key == "16QAM":
        candidates.append("QAM16")
    elif key == "QAM16":
        candidates.append("16QAM")
    elif key == "64QAM":
        candidates.append("QAM64")
    elif key == "QAM64":
        candidates.append("64QAM")

    for candidate in candidates:
        try:
            return _resolve_mod(dataset, candidate)
        except KeyError:
            continue
    return _resolve_mod(dataset, mod)


def cmd_summary(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.pkl)
    table = summary_table(dataset)
    print(dataset_summary_text(dataset))
    print()
    print(table.to_string(index=False))
    return 0


def cmd_plot_distribution(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.pkl)
    out = plot_class_snr_distribution(dataset, out_dir=args.outdir)
    print(f"Generated: {out}")
    print("Figure: class/SNR distribution.")
    return 0


def cmd_plot_constellation(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.pkl)
    mod = _resolve_mod(dataset, args.mod)
    frames = iq_to_complex(get_frames(dataset, mod=mod, snr=args.snr))
    out = plot_constellation(frames, mod=mod, snr=args.snr, n_points=args.n, out_dir=args.outdir)
    print(f"Generated: {out}")
    print(f"Constellation for mod={mod}, snr={args.snr}, n={args.n}.")
    return 0


def cmd_plot_constellation_sweep(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.pkl)
    mod = _resolve_mod(dataset, args.mod)
    snrs = _resolve_snrs(args.snrs, dataset)
    out = plot_constellation_sweep(
        dataset=dataset,
        mod=mod,
        snrs=snrs,
        n_points_per_snr=args.n,
        out_dir=args.outdir,
    )
    print(f"Generated: {out}")
    print(f"Constellation sweep for mod={mod} across {len(snrs)} SNR values.")
    return 0


def cmd_plot_psd(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.pkl)
    mod = _resolve_mod(dataset, args.mod)
    out_dir = ensure_output_dir(args.outdir)
    frames = iq_to_complex(get_frames(dataset, mod=mod, snr=args.snr))
    out, freq, pxx = plot_psd(
        complex_frames=frames,
        mod=mod,
        snr=args.snr,
        window=args.window,
        out_dir=out_dir,
        fs=args.fs,
    )
    csv_path = out_dir / f"psd_{_safe_tag(mod)}_{args.snr}_{args.window.lower()}.csv"
    pd.DataFrame({"f": freq, "psd": pxx}).to_csv(csv_path, index=False)
    win = get_window(args.window, n=frames.shape[1], fs=args.fs)
    print(f"Saved PNG: {out}; Saved CSV: {csv_path}")
    print(
        f"PSD for mod={mod}, snr={args.snr}, window={args.window}. "
        f"coherent_gain={win.coherent_gain:.6f}, ENBW={win.enbw_bins:.3f} bins ({win.enbw_hz:.6f} Hz)"
    )
    return 0


def cmd_plot_psd_sweep(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.pkl)
    mod = _resolve_mod(dataset, args.mod)
    snrs = _resolve_snrs(args.snrs, dataset)
    out_dir = ensure_output_dir(args.outdir)
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for snr in snrs:
        frames = iq_to_complex(get_frames(dataset, mod=mod, snr=snr))
        freq, pxx, _ = mean_psd(frames, fs=args.fs, window=args.window)
        curves[int(snr)] = (freq, pxx)

    out = plot_psd_sweep(
        dataset=dataset,
        mod=mod,
        snrs=snrs,
        window=args.window,
        out_dir=out_dir,
        fs=args.fs,
        curves=curves,
    )
    csv_paths = []
    for snr in snrs:
        freq, pxx = curves[int(snr)]
        csv_path = out_dir / f"psd_{_safe_tag(mod)}_{snr}_{args.window.lower()}.csv"
        pd.DataFrame({"f": freq, "psd": pxx}).to_csv(csv_path, index=False)
        csv_paths.append(csv_path)

    n = iq_to_complex(get_frames(dataset, mod=mod, snr=snrs[0])).shape[1]
    win = get_window(args.window, n=n, fs=args.fs)
    print(f"Saved PNG: {out}; Saved CSV: {', '.join(str(path) for path in csv_paths)}")
    print(
        f"PSD sweep for mod={mod}, window={args.window}, snr_count={len(snrs)}. "
        f"coherent_gain={win.coherent_gain:.6f}, ENBW={win.enbw_bins:.3f} bins ({win.enbw_hz:.6f} Hz)"
    )
    return 0


def cmd_window_demo(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.pkl)
    mod = _resolve_mod(dataset, args.mod)
    frames = iq_to_complex(get_frames(dataset, mod=mod, snr=args.snr))
    if args.frame_index < 0 or args.frame_index >= frames.shape[0]:
        raise IndexError(f"frame-index out of range [0, {frames.shape[0]-1}]")

    frame = frames[args.frame_index]
    out = plot_window_demo(one_frame=frame, mod=mod, snr=args.snr, out_dir=args.outdir, fs=args.fs)
    print(f"Generated: {out}")
    for window in ("rect", "hann", "hamming", "blackman"):
        win = get_window(window, n=frame.size, fs=args.fs)
        print(
            f"  {window:8s} coherent_gain={win.coherent_gain:.6f}, "
            f"ENBW={win.enbw_bins:.3f} bins ({win.enbw_hz:.6f} Hz)"
        )
    print(f"Window leakage demo for mod={mod}, snr={args.snr}, frame_index={args.frame_index}.")
    return 0


def cmd_plot_quality(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.pkl)
    mod = _resolve_mod(dataset, args.mod)
    out_dir = ensure_output_dir(args.outdir)

    power_df = metric_curve_vs_snr(dataset, mod=mod, metric_name="power", fs=args.fs)
    snr_df = metric_curve_vs_snr(dataset, mod=mod, metric_name="snr_est", fs=args.fs)

    power_png = plot_metric_vs_snr(
        power_df,
        mod=mod,
        metric_label="Frame Power",
        out_path=out_dir / f"power_vs_snr_{mod}.png",
    )
    snr_png = plot_metric_vs_snr(
        snr_df,
        mod=mod,
        metric_label="Empirical SNR Estimate (dB)",
        out_path=out_dir / f"snr_est_vs_snr_{mod}.png",
    )

    power_csv = out_dir / "quality_power_vs_snr.csv"
    snr_csv = out_dir / "quality_snr_est_vs_snr.csv"
    pd.DataFrame(
        {
            "snr_db": power_df["snr_label_db"].to_numpy(dtype=int),
            "mean_power": power_df["metric_mean"].to_numpy(dtype=float),
            "std_power": power_df["metric_std"].to_numpy(dtype=float),
            "n_frames": power_df["n_frames"].to_numpy(dtype=int),
        }
    ).to_csv(power_csv, index=False)
    pd.DataFrame(
        {
            "snr_db": snr_df["snr_label_db"].to_numpy(dtype=int),
            "snr_est_mean": snr_df["metric_mean"].to_numpy(dtype=float),
            "snr_est_std": snr_df["metric_std"].to_numpy(dtype=float),
            "n_frames": snr_df["n_frames"].to_numpy(dtype=int),
        }
    ).to_csv(snr_csv, index=False)

    print(f"Saved PNG: {power_png}; Saved CSV: {power_csv}")
    print(f"Saved PNG: {snr_png}; Saved CSV: {snr_csv}")
    print(f"Quality curves for mod={mod}: power vs SNR and SNR_est vs SNR.")
    return 0


def cmd_plot_evm_snr(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.pkl)
    out_dir = ensure_output_dir(args.outdir)
    mods = []
    for mod in args.mods:
        if not is_supported_evm_mod(mod):
            continue
        resolved = _resolve_evm_mod(dataset, mod)
        if resolved not in mods:
            mods.append(resolved)
    evm_df = evm_curves(dataset, mods=mods)
    if evm_df.empty:
        raise ValueError(f"No supported EVM modulations provided. Supported: {SUPPORTED_EVM_MODS}")

    out = plot_evm_vs_snr(evm_df, out_dir=out_dir)

    csv_paths = []
    for mod in sorted(evm_df["modulation"].unique().tolist()):
        part = evm_df[evm_df["modulation"] == mod].sort_values("snr_label_db")
        evm_rms = part["evm_rms_mean"].to_numpy(dtype=float)
        csv_path = out_dir / f"evm_vs_snr_{_safe_tag(mod)}.csv"
        pd.DataFrame(
            {
                "snr_db": part["snr_label_db"].to_numpy(dtype=int),
                "evm_rms": evm_rms,
                "evm_db": 20.0 * np.log10(np.maximum(evm_rms, 1e-15)),
                "n_frames": part["n_frames"].to_numpy(dtype=int),
            }
        ).to_csv(csv_path, index=False)
        csv_paths.append(csv_path)

    print(f"Saved PNG: {out}; Saved CSV: {', '.join(str(path) for path in csv_paths)}")
    print(f"EVM vs SNR for mods: {', '.join(sorted(evm_df['modulation'].unique().tolist()))}.")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.pkl)
    out_dir = ensure_output_dir(args.outdir)
    mod = _resolve_mod(dataset, args.mod)
    frames = iq_to_complex(get_frames(dataset, mod=mod, snr=args.snr))
    if args.idx < 0 or args.idx >= frames.shape[0]:
        raise IndexError(f"idx out of range [0, {frames.shape[0]-1}]")

    frame = frames[args.idx]
    png_path, freq, pxx = plot_inspect_frame(
        one_frame=frame,
        mod=mod,
        snr=args.snr,
        idx=args.idx,
        window=args.window,
        out_dir=out_dir,
        fs=args.fs,
    )
    csv_path = out_dir / f"inspect_psd_{_safe_tag(mod)}_{args.snr}_idx{args.idx}_{args.window.lower()}.csv"
    pd.DataFrame({"f": freq, "psd": pxx}).to_csv(csv_path, index=False)
    print(f"Inspect (mod={mod}, snr={args.snr}, idx={args.idx}): saved {png_path} and {csv_path}")
    return 0


def cmd_make_core_figures(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.pkl)
    mod = _resolve_mod(dataset, args.mod)
    snrs = _resolve_snrs(args.snrs, dataset)

    generated = []
    generated.append(plot_class_snr_distribution(dataset, out_dir=args.outdir))
    generated.append(
        plot_constellation_sweep(
            dataset=dataset,
            mod=mod,
            snrs=snrs,
            n_points_per_snr=args.n,
            out_dir=args.outdir,
        )
    )
    generated.append(
        plot_psd_sweep(
            dataset=dataset,
            mod=mod,
            snrs=snrs,
            window=args.window,
            out_dir=args.outdir,
            fs=args.fs,
        )
    )
    frame = iq_to_complex(get_frames(dataset, mod=mod, snr=args.window_demo_snr))[args.frame_index]
    generated.append(
        plot_window_demo(
            one_frame=frame,
            mod=mod,
            snr=args.window_demo_snr,
            out_dir=args.outdir,
            fs=args.fs,
        )
    )
    evm_mods = []
    for mod in args.evm_mods:
        if not is_supported_evm_mod(mod):
            continue
        resolved = _resolve_evm_mod(dataset, mod)
        if resolved not in evm_mods:
            evm_mods.append(resolved)
    evm_df = evm_curves(dataset, mods=evm_mods)
    if evm_df.empty:
        raise ValueError("No supported EVM modulations were provided for core figures.")
    generated.append(plot_evm_vs_snr(evm_df, out_dir=args.outdir))

    print("Generated core figures:")
    for path in generated:
        print(f"- {path}")
    print(
        "Core set includes: class/SNR distribution, constellation sweep, PSD sweep, "
        "window demo, and EVM vs SNR."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build top-level parser."""
    parser = argparse.ArgumentParser(prog="l1sa", description="Layer-1 signal analytics toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p_summary = sub.add_parser("summary", help="Print dataset summary")
    p_summary.add_argument("--pkl", type=str, default=DEFAULT_PKL)
    p_summary.set_defaults(func=cmd_summary)

    p_dist = sub.add_parser("plot-distribution", help="Plot class/SNR distribution")
    p_dist.add_argument("--pkl", type=str, default=DEFAULT_PKL)
    p_dist.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p_dist.set_defaults(func=cmd_plot_distribution)

    p_const = sub.add_parser("plot-constellation", help="Plot constellation for one (mod, snr)")
    p_const.add_argument("--pkl", type=str, default=DEFAULT_PKL)
    p_const.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p_const.add_argument("--mod", type=str, required=True)
    p_const.add_argument("--snr", type=int, required=True)
    p_const.add_argument("--n", type=int, default=2000, help="Number of points to plot")
    p_const.set_defaults(func=cmd_plot_constellation)

    p_const_sw = sub.add_parser(
        "plot-constellation-sweep",
        help="Plot constellation across multiple SNR values",
    )
    p_const_sw.add_argument("--pkl", type=str, default=DEFAULT_PKL)
    p_const_sw.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p_const_sw.add_argument("--mod", type=str, required=True)
    p_const_sw.add_argument("--snrs", type=int, nargs="+", default=None)
    p_const_sw.add_argument("--n", type=int, default=2000, help="Points per SNR panel")
    p_const_sw.set_defaults(func=cmd_plot_constellation_sweep)

    p_psd = sub.add_parser("plot-psd", help="Plot mean PSD for one (mod, snr)")
    p_psd.add_argument("--pkl", type=str, default=DEFAULT_PKL)
    p_psd.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p_psd.add_argument("--mod", type=str, required=True)
    p_psd.add_argument("--snr", type=int, required=True)
    p_psd.add_argument("--window", type=str, default="hann", choices=("rect", "hann", "hamming", "blackman"))
    p_psd.add_argument("--fs", type=float, default=1.0, help="Sampling rate for PSD axis scaling")
    p_psd.set_defaults(func=cmd_plot_psd)

    p_psd_sw = sub.add_parser("plot-psd-sweep", help="Plot PSD across multiple SNR values")
    p_psd_sw.add_argument("--pkl", type=str, default=DEFAULT_PKL)
    p_psd_sw.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p_psd_sw.add_argument("--mod", type=str, required=True)
    p_psd_sw.add_argument("--snrs", type=int, nargs="+", default=None)
    p_psd_sw.add_argument("--window", type=str, default="hann", choices=("rect", "hann", "hamming", "blackman"))
    p_psd_sw.add_argument("--fs", type=float, default=1.0)
    p_psd_sw.set_defaults(func=cmd_plot_psd_sweep)

    p_win = sub.add_parser("window-demo", help="Leakage demo comparing windows on same frame")
    p_win.add_argument("--pkl", type=str, default=DEFAULT_PKL)
    p_win.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p_win.add_argument("--mod", type=str, required=True)
    p_win.add_argument("--snr", type=int, required=True)
    p_win.add_argument("--frame-index", type=int, default=0)
    p_win.add_argument("--fs", type=float, default=1.0)
    p_win.set_defaults(func=cmd_window_demo)

    p_qual = sub.add_parser("plot-quality", help="Plot power-vs-SNR and SNR_est-vs-SNR")
    p_qual.add_argument("--pkl", type=str, default=DEFAULT_PKL)
    p_qual.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p_qual.add_argument("--mod", type=str, required=True)
    p_qual.add_argument("--fs", type=float, default=1.0)
    p_qual.set_defaults(func=cmd_plot_quality)

    p_evm = sub.add_parser("plot-evm-snr", help="Plot EVM vs SNR for one or more modulations")
    p_evm.add_argument("--pkl", type=str, default=DEFAULT_PKL)
    p_evm.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p_evm.add_argument("--mods", type=str, nargs="+", default=["QPSK", "16QAM", "64QAM"])
    p_evm.set_defaults(func=cmd_plot_evm_snr)

    p_inspect = sub.add_parser("inspect", help="Inspect one frame with waveform, constellation, and PSD")
    p_inspect.add_argument("--pkl", type=str, required=True)
    p_inspect.add_argument("--mod", type=str, required=True)
    p_inspect.add_argument("--snr", type=int, required=True)
    p_inspect.add_argument("--idx", type=int, default=0)
    p_inspect.add_argument(
        "--window",
        type=str,
        default="hann",
        choices=("rect", "hann", "hamming", "blackman"),
    )
    p_inspect.add_argument("--fs", type=float, default=1.0)
    p_inspect.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p_inspect.set_defaults(func=cmd_inspect)

    p_core = sub.add_parser("make-core-figures", help="Generate the 5 acceptance-core figures")
    p_core.add_argument("--pkl", type=str, default=DEFAULT_PKL)
    p_core.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p_core.add_argument("--mod", type=str, default="QPSK")
    p_core.add_argument("--snrs", type=int, nargs="+", default=[-20, -10, 0, 10, 18])
    p_core.add_argument("--n", type=int, default=2000)
    p_core.add_argument("--window", type=str, default="hann", choices=("rect", "hann", "hamming", "blackman"))
    p_core.add_argument("--window-demo-snr", type=int, default=0)
    p_core.add_argument("--frame-index", type=int, default=0)
    p_core.add_argument("--evm-mods", type=str, nargs="+", default=["QPSK", "16QAM", "64QAM"])
    p_core.add_argument("--fs", type=float, default=1.0)
    p_core.set_defaults(func=cmd_make_core_figures)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
