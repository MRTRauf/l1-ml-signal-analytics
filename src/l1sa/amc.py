"""Optional PyTorch AMC baseline utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import DatasetDict, ensure_output_dir


def _import_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install it first (see requirements-ml.txt), "
            "then rerun `python -m l1sa amc-baseline ...`."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def flatten_dataset(
    dataset: DatasetDict,
    max_per_snr: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Flatten dict dataset into arrays.

    Returns:
    - X: (N, 2, 128) float32
    - y: (N,) int64 modulation index
    - snr: (N,) int64 label SNR
    - mods: modulation list ordered by index
    """
    mods = sorted({mod for mod, _ in dataset})
    snrs = sorted({int(snr) for _, snr in dataset})
    mod_to_idx = {mod: idx for idx, mod in enumerate(mods)}
    rng = np.random.default_rng(int(seed))

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    snr_labels: list[np.ndarray] = []

    for snr_db in snrs:
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        s_parts: list[np.ndarray] = []

        for mod in mods:
            key = (mod, int(snr_db))
            if key not in dataset:
                continue
            frames = np.asarray(dataset[key], dtype=np.float32)
            if frames.ndim != 3 or frames.shape[1] != 2:
                raise ValueError(f"Expected (N, 2, L) frames for key {key!r}, got {frames.shape!r}")
            n = int(frames.shape[0])
            x_parts.append(frames)
            y_parts.append(np.full(n, mod_to_idx[mod], dtype=np.int64))
            s_parts.append(np.full(n, int(snr_db), dtype=np.int64))

        if not x_parts:
            continue

        x_snr = np.concatenate(x_parts, axis=0)
        y_snr = np.concatenate(y_parts, axis=0)
        s_snr = np.concatenate(s_parts, axis=0)

        if max_per_snr > 0 and x_snr.shape[0] > max_per_snr:
            keep = np.sort(rng.choice(x_snr.shape[0], size=int(max_per_snr), replace=False))
            x_snr = x_snr[keep]
            y_snr = y_snr[keep]
            s_snr = s_snr[keep]

        xs.append(x_snr)
        ys.append(y_snr)
        snr_labels.append(s_snr)

    if not xs:
        raise ValueError("No usable frames were found in dataset.")

    X = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(ys, axis=0).astype(np.int64, copy=False)
    snr = np.concatenate(snr_labels, axis=0).astype(np.int64, copy=False)

    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm], snr[perm], mods


def _stratified_split_indices(
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1).")

    rng = np.random.default_rng(int(seed))
    train_idx: list[np.ndarray] = []
    test_idx: list[np.ndarray] = []

    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        n_cls = cls_idx.size

        if n_cls <= 1:
            test_count = 1
        else:
            test_count = int(round(n_cls * test_size))
            test_count = min(max(1, test_count), n_cls - 1)

        test_idx.append(cls_idx[:test_count])
        train_idx.append(cls_idx[test_count:])

    train = np.concatenate(train_idx).astype(np.int64, copy=False)
    test = np.concatenate(test_idx).astype(np.int64, copy=False)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _build_model(nn: Any, n_classes: int) -> Any:
    class TinyAMCNet(nn.Module):
        def __init__(self, classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(2, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Linear(64, classes)

        def forward(self, x: Any) -> Any:
            z = self.features(x)
            z = z.squeeze(-1)
            return self.classifier(z)

    return TinyAMCNet(n_classes)


def _predict(model: Any, loader: Any, device: Any, torch: Any) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred)
            targets.append(yb.numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    return y_true, y_pred


def _plot_accuracy_vs_snr(acc_df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(
        acc_df["snr_db"].to_numpy(dtype=float),
        acc_df["accuracy"].to_numpy(dtype=float),
        marker="o",
        linewidth=1.4,
        color="#1D3557",
    )
    ax.set_title("AMC Accuracy vs SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_confusion_at_snr_zero(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr_test: np.ndarray,
    mods: list[str],
    out_path: Path,
) -> Path | None:
    mask = snr_test == 0
    if not np.any(mask):
        return None

    yt = y_true[mask]
    yp = y_pred[mask]
    n_cls = len(mods)
    conf = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(yt, yp):
        conf[int(t), int(p)] += 1

    fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
    im = ax.imshow(conf, cmap="Blues", aspect="auto")
    ax.set_title("AMC Confusion Matrix @ 0 dB")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ticks = np.arange(n_cls)
    ax.set_xticks(ticks, mods, rotation=45, ha="right")
    ax.set_yticks(ticks, mods)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Frames")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_amc_baseline(
    dataset: DatasetDict,
    out_dir: str | Path,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 1e-3,
    max_per_snr: int = 2000,
    seed: int = 42,
    test_size: float = 0.2,
    device: str = "cpu",
) -> dict[str, Any]:
    torch, nn, DataLoader, TensorDataset = _import_torch()

    out = ensure_output_dir(out_dir)
    X, y, snr, mods = flatten_dataset(dataset, max_per_snr=max_per_snr, seed=seed)
    train_idx, test_idx = _stratified_split_indices(y, test_size=test_size, seed=seed)

    X_train = torch.from_numpy(X[train_idx]).float()
    y_train = torch.from_numpy(y[train_idx]).long()
    X_test = torch.from_numpy(X[test_idx]).float()
    y_test = torch.from_numpy(y[test_idx]).long()
    snr_test = snr[test_idx]

    if device.lower() == "cuda":
        if torch.cuda.is_available():
            device_obj = torch.device("cuda")
        else:
            print("CUDA requested but not available. Falling back to CPU.")
            device_obj = torch.device("cpu")
    else:
        device_obj = torch.device("cpu")

    model = _build_model(nn, n_classes=len(mods)).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_gen = torch.Generator()
    train_gen.manual_seed(int(seed))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, generator=train_gen)
    test_loader = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False)

    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device_obj)
            yb = yb.to(device_obj)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            bs = int(yb.shape[0])
            total_loss += float(loss.item()) * bs
            total_count += bs

        y_true_epoch, y_pred_epoch = _predict(model, test_loader, device_obj, torch)
        val_acc = float(np.mean(y_true_epoch == y_pred_epoch))
        mean_loss = total_loss / max(total_count, 1)
        print(f"Epoch {epoch + 1}/{int(epochs)} - loss={mean_loss:.4f} - val_acc={val_acc:.4f}")

    y_true, y_pred = _predict(model, test_loader, device_obj, torch)
    overall_acc = float(np.mean(y_true == y_pred))

    rows = []
    for snr_db in sorted(np.unique(snr_test)):
        mask = snr_test == snr_db
        rows.append(
            {
                "snr_db": int(snr_db),
                "accuracy": float(np.mean(y_true[mask] == y_pred[mask])),
                "n_frames": int(np.sum(mask)),
            }
        )
    acc_df = pd.DataFrame(rows).sort_values("snr_db").reset_index(drop=True)

    csv_path = out / "amc_accuracy_vs_snr.csv"
    png_path = out / "amc_accuracy_vs_snr.png"
    acc_df.to_csv(csv_path, index=False)
    _plot_accuracy_vs_snr(acc_df, out_path=png_path)
    conf_path = _plot_confusion_at_snr_zero(
        y_true=y_true,
        y_pred=y_pred,
        snr_test=snr_test,
        mods=mods,
        out_path=out / "amc_confusion_snr_0.png",
    )

    return {
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "overall_accuracy": overall_acc,
        "mods": mods,
        "accuracy_csv": csv_path,
        "accuracy_png": png_path,
        "confusion_png": conf_path,
    }
