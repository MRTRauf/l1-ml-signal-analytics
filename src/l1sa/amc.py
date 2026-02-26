"""Optional PyTorch AMC baseline utilities."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import DatasetDict, ensure_output_dir


EPS = 1e-6


def _import_torch() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install it first (see requirements-ml.txt), "
            "then rerun `python -m l1sa amc-baseline ...`."
        ) from exc
    return torch, nn, DataLoader, TensorDataset, WeightedRandomSampler


def set_determinism(seed: int) -> None:
    np.random.seed(int(seed))


def normalize_frames(frames: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Per-frame per-channel normalization to approximately zero-mean unit-std."""
    x = np.asarray(frames, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"Expected shape (N, 2, L), got {x.shape!r}")
    mean = np.mean(x, axis=2, keepdims=True)
    std = np.std(x, axis=2, keepdims=True)
    return (x - mean) / np.maximum(std, float(eps))


def flatten_dataset(
    dataset: DatasetDict,
    max_per_cell: int = 1500,
    seed: int = 42,
    snr_min: int = -10,
    snr_max: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Flatten dict dataset into arrays with deterministic cap per (modulation, snr) cell.

    Returns:
    - X: (N, 2, 128) float32
    - y: (N,) int64 modulation index
    - snr: (N,) int64 label SNR
    - mods: deterministic modulation list (index mapping)
    """
    mods = sorted({mod for mod, _ in dataset})
    snrs = sorted({int(s) for _, s in dataset if int(snr_min) <= int(s) <= int(snr_max)})
    if not snrs:
        raise ValueError(f"No SNR labels in range [{snr_min}, {snr_max}].")

    mod_to_idx = {mod: idx for idx, mod in enumerate(mods)}
    rng = np.random.default_rng(int(seed))

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    snr_labels: list[np.ndarray] = []

    for snr_db in snrs:
        for mod in mods:
            key = (mod, int(snr_db))
            if key not in dataset:
                continue

            frames = np.asarray(dataset[key], dtype=np.float32)
            if frames.ndim != 3 or frames.shape[1] != 2:
                raise ValueError(f"Expected (N, 2, L) frames for key {key!r}, got {frames.shape!r}")

            n = int(frames.shape[0])
            if int(max_per_cell) > 0:
                take = min(int(max_per_cell), n)
            else:
                take = n

            if take < n:
                keep = np.sort(rng.choice(n, size=take, replace=False))
                frames = frames[keep]

            frames = normalize_frames(frames)
            xs.append(frames)
            ys.append(np.full(frames.shape[0], mod_to_idx[mod], dtype=np.int64))
            snr_labels.append(np.full(frames.shape[0], int(snr_db), dtype=np.int64))

    if not xs:
        raise ValueError("No usable frames were found after filtering/sampling.")

    X = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(ys, axis=0).astype(np.int64, copy=False)
    snr = np.concatenate(snr_labels, axis=0).astype(np.int64, copy=False)

    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm], snr[perm], mods


def _stratified_split_indices(y: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("split ratio must be in (0, 1).")

    rng = np.random.default_rng(int(seed))
    a_idx: list[np.ndarray] = []
    b_idx: list[np.ndarray] = []

    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        n = cls_idx.size
        if n <= 1:
            b_count = 1
        else:
            b_count = int(round(n * test_size))
            b_count = min(max(1, b_count), n - 1)
        b_idx.append(cls_idx[:b_count])
        a_idx.append(cls_idx[b_count:])

    a = np.concatenate(a_idx).astype(np.int64, copy=False)
    b = np.concatenate(b_idx).astype(np.int64, copy=False)
    rng.shuffle(a)
    rng.shuffle(b)
    return a, b


def _build_cnn_model(nn: Any, n_classes: int) -> Any:
    class CNNAMCNet(nn.Module):
        def __init__(self, classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(2, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(64, 96, kernel_size=3, padding=1),
                nn.BatchNorm1d(96),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(96, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, classes))

        def forward(self, x: Any) -> Any:
            z = self.features(x).squeeze(-1)
            return self.head(z)

    return CNNAMCNet(n_classes)


def _build_cldnn_model(nn: Any, n_classes: int) -> Any:
    class CLDNNLite(nn.Module):
        def __init__(self, classes: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(2, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.gru = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, classes))

        def forward(self, x: Any) -> Any:
            z = self.conv(x)
            z = z.transpose(1, 2)
            z, _ = self.gru(z)
            return self.head(z[:, -1, :])

    return CLDNNLite(n_classes)


def _build_rescnn_model(nn: Any, n_classes: int) -> Any:
    class ResidualBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int, stride: int = 1, p_drop: float = 0.25):
            super().__init__()
            self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(c_out)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(p_drop)
            self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm1d(c_out)
            if stride != 1 or c_in != c_out:
                self.skip = nn.Sequential(
                    nn.Conv1d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(c_out),
                )
            else:
                self.skip = nn.Identity()

        def forward(self, x: Any) -> Any:
            identity = self.skip(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.drop(out)
            out = self.bn2(self.conv2(out))
            return self.relu(out + identity)

    class ResCNNAMC(nn.Module):
        def __init__(self, classes: int):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(2, 64, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
            self.blocks = nn.Sequential(
                ResidualBlock(64, 64, stride=1, p_drop=0.2),
                ResidualBlock(64, 64, stride=1, p_drop=0.2),
                ResidualBlock(64, 64, stride=1, p_drop=0.2),
                ResidualBlock(64, 128, stride=2, p_drop=0.25),
                ResidualBlock(128, 128, stride=1, p_drop=0.25),
                ResidualBlock(128, 256, stride=2, p_drop=0.3),
                ResidualBlock(256, 256, stride=1, p_drop=0.3),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(256, classes))

        def forward(self, x: Any) -> Any:
            z = self.stem(x)
            z = self.blocks(z)
            z = self.pool(z).squeeze(-1)
            return self.head(z)

    return ResCNNAMC(n_classes)


def _build_model(nn: Any, n_classes: int, model_name: str) -> Any:
    name = str(model_name).lower()
    if name == "cnn":
        return _build_cnn_model(nn, n_classes)
    if name == "cldnn":
        return _build_cldnn_model(nn, n_classes)
    if name == "rescnn":
        return _build_rescnn_model(nn, n_classes)
    raise ValueError(f"Unknown model {model_name!r}. Use 'cnn', 'cldnn', or 'rescnn'.")


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

    return np.concatenate(targets), np.concatenate(preds)


def _evaluate(model: Any, loader: Any, device: Any, torch: Any) -> tuple[float, np.ndarray, np.ndarray]:
    y_true, y_pred = _predict(model, loader, device, torch)
    return float(np.mean(y_true == y_pred)), y_true, y_pred


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1_scores = []
    for cls in range(n_classes):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, EPS)
        f1_scores.append(float(f1))
    return float(np.mean(f1_scores))


def _plot_accuracy_vs_snr(acc_df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(
        acc_df["snr_db"].to_numpy(dtype=float),
        acc_df["accuracy"].to_numpy(dtype=float),
        marker="o",
        linewidth=1.5,
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


def _plot_confusion_at_snr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr_test: np.ndarray,
    mods: list[str],
    confusion_snr: int,
    out_path: Path,
) -> Path | None:
    mask = snr_test == int(confusion_snr)
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
    ax.set_title(f"AMC Confusion Matrix @ {int(confusion_snr)} dB")
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
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    max_per_cell: int = 1500,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.1,
    snr_min: int = -10,
    snr_max: int = 18,
    device: str = "cpu",
    model_name: str = "rescnn",
    confusion_snr: int = 10,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    grad_clip: float = 1.0,
    scheduler_name: str = "onecycle",
    patience: int = 5,
    min_epochs: int = 8,
) -> dict[str, Any]:
    torch, nn, DataLoader, TensorDataset, WeightedRandomSampler = _import_torch()
    set_determinism(seed)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    out = ensure_output_dir(out_dir)
    X, y, snr, mods = flatten_dataset(
        dataset,
        max_per_cell=max_per_cell,
        seed=seed,
        snr_min=snr_min,
        snr_max=snr_max,
    )

    train_all_idx, test_idx = _stratified_split_indices(y, test_size=test_size, seed=seed)
    train_rel_idx, val_rel_idx = _stratified_split_indices(
        y[train_all_idx],
        test_size=val_size,
        seed=seed + 1,
    )
    train_idx = train_all_idx[train_rel_idx]
    val_idx = train_all_idx[val_rel_idx]

    X_train = torch.from_numpy(X[train_idx]).float()
    y_train = torch.from_numpy(y[train_idx]).long()
    X_val = torch.from_numpy(X[val_idx]).float()
    y_val = torch.from_numpy(y[val_idx]).long()
    X_test = torch.from_numpy(X[test_idx]).float()
    y_test = torch.from_numpy(y[test_idx]).long()
    snr_test = snr[test_idx]

    if device.lower() == "cuda" and torch.cuda.is_available():
        device_obj = torch.device("cuda")
    else:
        if device.lower() == "cuda":
            print("CUDA requested but not available. Falling back to CPU.")
        device_obj = torch.device("cpu")

    model = _build_model(nn, n_classes=len(mods), model_name=model_name).to(device_obj)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(max(weight_decay, 0.0)),
    )

    loss_kwargs: dict[str, float] = {}
    if float(label_smoothing) > 0.0:
        loss_kwargs["label_smoothing"] = float(label_smoothing)
    try:
        loss_fn = nn.CrossEntropyLoss(**loss_kwargs)
    except TypeError:
        loss_fn = nn.CrossEntropyLoss()
        if loss_kwargs:
            print("label_smoothing is not supported in this torch version; using plain CrossEntropyLoss.")

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    y_train_np = y[train_idx]
    class_counts = np.bincount(y_train_np, minlength=len(mods)).astype(float)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    sample_weights = class_weights[y_train_np]

    train_gen = torch.Generator()
    train_gen.manual_seed(int(seed))
    train_sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=int(sample_weights.size),
        replacement=True,
        generator=train_gen,
    )

    train_loader = DataLoader(train_ds, batch_size=int(batch_size), sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False)

    scheduler = None
    sched_name = str(scheduler_name).lower()
    if sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-5
        )
    elif sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
            eta_min=max(float(lr) * 0.05, 1e-5),
        )
    elif sched_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(lr),
            steps_per_epoch=max(1, len(train_loader)),
            epochs=max(int(epochs), 1),
            pct_start=0.3,
            anneal_strategy="cos",
        )
    else:
        raise ValueError("scheduler_name must be one of: plateau, cosine, onecycle")

    best_val_acc = -1.0
    best_state = None
    patience_count = 0

    for epoch in range(int(epochs)):
        model.train()
        running_loss = 0.0
        train_true_batches: list[np.ndarray] = []
        train_pred_batches: list[np.ndarray] = []

        for xb, yb in train_loader:
            xb = xb.to(device_obj)
            yb = yb.to(device_obj)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()

            if float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            optimizer.step()
            if sched_name == "onecycle":
                scheduler.step()

            running_loss += float(loss.item()) * int(yb.size(0))
            train_true_batches.append(yb.detach().cpu().numpy())
            train_pred_batches.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

        y_train_true_epoch = np.concatenate(train_true_batches)
        y_train_pred_epoch = np.concatenate(train_pred_batches)
        train_acc = float(np.mean(y_train_true_epoch == y_train_pred_epoch))
        train_loss = running_loss / max(int(y_train_true_epoch.size), 1)

        val_acc, _, _ = _evaluate(model, val_loader, device_obj, torch)

        if sched_name == "plateau":
            scheduler.step(val_acc)
        elif sched_name == "cosine":
            scheduler.step()

        lr_now = float(optimizer.param_groups[0]["lr"])
        print(
            f"Epoch {epoch + 1}/{int(epochs)} - train_loss={train_loss:.4f} "
            f"- train_acc={train_acc:.4f} - val_acc={val_acc:.4f} - lr={lr_now:.6f}"
        )

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state = deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if (epoch + 1) >= int(min_epochs) and patience_count >= int(patience):
                print(f"Early stopping at epoch {epoch + 1} (best_val_acc={best_val_acc:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    overall_acc, y_true, y_pred = _evaluate(model, test_loader, device_obj, torch)

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

    high_mask = snr_test >= 10
    high_snr_acc = float(np.mean(y_true[high_mask] == y_pred[high_mask])) if np.any(high_mask) else float("nan")

    per_class_acc: dict[str, float] = {}
    for cls_idx, mod in enumerate(mods):
        m = y_true == cls_idx
        if int(np.sum(m)) == 0:
            per_class_acc[mod] = float("nan")
        else:
            per_class_acc[mod] = float(np.mean(y_pred[m] == y_true[m]))

    metrics = {
        "overall_accuracy": float(overall_acc),
        "macro_f1": float(_macro_f1(y_true, y_pred, n_classes=len(mods))),
        "high_snr_accuracy": float(high_snr_acc),
        "per_class_accuracy": per_class_acc,
        "label_map": {str(idx): mod for idx, mod in enumerate(mods)},
        "snr_min": int(snr_min),
        "snr_max": int(snr_max),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "best_val_accuracy": float(best_val_acc),
        "model": str(model_name),
        "scheduler": str(scheduler_name),
    }

    csv_path = out / "amc_accuracy_vs_snr.csv"
    png_path = out / "amc_accuracy_vs_snr.png"
    json_path = out / "amc_overall_metrics.json"
    acc_df.to_csv(csv_path, index=False)
    _plot_accuracy_vs_snr(acc_df, out_path=png_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    conf_path = _plot_confusion_at_snr(
        y_true=y_true,
        y_pred=y_pred,
        snr_test=snr_test,
        mods=mods,
        confusion_snr=confusion_snr,
        out_path=out / f"amc_confusion_snr_{int(confusion_snr)}.png",
    )

    return {
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "overall_accuracy": float(overall_acc),
        "macro_f1": float(metrics["macro_f1"]),
        "high_snr_accuracy": float(high_snr_acc),
        "mods": mods,
        "accuracy_csv": csv_path,
        "accuracy_png": png_path,
        "metrics_json": json_path,
        "confusion_png": conf_path,
        "label_map": metrics["label_map"],
    }
