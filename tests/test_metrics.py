from __future__ import annotations

import numpy as np

from l1sa.metrics import evm_rms_frame


def _synthetic_qpsk_symbols(n: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(2026)
    constellation = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex128) / np.sqrt(2.0)
    idx = rng.integers(0, len(constellation), size=n)
    return constellation[idx]


def test_evm_ideal_vs_noisy_qpsk() -> None:
    clean = _synthetic_qpsk_symbols()
    clean_evm = evm_rms_frame(clean, mod="QPSK")
    assert clean_evm < 1e-10

    rng = np.random.default_rng(99)
    noise = 0.2 * (rng.standard_normal(clean.size) + 1j * rng.standard_normal(clean.size))
    noisy = clean + noise
    noisy_evm = evm_rms_frame(noisy, mod="QPSK")
    assert noisy_evm > clean_evm + 1e-3
