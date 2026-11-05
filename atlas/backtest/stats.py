"""Natija tasodif emasligini tekshirish.

  * Stationary block bootstrap (Politis-Romano): Sharpe uchun 90% ishonch oralig'i, avtokorrelyatsiyani saqlaydi.
  * Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012): haqiqiy Sharpe > 0 ehtimoli, qiyshiqlik va
    "og'ir dum"ni hisobga oladi.
  * Deflated Sharpe Ratio (2014): ko'p sozlama sinab, eng yaxshisini tanlash (selection bias) uchun tuzatilgan PSR.
    Sinovlar soni = tekshirilgan barcha strategiya x chegara x variant x kechikish kombinatsiyalari.
  * Tasodifiy signal testi: model signallarining kunlar tartibi aralashtiriladi (pozitsiyalar soni, long/short
    nisbati, aylanma taxminan saqlanadi). Haqiqiy natija tasodifiy taqsimotning qayeriga tushadi?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER = 0.5772156649


def sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def block_bootstrap(r: pd.Series, n: int = 2000, block: int = 10, seed: int = 0) -> dict:
    x = r.dropna().to_numpy()
    T = len(x)
    rng = np.random.default_rng(seed)
    p = 1.0 / block
    jump = rng.random((n, T)) < p
    starts = rng.integers(T, size=(n, T))
    idx = np.empty((n, T), dtype=int)
    idx[:, 0] = starts[:, 0]
    for t in range(1, T):
        idx[:, t] = np.where(jump[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    xs = x[idx]
    sd = xs.std(1, ddof=1)
    out = np.where(sd > 0, xs.mean(1) / np.maximum(sd, 1e-12) * np.sqrt(252), 0.0)
    return {"sharpe_ci90": [float(np.quantile(out, 0.05)), float(np.quantile(out, 0.95))],
            "p_sharpe_le_0": float((out <= 0).mean())}


def psr(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """sr_benchmark — yillik Sharpe birligida."""
    x = r.dropna().to_numpy()
    T = len(x)
    sd = x.std(ddof=1)
    if T < 20 or sd == 0:
        return float("nan")
    sr = x.mean() / sd                                   # kunlik
    sr0 = sr_benchmark / np.sqrt(252)
    g3 = pd.Series(x).skew()
    g4 = pd.Series(x).kurt() + 3
    den = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / den))


def deflated_sr0(trial_sharpes: list[float], n_trials: int | None = None) -> float:
    """Kutilgan eng yuqori Sharpe (yillik), agar barcha sinovlar haqiqatda nol foydali bo'lsa."""
    s = np.asarray([v for v in trial_sharpes if np.isfinite(v)])
    N = n_trials or len(s)
    if N < 2 or len(s) < 2:
        return 0.0
    v = s.var(ddof=1)
    return float(np.sqrt(v) * ((1 - EULER) * norm.ppf(1 - 1 / N) + EULER * norm.ppf(1 - 1 / (N * np.e))))


def permutation_test(simulate_fn, s: np.ndarray, actual: float, n: int = 300, seed: int = 1) -> dict:
    """simulate_fn(signal) -> Sharpe. Kunlar tartibi aralashtirilgan signallar bilan taqsimot."""
    rng = np.random.default_rng(seed)
    T = len(s)
    vals = np.array([simulate_fn(s[rng.permutation(T)]) for _ in range(n)])
    return {"random_mean": float(vals.mean()), "random_p95": float(np.quantile(vals, 0.95)),
            "percentile": float((vals < actual).mean() * 100), "p_value": float((vals >= actual).mean()),
            "n": n}
