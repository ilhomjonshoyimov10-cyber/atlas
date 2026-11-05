"""Portfel ko'rsatkichlari. Kirish: kunlik oddiy (arifmetik) portfel daromadlari, pd.Series (indeks = sana)."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def drawdown(r: pd.Series) -> pd.Series:
    eq = (1 + r).cumprod()
    return eq / eq.cummax() - 1


def _dd_duration(dd: pd.Series) -> int:
    """Eng uzun 'cho'qqidan pastda' davr (savdo kunlari)."""
    best = cur = 0
    for v in (dd < -1e-12).to_numpy():
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def summary(r: pd.Series, bench: pd.Series | None = None, turnover: float | None = None,
            exposure: float | None = None) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    mu, sd = r.mean(), r.std(ddof=1)
    down = r[r < 0]
    dsd = np.sqrt((np.minimum(r, 0) ** 2).mean())
    eq = (1 + r).cumprod()
    years = len(r) / ANN
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    dd = drawdown(r)
    mdd = dd.min()
    wins, losses = r[r > 0], r[r < 0]
    q05 = r.quantile(0.05)
    out = {
        "days": int(len(r)), "total_ret": float((eq.iloc[-1] - 1) * 100), "cagr": float(cagr * 100),
        "ann_vol": float(sd * np.sqrt(ANN) * 100),
        "sharpe": float(mu / sd * np.sqrt(ANN)) if sd > 0 else 0.0,
        "sortino": float(mu / dsd * np.sqrt(ANN)) if dsd > 0 else 0.0,
        "max_dd": float(mdd * 100), "dd_days": _dd_duration(dd),
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else 0.0,
        "hit_rate": float((r > 0).mean() * 100),
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else None,
        "win_loss": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else None,
        "skew": float(r.skew()), "kurt": float(r.kurt() + 3),
        "var95": float(-q05 * 100), "cvar95": float(-r[r <= q05].mean() * 100),
        "best_day": float(r.max() * 100), "worst_day": float(r.min() * 100),
    }
    if len(down) == 0:
        out["sortino"] = None
    if turnover is not None:
        out["turnover_ann"] = float(turnover * ANN)          # yillik aylanma (portfel kapitaliga nisbatan, marta)
    if exposure is not None:
        out["exposure"] = float(exposure * 100)              # o'rtacha bozorda bo'lish (yalpi, % kapital)
    if bench is not None:
        b = bench.reindex(r.index).fillna(0)
        if b.std() > 0 and sd > 0:
            beta = float(np.cov(r, b)[0, 1] / b.var())
            act = r - b
            out.update({"beta": beta, "corr": float(np.corrcoef(r, b)[0, 1]),
                        "alpha_ann": float((r.mean() - beta * b.mean()) * ANN * 100),
                        "info_ratio": float(act.mean() / act.std() * np.sqrt(ANN)) if act.std() > 0 else 0.0})
    return out


def monthly(r: pd.Series) -> list[list]:
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    return [[int(y), int(mo), round(float(v) * 100, 3)] for (y, mo), v in m.items()]


def yearly(r: pd.Series) -> dict:
    y = (1 + r).groupby(r.index.year).prod() - 1
    return {str(k): round(float(v) * 100, 2) for k, v in y.items()}


def rolling_sharpe(r: pd.Series, window: int = 126) -> pd.Series:
    m = r.rolling(window, min_periods=window // 2)
    return (m.mean() / m.std() * np.sqrt(ANN)).dropna()
