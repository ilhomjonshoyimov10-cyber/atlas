"""Hisobotlar: bitta sozlama uchun batafsil natija va barcha sozlamalar bo'yicha barqarorlik (robustness) jadvali."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from . import stats as S
from .engine import STRATEGIES, Market, simulate, signals

THRESHOLDS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]   # P(o'sish) 0.5 dan qancha uzoq bo'lsa savdo
COSTS = ["real", 0, 5, 10, 20]
VARIANT_NAMES = {"full": "Model (to'liq, yangilik bilan)", "full_scratch": "Model (oldindan o'qitishsiz)",
                 "no_news": "Model (yangiliksiz)", "price+season": "Narx + mavsum", "price": "Faqat narx",
                 "climatology": "Statistik baza (klimatologiya)", "buy_hold": "Sotib olib ushlash (benchmark)"}


def _cost(c):
    return None if c in (None, "real") else float(c)


def configs(mkt: Market) -> list[tuple[str, str, float | None]]:
    out = []
    for v in mkt.q:
        for st in STRATEGIES:
            for t in (THRESHOLDS if st != "cross_sectional" else [None]):
                out.append((v, st, t))
    return out


def robustness(mkt: Market) -> dict:
    """Har bir (variant, strategiya, chegara) x xarajat x kechikish uchun Sharpe va CAGR.
    Shu jadvaldan Deflated Sharpe uchun 'sinovlar soni' va ularning tarqoqligi olinadi."""
    grid = []
    sig_cache = {}
    for v, st, t in configs(mkt):
        s = signals(mkt, v, st, t or 0.0)
        sig_cache[(v, st, t)] = s
        for c in COSTS:
            for lag in (0, 1):
                r = simulate(mkt, s, _cost(c), lag)["r"]
                sm = M.summary(r)
                grid.append({"variant": v, "strategy": st, "threshold": t, "cost": c, "lag": lag,
                             "sharpe": round(sm.get("sharpe", float("nan")), 3),
                             "cagr": round(sm.get("cagr", float("nan")), 2),
                             "max_dd": round(sm.get("max_dd", float("nan")), 2)})
    bh = simulate(mkt, signals(mkt, "", "buy_hold"), None, 0)["r"]
    base = [g["sharpe"] for g in grid if g["cost"] == "real" and g["lag"] == 0]
    return {"grid": grid, "n_trials": len(base), "trial_sharpe_std": float(np.nanstd(base, ddof=1)),
            "sr0_deflated": S.deflated_sr0(base), "buy_hold_sharpe": round(M.summary(bh).get("sharpe", 0), 3),
            "_signals": sig_cache}


def _series(x: pd.Series, every: int = 5, nd: int = 5) -> list:
    x = x.dropna()
    if len(x) == 0:
        return []
    keep = list(range(0, len(x), every))
    if keep[-1] != len(x) - 1:
        keep.append(len(x) - 1)
    return [[str(x.index[i].date()), round(float(x.iloc[i]), nd)] for i in keep]


def build_report(mkt: Market, strategy: str = "direction", threshold: float = 0.01, cost="real",
                 rob: dict | None = None, perm_n: int = 200, perm_cache: dict | None = None) -> dict:
    """Bitta sozlama uchun barcha model variantlari + benchmark: ko'rsatkichlar, egri chiziqlar, statistika."""
    thr = None if strategy == "cross_sectional" else threshold
    sig_cache = (rob or {}).get("_signals", {})
    bh = simulate(mkt, signals(mkt, "", "buy_hold"), _cost(cost))
    out = {"segment": mkt.segment, "strategy": strategy, "strategy_name": STRATEGIES[strategy],
           "threshold": thr, "cost": cost, "period": [str(mkt.dates[0].date()), str(mkt.dates[-1].date())],
           "folds": mkt.folds, "n_assets": len(mkt.symbols), "variants": {}}
    sr0 = (rob or {}).get("sr0_deflated", 0.0)
    out["deflated"] = {"n_trials": (rob or {}).get("n_trials"), "sr0": sr0}
    for v in list(mkt.q) + ["buy_hold"]:
        if v == "buy_hold":
            s, sim = None, bh
        else:
            s = sig_cache.get((v, strategy, thr))
            if s is None:
                s = signals(mkt, v, strategy, thr or 0.0)
            sim = simulate(mkt, s, _cost(cost))
        r = sim["r"]
        st = M.summary(r, None if v == "buy_hold" else bh["r"], sim["turnover"], sim["exposure"])
        if not st:
            continue
        st["cost_drag"] = sim["cost_drag_ann"]
        lag1 = simulate(mkt, s, _cost(cost), lag=1)["r"] if s is not None else None
        stat = {**S.block_bootstrap(r), "psr": S.psr(r), "dsr": S.psr(r, sr0) if v != "buy_hold" else None,
                "sharpe_lag1": M.summary(lag1).get("sharpe") if lag1 is not None else None}
        if s is not None and perm_n:
            key = (mkt.segment, v, strategy, thr)
            if perm_cache is not None and key in perm_cache:
                stat["random"] = perm_cache[key]
            else:
                # xarajatsiz (gross) solishtiriladi: aralashtirilgan signal aylanmasi yuqori bo'ladi,
                # xarajat bilan solishtirsa model nohaq "yaxshi" ko'rinardi
                stat["random"] = S.permutation_test(
                    lambda x: S.sharpe(simulate(mkt, x, 0.0)["r"].to_numpy()), s,
                    S.sharpe(simulate(mkt, s, 0.0)["r"].to_numpy()), n=perm_n)
                if perm_cache is not None:
                    perm_cache[key] = stat["random"]
        contrib = pd.DataFrame(sim["contrib"], index=mkt.dates, columns=mkt.symbols)
        per_asset = []
        for a, g in zip(mkt.symbols, mkt.groups):
            x = contrib[a]
            if (x != 0).sum() > 20:
                per_asset.append({"symbol": a, "group": g, "pnl": round(float(x.sum() * 100), 3),
                                  "sharpe": round(float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0, 2)})
        per_class = contrib.T.groupby(mkt.groups).sum().T.sum().mul(100).round(3).to_dict()
        eq = (1 + r).cumprod() - 1
        out["variants"][v] = {
            "name": VARIANT_NAMES.get(v, v), "stats": st, "tests": stat,
            "equity": _series(eq * 100, nd=3), "drawdown": _series(M.drawdown(r) * 100, nd=2),
            "rolling_sharpe": _series(M.rolling_sharpe(r), nd=2),
            "monthly": M.monthly(r), "yearly": M.yearly(r),
            "per_asset": sorted(per_asset, key=lambda x: -x["pnl"]), "per_class": per_class,
        }
    return out


def clean(o):
    """JSON uchun: NaN/inf -> None (brauzer JSON.parse NaN ni qabul qilmaydi)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o
