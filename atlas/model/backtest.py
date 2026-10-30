"""Backtest: model signallari bo'yicha savdo qilinsa nima bo'lardi — FAQAT tashqi (out-of-sample) bashoratlarda.

Qoidalar (oddiy va shaffof):
  * Signal: 1 kunlik gorizont. O'sish ehtimoli p_up > 0.5 + chegara -> long, < 0.5 - chegara -> short, aks holda 0.
  * Hajm: volatillikka teskari (har bir aktiv portfelga bir xil risk qo'shadi), aktivlar teng vaznli.
  * Xarajat: pozitsiya o'zgarishiga `cost_bps` (komissiya + spread).
  * Daromad: ertangi haqiqiy log-qaytim (y_1 * sigma). Bashorat t kuni yopilishida, savdo t+1 yopilishigacha.
  * Taqqoslash: "sotib olib ushlash" (teng vaznli, vol bo'yicha) va yangiliksiz model.
Bu tadqiqot vositasi — natija investitsiya tavsiyasi emas. Slippage, likvidlik va soliqlar hisobga olinmagan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import REPORTS_DIR
from ..features.dataset import Panel


def p_up_from_q(q: np.ndarray) -> np.ndarray:
    """q: [N, 3] (q05, q50, q95) -> P(y>0), chiziqli interpolyatsiya, [0.03, 0.97] ga kesilgan."""
    lo, md, hi = q[:, 0], q[:, 1], q[:, 2]
    cdf0 = np.where(0 <= md, 0.05 + 0.45 * np.clip((0 - lo) / np.maximum(md - lo, 1e-9), 0, 1),
                    0.5 + 0.45 * np.clip((0 - md) / np.maximum(hi - md, 1e-9), 0, 1))
    cdf0 = np.where(0 < lo, 0.03, np.where(0 > hi, 0.97, cdf0))
    return 1 - cdf0


def remap_rows(panel: Panel, assets: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """(aktiv, sana) juftliklarini joriy paneldagi qator raqamlariga o'giradi; topilmasa -1."""
    key = pd.MultiIndex.from_arrays([panel.asset, pd.DatetimeIndex(panel.date)])
    idx = pd.Series(np.arange(len(key)), index=key)
    idx = idx[~idx.index.duplicated()]
    want = pd.MultiIndex.from_arrays([np.asarray(assets), pd.DatetimeIndex(dates)])
    return idx.reindex(want).fillna(-1).astype(int).to_numpy()


def _stats(r: pd.Series, turnover: float | None = None) -> dict:
    r = r.dropna()
    if len(r) < 5:
        return {}
    eq = r.cumsum()
    dd = eq - eq.cummax()
    ann = 252
    out = {"total_ret": float(np.expm1(r.sum()) * 100), "ann_ret": float(r.mean() * ann * 100),
           "ann_vol": float(r.std() * np.sqrt(ann) * 100),
           "sharpe": float(r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else 0.0,
           "max_dd": float(np.expm1(dd.min()) * 100), "hit_rate": float((r > 0).mean() * 100), "days": int(len(r))}
    if turnover is not None:
        out["turnover"] = float(turnover)
    return out


def run_backtest(panel: Panel, tag: str = "HOLDOUT", variant: str = "full", threshold: float = 0.05,
                 cost_bps: float = 5.0) -> dict:
    path = REPORTS_DIR / f"oos_{tag}.npz"
    if not path.exists():
        raise FileNotFoundError("Out-of-sample bashoratlar yo'q. Avval: python -m atlas.cli evaluate --final")
    z = np.load(path)
    rows = z["rows"]
    if "dates" in z.files:
        # Panelga yangi kunlar qo'shilganda qator raqamlari siljiydi -> (aktiv, sana) bo'yicha qayta topamiz
        ok = bool((rows < len(panel.date)).all()) and np.array_equal(panel.date[rows], z["dates"]) \
            and np.array_equal(panel.asset[rows], z["assets"])
        if not ok:
            rows = remap_rows(panel, z["assets"], z["dates"])
            if (rows < 0).any():
                raise FileNotFoundError("Saqlangan bashoratlar joriy ma'lumotga mos emas — evaluate --final ni qayta ishga tushiring")
    df = pd.DataFrame({"date": pd.DatetimeIndex(panel.date[rows]), "asset": panel.asset[rows],
                       "ret": panel.y[rows, 0] * panel.vol[rows]})   # ertangi log-qaytim
    target_vol = 0.01  # har bir pozitsiya kunlik ~1% risk
    lev = target_vol / panel.vol[rows]

    def strat(q: np.ndarray) -> tuple[pd.Series, float, pd.DataFrame]:
        p = p_up_from_q(q[:, 0])
        pos = np.where(p > 0.5 + threshold, 1.0, np.where(p < 0.5 - threshold, -1.0, 0.0)) * lev
        d = df.assign(pos=pos).sort_values(["asset", "date"])
        d["dpos"] = d.groupby("asset")["pos"].diff().abs().fillna(d["pos"].abs())
        d["pnl"] = d["pos"] * d["ret"] - d["dpos"] * cost_bps / 1e4
        daily = d.groupby("date")["pnl"].mean()
        return daily, float(d["dpos"].mean()), d

    res = {"tag": tag, "variant": variant, "threshold": threshold, "cost_bps": cost_bps,
           "period": [str(df.date.min().date()), str(df.date.max().date())], "series": {}, "stats": {}}
    variants = [v for v in (variant, "no_news", "price") if v in z.files]
    per_asset = None
    for v in dict.fromkeys(variants):
        daily, to, d = strat(z[v])
        res["stats"][v] = _stats(daily, to)
        res["series"][v] = [[str(k.date()), float(x)] for k, x in daily.cumsum().items()]
        if v == variant:
            per_asset = d
    bh = df.assign(pnl=df["ret"] * lev).groupby("date")["pnl"].mean()
    res["stats"]["buy_hold"] = _stats(bh)
    res["series"]["buy_hold"] = [[str(k.date()), float(x)] for k, x in bh.cumsum().items()]
    if per_asset is not None:
        rows_ = []
        for a, g in per_asset.groupby("asset"):
            st = _stats(g.set_index("date")["pnl"])
            if st:
                rows_.append({"symbol": panel.symbols[a], **{k: st[k] for k in ("sharpe", "total_ret", "hit_rate")}})
        res["per_asset"] = sorted(rows_, key=lambda r: -r["sharpe"])
    return res
