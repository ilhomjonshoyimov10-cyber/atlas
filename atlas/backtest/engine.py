"""Savdo simulyatsiyasi (vektorlashgan, sana x aktiv matritsalari).

Qoidalar — barchasi oldindan belgilangan, natijaga qarab tanlanmagan:
  * Faqat out-of-sample bashoratlar: walk-forward foldlari (har biri faqat o'tmishda o'qitilgan) yoki final holdout.
  * Qaror t kuni 20:00 UTC da (yangiliklar shu vaqtgacha), pozitsiya t yopilishidan t+1 yopilishigacha ushlanadi.
    Ijro kechikishi testi (lag=1): pozitsiya bir kun kechikib ochiladi — signal "eskirsa" natija qancha yomonlashadi.
  * Har bir aktiv bir xil risk oladi: vazn ∝ 1 / kunlik volatillik (ex-ante, faqat o'tmish narxlaridan).
  * Portfel volatillik maqsadi: yillik 10%. Masshtab o'tgan 60 kunlik haqiqiy volatillikdan (1 kun lag bilan),
    leverage 3x bilan cheklangan. Shu sababli barcha strategiyalar va benchmark bir xil risk darajasida solishtiriladi.
  * Xarajat: har bir pozitsiya o'zgarishiga (notional) aktiv sinfiga xos bir tomonlama xarajat
    (komissiya + yarim spread + slippage) yoki foydalanuvchi bergan bir xil bps.
  * Faqat ish kunlari: kripto dam olish kunlari savdo qilinmaydi (soddalik, natijani yaxshilamaydi).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ASSETS, REPORTS_DIR

# bir tomonlama xarajat, bps (komissiya + yarim spread + slippage) — konservativ, retail darajasi
CLASS_COST_BPS = {"forex": 1.0, "crypto": 8.0, "equity": 2.0, "index": 1.5, "energy": 3.0,
                  "agri": 4.0, "metal": 2.0, "rates": 1.5}
STRATEGIES = {
    "direction": "Yo'nalish (long/short)",
    "long_only": "Faqat long",
    "cross_sectional": "Kesishma reyting (top 20% long / past 20% short)",
}
VOL_TARGET = 0.10
MAX_LEV = 3.0
RISK_PER_ASSET = 0.01


def p_up(q: np.ndarray) -> np.ndarray:
    """q[..., 3] (q05, q50, q95) -> P(qaytim > 0), chiziqli interpolyatsiya, [0.03, 0.97]."""
    lo, md, hi = q[..., 0], q[..., 1], q[..., 2]
    cdf0 = np.where(0 <= md, 0.05 + 0.45 * np.clip((0 - lo) / np.maximum(md - lo, 1e-9), 0, 1),
                    0.5 + 0.45 * np.clip((0 - md) / np.maximum(hi - md, 1e-9), 0, 1))
    cdf0 = np.where(0 < lo, 0.03, np.where(0 > hi, 0.97, cdf0))
    return 1 - cdf0


@dataclass
class Market:
    segment: str
    dates: pd.DatetimeIndex          # [T] ish kunlari
    symbols: list[str]               # [A]
    groups: list[str]                # [A]
    ret: np.ndarray                  # [T, A] t -> t+1 oddiy qaytim (NaN = savdo yo'q)
    ret_next: np.ndarray             # [T, A] t+1 -> t+2 (lag testi uchun)
    vol: np.ndarray                  # [T, A] ex-ante kunlik volatillik
    q: dict                          # variant -> [T, A, 3] h1 kvantillar (NaN = bashorat yo'q)
    folds: list                      # [(nom, boshlanish, tugash)]


def _segment_files(segment: str) -> list:
    if segment == "holdout":
        return [REPORTS_DIR / "oos_HOLDOUT.npz"]
    return sorted(REPORTS_DIR.glob("oos_fold*.npz"), key=lambda p: int(p.stem.replace("oos_fold", "") or 0))


def load_market(panel, segment: str = "wf") -> Market:
    from ..model.backtest import remap_rows

    files = [f for f in _segment_files(segment) if f.exists()]
    if not files:
        raise FileNotFoundError(f"'{segment}' uchun out-of-sample bashoratlar yo'q (reports/oos_*.npz)")
    parts, folds, variants = [], [], None
    for f in files:
        z = np.load(f)
        rows = remap_rows(panel, z["assets"], z["dates"]) if "dates" in z.files else z["rows"]
        ok = rows >= 0
        if not ok.any():
            continue
        vs = [v for v in z.files if v not in ("rows", "dates", "assets")]
        variants = vs if variants is None else [v for v in variants if v in vs]
        d = pd.DataFrame({"row": rows[ok]})
        for v in vs:
            d[v] = list(z[v][ok][:, 0, :])          # h1 kvantillari
        parts.append(d)
        dts = panel.date[rows[ok]]
        folds.append([f.stem.replace("oos_", ""), str(pd.Timestamp(dts.min()).date()), str(pd.Timestamp(dts.max()).date())])
    if not parts:
        raise FileNotFoundError("Saqlangan bashoratlar joriy ma'lumotga mos emas — evaluate ni qayta ishga tushiring")
    df = pd.concat(parts, ignore_index=True)
    rows = df["row"].to_numpy()
    df["date"] = pd.DatetimeIndex(panel.date[rows])
    df["asset"] = panel.asset[rows]
    df = df.drop_duplicates(["asset", "date"], keep="last")
    df = df[df["date"].dt.dayofweek < 5].sort_values(["asset", "date"])
    rows = df["row"].to_numpy()
    df["ret"] = np.expm1(panel.y[rows, 0] * panel.vol[rows])
    df["vol"] = panel.vol[rows]
    df["ret_next"] = df.groupby("asset")["ret"].shift(-1)

    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    A = len(panel.symbols)
    ti = dates.get_indexer(df["date"])
    ai = df["asset"].to_numpy()

    def mat(col):
        m = np.full((len(dates), A), np.nan)
        m[ti, ai] = df[col].to_numpy(float)
        return m

    q = {}
    for v in variants:
        m = np.full((len(dates), A, 3), np.nan)
        m[ti, ai] = np.stack(df[v].to_numpy())
        q[v] = m
    grp = {a.symbol: a.group for a in ASSETS}
    return Market(segment, dates, list(panel.symbols), [grp.get(s, "other") for s in panel.symbols],
                  mat("ret"), mat("ret_next"), mat("vol"), q, folds)


def signals(mkt: Market, variant: str, strategy: str, threshold: float = 0.04) -> np.ndarray:
    """[T, A] pozitsiya yo'nalishi/kuchi, -1..1 (NaN = bashorat yo'q)."""
    if strategy == "buy_hold":
        return np.where(np.isnan(mkt.ret), np.nan, 1.0)
    q = mkt.q[variant]
    has = ~np.isnan(q[..., 1])
    if strategy in ("direction", "long_only"):
        p = p_up(np.nan_to_num(q))
        s = np.where(p > 0.5 + threshold, 1.0, np.where(p < 0.5 - threshold, -1.0, 0.0))
        if strategy == "long_only":
            s = np.maximum(s, 0.0)
        return np.where(has, s, np.nan)
    if strategy == "cross_sectional":
        score = q[..., 1] / np.maximum(q[..., 2] - q[..., 0], 1e-9)       # kutilgan qaytim / noaniqlik
        s = np.zeros_like(score)
        for t in range(len(score)):
            ok = np.where(has[t])[0]
            if len(ok) < 5:
                continue
            k = max(1, int(round(len(ok) * 0.2)))
            order = ok[np.argsort(score[t, ok])]
            s[t, order[-k:]] = 1.0
            s[t, order[:k]] = -1.0
        return np.where(has, s, np.nan)
    raise ValueError(f"Noma'lum strategiya: {strategy}")


def cost_vector(mkt: Market, cost_bps: float | None) -> np.ndarray:
    if cost_bps is None:          # "real": aktiv sinfi bo'yicha
        return np.array([CLASS_COST_BPS.get(g, 3.0) for g in mkt.groups]) / 1e4
    return np.full(len(mkt.symbols), cost_bps / 1e4)


def simulate(mkt: Market, s: np.ndarray, cost_bps: float | None = None, lag: int = 0,
             vol_target: float = VOL_TARGET, max_lev: float = MAX_LEV) -> dict:
    """s: [T, A] signal. Natija: kunlik portfel qaytimi (xarajatdan keyin), vaznlar, aylanma va hissalar."""
    T, A = s.shape
    ret = mkt.ret_next if lag else mkt.ret
    sig = pd.DataFrame(s).ffill().fillna(0.0).to_numpy()     # savdo yo'q kuni pozitsiya o'zgarmaydi
    vol = pd.DataFrame(mkt.vol).ffill().bfill().to_numpy()
    lev = RISK_PER_ASSET / np.maximum(vol, 1e-4)
    w_raw = sig * lev / A                                    # teng risk byudjeti (universal = A aktiv)
    r_raw = np.nansum(w_raw * np.nan_to_num(ret), axis=1)
    # ex-ante (korrelyatsiyasiz) baho — tarix yetarli bo'lmagan dastlabki kunlar uchun
    ex_ante = RISK_PER_ASSET * np.sqrt((sig ** 2).sum(1)) / A
    realized = pd.Series(r_raw).rolling(60, min_periods=20).std().shift(1).to_numpy()
    est = np.where(np.isnan(realized), ex_ante, realized)
    scale = np.clip(vol_target / np.sqrt(252) / np.maximum(est, 1e-6), 0, max_lev)
    scale = np.where((sig != 0).any(1), scale, 0.0)
    w = w_raw * scale[:, None]
    gross = np.nansum(w * np.nan_to_num(ret), axis=1)
    dw = np.abs(np.diff(np.vstack([np.zeros((1, A)), w]), axis=0))
    cost = dw @ cost_vector(mkt, cost_bps)
    contrib = w * np.nan_to_num(ret) - dw * cost_vector(mkt, cost_bps)[None, :]
    r = pd.Series(gross - cost, index=mkt.dates)
    return {"r": r, "gross": pd.Series(gross, index=mkt.dates), "w": w, "contrib": contrib,
            "turnover": float(dw.sum(1).mean()), "exposure": float(np.abs(w).sum(1).mean()),
            "cost_drag_ann": float(cost.mean() * 252 * 100)}
