"""Strategiya dataseti: har bir signal (setup) — bitta yozuv; natija triple-barrier bilan belgilanadi.

Belgilash (label): kirish = t kunining yopilish narxi.
  * Maqsad (TP) = kirish ± 1.5·ATR, stop (SL) = kirish ∓ 1.0·ATR, muddat = 20 bar.
  * Qaysi biri birinchi tegsa — o'sha. Bitta barda ikkalasi tegsa — stop (ehtiyotkor taxmin).
  * R natija: TP = +1.5, SL = −1, muddat tugasa = yo'nalishdagi o'zgarish / ATR ([−1, 1.5] oralig'ida).

Taqqoslash (eng muhim qism): har bir signal o'sha aktiv, o'sha yo'nalish va O'SHA KUNdagi "tasodifiy kirish"
bilan solishtiriladi (excess R = R_signal − shu aktivda ±6 oy ichida har kuni kirishning o'rtacha R'i). Shunda oltinning 10 yillik o'sishi kabi umumiy
trend "long strategiyalar yaxshi" degan yolg'on xulosaga olib kelmaydi.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import ASSET_BY_SYMBOL, ASSETS
from .detect import SETUP_NAMES, SETUPS, sanitize_ohlc, scan, structure_at

log = logging.getLogger(__name__)
TP_ATR, SL_ATR, HORIZON = 1.5, 1.0, 20
SEQ = 64
GROUPS = sorted({a.group for a in ASSETS})


def barrier_outcomes(df: pd.DataFrame, atr: np.ndarray, direction: int) -> tuple[np.ndarray, np.ndarray]:
    """Har bir t uchun (yo'nalish berilgan) natija: win (0/1, NaN = hali noma'lum) va R."""
    c, h, l = (df[k].to_numpy(float) for k in ("close", "high", "low"))
    n = len(c)
    win = np.full(n, np.nan)
    R = np.full(n, np.nan)
    for t in range(n - 1):
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
        end = min(n, t + 1 + HORIZON)
        if end - (t + 1) < HORIZON:      # muddat to'lmagan — natija hali noma'lum
            continue
        tp = c[t] + direction * TP_ATR * a
        sl = c[t] - direction * SL_ATR * a
        hh, ll = h[t + 1:end], l[t + 1:end]
        hit_tp = (hh >= tp) if direction > 0 else (ll <= tp)
        hit_sl = (ll <= sl) if direction > 0 else (hh >= sl)
        i_tp = int(np.argmax(hit_tp)) if hit_tp.any() else 10**9
        i_sl = int(np.argmax(hit_sl)) if hit_sl.any() else 10**9
        if i_sl <= i_tp and i_sl < 10**9:
            win[t], R[t] = 0, -SL_ATR
        elif i_tp < 10**9:
            win[t], R[t] = 1, TP_ATR
        else:
            win[t] = 0
            R[t] = float(np.clip(direction * (c[end - 1] - c[t]) / a, -SL_ATR, TP_ATR))
    return win, R


def _seq(df: pd.DataFrame, t: int, atr: float, direction: int, vol_z: np.ndarray) -> np.ndarray:
    """Oxirgi SEQ bar: OHLC (kirish narxiga nisbatan, ATR birligida, yo'nalishga normallashgan) + hajm z."""
    s = max(0, t - SEQ + 1)
    block = df[["open", "high", "low", "close"]].to_numpy(float)[s:t + 1]
    x = (block - block[-1, 3]) / atr * direction
    if direction < 0:                          # short uchun high/low o'rin almashadi
        x = x[:, [0, 2, 1, 3]]
    x = np.clip(x, -15, 15)
    v = vol_z[s:t + 1, None]
    out = np.zeros((SEQ, 5), np.float32)
    out[SEQ - len(x):] = np.concatenate([x, v], 1)
    return out


def _tab(ev: dict, ind: pd.DataFrame, pv, df: pd.DataFrame) -> list[float]:
    t, d, a = ev["t"], ev["dir"], ev["atr"]
    c = float(df["close"].iat[t])
    r = ind.iloc[t]
    st = structure_at(pv, t, c)
    trend = {"ko'tarilish": 1, "pasayish": -1}.get(st["trend"], 0) * d
    lvl = ev.get("level")
    dist = 0.0 if lvl is None else np.clip((c - lvl) / a * d, -10, 10)
    s200 = 0.0 if not np.isfinite(r["sma200"]) else np.clip((c - r["sma200"]) / a * d, -30, 30) / 10
    s50 = 0.0 if not np.isfinite(r["sma50"]) else np.clip((c - r["sma50"]) / a * d, -20, 20) / 10
    return [float(dist), float(min(ev.get("touches", 0), 10)) / 5, (float(r["rsi"]) - 50) / 50 * d,
            float(np.clip(r["vol_z"], -4, 4)) / 2, float(trend), s200, s50,
            float(np.clip(r["macd_hist"] / a * d, -3, 3)),
            float(np.clip(r["bb_pctb"], -1, 2)) if np.isfinite(r["bb_pctb"]) else 0.5,
            float(ev.get("impulse", ev.get("height_atr", 0.0))) / 5]


TAB_NAMES = ["darajagacha masofa", "tegishlar", "RSI", "volatillik", "trend bilan mosligi", "SMA200 holati",
             "SMA50 holati", "MACD", "Bollinger %b", "impuls/figura balandligi"]


def build(prices: dict[str, pd.DataFrame]) -> dict:
    """Barcha aktivlar bo'yicha dataset: X_seq [N,64,5], X_tab [N,10], setup/guruh id, win, R, excess R."""
    rows, seqs, tabs = [], [], []
    for a in ASSETS:
        df = prices.get(a.symbol)
        if df is None or len(df) < 400:
            continue
        df = sanitize_ohlc(df.reset_index(drop=True))
        events, aux = scan(df)
        ind, pv = aux["ind"], aux["pv"]
        atr = ind["atr"].to_numpy()
        vol_z = ind["vol_z"].to_numpy()
        if not np.isfinite(df["volume"].to_numpy(float)).all() or df["volume"].sum() == 0:
            vol_z = np.zeros(len(df))
        outc = {d: barrier_outcomes(df, atr, d) for d in (1, -1)}
        dates = pd.DatetimeIndex(df["date"])
        for ev in events:
            t, d = ev["t"], ev["dir"]
            w, R = outc[d][0][t], outc[d][1][t]
            rows.append({"symbol": a.symbol, "group": a.group, "date": dates[t], "t": t, "setup": ev["setup"],
                         "dir": d, "win": w, "R": R, "level": ev.get("level")})
            seqs.append(_seq(df, t, ev["atr"], d, vol_z))
            tabs.append(_tab(ev, ind, pv, df))
        log.info("%s: %d signal", a.symbol, len(events))
    meta = pd.DataFrame(rows)
    meta["base_nb"] = neighborhood_baseline(prices, meta)
    return {"meta": meta, "seq": np.stack(seqs).astype(np.float32), "tab": np.asarray(tabs, np.float32)}


def neighborhood_baseline(prices: dict[str, pd.DataFrame], meta: pd.DataFrame, half: int = 126) -> np.ndarray:
    """Har bir signal uchun: o'sha aktivda, o'sha yo'nalishda, t±half (≈ ±6 oy) kunlarda tasodifiy kirishning o'rtacha R'i.

    Bu "shu davrda shunchaki kirsang nima bo'lardi" — trend va volatillik rejimini nazorat qiladi.
    """
    base = np.full(len(meta), np.nan)
    from .detect import indicators
    for sym, g in meta.groupby("symbol"):
        df = sanitize_ohlc(prices[sym].reset_index(drop=True))
        atr = indicators(df)["atr"].to_numpy()
        for d in (1, -1):
            _, R = barrier_outcomes(df, atr, d)
            Rs = pd.Series(R).rolling(2 * half + 1, center=True, min_periods=half // 2).mean().to_numpy()
            m = g["dir"].to_numpy() == d
            base[g.index[m]] = Rs[g["t"].to_numpy()[m]]
    return base


def setup_stats(meta: pd.DataFrame, mask: np.ndarray | None = None) -> list[dict]:
    """Har bir strategiya: soni, yutish %, o'rtacha R, tasodifiy kirishga nisbatan ustunlik va Holm p-qiymat."""
    from math import erf, sqrt

    m = meta if mask is None else meta[mask]
    m = m.dropna(subset=["R", "base_nb"])
    out = []
    for name in SETUP_NAMES:
        g = m[m["setup"] == name]
        if len(g) < 20:
            out.append({"setup": name, "n": int(len(g))})
            continue
        ex = (g["R"] - g["base_nb"]).to_numpy()
        # signallar bir-biriga yaqin kunlarda to'planadi -> samarali soni kamroq (oddiy konservativ tuzatish: /2)
        n_eff = max(5, len(ex) / 2)
        se = ex.std(ddof=1) / sqrt(n_eff)
        z = ex.mean() / se if se > 0 else 0.0
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        out.append({"setup": name, "n": int(len(g)), "win_rate": float(g["win"].mean() * 100),
                    "avg_R": float(g["R"].mean()), "base_R": float(g["base_nb"].mean()),
                    "excess_R": float(ex.mean()), "z": float(z), "p": float(p)})
    # Holm tuzatishi (ko'p strategiya sinalganda tasodifiy "topilma"larning oldini oladi)
    tested = sorted([r for r in out if "p" in r], key=lambda r: r["p"])
    k = len(tested)
    running = 0.0
    for i, r in enumerate(tested):
        running = max(running, min(1.0, r["p"] * (k - i)))
        r["p_holm"] = running
    for r in out:
        r["name"], r["family"] = SETUPS[r["setup"]][1], SETUPS[r["setup"]][2]
        r["dir"] = SETUPS[r["setup"]][0]
    return out
