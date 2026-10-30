"""Texnik tahlil detektorlari: SNR (support/resistance), SND (supply/demand zonalari), bozor tuzilmasi,
klassik figuralar, indikatorlar va sham figuralari.

Leakage qoidasi: t kuni ma'lum bo'lmagan narsa t kuni ishlatilmaydi.
  * Pivot (fraktal) cho'qqi/tub i-barda bo'lsa, u faqat i+K barda "tasdiqlanadi" (o'ngdagi K bar kerak).
    Shuning uchun t kunidagi darajalar faqat `confirm <= t` bo'lgan pivotlardan quriladi.
  * SND zonasi impuls bari yopilgandan keyingina mavjud bo'ladi.
  * Indikatorlar faqat t gacha bo'lgan barlardan hisoblanadi (rolling/ewm).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PIVOT_K = 5          # fraktal kengligi (har tomonda K bar)
LEVEL_LOOKBACK = 400  # darajalar uchun qancha bar orqaga qaraladi
LEVEL_TOL = 0.5      # bir darajaga birlashtirish chegarasi, ATR birligida


def sanitize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Manba xatolarini tuzatadi: open [low, high] dan tashqarida bo'lsa (Yahoo'da uchraydi) — oldingi yopilish;
    high/low sham tanasini o'z ichiga olishi shart."""
    d = df.copy()
    o, h, l, c = (d[k].to_numpy(float).copy() for k in ("open", "high", "low", "close"))
    prev = np.r_[c[0], c[:-1]]
    bad = ~np.isfinite(o) | (o > h * 1.0001) | (o < l * 0.9999)
    o[bad] = np.clip(prev[bad], l[bad], h[bad])
    h = np.maximum.reduce([h, o, c])
    l = np.minimum.reduce([l, o, c])
    d["open"], d["high"], d["low"] = o, h, l
    return d


# ------------------------------------------------------------------ indikatorlar
def indicators(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    out = pd.DataFrame(index=df.index)
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - prev), abs(l - prev)))
    out["atr"] = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / 14, adjust=False).mean()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50).to_numpy()
    cs = pd.Series(c)
    out["sma20"] = cs.rolling(20).mean().to_numpy()
    out["sma50"] = cs.rolling(50).mean().to_numpy()
    out["sma200"] = cs.rolling(200).mean().to_numpy()
    ema12, ema26 = cs.ewm(span=12, adjust=False).mean(), cs.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()).to_numpy()
    sd = cs.rolling(20).std()
    out["bb_pctb"] = ((cs - (out["sma20"] - 2 * sd)) / (4 * sd)).to_numpy()
    out["vol_z"] = ((pd.Series(np.log(np.maximum(tr, 1e-12))) -
                     pd.Series(np.log(np.maximum(tr, 1e-12))).rolling(100).mean()) /
                    pd.Series(np.log(np.maximum(tr, 1e-12))).rolling(100).std()).fillna(0).to_numpy()
    body = abs(c - o)
    rng = np.maximum(h - l, 1e-12)
    po, pc = np.r_[o[0], o[:-1]], prev
    out["engulf_bull"] = (c > o) & (pc < po) & (c >= po) & (o <= pc) & (body > 0.6 * rng)
    out["engulf_bear"] = (c < o) & (pc > po) & (c <= po) & (o >= pc) & (body > 0.6 * rng)
    lower_wick = np.minimum(o, c) - l
    upper_wick = h - np.maximum(o, c)
    out["pin_bull"] = (lower_wick > 2 * body) & (lower_wick > 0.6 * rng)       # hammer
    out["pin_bear"] = (upper_wick > 2 * body) & (upper_wick > 0.6 * rng)       # shooting star
    out["doji"] = body < 0.1 * rng
    return out


# ------------------------------------------------------------------ pivotlar
@dataclass
class Pivots:
    idx: np.ndarray      # pivot bar indeksi
    price: np.ndarray
    kind: np.ndarray     # +1 cho'qqi (high), -1 tub (low)
    confirm: np.ndarray  # tasdiqlangan bar = idx + K


def pivots(df: pd.DataFrame, k: int = PIVOT_K) -> Pivots:
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(h)
    hi_max = pd.Series(h).rolling(2 * k + 1, center=True).max().to_numpy()
    lo_min = pd.Series(l).rolling(2 * k + 1, center=True).min().to_numpy()
    ph = np.where((h == hi_max) & (np.arange(n) < n - k))[0]
    pl = np.where((l == lo_min) & (np.arange(n) < n - k))[0]
    idx = np.r_[ph, pl]
    kind = np.r_[np.ones(len(ph), int), -np.ones(len(pl), int)]
    price = np.r_[h[ph], l[pl]]
    o = np.argsort(idx, kind="stable")
    idx, kind, price = idx[o], kind[o], price[o]
    return Pivots(idx, price, kind, idx + k)


def zigzag(idx: np.ndarray, price: np.ndarray, kind: np.ndarray):
    """Ketma-ket bir xil turdagi pivotlardan eng ekstremalini qoldiradi.

    Faqat allaqachon tasdiqlangan pivotlar ustida chaqiriladi — kelajakdagi pivot o'tmishdagisini o'chirmaydi.
    """
    ki, kp, kk = [], [], []
    for i, p, k in zip(idx, price, kind):
        if kk and kk[-1] == k:
            if (k == 1 and p > kp[-1]) or (k == -1 and p < kp[-1]):
                ki[-1], kp[-1] = i, p
            continue
        ki.append(i); kp.append(p); kk.append(k)
    return np.array(ki, int), np.array(kp, float), np.array(kk, int)


def confirmed(pv: "Pivots", t: int, last: int = 15):
    m = pv.confirm <= t
    return zigzag(pv.idx[m][-3 * last:], pv.price[m][-3 * last:], pv.kind[m][-3 * last:])


# ------------------------------------------------------------------ SNR darajalari
def levels_at(pv: Pivots, t: int, atr: float, close: float, lookback: int = LEVEL_LOOKBACK,
              tol: float = LEVEL_TOL) -> list[dict]:
    """t barda ma'lum bo'lgan S/R darajalari: yaqin pivotlar klasterlanadi, tegishlar soni = kuch."""
    m = (pv.confirm <= t) & (pv.idx >= t - lookback)
    if m.sum() == 0 or not np.isfinite(atr) or atr <= 0:
        return []
    p, ix = pv.price[m], pv.idx[m]
    order = np.argsort(p)
    p, ix = p[order], ix[order]
    out, start = [], 0
    for i in range(1, len(p) + 1):
        if i == len(p) or p[i] - p[start] > tol * atr:
            grp = slice(start, i)
            lvl = float(np.mean(p[grp]))
            out.append({"price": lvl, "touches": int(i - start), "last": int(ix[grp].max()),
                        "first": int(ix[grp].min()),
                        "type": "support" if lvl < close else "resistance"})
            start = i
    for d in out:   # kuch: tegishlar + yangilik (oxirgi tegish qanchalik yaqin)
        d["strength"] = round(d["touches"] + 2.0 * np.exp(-(t - d["last"]) / 120), 2)
    return out


def nearest_levels(levels: list[dict], close: float, n: int = 3, min_touch: int = 2):
    sup = sorted([d for d in levels if d["price"] < close and d["touches"] >= min_touch], key=lambda d: -d["price"])[:n]
    res = sorted([d for d in levels if d["price"] >= close and d["touches"] >= min_touch], key=lambda d: d["price"])[:n]
    return sup, res


# ------------------------------------------------------------------ SND zonalari
def snd_zones(df: pd.DataFrame, atr: np.ndarray, max_base: int = 4, impulse: float = 1.5) -> list[dict]:
    """Supply/Demand: kichik "baza" shamlaridan keyin kuchli impuls.

    demand: baza -> impuls yuqoriga (rally-base-rally / drop-base-rally)
    supply: baza -> impuls pastga (drop-base-drop / rally-base-drop)
    Zona impuls bari yopilganda paydo bo'ladi (`born`), narx keyin qaytib zonaga kirsa — test.
    """
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    rng = h - l
    zones = []
    n = len(c)
    for t in range(max_base + 1, n):
        a = atr[t - 1]
        if not np.isfinite(a) or a <= 0:
            continue
        move = c[t] - o[t]
        if abs(move) < impulse * a or rng[t] < impulse * a:
            continue
        # baza: impulsdan oldingi 1..max_base ta kichik sham
        b = t - 1
        while b > t - 1 - max_base and rng[b] < 0.8 * atr[b]:
            b -= 1
        base = slice(b + 1, t)
        if base.start >= base.stop:
            continue
        if move > 0:
            lo, hi = float(l[base].min()), float(np.maximum(o[base], c[base]).max())
            kind = "demand"
        else:
            lo, hi = float(np.minimum(o[base], c[base]).min()), float(h[base].max())
            kind = "supply"
        if hi - lo <= 0:
            hi = lo + 0.1 * a
        zones.append({"kind": kind, "lo": lo, "hi": hi, "born": t, "base_start": int(base.start),
                      "impulse_atr": round(abs(move) / a, 2)})
    return zones


def zone_status(z: dict, df: pd.DataFrame, t: int) -> dict:
    """t gacha zona necha marta test qilingan va buzilganmi."""
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    seg = slice(z["born"] + 1, t + 1)
    if z["kind"] == "demand":
        touched = l[seg] <= z["hi"]
        broken = c[seg] < z["lo"]
    else:
        touched = h[seg] >= z["lo"]
        broken = c[seg] > z["hi"]
    first_touch = int(np.argmax(touched)) + z["born"] + 1 if touched.any() else None
    return {"tests": int(np.sum(np.diff(np.r_[0, touched.astype(int)]) == 1)),
            "broken": bool(broken.any()), "first_touch": first_touch}


# ------------------------------------------------------------------ bozor tuzilmasi
def structure_at(pv: Pivots, t: int, close: float) -> dict:
    """Oxirgi tasdiqlangan swinglar bo'yicha trend: HH+HL -> ko'tarilish, LH+LL -> pasayish."""
    _, pr, kd = confirmed(pv, t)
    hi = pr[kd == 1][-2:]
    lo = pr[kd == -1][-2:]
    if len(hi) < 2 or len(lo) < 2:
        return {"trend": "noaniq", "last_high": None, "last_low": None}
    hh, hl = hi[1] > hi[0], lo[1] > lo[0]
    trend = "ko'tarilish" if hh and hl else "pasayish" if (not hh and not hl) else "yonbosh"
    return {"trend": trend, "last_high": float(hi[1]), "last_low": float(lo[1]),
            "hh": bool(hh), "hl": bool(hl), "bos_up": bool(close > hi[1]), "bos_down": bool(close < lo[1])}


# ------------------------------------------------------------------ klassik figuralar
def classic_patterns_at(pv: Pivots, t: int, close: float, prev_close: float, atr: float) -> list[dict]:
    """t barda bo'yin chizig'i (neckline) sinishi bilan tugagan klassik figuralar."""
    idx, pr, kd = confirmed(pv, t)
    idx, pr, kd = idx[-7:], pr[-7:], kd[-7:]
    out = []
    if len(idx) < 4 or not np.isfinite(atr) or atr <= 0:
        return out
    tol = 0.6 * atr
    # Qo'sh cho'qqi: H L H (H'lar teng), bo'yin = L; sinish: close < L
    for j in range(len(idx) - 3, len(idx) - 2):
        a, b, cc = j, j + 1, j + 2
        if cc >= len(idx):
            continue
        if kd[a] == 1 and kd[b] == -1 and kd[cc] == 1 and abs(pr[a] - pr[cc]) < tol and idx[cc] - idx[a] >= 8:
            neck = pr[b]
            if prev_close >= neck > close:
                out.append({"name": "double_top", "dir": -1, "neck": float(neck), "height": float(max(pr[a], pr[cc]) - neck)})
        if kd[a] == -1 and kd[b] == 1 and kd[cc] == -1 and abs(pr[a] - pr[cc]) < tol and idx[cc] - idx[a] >= 8:
            neck = pr[b]
            if prev_close <= neck < close:
                out.append({"name": "double_bottom", "dir": 1, "neck": float(neck), "height": float(neck - min(pr[a], pr[cc]))})
    # Bosh-yelka: H L H(bosh) L H, yelkalar teng, bosh balandroq; bo'yin = ikki L o'rtachasi
    if len(idx) >= 5:
        k5, p5, i5 = kd[-5:], pr[-5:], idx[-5:]
        if list(k5) == [1, -1, 1, -1, 1] and p5[2] > max(p5[0], p5[4]) + 0.5 * atr and abs(p5[0] - p5[4]) < 1.5 * tol:
            neck = (p5[1] + p5[3]) / 2
            if prev_close >= neck > close:
                out.append({"name": "head_shoulders", "dir": -1, "neck": float(neck), "height": float(p5[2] - neck)})
        if list(k5) == [-1, 1, -1, 1, -1] and p5[2] < min(p5[0], p5[4]) - 0.5 * atr and abs(p5[0] - p5[4]) < 1.5 * tol:
            neck = (p5[1] + p5[3]) / 2
            if prev_close <= neck < close:
                out.append({"name": "inv_head_shoulders", "dir": 1, "neck": float(neck), "height": float(neck - p5[2])})
    # Uchburchaklar: oxirgi 2 cho'qqi va 2 tub
    hs_ = pr[kd == 1][-2:]
    ls_ = pr[kd == -1][-2:]
    if len(hs_) == 2 and len(ls_) == 2:
        if abs(hs_[1] - hs_[0]) < tol and ls_[1] > ls_[0] + 0.3 * atr and prev_close <= hs_.max() < close:
            out.append({"name": "asc_triangle", "dir": 1, "neck": float(hs_.max()), "height": float(hs_.max() - ls_[0])})
        if abs(ls_[1] - ls_[0]) < tol and hs_[1] < hs_[0] - 0.3 * atr and prev_close >= ls_.min() > close:
            out.append({"name": "desc_triangle", "dir": -1, "neck": float(ls_.min()), "height": float(hs_[0] - ls_.min())})
    return out


# ------------------------------------------------------------------ strategiya signallari (setup)
SETUPS = {
    # nom: (yo'nalish, o'zbekcha tavsif, oila)
    "sr_bounce_long": (1, "Supportdan qaytish", "SNR"),
    "sr_bounce_short": (-1, "Resistancedan qaytish", "SNR"),
    "sr_breakout_long": (1, "Resistance sinishi (yuqoriga)", "SNR"),
    "sr_breakdown_short": (-1, "Support sinishi (pastga)", "SNR"),
    "snd_demand_long": (1, "Demand zonasiga qaytish", "SND"),
    "snd_supply_short": (-1, "Supply zonasiga qaytish", "SND"),
    "bos_long": (1, "Struktura sinishi yuqoriga (BOS)", "Struktura"),
    "bos_short": (-1, "Struktura sinishi pastga (BOS)", "Struktura"),
    "double_bottom": (1, "Qo'sh tub", "Klassik"),
    "double_top": (-1, "Qo'sh cho'qqi", "Klassik"),
    "inv_head_shoulders": (1, "Teskari bosh-yelka", "Klassik"),
    "head_shoulders": (-1, "Bosh-yelka", "Klassik"),
    "asc_triangle": (1, "Ko'tariluvchi uchburchak sinishi", "Klassik"),
    "desc_triangle": (-1, "Tushuvchi uchburchak sinishi", "Klassik"),
    "golden_cross": (1, "Oltin kesishish (SMA50 > SMA200)", "Indikator"),
    "death_cross": (-1, "O'lim kesishishi (SMA50 < SMA200)", "Indikator"),
    "rsi_oversold": (1, "RSI 30 dan yuqoriga qaytdi", "Indikator"),
    "rsi_overbought": (-1, "RSI 70 dan pastga qaytdi", "Indikator"),
    "engulf_support": (1, "Supportda yutuvchi (engulfing) sham", "Sham"),
    "engulf_resistance": (-1, "Resistanceda yutuvchi sham", "Sham"),
    "pin_support": (1, "Supportda pin-bar (hammer)", "Sham"),
    "pin_resistance": (-1, "Resistanceda pin-bar (shooting star)", "Sham"),
}
SETUP_NAMES = list(SETUPS)


def scan(df: pd.DataFrame, start: int = 220) -> tuple[list[dict], dict]:
    """Butun tarix bo'ylab barcha signallarni topadi (har biri faqat t gacha ma'lumotdan).

    Natija: signallar ro'yxati va qayta ishlatiladigan oraliq hisoblar (indikatorlar, pivotlar, zonalar).
    """
    df = sanitize_ohlc(df)
    ind = indicators(df)
    pv = pivots(df)
    atr = ind["atr"].to_numpy()
    c, h, l = (df[k].to_numpy(float) for k in ("close", "high", "low"))
    zones = snd_zones(df, atr)
    rsi, s50, s200 = ind["rsi"].to_numpy(), ind["sma50"].to_numpy(), ind["sma200"].to_numpy()
    events: list[dict] = []
    zone_touched: set[int] = set()
    born_by_t: dict[int, list[int]] = {}
    for zi, z in enumerate(zones):
        born_by_t.setdefault(z["born"], []).append(zi)
    live: list[int] = []
    last_struct = None
    for t in range(start, len(c)):
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
        lv = levels_at(pv, t - 1, atr[t - 1], c[t - 1])     # kechagi holatdagi darajalar
        sup, res = nearest_levels(lv, c[t - 1], n=1)
        ctx = {"atr": float(a)}

        def add(name, **extra):
            events.append({"t": t, "setup": name, "dir": SETUPS[name][0], **ctx, **extra})

        if sup:
            s = sup[0]
            near = l[t] <= s["price"] + 0.25 * a
            if near and c[t] > s["price"]:
                add("sr_bounce_long", level=s["price"], touches=s["touches"])
                if ind["engulf_bull"].iat[t]:
                    add("engulf_support", level=s["price"], touches=s["touches"])
                if ind["pin_bull"].iat[t]:
                    add("pin_support", level=s["price"], touches=s["touches"])
            if c[t - 1] >= s["price"] and c[t] < s["price"] - 0.25 * a:
                add("sr_breakdown_short", level=s["price"], touches=s["touches"])
        if res:
            r = res[0]
            near = h[t] >= r["price"] - 0.25 * a
            if near and c[t] < r["price"]:
                add("sr_bounce_short", level=r["price"], touches=r["touches"])
                if ind["engulf_bear"].iat[t]:
                    add("engulf_resistance", level=r["price"], touches=r["touches"])
                if ind["pin_bear"].iat[t]:
                    add("pin_resistance", level=r["price"], touches=r["touches"])
            if c[t - 1] <= r["price"] and c[t] > r["price"] + 0.25 * a:
                add("sr_breakout_long", level=r["price"], touches=r["touches"])
        # SND: zonaga birinchi qaytish (yangi, buzilmagan zona)
        live.extend(born_by_t.get(t - 1, []))
        still = []
        for zi in live:
            z = zones[zi]
            if t - z["born"] > 250:
                continue
            if z["kind"] == "demand":
                if c[t] < z["lo"]:
                    continue
                if zi not in zone_touched and l[t] <= z["hi"]:
                    zone_touched.add(zi)
                    if c[t] >= z["lo"]:
                        add("snd_demand_long", level=z["hi"], zone_lo=z["lo"], zone_hi=z["hi"], impulse=z["impulse_atr"])
            else:
                if c[t] > z["hi"]:
                    continue
                if zi not in zone_touched and h[t] >= z["lo"]:
                    zone_touched.add(zi)
                    if c[t] <= z["hi"]:
                        add("snd_supply_short", level=z["lo"], zone_lo=z["lo"], zone_hi=z["hi"], impulse=z["impulse_atr"])
            still.append(zi)
        live = still
        # Struktura sinishi (trend yo'nalishida BOS)
        st = structure_at(pv, t, c[t])
        if last_struct is not None and st["trend"] != "noaniq":
            if st["trend"] == "ko'tarilish" and c[t - 1] <= st["last_high"] < c[t]:
                add("bos_long", level=st["last_high"])
            if st["trend"] == "pasayish" and c[t - 1] >= st["last_low"] > c[t]:
                add("bos_short", level=st["last_low"])
        last_struct = st
        for p in classic_patterns_at(pv, t, c[t], c[t - 1], a):
            add(p["name"], level=p["neck"], height_atr=p["height"] / a)
        if np.isfinite(s200[t - 1]) and np.isfinite(s200[t]):
            if s50[t - 1] <= s200[t - 1] and s50[t] > s200[t]:
                add("golden_cross")
            if s50[t - 1] >= s200[t - 1] and s50[t] < s200[t]:
                add("death_cross")
        if rsi[t - 1] < 30 <= rsi[t]:
            add("rsi_oversold", rsi=float(rsi[t]))
        if rsi[t - 1] > 70 >= rsi[t]:
            add("rsi_overbought", rsi=float(rsi[t]))
    return events, {"ind": ind, "pv": pv, "zones": zones}
