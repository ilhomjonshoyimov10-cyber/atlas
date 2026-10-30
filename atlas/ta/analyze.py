"""'Analiz qil' — bitta aktiv bo'yicha to'liq texnik tahlil (o'zbek tilida, raqamlar bilan).

Har bir xulosa ortida o'lchangan narsa turadi:
  * darajalar va zonalar — faqat bugungacha tasdiqlangan pivot/impulslardan;
  * har bir faol signal uchun — shu strategiyaning 10+ yillik dataset bo'yicha natijasi va
    SetupNet modelining shu aniq holat uchun bahosi;
  * ATLAS asosiy modelining prognozi (bo'lsa) — mustaqil ikkinchi fikr sifatida.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..config import ASSET_BY_SYMBOL
from .dataset import GROUPS, _seq, _tab
from .detect import SETUP_NAMES, SETUPS, sanitize_ohlc, classic_patterns_at, indicators, levels_at, nearest_levels, pivots, \
    scan, snd_zones, structure_at, zone_status

TAIL = 900          # tahlil uchun oxirgi barlar (SMA200 + 400 barlik darajalar uchun yetarli)
RECENT = 5          # "faol" signal — oxirgi 5 bar ichida paydo bo'lgan


def _fmt(v: float) -> str:
    a = abs(v)
    d = 2 if a >= 10 else 4 if a >= 1 else 5
    return f"{v:,.{d}f}".replace(",", " ")


def _p_model(nets, z, df, ev, ind, pv, vol_z, group) -> float | None:
    if not nets:
        return None
    seq = torch.from_numpy(_seq(df, ev["t"], ev["atr"], ev["dir"], vol_z)[None])
    tab = torch.tensor([_tab(ev, ind, pv, df)], dtype=torch.float32)
    s = torch.tensor([SETUP_NAMES.index(ev["setup"])])
    g = torch.tensor([GROUPS.index(group)])
    with torch.no_grad():
        return float(np.mean([torch.sigmoid(n(seq, tab, s, g)).item() for n in nets]))


def analyze(symbol: str, prices: dict[str, pd.DataFrame], nets=None, z: dict | None = None,
            forecast: dict | None = None) -> dict:
    asset = ASSET_BY_SYMBOL[symbol]
    df = sanitize_ohlc(prices[symbol].tail(TAIL).reset_index(drop=True))
    n = len(df)
    t = n - 1
    c = df["close"].to_numpy(float)
    ind = indicators(df)
    pv = pivots(df)
    atr = float(ind["atr"].iat[t])
    close = float(c[t])
    dates = pd.DatetimeIndex(df["date"])

    # ---- darajalar (SNR)
    lv = levels_at(pv, t, atr, close)
    sup, res = nearest_levels(lv, close, n=3)
    lvl_out = [{"price": d["price"], "type": d["type"], "touches": d["touches"], "strength": d["strength"],
                "last": str(dates[d["last"]].date()), "dist_pct": (d["price"] / close - 1) * 100,
                "dist_atr": (d["price"] - close) / atr} for d in sup + res]

    # ---- SND zonalari (buzilmagan, yaqin)
    zones = []
    for zn in snd_zones(df, ind["atr"].to_numpy()):
        if t - zn["born"] > 250 or zn["born"] >= t:
            continue
        st = zone_status(zn, df, t)
        if st["broken"]:
            continue
        mid = (zn["lo"] + zn["hi"]) / 2
        if abs(mid - close) > 6 * atr:
            continue
        if (zn["kind"] == "demand" and zn["lo"] > close) or (zn["kind"] == "supply" and zn["hi"] < close):
            continue
        zones.append({"kind": zn["kind"], "lo": zn["lo"], "hi": zn["hi"], "born": str(dates[zn["born"]].date()),
                      "from": str(dates[zn["base_start"]].date()), "tests": st["tests"],
                      "impulse_atr": zn["impulse_atr"], "dist_atr": (mid - close) / atr})
    zones.sort(key=lambda d: abs(d["dist_atr"]))
    zones = [d for d in zones if d["kind"] == "demand"][:2] + [d for d in zones if d["kind"] == "supply"][:2]

    # ---- tuzilma, figuralar, indikatorlar
    st = structure_at(pv, t, close)
    pats = []
    for k in range(max(1, t - 9), t + 1):
        for p in classic_patterns_at(pv, k, c[k], c[k - 1], float(ind["atr"].iat[k])):
            pats.append({**p, "date": str(dates[k].date()), "label": SETUPS[p["name"]][1],
                         "target": p["neck"] + p["dir"] * p["height"]})
    r = ind.iloc[t]
    indic = {"rsi": float(r["rsi"]), "macd_hist": float(r["macd_hist"]), "sma50": float(r["sma50"]),
             "sma200": float(r["sma200"]) if np.isfinite(r["sma200"]) else None,
             "bb_pctb": float(r["bb_pctb"]) if np.isfinite(r["bb_pctb"]) else None,
             "candle": "bullish engulfing" if r["engulf_bull"] else "bearish engulfing" if r["engulf_bear"] else
             "hammer (pin-bar)" if r["pin_bull"] else "shooting star (pin-bar)" if r["pin_bear"] else
             "doji" if r["doji"] else None}

    # ---- faol signallar + tarixiy statistika + SetupNet bahosi
    events, _ = scan(df, start=max(220, t - RECENT - 2))
    vol_z = ind["vol_z"].to_numpy() if df["volume"].sum() > 0 else np.zeros(n)
    rep = (z or {}).get("report", {})
    hold = {s["setup"]: s for s in rep.get("stats_holdout", [])}
    train = {s["setup"]: s for s in rep.get("stats_train", [])}
    thr = rep.get("thresholds", {})
    br = (z or {}).get("base_rates", {})
    setups = []
    latest: dict[str, dict] = {}
    for ev in events:                        # har bir strategiyadan faqat eng so'nggi signal
        if ev["t"] >= t - RECENT + 1:
            latest[ev["setup"]] = ev
    for ev in latest.values():
        name = ev["setup"]
        p = _p_model(nets, z, df, ev, ind, pv, vol_z, asset.group)
        base = br.get("setup_group", {}).get(f"{name}|{asset.group}", br.get("setup", {}).get(name))
        tr_s = train.get(name, {})
        setups.append({
            "setup": name, "label": SETUPS[name][1], "family": SETUPS[name][2], "dir": ev["dir"],
            "date": str(dates[ev["t"]].date()), "level": ev.get("level"),
            "p_model": p, "base_win": base, "n_hist": br.get("setup_n", {}).get(name),
            "hist_excess_R": tr_s.get("excess_R"), "hist_p_holm": tr_s.get("p_holm"),
            "holdout_win": hold.get(name, {}).get("win_rate"),
            "strong": bool(p is not None and name in thr and p >= thr[name]),
            "entry": float(c[ev["t"]]),
            "tp": float(c[ev["t"]] + ev["dir"] * 1.5 * ev["atr"]), "sl": float(c[ev["t"]] - ev["dir"] * 1.0 * ev["atr"]),
        })
    setups.sort(key=lambda s: -(s["p_model"] or 0))

    # ---- umumiy baho: har bir omil -1..+1, og'irliklar bilan
    factors = []
    tr_map = {"ko'tarilish": 1, "pasayish": -1}
    if st["trend"] in tr_map:
        factors.append(("Bozor tuzilmasi", tr_map[st["trend"]], 1.0,
                        f"swinglar {'HH+HL (yuqoriga)' if tr_map[st['trend']] > 0 else 'LH+LL (pastga)'}"))
    if indic["sma200"]:
        s = 1 if close > indic["sma200"] else -1
        factors.append(("SMA200", s, 0.7, f"narx SMA200 ({_fmt(indic['sma200'])}) dan {'yuqorida' if s > 0 else 'pastda'}"))
    if indic["sma200"] and np.isfinite(indic["sma50"]):
        s = 1 if indic["sma50"] > indic["sma200"] else -1
        factors.append(("SMA50/200", s, 0.4, "SMA50 SMA200 dan " + ("yuqorida" if s > 0 else "pastda")))
    m = 1 if indic["macd_hist"] > 0 else -1
    factors.append(("MACD", m, 0.3, f"gistogramma {'musbat' if m > 0 else 'manfiy'}"))
    for s_ in setups:
        if s_["p_model"] is None or s_["base_win"] is None:
            continue
        edge = float(np.clip((s_["p_model"] - s_["base_win"]) * 5, -1, 1))   # tayanchdan farq
        w = 1.2 if s_["strong"] else 0.6
        factors.append((s_["label"], s_["dir"] * (0.5 + 0.5 * edge), w,
                        f"model {s_['p_model'] * 100:.0f}% vs tarixiy {s_['base_win'] * 100:.0f}%"))
    fc5 = None
    if forecast:
        fc5 = next((h for h in forecast.get("horizons", []) if h["h"] == 5), None)
        if fc5:
            s = float(np.clip((fc5["p_up"] - 0.5) * 6, -1, 1))
            factors.append(("ATLAS prognozi (5 kun)", s, 1.0, f"o'sish ehtimoli {fc5['p_up'] * 100:.0f}%"))
    wsum = sum(f[2] for f in factors) or 1
    score = sum(f[1] * f[2] for f in factors) / wsum
    bias = "ko'tarilish" if score > 0.2 else "pasayish" if score < -0.2 else "neytral"
    conf = "kuchli" if abs(score) > 0.5 else "o'rtacha" if abs(score) > 0.2 else "past"

    # ---- yozma tahlil
    text = []
    nm = asset.name
    text.append(f"{nm} ({symbol}) {dates[t].date()} yopilishida {_fmt(close)}. Kunlik o'rtacha harakat (ATR) "
                f"{_fmt(atr)} ({atr / close * 100:.2f}%).")
    if st["trend"] == "noaniq":
        text.append("Bozor tuzilmasi: tasdiqlangan swinglar yetarli emas.")
    else:
        extra = ""
        if st.get("bos_up"):
            extra = f" Narx oxirgi swing cho'qqisi ({_fmt(st['last_high'])}) dan yuqorida — tuzilma yuqoriga sindirilgan."
        elif st.get("bos_down"):
            extra = f" Narx oxirgi swing tubi ({_fmt(st['last_low'])}) dan pastda — tuzilma pastga sindirilgan."
        text.append(f"Bozor tuzilmasi: {st['trend']} (oxirgi cho'qqi {_fmt(st['last_high'])}, oxirgi tub "
                    f"{_fmt(st['last_low'])}).{extra}")
    if sup:
        text.append("Supportlar: " + "; ".join(f"{_fmt(d['price'])} ({d['touches']} marta tegilgan, "
                                               f"{(d['price'] / close - 1) * 100:+.1f}%)" for d in sup) + ".")
    if res:
        text.append("Resistancelar: " + "; ".join(f"{_fmt(d['price'])} ({d['touches']} marta tegilgan, "
                                                   f"{(d['price'] / close - 1) * 100:+.1f}%)" for d in res) + ".")
    for zn in zones:
        text.append(f"{'Demand' if zn['kind'] == 'demand' else 'Supply'} zonasi {_fmt(zn['lo'])}–{_fmt(zn['hi'])} "
                    f"({zn['born']} dagi {zn['impulse_atr']:.1f}·ATR impuls bilan hosil bo'lgan, "
                    + ("hali test qilinmagan" if zn["tests"] == 0 else f"{zn['tests']} marta test qilingan") + ").")
    for p in pats:
        text.append(f"Klassik figura: {p['label']} — {p['date']} da bo'yin chizig'i {_fmt(p['neck'])} sindi; "
                    f"o'lchov maqsadi ≈ {_fmt(p['target'])}.")
    rsi_t = "o'ta sotilgan" if indic["rsi"] < 30 else "o'ta sotib olingan" if indic["rsi"] > 70 else "neytral"
    text.append(f"Indikatorlar: RSI {indic['rsi']:.0f} ({rsi_t}), MACD gistogrammasi "
                f"{'musbat' if indic['macd_hist'] > 0 else 'manfiy'}"
                + (f", oxirgi sham — {indic['candle']}" if indic["candle"] else "") + ".")
    if setups:
        for s_ in setups[:4]:
            hist = f"tarixda {s_['n_hist']} ta holatda {s_['base_win'] * 100:.0f}% maqsadga yetgan" if s_["base_win"] else ""
            mdl = f"SetupNet bahosi {s_['p_model'] * 100:.0f}%" if s_["p_model"] is not None else ""
            tag = " — model uni yuqori sifatli deb hisoblaydi" if s_["strong"] else ""
            text.append(f"Signal: {s_['label']} ({'long' if s_['dir'] > 0 else 'short'}, {s_['date']}). "
                        f"{hist}; {mdl}{tag}. Strategiya qoidasi bo'yicha: kirish {_fmt(s_['entry'])}, maqsad {_fmt(s_['tp'])}, "
                        f"stop {_fmt(s_['sl'])} (1.5:1).")
    else:
        text.append("Oxirgi 5 kunda datasetdagi strategiyalardan birortasi ham signal bermagan.")
    if fc5:
        text.append(f"ATLAS deep learning prognozi: 5 savdo kunida median {fc5['ret_q50']:+.2f}%, "
                    f"90% oraliq {_fmt(fc5['q05'])}–{_fmt(fc5['q95'])}, o'sish ehtimoli {fc5['p_up'] * 100:.0f}%.")
    verdict = {"ko'tarilish": "ko'tarilish tomoni ustun", "pasayish": "pasayish tomoni ustun",
               "neytral": "aniq ustunlik yo'q"}[bias]
    text.append(f"Xulosa: {verdict} (ishonch: {conf}, umumiy ball {score:+.2f}). "
                + ("Kuzatiladigan darajalar: " + ", ".join(_fmt(d["price"]) for d in (sup[:1] + res[:1])) + "." if sup or res else ""))
    return {"symbol": symbol, "name": nm, "as_of": str(dates[t].date()), "close": close, "atr": atr,
            "structure": st, "levels": lvl_out, "zones": zones, "patterns": pats, "indicators": indic,
            "setups": setups, "factors": [{"name": f[0], "score": f[1], "weight": f[2], "note": f[3]} for f in factors],
            "score": score, "bias": bias, "confidence": conf, "text": text,
            "disclaimer": "Texnik tahlil ehtimollik bilan ishlaydi; bu investitsiya tavsiyasi emas."}
