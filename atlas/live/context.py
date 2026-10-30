"""Ichki kun signallari uchun qo'shimcha kontekst (faqat signal paytida ma'lum bo'lgan narsalar):

  * savdo sessiyasi: bar yopilish soati (UTC) — sin/cos (Osiyo / London / Nyu-York);
  * katta vaqt oralig'i trendi: oxirgi YOPILGAN katta bar bo'yicha close − EMA20 (ATR birligida) va uning ishorasi.
    5m/15m/30m -> 1h, 1h -> 4h, 4h -> kunlik.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .intraday import TF_MINUTES, resample

HTF = {"5m": "1h", "15m": "1h", "30m": "1h", "1h": "4h", "4h": "1D"}
HTF_MIN = {"1h": 60, "4h": 240, "1D": 1440}
CTX_NAMES = ["sessiya sin", "sessiya cos", "katta TF trendi", "katta TF masofasi"]


def _htf_table(frames: dict[str, pd.DataFrame], htf: str) -> pd.DataFrame:
    base = frames.get("1h")
    if base is None:
        return pd.DataFrame(columns=["close_time", "up", "dist"])
    df = base if htf == "1h" else resample(base, htf)
    c = df["close"]
    ema = c.ewm(span=20, adjust=False).mean()
    tr = (df["high"] - df["low"]).ewm(alpha=1 / 14, adjust=False).mean()
    return pd.DataFrame({"close_time": df["date"] + pd.Timedelta(minutes=HTF_MIN[htf]),
                         "up": np.sign(c - ema).to_numpy(), "dist": ((c - ema) / tr).clip(-6, 6).to_numpy()})


def context_frame(frames: dict[str, pd.DataFrame], tf: str, df: pd.DataFrame) -> pd.DataFrame:
    """df ning har bir bari uchun kontekst (yo'nalishga ko'paytirilmagan)."""
    close_time = df["date"] + pd.Timedelta(minutes=TF_MINUTES[tf])
    h = close_time.dt.hour + close_time.dt.minute / 60
    out = pd.DataFrame({"hs": np.sin(2 * np.pi * h / 24), "hc": np.cos(2 * np.pi * h / 24)}, index=df.index)
    tab = _htf_table(frames, HTF[tf]).sort_values("close_time")
    if len(tab):
        m = pd.merge_asof(pd.DataFrame({"t": close_time}).reset_index(), tab, left_on="t", right_on="close_time",
                          direction="backward", allow_exact_matches=True).set_index("index")
        out["up"] = m["up"].fillna(0).to_numpy()
        out["dist"] = m["dist"].fillna(0).to_numpy()
    else:
        out["up"], out["dist"] = 0.0, 0.0
    return out


def ctx_vector(ctx: pd.DataFrame, t: int, direction: int) -> list[float]:
    r = ctx.iloc[t]
    return [float(r["hs"]), float(r["hc"]), float(r["up"] * direction), float(r["dist"] * direction) / 3]
