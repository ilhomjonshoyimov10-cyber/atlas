"""Sintetik bozor + yangiliklar generatori — quvurni internet/GPU'siz sinash uchun.

Ichiga MA'LUM signallar joylangan:
  * mavsumiylik (bug'doy: hosil oylarida pasayish, gaz: qishda o'sish ...)
  * yangilik ta'siri: "OPEC cuts output" -> neft keyingi kunlarda o'sadi va h.k.
    Ta'sir FAQAT yangilik `available_at` dan keyingi yopilishdan boshlanadi (leakage yo'q).
Agar quvur to'g'ri ishlasa, ablationda `full` > `price+season` > `price` chiqishi kerak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ASSETS, DECISION_HOUR_UTC

# (shablon, aktivlar, yo'nalish)
TEMPLATES = [
    ("OPEC agrees surprise output cut as Saudi pushes for higher crude prices", ["CL", "XOM"], +1),
    ("Crude oil inventories jump far above forecasts, EIA says", ["CL"], -1),
    ("Severe drought threatens wheat harvest in Black Sea region", ["WHEAT", "CORN"], +1),
    ("USDA raises wheat crop forecast to record high in WASDE report", ["WHEAT"], -1),
    ("Fed signals rate cut as inflation cools, Powell says", ["SPX", "NDX", "GOLD", "EURUSD"], +1),
    ("Fed raises rates and warns of further hikes to fight inflation", ["SPX", "NDX", "GOLD", "EURUSD"], -1),
    ("SEC approves spot bitcoin ETF in landmark crypto decision", ["BTC", "ETH"], +1),
    ("Major crypto exchange hack drains millions, bitcoin slides", ["BTC", "ETH", "SOL"], -1),
    ("Nvidia earnings beat estimates on surging AI chip demand", ["NVDA", "NDX"], +1),
    ("Apple misses iPhone revenue estimates, shares fall", ["AAPL"], -1),
    ("ECB's Lagarde hints at more tightening, euro strengthens", ["EURUSD"], +1),
    ("Geopolitical tensions escalate, investors rush to gold as safe haven", ["GOLD", "SILVER"], +1),
    ("China demand for copper weakens as factory output slumps", ["COPPER"], -1),
    ("Frost in Brazil damages coffee crop", ["COFFEE"], +1),
    ("Cocoa prices surge as Ivory Coast harvest disappoints", ["COCOA"], +1),
    ("Tesla deliveries drop amid weak electric vehicle demand", ["TSLA"], -1),
]
NOISE = [
    "Markets await fresh economic data this week", "Analysts debate outlook for second half",
    "Trading volumes thin ahead of holiday", "Company announces new board appointment",
    "Local elections draw attention of investors", "Weather forecasts mixed for the weekend",
    "Retail sales data due on Thursday", "Central bank officials to speak at conference",
]
SEASON = {  # oy -> kunlik drift (sigma birliklarida)
    "WHEAT": {4: 0.08, 5: 0.06, 7: -0.10, 8: -0.08}, "CORN": {5: 0.06, 9: -0.08, 10: -0.06},
    "NG": {11: 0.08, 12: 0.06, 1: 0.04, 4: -0.06, 5: -0.04}, "COFFEE": {6: 0.05, 7: 0.05},
    "SPX": {11: 0.05, 12: 0.05, 9: -0.05}, "BTC": {10: 0.06, 11: 0.05, 6: -0.04},
}
BASE_PRICE = {"forex": 1.2, "crypto": 20000, "index": 3000, "equity": 150, "energy": 60, "agri": 500,
              "metal": 1500, "rates": 2.5}
BASE_VOL = {"forex": 0.005, "crypto": 0.035, "index": 0.011, "equity": 0.018, "energy": 0.022, "agri": 0.017,
            "metal": 0.011, "rates": 0.02}


def generate(start="2014-01-01", end="2026-10-28", seed=7, news_per_day=12.0):
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    T = len(days)
    # 1) yangiliklar
    rows, effect = [], {a.symbol: np.zeros(T) for a in ASSETS}
    for i, d in enumerate(days):
        for _ in range(rng.poisson(news_per_day)):
            avail = d + pd.Timedelta(hours=float(rng.uniform(0, 24)))
            if rng.random() < 0.25:
                tpl, assets, sign = TEMPLATES[rng.integers(len(TEMPLATES))]
                title = tpl
                # ta'sir: use_date = ceil(avail - decision_hour) -> o'sha kun yopilishidan keyingi qaytimlar
                ud = (avail - pd.Timedelta(hours=DECISION_HOUR_UTC)).ceil("D")
                j = days.searchsorted(ud)
                strength = rng.uniform(0.4, 1.0)
                for s in assets:
                    for lag, w in ((1, 0.8), (2, 0.4), (3, 0.2)):   # bir necha kunga yoyilgan ta'sir
                        if j + lag < T:
                            effect[s][j + lag] += sign * strength * w
            else:
                title = NOISE[rng.integers(len(NOISE))]
            rows.append({"origin": "synthetic", "source": "synth", "title": title, "url": "",
                         "published_at": avail.tz_localize("UTC"), "available_at": avail.tz_localize("UTC"),
                         "assets": ""})
    news = pd.DataFrame(rows)
    news["id"] = [f"s{i}" for i in range(len(news))]
    # 2) narxlar
    prices = {}
    common = rng.standard_normal(T)
    for a in ASSETS:
        vol0 = BASE_VOL[a.group]
        h = np.empty(T)
        h[0] = vol0 ** 2
        z = 0.4 * common + np.sqrt(1 - 0.16) * rng.standard_normal(T)
        r = np.empty(T)
        seas = SEASON.get(a.symbol, {})
        for t in range(T):
            if t:
                h[t] = vol0 ** 2 * 0.05 + 0.08 * r[t - 1] ** 2 + 0.87 * h[t - 1]
            sig = np.sqrt(h[t])
            drift = seas.get(days[t].month, 0.0) + 0.6 * effect[a.symbol][t]
            r[t] = sig * (z[t] + drift)
        dd = days if a.group == "crypto" else days[days.dayofweek < 5]
        keep = np.isin(days, dd)
        logc = np.log(BASE_PRICE[a.group] * rng.uniform(0.7, 1.3)) + np.cumsum(r)
        # dam olish kunlari qaytimlari dushanbaga qo'shiladi (cumsum orqali avtomatik)
        c = np.exp(logc[keep])
        rngh = np.abs(rng.standard_normal(keep.sum())) * np.sqrt(h[keep])
        prices[a.symbol] = pd.DataFrame({
            "date": days[keep], "open": c * (1 - rngh / 4), "high": c * (1 + rngh / 2), "low": c * (1 - rngh / 2),
            "close": c, "volume": rng.lognormal(10, 0.3, keep.sum()) * (1 + np.abs(r[keep]) / np.sqrt(h[keep]) / 3)})
    return prices, news
