"""Jonli narxlar: barcha aktivlar uchun so'nggi narx (server tomonida, har N soniyada).

Manbalar:
  * Kripto  — Binance ochiq API (real vaqt, kalitsiz)
  * Qolganlari (forex, aksiya, indeks, fyuchers) — Yahoo Finance 1 daqiqalik barlar.
    Eslatma: Yahoo birja ma'lumotlari ~0–15 daqiqa kechikishi mumkin; oltin/kumush/neft — COMEX/NYMEX
    fyucherslari (GC=F, SI=F, CL=F), shuning uchun spot XAUUSD (OANDA) dan bir necha dollar farq qiladi.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from ..config import ASSETS

log = logging.getLogger(__name__)


class LiveQuotes:
    def __init__(self, interval: int = 30):
        self.interval = interval
        self.quotes: dict[str, dict] = {}
        self.prev_close: dict[str, float] = {}
        self._prev_day = None
        self.error: str | None = None
        self._stop = threading.Event()

    # ---- kripto: Binance ----
    def _binance(self):
        import httpx

        syms = [a for a in ASSETS if a.binance]
        q = '["' + '","'.join(a.binance.upper() for a in syms) + '"]'
        r = httpx.get("https://api.binance.com/api/v3/ticker/24hr", params={"symbols": q}, timeout=15)
        r.raise_for_status()
        by = {d["symbol"]: d for d in r.json()}
        now = datetime.now(timezone.utc).isoformat()
        for a in syms:
            d = by.get(a.binance.upper())
            if d:
                price = float(d["lastPrice"])
                self.quotes[a.symbol] = {"symbol": a.symbol, "price": price,
                                         "change_pct": float(d["priceChangePercent"]),
                                         "time": now, "source": "Binance"}

    # ---- qolganlari: Yahoo ----
    def _yahoo(self):
        import pandas as pd
        import yfinance as yf

        assets = [a for a in ASSETS if not a.binance]
        tickers = [a.yf for a in assets]
        today = datetime.now(timezone.utc).date()
        if self._prev_day != today:   # kunlik yopilishlar (kuniga bir marta)
            d = yf.download(tickers, period="7d", interval="1d", progress=False, auto_adjust=False,
                            group_by="ticker", threads=True)
            for a in assets:
                try:
                    c = d[a.yf]["Close"].dropna()
                    # oxirgi to'liq kun yopilishi (bugungi bar hali yopilmagan bo'lishi mumkin)
                    last_day = c.index[-1].date() if len(c) else None
                    prev = c.iloc[-2] if (len(c) > 1 and last_day == today) else c.iloc[-1]
                    self.prev_close[a.symbol] = float(prev)
                except Exception:
                    pass
            self._prev_day = today
        m = yf.download(tickers, period="1d", interval="1m", progress=False, auto_adjust=False,
                        group_by="ticker", threads=True, prepost=True)
        if m is None or m.empty:   # dam olish kunlari: 5 kunlik oraliqdan oxirgi narx
            m = yf.download(tickers, period="5d", interval="5m", progress=False, auto_adjust=False,
                            group_by="ticker", threads=True)
        for a in assets:
            try:
                c = m[a.yf]["Close"].dropna()
                if not len(c):
                    continue
                price = float(c.iloc[-1])
                ts = pd.Timestamp(c.index[-1])
                ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
                prev = self.prev_close.get(a.symbol)
                self.quotes[a.symbol] = {"symbol": a.symbol, "price": price,
                                         "change_pct": (price / prev - 1) * 100 if prev else None,
                                         "time": ts.isoformat(), "source": f"Yahoo {a.yf}"}
            except Exception:
                pass

    def poll_once(self):
        errs = []
        for fn in (self._binance, self._yahoo):
            try:
                fn()
            except Exception as e:
                errs.append(f"{fn.__name__}: {e}")
        self.error = "; ".join(errs) or None
        if errs:
            log.warning("Jonli narx xatosi: %s", self.error)

    def run_forever(self):
        while not self._stop.is_set():
            t0 = time.time()
            self.poll_once()
            self._stop.wait(max(5, self.interval - (time.time() - t0)))

    def start(self):
        threading.Thread(target=self.run_forever, daemon=True, name="live-quotes").start()

    def stop(self):
        self._stop.set()
