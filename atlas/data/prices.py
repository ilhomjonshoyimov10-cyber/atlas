"""Tarixiy narxlarni yuklab olish (10+ yil, kunlik) va saqlash.

Manba: Yahoo Finance (yfinance) — forex, kripto, aksiya, indeks, fyuchers bir joyda.
Har bir aktiv `data/prices/<SYMBOL>.parquet` ga yoziladi: date, open, high, low, close, volume.
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from ..config import ASSETS, ASSET_BY_SYMBOL, PRICES_DIR, Asset

log = logging.getLogger(__name__)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower).reset_index()
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna(subset=["close"])
    df = df[df["close"] > 0].drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if {"open", "high", "low"} <= set(df.columns):
        from ..ta.detect import sanitize_ohlc   # Yahoo ba'zan noto'g'ri open beradi (masalan, 63 o'rniga 78)
        df = sanitize_ohlc(df.fillna({"open": df["close"], "high": df["close"], "low": df["close"]}))
    return df


def download_asset(asset: Asset, years: int = 12, retries: int = 3, start: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    last_err: Exception | None = None
    for i in range(retries):
        try:
            # Eslatma: yfinance "12y" kabi davrni qabul qilmaydi — boshlanish sanasini beramiz
            st = start or (pd.Timestamp.utcnow() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
            raw = yf.download(asset.yf, start=st, interval="1d",
                              auto_adjust=True, progress=False, threads=False)
            if raw is None or raw.empty:
                raise RuntimeError("bo'sh javob")
            return _clean(raw)
        except Exception as e:  # tarmoq xatolari va h.k.
            last_err = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{asset.symbol} ({asset.yf}) yuklanmadi: {last_err}")


def download_all(years: int = 12, symbols: list[str] | None = None) -> dict[str, int]:
    from ..config import DATA_DIR
    (DATA_DIR / "DEMO_SYNTHETIC").unlink(missing_ok=True)  # haqiqiy narxlar sintetik demoni almashtiradi
    out: dict[str, int] = {}
    for a in ASSETS:
        if symbols and a.symbol not in symbols:
            continue
        try:
            df = download_asset(a, years)
            df.to_parquet(PRICES_DIR / f"{a.symbol}.parquet", index=False)
            out[a.symbol] = len(df)
            log.info("%s: %d qator (%s .. %s)", a.symbol, len(df), df.date.min().date(), df.date.max().date())
        except Exception as e:
            log.warning("%s o'tkazib yuborildi: %s", a.symbol, e)
    return out


def load_prices(symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    res: dict[str, pd.DataFrame] = {}
    for p in sorted(PRICES_DIR.glob("*.parquet")):
        sym = p.stem
        if sym not in ASSET_BY_SYMBOL or (symbols and sym not in symbols):
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        res[sym] = df.sort_values("date").reset_index(drop=True)
    return res


def latest_quote(symbol: str) -> dict:
    """Bitta aktiv uchun so'nggi narx (yfinance, ~15 daqiqa kechikishi mumkin)."""
    import yfinance as yf

    a = ASSET_BY_SYMBOL[symbol]
    t = yf.Ticker(a.yf)
    h = t.history(period="5d", interval="1m" if a.group in ("crypto", "forex") else "5m")
    if h.empty:
        h = t.history(period="5d", interval="1d")
    last = h.iloc[-1]
    prev_close = t.history(period="5d", interval="1d")["Close"]
    prev = float(prev_close.iloc[-2]) if len(prev_close) > 1 else float(last["Close"])
    price = float(last["Close"])
    return {
        "symbol": symbol,
        "price": price,
        "change_pct": (price / prev - 1) * 100 if prev else 0.0,
        "time": pd.Timestamp(h.index[-1]).tz_convert("UTC").isoformat() if h.index.tz is not None
        else pd.Timestamp(h.index[-1]).isoformat(),
    }
