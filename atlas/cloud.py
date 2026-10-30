"""Bulut rejimi: serversiz (statik) sayt uchun ma'lumot tayyorlash.

GitHub Actions har 30 daqiqada `python -m atlas.cli cloud-update` ni ishga tushiradi:
  1. narxlarni yangilaydi (faqat oxirgi kunlar yuklanadi, eski tarix saqlanadi);
  2. RSS manbalaridan yangiliklarni yig'adi (first_seen = birinchi ko'rilgan vaqt);
  3. yangi xabarlarning ta'sirini model orqali baholaydi;
  4. prognozlarni hisoblaydi va "prognoz jurnali"ga yozadi (keyin haqiqat bilan solishtiriladi);
  5. barcha API javoblarini `site/` ichiga JSON fayl qilib yozadi — sayt ularni to'g'ridan-to'g'ri o'qiydi.

Shunday qilib sayt doim ishlaydi (uxlaydigan server yo'q), hisob-kitob esa bepul GitHub Actions'da.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ASSETS, DATA_DIR, PRICES_DIR, REPORTS_DIR

log = logging.getLogger(__name__)
SITE = DATA_DIR / "site"
LOG_PATH = DATA_DIR / "forecast_log.jsonl"
BT_THRESHOLDS = [round(0.02 * i, 2) for i in range(11)]      # 0 .. 0.20
BT_COSTS = [0, 2, 5, 10, 20]
CLOSE_HOUR_UTC = 21   # shu soatdan keyin bugungi kunlik sham "yopilgan" deb hisoblanadi


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _write(rel: str, obj) -> None:
    p = SITE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str))


def closed_bars_only(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Hali yopilmagan bugungi shamni olib tashlaydi — prognoz faqat yopilgan kunlardan qilinadi."""
    now = now or _now()
    today = now.tz_localize(None).normalize() if now.tzinfo else now.normalize()
    if now.hour >= CLOSE_HOUR_UTC:
        return df[df["date"] <= today]
    return df[df["date"] < today]


# ---------------------------------------------------------------- 1. narxlar
def update_prices(lookback_days: int = 10) -> dict:
    from .data.prices import download_asset

    stats = {"ok": 0, "failed": []}
    for a in ASSETS:
        p = PRICES_DIR / f"{a.symbol}.parquet"
        try:
            if p.exists():
                old = pd.read_parquet(p)
                old["date"] = pd.to_datetime(old["date"])
                start = (old["date"].max() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
                new = download_asset(a, start=start)
                df = pd.concat([old[old["date"] < new["date"].min()], new])
            else:
                df = download_asset(a)
            df = closed_bars_only(df.drop_duplicates("date", keep="last").sort_values("date"))
            df.reset_index(drop=True).to_parquet(p, index=False)
            stats["ok"] += 1
        except Exception as e:   # bitta aktiv xatosi butun yangilanishni to'xtatmasin
            log.warning("%s narxi yangilanmadi: %s", a.symbol, e)
            stats["failed"].append(a.symbol)
    return stats


# ---------------------------------------------------------------- 2. yangiliklar
def collect_news() -> tuple[list[dict], dict]:
    from .data.collector import Collector

    col = Collector()
    new = asyncio.run(col.poll_once())
    return new, col.stats


# ---------------------------------------------------------------- 4. prognoz jurnali
def append_forecast_log(fc: dict) -> int:
    seen = set()
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text().splitlines():
            try:
                r = json.loads(line)
                seen.add((r["symbol"], r["as_of"]))
            except Exception:
                pass
    made = _now().isoformat()
    n = 0
    with LOG_PATH.open("a") as f:
        for s, item in fc.items():
            if (s, item["as_of"]) in seen:
                continue
            f.write(json.dumps({"symbol": s, "as_of": item["as_of"], "made_at": made, "last": item["last"],
                                "h": [{k: hz[k] for k in ("h", "q05", "q50", "q95", "p_up")} for hz in item["horizons"]]})
                    + "\n")
            n += 1
    return n


def track_record(prices: dict[str, pd.DataFrame]) -> dict:
    """Jurnaldagi prognozlarni haqiqiy narx bilan solishtirish (faqat muddati o'tganlari)."""
    if not LOG_PATH.exists():
        return {"summary": [], "recent": [], "pending": 0, "started": None}
    rows, pending, started = [], 0, None
    closes = {s: (pd.DatetimeIndex(df["date"]), df["close"].to_numpy()) for s, df in prices.items()}
    for line in LOG_PATH.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        started = min(started or r["as_of"], r["as_of"])
        if r["symbol"] not in closes:
            continue
        dates, close = closes[r["symbol"]]
        i = dates.searchsorted(pd.Timestamp(r["as_of"]))
        if i >= len(dates) or dates[i] != pd.Timestamp(r["as_of"]):
            continue
        for hz in r["h"]:
            j = i + hz["h"]
            if j >= len(dates):
                pending += 1
                continue
            real = float(close[j])
            d_pred, d_real = np.sign(hz["q50"] - r["last"]), np.sign(real - r["last"])
            rows.append({"symbol": r["symbol"], "as_of": r["as_of"], "target": str(dates[j].date()), "h": hz["h"],
                         "last": r["last"], "q05": hz["q05"], "q50": hz["q50"], "q95": hz["q95"], "real": real,
                         "inside": bool(hz["q05"] <= real <= hz["q95"]),
                         "dir_ok": None if d_pred == 0 or d_real == 0 else bool(d_pred == d_real),
                         "err_pct": abs(real / hz["q50"] - 1) * 100})
    summary = []
    if rows:
        d = pd.DataFrame(rows)
        for h, g in d.groupby("h"):
            dr = g["dir_ok"].dropna()
            summary.append({"h": int(h), "n": int(len(g)), "coverage90": float(g["inside"].mean() * 100),
                            "direction": float(dr.mean() * 100) if len(dr) else None,
                            "mae_pct": float(g["err_pct"].mean())})
    rows.sort(key=lambda r: (r["target"], r["symbol"]), reverse=True)
    return {"summary": summary, "recent": rows[:300], "pending": pending, "started": started}


# ---------------------------------------------------------------- 5. eksport
def _hash(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()[:16] if p.exists() else ""


def export_backtests(svc, force: bool = False) -> int:
    from .model.backtest import run_backtest

    oos = REPORTS_DIR / "oos_HOLDOUT.npz"
    stamp = SITE / "backtest" / "_source.txt"
    h = _hash(oos)
    if not h:
        return 0
    if not force and stamp.exists() and stamp.read_text() == h:
        return 0   # bashoratlar o'zgarmagan -> qayta hisoblash shart emas
    n = 0
    for t in BT_THRESHOLDS:
        for c in BT_COSTS:
            try:
                _write(f"backtest/t{t:.2f}_c{c}.json", run_backtest(svc.panel, "HOLDOUT", "full", t, c))
                n += 1
            except FileNotFoundError as e:
                log.warning("Backtest: %s", e)
                return 0
    stamp.write_text(h)
    return n


def score_news(svc, limit: int = 200) -> int:
    """So'nggi xabarlar uchun to'liq ta'sir tahlili (saytdagi 'Agar shunday bo'lsa' bo'limi uchun)."""
    if svc.cube is None:
        return 0
    recent = svc.store.recent(limit)
    keep = {it["id"] for it in recent}
    adir = SITE / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    for f in adir.glob("*.json"):
        if f.stem not in keep:
            f.unlink()
    n = 0
    for it in recent:
        f = adir / f"{it['id']}.json"
        if f.exists() and it.get("impact"):
            continue
        try:
            res = svc.analyze_text(it["title"])
            f.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")))
            if not it.get("impact"):
                h1, h5 = (f"s_ret_h{h}" for h in svc.cfg.horizons[:2])
                k1 = f"sigma_h{svc.cfg.horizons[0]}"
                top = sorted(res.get("impact", []), key=lambda r: -abs(r[k1]))[:4]
                quick = [{"symbol": r["symbol"], "name": r["name"], "tagged": r["tagged"],
                          "ret_h1": round(r[h1], 4), "ret_h5": round(r[h5], 4), "sigma_h1": round(r[k1], 3)}
                         for r in top]
                svc.store.set_impact(it["id"], json.dumps(quick))
            n += 1
        except Exception as e:
            log.warning("Xabar baholanmadi (%s): %s", it["id"], e)
    return n


def export_site(svc, collector_stats: dict | None = None, price_stats: dict | None = None) -> dict:
    from .api import main as api

    api.SVC = svc
    for d in ("forecast", "history", "seasonality"):
        shutil.rmtree(SITE / d, ignore_errors=True)
    _write("assets.json", api.assets())
    fc = svc.forecast()
    _write("forecast.json", fc)
    for a in ASSETS:
        if a.symbol not in svc.latest:
            continue
        _write(f"forecast/{a.symbol}.json", api.forecast_one(a.symbol))
        df = svc.prices[a.symbol]
        df = df[df["date"] >= df["date"].max() - pd.Timedelta(days=int(365.25 * 11))]
        # ixcham ustunli format (grafik faqat yopilish narxini chizadi): t — sanalar, c — yopilish
        _write(f"history/{a.symbol}.json", {"symbol": a.symbol, "t": [str(d.date()) for d in df["date"]],
                                            "c": [float(f"{v:.6g}") for v in df["close"]]})
        for by in ("month", "week", "dow"):
            _write(f"seasonality/{a.symbol}_{by}.json", api.seasonality(a.symbol, by))
    _write("news.json", svc.store.recent(200))
    _write("report.json", svc.report)
    _write("experiments.json", api.experiments())
    _write("track.json", track_record(svc.prices))
    _write("backtest/index.json", {"thresholds": BT_THRESHOLDS, "costs": BT_COSTS})
    _write("health.json", {
        "ready": True, "error": None, "news_model": svc.cube is not None, "demo": False, "version": "atlas-3",
        "mode": "static", "updated_at": _now().isoformat(),
        "data_until": max((str(df["date"].max().date()) for df in svc.prices.values()), default=None),
        "collector": collector_stats or {}, "prices": price_stats or {},
    })
    return fc


def cloud_update(skip_prices: bool = False, skip_news: bool = False) -> dict:
    from .api.service import Service

    t0 = datetime.now(timezone.utc)
    price_stats = {} if skip_prices else update_prices()
    new, col_stats = ([], {}) if skip_news else collect_news()
    svc = Service()
    if not svc.ready:
        raise RuntimeError(f"Servis yuklanmadi: {svc.error}")
    scored = score_news(svc)
    fc = export_site(svc, col_stats, price_stats)
    logged = append_forecast_log(fc)
    _write("track.json", track_record(svc.prices))
    bt = export_backtests(svc)
    out = {"prices": price_stats, "news_new": len(new), "scored": scored, "logged": logged, "backtests": bt,
           "seconds": (datetime.now(timezone.utc) - t0).total_seconds()}
    log.info("cloud-update: %s", out)
    return out
