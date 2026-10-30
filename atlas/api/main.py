"""FastAPI: REST + WebSocket. Web interfeys `web/dist` dan beriladi."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import ASSET_BY_SYMBOL, DATA_DIR, NEWS_FEEDS, ROOT
from ..data.collector import Collector
from ..data.live import LiveQuotes
from ..features.seasonality import seasonal_profile
from .service import Service

log = logging.getLogger(__name__)
SVC: Service | None = None
CLIENTS: set[WebSocket] = set()
COLLECTOR: Collector | None = None
ENGINE = None          # jonli signal dvigateli (oltin 5m–4h + yangilik shoki)
_NEWS_THR = {"value": float("inf"), "at": 0.0}
LIVE = LiveQuotes(interval=int(os.environ.get("ATLAS_LIVE_INTERVAL", "30")))


async def _broadcast(msg: dict):
    dead = []
    for ws in CLIENTS:
        try:
            await ws.send_text(json.dumps(msg, default=str))
        except Exception:
            dead.append(ws)
    for ws in dead:
        CLIENTS.discard(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SVC, COLLECTOR
    SVC = Service()
    loop = asyncio.get_running_loop()

    def analyze_items(items, alert: bool = True):
        """Har bir yangi xabar uchun model ta'sirini hisoblaydi (alohida oqimda — API bloklanmaydi)
        va ta'sir kuchli bo'lsa push xabarnoma yuboradi."""
        import time as _t
        from ..live.engine import news_alert, news_threshold
        if _t.time() - _NEWS_THR["at"] > 600:
            _NEWS_THR["value"], _NEWS_THR["at"] = news_threshold(SVC.store), _t.time()
        for it in items:
            try:
                imp = SVC.quick_impact(it["title"])
                it["impact"] = json.dumps(imp)
                SVC.store.set_impact(it["id"], it["impact"])
                if alert and os.environ.get("ATLAS_SIGNALS", "1") == "1":
                    news_alert(it, imp, _NEWS_THR["value"])
            except Exception as e:  # tahlil xatosi lentani to'xtatmasin
                log.warning("Ta'sir hisoblanmadi: %s", e)
        return items

    def on_new(items):
        SVC.add_news(items)
        if SVC.ready:
            async def _work():
                done = await asyncio.to_thread(analyze_items, items[:50])
                await _broadcast({"type": "news", "items": done})
            loop.create_task(_work())
        else:
            loop.create_task(_broadcast({"type": "news", "items": items}))

    if SVC.ready:   # bazadagi ta'siri hisoblanmagan so'nggi xabarlar
        backlog = [it for it in SVC.store.recent(60) if not it.get("impact")]
        if backlog:
            loop.create_task(asyncio.to_thread(analyze_items, backlog, False))   # eski xabarlar uchun push yo'q

    if os.environ.get("ATLAS_LIVE", "1") == "1":
        LIVE.start()   # jonli narxlar (Binance + Yahoo), alohida oqimda

    global ENGINE
    if os.environ.get("ATLAS_SIGNALS", "1") == "1":
        from ..live.engine import LiveEngine
        ENGINE = LiveEngine(SVC.store, interval=int(os.environ.get("ATLAS_SIGNALS_INTERVAL", "60")))
        ENGINE.start()

    task = None
    if os.environ.get("ATLAS_COLLECT", "1") == "1":
        COLLECTOR = Collector(store=SVC.store, interval=int(os.environ.get("ATLAS_COLLECT_INTERVAL", "60")),
                              on_new=on_new)
        task = asyncio.create_task(COLLECTOR.run_forever())
    yield
    LIVE.stop()
    if ENGINE:
        ENGINE.stop()
    if task:
        task.cancel()


app = FastAPI(title="ATLAS API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _svc() -> Service:
    if SVC is None or not SVC.ready:
        raise HTTPException(503, detail=f"Model tayyor emas: {SVC.error if SVC else 'yuklanmoqda'}")
    return SVC


@app.get("/api/health")
def health():
    return {"ready": bool(SVC and SVC.ready), "error": SVC.error if SVC else None,
            "news_model": bool(SVC and SVC.ready and SVC.cube is not None),
            "demo": (DATA_DIR / "DEMO_SYNTHETIC").exists(),
            "version": "atlas-2", "live_quotes": len(LIVE.quotes),
            "collector": COLLECTOR.stats if COLLECTOR else {}}


@app.get("/api/quotes")
def quotes():
    """Barcha aktivlar uchun jonli narx (demo rejimda ham haqiqiy bozor narxi)."""
    return {"quotes": LIVE.quotes, "error": LIVE.error}


@app.get("/api/assets")
def assets():
    return _svc().assets_overview()


@app.get("/api/history/{symbol}")
def history(symbol: str, years: float = 10):
    s = _svc()
    df = s.prices.get(symbol)
    if df is None:
        raise HTTPException(404, "Aktiv topilmadi")
    df = df[df["date"] >= df["date"].max() - pd.Timedelta(days=int(365.25 * years))]
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return {"symbol": symbol, "name": ASSET_BY_SYMBOL[symbol].name,
            "rows": [{"t": str(d.date()), **{c: (None if pd.isna(v) else float(v)) for c, v in zip(cols, vals)}}
                     for d, *vals in df[["date"] + cols].itertuples(index=False)]}


@app.get("/api/forecast")
def forecast_all():
    return _svc().forecast()


@app.get("/api/forecast/{symbol}")
def forecast_one(symbol: str):
    r = _svc().forecast([symbol], explain=True)
    if symbol not in r:
        raise HTTPException(404, "Aktiv topilmadi")
    return r[symbol]


@app.get("/api/seasonality/{symbol}")
def seasonality(symbol: str, by: str = Query("month", pattern="^(month|week|dow)$")):
    s = _svc()
    df = s.prices.get(symbol)
    if df is None:
        raise HTTPException(404, "Aktiv topilmadi")
    prof, by_year = seasonal_profile(df, by)
    return {"symbol": symbol, "by": by,
            "profile": json.loads(prof.to_json(orient="records")),
            "by_year": {"years": [int(y) for y in by_year.index], "keys": [int(k) for k in by_year.columns],
                        "values": json.loads(by_year.round(2).to_json(orient="values"))}}


@app.get("/api/news")
def news(limit: int = 60, asset: str | None = None):
    s = SVC
    if s is None:
        return []
    return s.store.recent(limit, asset)


class AnalyzeIn(BaseModel):
    text: str


@app.post("/api/news/analyze")
def analyze(inp: AnalyzeIn):
    if len(inp.text.strip()) < 8:
        raise HTTPException(400, "Matn juda qisqa")
    return _svc().analyze_text(inp.text.strip())


@app.get("/api/report")
def report():
    return _svc().report


@app.get("/api/backtest")
def backtest(variant: str = "full", threshold: float = Query(0.05, ge=0, le=0.4),
             cost_bps: float = Query(5.0, ge=0, le=100)):
    from ..model.backtest import run_backtest
    s = _svc()
    key = (variant, round(threshold, 3), round(cost_bps, 2))
    cache = getattr(s, "_bt_cache", {})
    if key not in cache:
        try:
            cache[key] = run_backtest(s.panel, "HOLDOUT", variant, threshold, cost_bps)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        s._bt_cache = cache
    return cache[key]


@app.get("/api/track")
def track():
    """Prognoz jurnali: oldin berilgan prognozlar va ularning haqiqatdagi natijasi."""
    from ..cloud import track_record
    return track_record(_svc().prices)


@app.get("/api/ta/{symbol}")
def ta(symbol: str):
    """Texnik tahlil: SNR, SND, tuzilma, klassik figuralar, faol signallar va SetupNet bahosi."""
    from ..ta.service import analysis
    s = _svc()
    if symbol not in s.prices:
        raise HTTPException(404, "Aktiv topilmadi")
    fc = s.forecast([symbol]).get(symbol)
    return analysis(symbol, s.prices, fc)


@app.get("/api/signals")
def signals():
    """Jonli signallar: oltin 5m–4h va kuchli yangiliklar, natijalari bilan."""
    from ..live.engine import summary
    out = summary()
    out["engine"] = {"running": ENGINE is not None, "last_run": getattr(ENGINE, "last_run", None),
                     "error": getattr(ENGINE, "last_error", None)}
    return out


@app.get("/api/strategies")
def strategies():
    from ..ta.service import strategies as st
    return st()


@app.get("/api/experiments")
def experiments():
    from ..config import REPORTS_DIR
    reg = REPORTS_DIR / "experiments.jsonl"
    return [json.loads(l) for l in reg.read_text(encoding="utf-8", errors="replace").splitlines()] if reg.exists() else []


@app.get("/api/sources")
def sources():
    return {"feeds": NEWS_FEEDS, "stats": COLLECTOR.stats if COLLECTOR else {}}


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    CLIENTS.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        CLIENTS.discard(ws)


DIST = Path(os.environ.get("ATLAS_WEB", ROOT / "web" / "dist"))
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="static")

    @app.get("/{path:path}")
    def spa(path: str):
        if path.startswith("api/") or path == "api":   # noma'lum API manzili -> JSON 404, HTML emas
            raise HTTPException(404, f"API manzili topilmadi: /{path}")
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
