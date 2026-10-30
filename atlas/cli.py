"""Buyruqlar:

  python -m atlas.cli real                     # BITTA BUYRUQ: haqiqiy narx + yangilik -> o'qitish -> baholash
  python -m atlas.cli download                 # 12 yillik narxlar (Yahoo Finance)
  python -m atlas.cli gdelt --days 90          # GDELT so'nggi yangiliklari (ixtiyoriy)
  python -m atlas.cli collect --interval 60    # real-time yangilik kollektori (doimiy)
  python -m atlas.cli features --embedder finbert
  python -m atlas.cli evaluate                 # walk-forward + ablation (holdoutga tegmaydi)
  python -m atlas.cli evaluate --final         # + final holdout (oxirida, BIR MARTA)
  python -m atlas.cli train                    # servis modeli
  python -m atlas.cli serve                    # API + web
  python -m atlas.cli backtest                 # holdout'da model signallari bo'yicha savdo natijasi
  python -m atlas.cli demo                     # hammasi sintetik ma'lumotda (internet kerak emas)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pickle

import numpy as np

from .config import ModelConfig

log = logging.getLogger("atlas")


def _device(arg: str) -> str:
    if arg != "auto":
        return arg
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_cached(cfg):
    from .config import NEWS_DIR
    from .data.prices import load_prices
    from .features.dataset import Panel
    from .features.news_agg import NewsCube
    from .pipeline import REDUCER_PATH

    panel = Panel.build(load_prices(), cfg.horizons)
    try:
        cube = NewsCube.load()
        emb = pickle.load(open(REDUCER_PATH, "rb"))["embedder"]
    except FileNotFoundError:
        cube, emb = None, "none"
    return panel, cube, emb


def main(argv=None):
    ap = argparse.ArgumentParser(prog="atlas")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download"); d.add_argument("--years", type=int, default=12)
    g = sub.add_parser("gdelt"); g.add_argument("--days", type=int, default=90)
    rl = sub.add_parser("real", help="haqiqiy ma'lumot bilan to'liq sozlash")
    rl.add_argument("--years", type=int, default=12)
    rl.add_argument("--gdelt-days", type=int, default=30)
    rl.add_argument("--embedder", default="auto", choices=["auto", "finbert", "hashing"])
    rl.add_argument("--full", action="store_true", help="to'liq rejim (GPU tavsiya etiladi)")
    rl.add_argument("--skip-eval", action="store_true", help="baholash/backtestni o'tkazib yuborish (tezroq)")
    c = sub.add_parser("collect"); c.add_argument("--interval", type=int, default=60)
    f = sub.add_parser("features"); f.add_argument("--embedder", default="auto", choices=["auto", "finbert", "hashing"])
    e = sub.add_parser("evaluate"); e.add_argument("--final", action="store_true")
    t = sub.add_parser("train")
    s = sub.add_parser("serve"); s.add_argument("--host", default="0.0.0.0"); s.add_argument("--port", type=int, default=8000)
    dm = sub.add_parser("demo"); dm.add_argument("--quick", action="store_true")
    bt = sub.add_parser("backtest"); bt.add_argument("--threshold", type=float, default=0.05)
    bt.add_argument("--cost-bps", type=float, default=5.0)
    lv = sub.add_parser("live", help="jonli signallar: yangilik shoki + oltin 5m–4h (push)")
    lv.add_argument("--once", action="store_true", help="bir marta ishlab chiqish (bulut/cron uchun)")
    lv.add_argument("--no-push", action="store_true")
    sub.add_parser("intraday-train", help="oltin ichki kun signal modelini qayta o'qitish")
    sub.add_parser("ta-train", help="texnik tahlil dataseti + SetupNet o'qitish")
    ta = sub.add_parser("ta", help="bitta aktiv bo'yicha texnik tahlil"); ta.add_argument("symbol")
    cu = sub.add_parser("cloud-update", help="statik sayt uchun ma'lumotni yangilash (GitHub Actions)")
    cu.add_argument("--skip-prices", action="store_true"); cu.add_argument("--skip-news", action="store_true")
    for p in (e, t, dm, rl):
        p.add_argument("--epochs", type=int, default=None)
        p.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = ModelConfig()
    if getattr(a, "epochs", None):
        cfg.epochs = a.epochs

    if a.cmd == "download":
        from .data.prices import download_all
        print(json.dumps(download_all(a.years), indent=1))
    elif a.cmd == "gdelt":
        _gdelt(a.days)
    elif a.cmd == "real":
        _real(a, cfg)
    elif a.cmd == "collect":
        from .data.collector import Collector
        asyncio.run(Collector(interval=a.interval).run_forever())
    elif a.cmd == "features":
        from .pipeline import build_features, load_inputs
        prices, news = load_inputs("real")
        build_features(prices, news, cfg, a.embedder)
    elif a.cmd == "evaluate":
        from .model.evaluate import walk_forward
        panel, cube, _ = _load_cached(cfg)
        rep = walk_forward(panel, cube, cfg, _device(a.device), final=a.final)
        print(json.dumps({"walk_forward": rep["walk_forward_summary"], "ablation": rep["ablation_tests"]}, indent=1))
    elif a.cmd == "train":
        from .pipeline import train_final
        panel, cube, emb = _load_cached(cfg)
        train_final(panel, cube, cfg, _device(a.device), emb)
    elif a.cmd == "backtest":
        from .model.backtest import run_backtest
        panel, _, _ = _load_cached(cfg)
        r = run_backtest(panel, "HOLDOUT", "full", a.threshold, a.cost_bps)
        print(json.dumps({"period": r["period"], "stats": r["stats"]}, indent=1))
    elif a.cmd == "live":
        _live(a)
    elif a.cmd == "intraday-train":
        from .live.engine import fetch_intraday
        from .live.intraday import load_frames, train as intra_train
        fetch_intraday()
        r = intra_train(load_frames())
        print(json.dumps({tf: {k: v.get(k) for k in ("enabled", "holdout_precision", "holdout_signals", "base_win")}
                          for tf, v in r["timeframes"].items()}, indent=1, default=float))
    elif a.cmd == "ta-train":
        from .ta.pipeline import train as ta_train
        r = ta_train()
        print(json.dumps({k: r[k] for k in ("auc", "auc_baseline", "brier_skill", "filter")}, indent=1))
    elif a.cmd == "ta":
        from .data.prices import load_prices
        from .ta.service import analysis
        print("\n".join(analysis(a.symbol.upper(), load_prices())["text"]))
    elif a.cmd == "cloud-update":
        from .cloud import cloud_update
        print(json.dumps(cloud_update(a.skip_prices, a.skip_news), indent=1, default=str))
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("atlas.api.main:app", host=a.host, port=a.port, reload=False)
    elif a.cmd == "demo":
        from .model.evaluate import walk_forward
        from .pipeline import build_features, load_inputs, train_final
        if a.quick:
            # Tez tekshiruv (CPU): kamroq epoch/fold, qisqa oyna
            cfg.epochs, cfg.wf_folds, cfg.hidden = min(cfg.epochs, 5), 2, 48
            cfg.lookback, cfg.batch_size, cfg.patience = 30, 1024, 2
            cfg.layers, cfg.ensemble = 1, 2
        prices, news = load_inputs("synthetic")
        panel, cube, _, emb = build_features(prices, news, cfg, "hashing")
        rep = walk_forward(panel, cube, cfg, _device(a.device), final=True)
        print(json.dumps({"walk_forward": rep["walk_forward_summary"], "ablation": rep["ablation_tests"]}, indent=1))
        train_final(panel, cube, cfg, _device(a.device), emb)
        _seed_demo_news(news)


def _gdelt(days: int):
    """GDELT DOC API'dan so'nggi kunlar yangiliklari (API so'rovlar orasida ~5 s kutishni so'raydi)."""
    import time
    import pandas as pd
    from .config import NEWS_DIR
    from .data.news_open import fetch_gdelt_doc
    end = pd.Timestamp.utcnow().tz_localize(None)
    frames = []
    for k in range(days):
        day_end = end - pd.Timedelta(days=k)
        try:
            frames.append(fetch_gdelt_doc(start=day_end - pd.Timedelta(days=1), end=day_end))
        except Exception as ex:
            log.warning("GDELT %s: %s", day_end.date(), ex)
        time.sleep(5)
    frames = [f for f in frames if len(f)]
    if not frames:
        log.warning("GDELT: hech narsa olinmadi")
        return 0
    df = pd.concat(frames).drop_duplicates("id")
    df.to_parquet(NEWS_DIR / "extra.parquet", index=False)
    log.info("GDELT: %d xabar saqlandi", len(df))
    return len(df)


def _real(a, cfg: ModelConfig):
    """Demo'dan haqiqiy ma'lumotga o'tish: narxlar -> yangiliklar -> embedding -> baholash -> servis modeli."""
    from .config import REPORTS_DIR
    from .data.collector import NewsStore
    from .data.prices import download_all
    from .model.evaluate import walk_forward
    from .pipeline import build_features, load_inputs, train_final

    if not a.full:   # CPU uchun tez sozlamalar
        cfg.epochs, cfg.wf_folds, cfg.hidden = min(cfg.epochs, 6), 1, 48
        cfg.lookback, cfg.batch_size, cfg.patience = 30, 1024, 2
        cfg.layers, cfg.ensemble = 1, 2
    if a.epochs:
        cfg.epochs = a.epochs
    log.info("1/5 Narxlar yuklanmoqda (%d yil)...", a.years)
    got = download_all(a.years)
    if len(got) < 5:
        raise SystemExit("Narxlar yuklanmadi — internet aloqasini tekshiring (Yahoo Finance).")
    log.info("2/5 Demo yangiliklari o'chirildi: %d; GDELT %d kun...", NewsStore().purge_demo(), a.gdelt_days)
    if a.gdelt_days:
        _gdelt(a.gdelt_days)
    log.info("3/5 Yangilik embeddinglari...")
    prices, news = load_inputs("real")
    panel, cube, _, emb = build_features(prices, news, cfg, a.embedder)
    for f in REPORTS_DIR.glob("oos_*.npz"):
        f.unlink()
    if not a.skip_eval:
        log.info("4/5 Baholash (walk-forward + holdout + backtest uchun bashoratlar)...")
        variants = ("climatology", "price", "no_news", "full") if not a.full else \
            ("climatology", "price", "price+season", "no_news", "full")
        walk_forward(panel, cube, cfg, _device(a.device), final=True, variants=variants)
    log.info("5/5 Servis modeli o'qitilmoqda...")
    train_final(panel, cube, cfg, _device(a.device), emb)
    log.info("TAYYOR. Endi: python -m atlas.cli serve  ->  http://localhost:8000")


def _live(a):
    """Jonli signallar. --once: yangiliklarni bir marta yig'ib baholaydi, oltin signallarini tekshiradi va chiqadi."""
    import time as _t
    from .live.engine import LiveEngine, fetch_intraday, gold_tick, news_alert, news_threshold, recent_gold_news, score_open
    from .api.service import Service
    svc = Service()
    push = not a.no_push
    def news_round():
        from .cloud import collect_news
        new, _ = collect_news()
        thr = news_threshold(svc.store)
        svc.add_news(new)
        for it in new[:40]:
            try:
                imp = svc.quick_impact(it["title"])
                svc.store.set_impact(it["id"], json.dumps(imp))
                news_alert(it, imp, thr, push)
            except Exception as e:
                log.warning("Xabar baholanmadi: %s", e)
        return len(new)
    while True:
        n = news_round()
        frames = fetch_intraday()
        fired = gold_tick(frames, recent_gold_news(svc.store), push)
        score_open(frames)
        log.info("live: %d yangi xabar, %d kuchli oltin signali", n, len(fired))
        if a.once:
            break
        _t.sleep(60)


def _seed_demo_news(news):
    """Demo: so'nggi sintetik xabarlarni kollektor bazasiga yozadi (UI lentasi bo'sh qolmasin)."""
    from .data.collector import NewsStore
    from .features.tagging import detect_event, tag_assets
    last = news.sort_values("available_at").tail(40)
    NewsStore().insert([{
        "id": f"demo-{r.id}", "origin": "synthetic", "source": "demo", "title": r.title, "url": "",
        "published_at": r.available_at.isoformat(), "first_seen": r.available_at.isoformat(),
        "assets": ",".join(tag_assets(r.title)), "event": detect_event(r.title)} for r in last.itertuples()])


if __name__ == "__main__":
    main()
