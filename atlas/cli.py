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
  python -m atlas.cli backtest --segment wf    # walk-forward OOS'da professional backtest (Sharpe, DSR, ...)
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
    bt = sub.add_parser("backtest", help="professional backtest (out-of-sample, xarajat, statistik testlar)")
    bt.add_argument("--threshold", type=float, default=0.01)
    bt.add_argument("--segment", choices=["wf", "holdout"], default="wf")
    bt.add_argument("--strategy", choices=["direction", "long_only", "cross_sectional"], default="direction")
    bt.add_argument("--cost", default="real", help="'real' (aktiv sinfi bo'yicha) yoki bps raqam")
    bt.add_argument("--export", action="store_true", help="saytdagi barcha backtest fayllarini qayta hisoblash")
    bt.add_argument("--cost-bps", type=float, default=5.0)
    lv = sub.add_parser("live", help="jonli signallar: yangilik shoki + oltin 5m–4h (push)")
    lv.add_argument("--once", action="store_true", help="bir marta ishlab chiqish (bulut/cron uchun)")
    lv.add_argument("--no-push", action="store_true")
    sub.add_parser("intraday-train", help="oltin ichki kun signal modelini qayta o'qitish")
    sub.add_parser("ta-train", help="texnik tahlil dataseti + SetupNet o'qitish")
    ta = sub.add_parser("ta", help="bitta aktiv bo'yicha texnik tahlil"); ta.add_argument("symbol")
    cu = sub.add_parser("cloud-update", help="statik sayt uchun ma'lumotni yangilash (GitHub Actions)")
    cu.add_argument("--skip-prices", action="store_true"); cu.add_argument("--skip-news", action="store_true")
    pt = sub.add_parser("pretrain", help="self-supervised: narx Transformerini oldindan o'qitish")
    pt.add_argument("--pt-epochs", type=int, default=20); pt.add_argument("--mask-ratio", type=float, default=0.3)
    tu = sub.add_parser("tune", help="Optuna giperparametr qidiruvi (holdout va oxirgi foldga tegmaydi)")
    tu.add_argument("--trials", type=int, default=30); tu.add_argument("--timeout", type=int, default=None)
    for p in (e, t, dm, rl, pt, tu):
        p.add_argument("--epochs", type=int, default=None)
        p.add_argument("--device", default="auto")
        p.add_argument("--preset", default=None, help="cpu | default | gpu | gpu_xl")
        p.add_argument("--tuned", action="store_true", help="data/models/best_config.json dagi sozlamalarni qo'llash")
        p.add_argument("--pretrained", default=None, help="oldindan o'qitilgan kodlovchi (pretrain natijasi)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = ModelConfig()
    if getattr(a, "preset", None):
        from .presets import apply_preset
        cfg = apply_preset(cfg, a.preset)
    if getattr(a, "tuned", False):
        from dataclasses import replace as _rep
        from .config import MODELS_DIR
        cfg = _rep(cfg, **json.loads((MODELS_DIR / "best_config.json").read_text(encoding="utf-8")))
    if getattr(a, "pretrained", None):
        cfg.extra = {**cfg.extra, "pretrained": a.pretrained}
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
        panel, _, _ = _load_cached(cfg)
        if a.export:
            from types import SimpleNamespace
            from .cloud import export_backtests
            print("fayllar:", export_backtests(SimpleNamespace(panel=panel), force=True))
        else:
            from .backtest import build_report, clean, load_market, robustness
            mkt = load_market(panel, a.segment)
            rob = robustness(mkt)
            cost = a.cost if a.cost == "real" else float(a.cost)
            r = clean(build_report(mkt, a.strategy, a.threshold, cost, rob, perm_n=200))
            print(f"{r['strategy_name']} | {r['period'][0]} — {r['period'][1]} | xarajat: {a.cost} | "
                  f"sinovlar: {rob['n_trials']}, deflated SR0 = {rob['sr0_deflated']:.2f}")
            for v, d in r["variants"].items():
                s_, t_ = d["stats"], d["tests"]
                print(f"  {v:13s} Sharpe {s_['sharpe']:6.2f}  CAGR {s_['cagr']:7.2f}%  MaxDD {s_['max_dd']:7.2f}%  "
                      f"CI90 {t_['sharpe_ci90']}  PSR {t_['psr']}  DSR {t_.get('dsr')}")
    elif a.cmd == "pretrain":
        _pretrain(a, cfg)
    elif a.cmd == "tune":
        from .model.tune import tune
        panel, cube, _ = _load_cached(cfg)
        r = tune(panel, cube, cfg, _device(a.device), a.trials, a.timeout)
        print(json.dumps({k: r[k] for k in ("best_params", "best_search_loss", "check_fold_loss_best",
                                            "check_fold_loss_base", "check_verdict")}, indent=1, default=float))
    elif a.cmd == "live":
        _live(a)
    elif a.cmd == "intraday-train":
        from .live.engine import fetch_intraday
        from .live.intraday import load_frames, train_final as intra_final
        from .live.walkforward import walk_forward
        fetch_intraday()
        frames = load_frames()
        r = intra_final(frames, walk_forward(frames, use_ctx=True))
        print(json.dumps({tf: {k: v.get(k) for k in ("enabled", "holdout_precision", "holdout_ci90", "holdout_signals", "base_win")}
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
    fails = 0
    for k in range(days):
        day_end = end - pd.Timedelta(days=k)
        for attempt in range(3):
            try:
                frames.append(fetch_gdelt_doc(start=day_end - pd.Timedelta(days=1), end=day_end))
                fails = 0
                break
            except Exception as ex:
                busy = "429" in str(ex)
                log.warning("GDELT %s: %s", day_end.date(), "429 — server band, kutilmoqda" if busy else ex)
                if not busy:
                    break
                time.sleep(15 * (attempt + 1))      # 429: sekinlashtirib qayta urinish
        else:
            fails += 1
        if fails >= 5:
            log.warning("GDELT ketma-ket 5 kun javob bermadi — to'xtatildi (boshqa yangilik manbalari bilan davom etiladi)")
            break
        time.sleep(6)
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

    if not a.full and not getattr(a, "preset", None):   # CPU uchun tez sozlamalar (preset berilmagan bo'lsa)
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


def _pretrain(a, cfg):
    """Eng birinchi walk-forward test davridan OLDINGI qatorlarda masked-reconstruction oldindan o'qitish."""
    import pandas as pd
    import torch
    from .config import MODELS_DIR, REPORTS_DIR
    from .features.dataset import BatchMaker
    from .model.evaluate import split_dates
    from .model.pretrain import pretrain
    panel, _, _ = _load_cached(cfg)
    folds, fin = split_dates(panel, cfg)
    first_test = min([pd.Timestamp(f["test_start"]) for f in folds] + [pd.Timestamp(fin["test_start"])])
    rows = panel.sample_rows(cfg.lookback, need_target=False)
    rows = rows[pd.DatetimeIndex(panel.date[rows]) < first_test - pd.Timedelta(days=5)]
    bm = BatchMaker(panel, None, cfg, _device(a.device))
    state, info = pretrain(cfg, bm, rows, epochs=a.pt_epochs, ratio=a.mask_ratio)
    info["cutoff"] = str(first_test.date())
    torch.save(state, MODELS_DIR / "pretrained.pt")
    (REPORTS_DIR / "pretrain.json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: info[k] for k in ("best_val_mse", "baseline_mean_fill_mse", "rows", "cutoff")}, indent=1))


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
