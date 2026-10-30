"""Uchidan-uchigacha quvur: data -> featurelar -> o'qitish -> baholash -> servis uchun model."""
from __future__ import annotations

import logging
import pickle
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import EMB_DIR, MODELS_DIR, PRICES_DIR, ModelConfig
from .features.dataset import BatchMaker, Panel
from .features.news_agg import NewsCube, build_from_news

log = logging.getLogger(__name__)
REDUCER_PATH = EMB_DIR / "reducer.pkl"
BUNDLE_PATH = MODELS_DIR / "atlas.pt"


def load_inputs(source: str = "real"):
    from .config import DATA_DIR
    marker = DATA_DIR / "DEMO_SYNTHETIC"
    if source == "synthetic":
        from .data.synthetic import generate
        prices, news = generate()
        marker.write_text("Bu ma'lumotlar SINTETIK (demo). Haqiqiy narxlar uchun: python -m atlas.cli download\n")
        for s, df in prices.items():
            df.to_parquet(PRICES_DIR / f"{s}.parquet", index=False)
        return prices, news
    marker.unlink(missing_ok=True)
    from .data.news_open import load_all_news
    from .data.prices import load_prices
    return load_prices(), load_all_news()


def build_features(prices, news, cfg: ModelConfig, embedder_kind: str = "auto"):
    from .features.text_embed import get_embedder

    panel = Panel.build(prices, cfg.horizons)
    start = pd.Timestamp(panel.date.min()) - pd.Timedelta(days=cfg.news_days + 1)
    end = pd.Timestamp(panel.date.max()) + pd.Timedelta(days=2)
    holdout_start = pd.Timestamp(panel.date.max()) - pd.Timedelta(days=cfg.holdout_days)
    if news is None or len(news) == 0:
        log.warning("Yangilik yo'q — faqat narx+mavsumiylik modeli o'qitiladi")
        return panel, None, None, "none"
    emb = get_embedder(embedder_kind)
    cube, reducer, _ = build_from_news(news, emb, cfg.emb_dim, fit_before=holdout_start, start=start, end=end)
    cube.save()
    with open(REDUCER_PATH, "wb") as f:
        pickle.dump({"reducer": reducer.state(), "embedder": emb.name}, f)
    return panel, cube, reducer, emb.name


def train_final(panel: Panel, cube, cfg: ModelConfig, device="cpu", embedder="hashing"):
    """Servis modeli: holdoutdan tashqari BARCHA ma'lumotda emas — eng oxirgi ma'lumotgacha o'qitiladi.

    Baholash (walk-forward + holdout) alohida bajariladi va hisobotga yoziladi; servis uchun esa
    eng so'nggi bozor holatini bilish muhim, shuning uchun oxirgi 10% validatsiya bilan to'liq o'qitamiz.
    """
    from .model.train import conformal_offsets, fit_ensemble, predict, save_bundle

    bm = BatchMaker(panel, cube, cfg, device)
    rows = panel.sample_rows(cfg.lookback, need_target=True)
    dates = pd.DatetimeIndex(panel.date[rows])
    cut = dates.max() - (dates.max() - dates.min()) * 0.10
    emb = pd.Timedelta(days=int(max(cfg.horizons) * 1.5) + 2)
    fit_rows, val_rows = rows[dates < cut - emb], rows[dates >= cut]
    cfg_s = replace(cfg, use_news=cube is not None)
    model, info = fit_ensemble(cfg_s, bm, fit_rows, val_rows)
    off = conformal_offsets(predict(model, bm, val_rows), panel.y[val_rows])
    import json
    from .config import REPORTS_DIR
    rep = {}
    for p in ("report_final.json", "report_walkforward.json"):
        if (REPORTS_DIR / p).exists():
            rep[p] = json.loads((REPORTS_DIR / p).read_text())
    save_bundle(BUNDLE_PATH, model, cfg_s, off, bm.news_dim, None, rep, embedder)
    log.info("Servis modeli saqlandi: %s (konformal: %s)", BUNDLE_PATH, np.round(off, 3))
    return model, off
