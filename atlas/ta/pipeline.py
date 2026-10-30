"""Texnik tahlil quvuri: dataset -> statistika -> SetupNet (holdout baholash) -> servis modeli."""
from __future__ import annotations

import json
import logging

from ..config import MODELS_DIR, REPORTS_DIR

log = logging.getLogger(__name__)
TA_MODEL = MODELS_DIR / "ta_setup.pt"
TA_REPORT = REPORTS_DIR / "ta_report.json"


def train(prices=None, n_models: int = 3) -> dict:
    from ..data.prices import load_prices
    from .dataset import build
    from .model import save, train_and_evaluate, train_final

    prices = prices or load_prices()
    ds = build(prices)
    meta = ds["meta"]
    log.info("TA dataset: %d signal, %d aktiv", len(meta), meta["symbol"].nunique())
    _, report = train_and_evaluate(ds, n_models)
    report["dataset"] = {"n_events": int(len(meta)), "n_labeled": int(meta["win"].notna().sum()),
                         "assets": int(meta["symbol"].nunique()),
                         "from": str(meta["date"].min().date()), "to": str(meta["date"].max().date()),
                         "by_setup": meta["setup"].value_counts().to_dict()}
    nets = train_final(ds, n_models)
    lab = meta[meta["win"].notna()]
    base_rates = {"setup": lab.groupby("setup")["win"].mean().to_dict(),
                  "setup_group": {f"{k[0]}|{k[1]}": v for k, v in lab.groupby(["setup", "group"])["win"].mean().items()},
                  "setup_n": lab.groupby("setup").size().to_dict()}
    save(nets, TA_MODEL, base_rates, report)
    TA_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return report
