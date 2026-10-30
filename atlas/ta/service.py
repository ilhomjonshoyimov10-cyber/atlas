"""Texnik tahlil servisi: model bir marta yuklanadi, tahlil va strategiya hisobotini beradi."""
from __future__ import annotations

import json
import logging
from functools import lru_cache

from .pipeline import TA_MODEL, TA_REPORT

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def model():
    from .model import load
    if not TA_MODEL.exists():
        log.warning("SetupNet topilmadi (%s) — tahlil modelsiz, faqat tarixiy statistika bilan", TA_MODEL)
        return [], {}
    return load(TA_MODEL)


def analysis(symbol: str, prices: dict, forecast: dict | None = None) -> dict:
    from .analyze import analyze
    nets, z = model()
    return analyze(symbol, prices, nets, z, forecast)


def strategies(today: list[dict] | None = None) -> dict:
    """Strategiyalar sahifasi: dataset, har bir strategiyaning natijasi, SetupNet holdout bahosi."""
    if not TA_REPORT.exists():
        return {"ready": False}
    rep = json.loads(TA_REPORT.read_text())
    keep = ("holdout_start", "holdout_end", "train_end", "n_train", "n_val", "n_holdout", "auc", "auc_baseline",
            "brier", "brier_baseline", "brier_skill", "holdout_win_rate", "filter", "stats_train", "stats_holdout",
            "per_setup_holdout", "dataset")
    out = {"ready": True, **{k: rep.get(k) for k in keep}}
    out["rules"] = {"tp_atr": 1.5, "sl_atr": 1.0, "horizon": 20}
    if today is not None:
        out["today"] = today
    return out


def today_signals(analyses: dict[str, dict]) -> list[dict]:
    rows = []
    for sym, a in analyses.items():
        for s in a.get("setups", []):
            if s["date"] == a["as_of"]:
                rows.append({"symbol": sym, "name": a["name"], **{k: s[k] for k in
                             ("setup", "label", "family", "dir", "p_model", "base_win", "strong", "entry", "tp", "sl")}})
    return sorted(rows, key=lambda r: -((r["p_model"] or 0) - (r["base_win"] or 0)))
