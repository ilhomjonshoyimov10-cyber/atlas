"""Ichki kun signallarini qayta tahlil: walk-forward (bir nechta ketma-ket sinov davri).

Bitta holdout kichik (5m–30m da 60 kunlik tarixning 15%i) va tasodifga juda sezgir. Shuning uchun:
  * har bir vaqt oralig'i tarixi K ta ketma-ket sinov blokiga bo'linadi;
  * har bir blok uchun model faqat undan OLDINGI ma'lumotda o'qitiladi (embargo 20 bar),
    "kuchli" chegarasi ham faqat oldingi ma'lumotdagi validatsiyada tanlanadi;
  * barcha bloklardagi sinov natijalari birlashtiriladi -> aniqlik, 90% ishonch oralig'i, AUC.
Shu yo'l bilan "kontekst bilan" va "kontekstsiz" variantlar halol solishtiriladi.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..ta.model import _auc, _fit_one, predict
from .intraday import MIN_SIGNALS_VAL, TIMEFRAMES, _tensors, build_dataset, wilson

log = logging.getLogger(__name__)


def _threshold(v: pd.DataFrame) -> float | None:
    best = None
    for q in np.arange(0.70, 0.99, 0.01):
        thr = float(v["p"].quantile(q))
        sel = v[v["p"] >= thr]
        if len(sel) < MIN_SIGNALS_VAL:
            break
        prec = sel["win"].mean()
        if best is None or prec > best[1]:
            best = (thr, prec)
    return best[0] if best else None


def walk_forward(frames: dict[str, pd.DataFrame], use_ctx: bool, folds: int = 4, seeds: int = 2,
                 test_frac: float = 0.40) -> dict:
    ds = build_dataset(frames, use_ctx=use_ctx)
    meta = ds["meta"]
    lab = meta["win"].notna().to_numpy()
    # har bir TF uchun nisbiy vaqt (0..1) — bloklar TF ichida xronologik
    rel = meta.groupby("tf")["t"].transform(lambda s: s / s.max()).to_numpy()
    edges = np.linspace(1 - test_frac, 1.0, folds + 1)
    preds = []
    for k in range(folds):
        lo, hi = edges[k], edges[k + 1]
        test = np.where(lab & (rel >= lo) & (rel < hi + (1e-9 if k == folds - 1 else 0)))[0]
        past = np.where(lab & (rel < lo))[0]
        # embargo: har bir TF da test boshlanishidan 20 bar oldingi signallar olib tashlanadi
        tmax = meta["tf"].map(meta.groupby("tf")["t"].max()).to_numpy()
        emb_ok = (meta["t"].to_numpy() < lo * tmax - 20)
        past = past[emb_ok[past]]
        pr = rel[past]
        cut = np.quantile(pr, 0.85)
        tr, va = past[pr < cut - 0.002], past[pr >= cut]
        Tr, Va, Te = _tensors(ds, tr), _tensors(ds, va), _tensors(ds, test)
        nets = [_fit_one(Tr, Va, 10 * k + s, n_groups=len(TIMEFRAMES)) for s in range(seeds)]
        vm = meta.iloc[va].assign(p=predict(nets, Va))
        tm = meta.iloc[test].assign(p=predict(nets, Te), fold=k)
        for tf in TIMEFRAMES:
            thr = _threshold(vm[vm["tf"] == tf])
            m = tm["tf"] == tf
            tm.loc[m, "thr"] = thr if thr is not None else np.inf
        preds.append(tm)
        log.info("fold %d (%.2f–%.2f): %d test signal", k, lo, hi, len(test))
    P = pd.concat(preds)
    out = {}
    for tf in TIMEFRAMES:
        g = P[P["tf"] == tf]
        if not len(g):
            continue
        s = g[g["p"] >= g["thr"]]
        k_, n_ = int(s["win"].sum()), int(len(s))
        lo_, hi_ = wilson(k_, n_)
        base = float(g["win"].mean())
        out[tf] = {"n_test": int(len(g)), "base_win": base * 100, "auc": _auc(g["win"].to_numpy(), g["p"].to_numpy()),
                   "strong_n": n_, "strong_win": (k_ / n_ * 100) if n_ else None,
                   "ci90": [lo_ * 100, hi_ * 100] if n_ else None,
                   "strong_avg_R": float(s["R"].mean()) if n_ else None,
                   "per_fold": [int(((g["fold"] == f) & (g["p"] >= g["thr"])).sum()) for f in range(folds)],
                   "enabled": bool(n_ >= 20 and lo_ > base)}
    return out
