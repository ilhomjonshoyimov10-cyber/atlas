"""Walk-forward baholash + tegilmagan final holdout + ablation (yangilik foyda beryaptimi?).

Protokol:
  1. Oxirgi `holdout_days` kalendar kuni — FINAL HOLDOUT. Unga faqat `--final` bilan, oxirida, bir marta qaraladi.
  2. Holdoutdan oldingi davrda `wf_folds` ta walk-forward fold: [train | embargo | test].
     Train ichida: oxirgi 15% — validatsiya (early stopping + konformal kalibratsiya).
     Embargo = maksimal gorizont (20 savdo kuni ~ 32 kalendar kun): target oynalari test bilan ustma-ust tushmaydi.
  3. Har bir foldda bir xil sharoitda variantlar o'qitiladi:
        climatology  — har bir aktiv uchun shartsiz kvantillar (eng oddiy baza)
        price        — faqat narx shoxi
        price+season — narx + mavsumiylik
        full         — narx + mavsumiylik + yangilik
  4. Diebold–Mariano testi (Newey–West HAC) + Holm tuzatishi — ko'p taqqoslash uchun.
  QOIDA: feature yoki giperparametrlarni holdout/test natijasiga qarab TANLAMANG.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import replace

import numpy as np
import pandas as pd

from ..config import REPORTS_DIR, ModelConfig
from ..features.dataset import BatchMaker, Panel
from .train import apply_conformal, conformal_offsets, fit_ensemble, metrics, pinball_per_row, predict

log = logging.getLogger(__name__)
# variant: (yangilik, mavsumiylik, bozorlararo)
VARIANTS = {"price": (False, False, False), "price+season": (False, True, False),
            "no_news": (False, True, True), "full": (True, True, True)}


def split_dates(panel: Panel, cfg: ModelConfig):
    rows = panel.sample_rows(cfg.lookback, need_target=True)
    dates = pd.DatetimeIndex(panel.date[rows])
    end = dates.max()
    hold_start = end - pd.Timedelta(days=cfg.holdout_days)
    embargo = pd.Timedelta(days=int(max(cfg.horizons) * 1.5) + 2)
    test_len = pd.Timedelta(days=cfg.wf_test_days)
    folds = []
    for k in range(cfg.wf_folds):
        t0 = hold_start - (cfg.wf_folds - k) * test_len
        t1 = t0 + test_len - embargo  # test oxiri ham keyingi bilan ustma-ust tushmasin
        train_end = t0 - embargo
        tr_dates = dates[dates < train_end]
        if len(tr_dates) == 0:
            continue
        span = train_end - tr_dates.min()
        val_start = train_end - span * 0.15
        fit_mask = dates < (val_start - embargo)
        val_mask = (dates >= val_start) & (dates < train_end)
        test_mask = (dates >= t0) & (dates < t1)
        if fit_mask.sum() < 1000 or test_mask.sum() < 100:
            continue
        folds.append({"k": k, "fit": rows[fit_mask], "val": rows[val_mask], "test": rows[test_mask],
                      "test_start": str(t0.date()), "test_end": str(t1.date())})
    final_train_end = hold_start - embargo
    span = final_train_end - dates.min()
    fval_start = final_train_end - span * 0.10
    final = {"fit": rows[dates < fval_start - embargo],
             "val": rows[(dates >= fval_start) & (dates < final_train_end)],
             "test": rows[dates >= hold_start], "test_start": str(hold_start.date()), "test_end": str(end.date())}
    return folds, final


def climatology(panel: Panel, fit_rows, test_rows, quantiles) -> np.ndarray:
    H = panel.y.shape[1]
    q = np.zeros((len(test_rows), H, 3), "float32")
    a_fit, a_test = panel.asset[fit_rows], panel.asset[test_rows]
    for a in np.unique(a_test):
        src = panel.y[fit_rows[a_fit == a]] if (a_fit == a).sum() > 50 else panel.y[fit_rows]
        q[a_test == a] = np.quantile(src, quantiles, axis=0).T
    return q


def newey_west_dm(d: np.ndarray, lag: int) -> tuple[float, float]:
    """d — kunlik yo'qotish farqi (A - B). Manfiy => A yaxshiroq. (stat, p-qiymat ikki tomonlama)."""
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 30:
        return float("nan"), float("nan")
    m = d.mean()
    x = d - m
    v = x @ x / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        v += 2 * w * (x[k:] @ x[:-k]) / n
    se = math.sqrt(max(v, 1e-18) / n)
    stat = m / se
    p = math.erfc(abs(stat) / math.sqrt(2))
    return float(stat), float(p)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def _daily(panel: Panel, rows, loss) -> pd.Series:
    return pd.Series(loss, index=pd.DatetimeIndex(panel.date[rows])).groupby(level=0).mean()


def run_variant(name, cfg, bm_full, bm_none, split, panel):
    if name == "climatology":
        q = climatology(panel, split["fit"], split["test"], cfg.quantiles)
        return q, None, None
    use_news, use_season, use_market = VARIANTS[name]
    vcfg = replace(cfg, use_news=use_news, use_season=use_season, use_market=use_market)
    bm = bm_full if use_news else bm_none
    model, info = fit_ensemble(vcfg, bm, split["fit"], split["val"], n=cfg.extra.get("wf_ensemble", 1),
                               verbose=False)
    off = conformal_offsets(predict(model, bm, split["val"]), panel.y[split["val"]])
    q = apply_conformal(predict(model, bm, split["test"]), off)
    return q, model, off


def walk_forward(panel: Panel, cube, cfg: ModelConfig, device="cpu", final: bool = False,
                 variants=("climatology", "price", "price+season", "no_news", "full")) -> dict:
    bm_full = BatchMaker(panel, cube, cfg, device)
    bm_none = BatchMaker(panel, None, cfg, device) if cube is not None else bm_full
    folds, fin = split_dates(panel, cfg)
    if cube is None:
        variants = tuple(v for v in variants if v != "full")
    report = {"config": {k: v for k, v in cfg.__dict__.items() if k != "extra"}, "folds": [], "variants": list(variants)}
    daily: dict[str, list[pd.Series]] = {v: [] for v in variants}
    splits = [(f"fold{f['k']}", f) for f in folds] + ([("HOLDOUT", fin)] if final else [])
    for tag, sp in splits:
        log.info("== %s  test %s .. %s  (fit=%d val=%d test=%d)", tag, sp["test_start"], sp["test_end"],
                 len(sp["fit"]), len(sp["val"]), len(sp["test"]))
        res = {"fold": tag, "test_start": sp["test_start"], "test_end": sp["test_end"], "metrics": {}}
        y = panel.y[sp["test"]]
        preds = {}
        for v in variants:
            q, _, _ = run_variant(v, cfg, bm_full, bm_none, sp, panel)
            preds[v] = q
            res["metrics"][v] = metrics(q, y, cfg.horizons, cfg.quantiles)
            if tag != "HOLDOUT":
                daily[v].append(_daily(panel, sp["test"], pinball_per_row(q, y, cfg.quantiles)))
            log.info("   %-13s pinball %.4f  cov90(h1) %.3f", v, res["metrics"][v]["pinball"],
                     res["metrics"][v][f"h{cfg.horizons[0]}"]["coverage90"])
        report["folds"].append(res)
        # Tashqi (out-of-sample) bashoratlarni saqlaymiz — backtest faqat shulardan foydalanadi
        np.savez_compressed(REPORTS_DIR / f"oos_{tag}.npz", rows=sp["test"], dates=panel.date[sp["test"]],
                            assets=panel.asset[sp["test"]], **{k: v for k, v in preds.items()})

    # Walk-forward bo'yicha yig'indi + DM testlar (holdout bunga kirmaydi)
    summ = {}
    for v in variants:
        ms = [f["metrics"][v] for f in report["folds"] if f["fold"] != "HOLDOUT"]
        if ms:
            summ[v] = {"pinball": float(np.mean([m["pinball"] for m in ms]))}
            for h in cfg.horizons:
                summ[v][f"h{h}_coverage90"] = float(np.mean([m[f"h{h}"]["coverage90"] for m in ms]))
                summ[v][f"h{h}_dir_acc"] = float(np.mean([m[f"h{h}"]["dir_acc"] for m in ms]))
    report["walk_forward_summary"] = summ
    # Asosiy savol: yangilik foyda beradimi? -> full vs no_news (boshqa hamma narsa bir xil)
    comps = [("full", "no_news"), ("no_news", "price+season"), ("price+season", "price"),
             ("price", "climatology"), ("full", "climatology")]
    raw = {}
    for a, b in comps:
        if daily.get(a) and daily.get(b):
            da, db = pd.concat(daily[a]), pd.concat(daily[b])
            d = (da - db.reindex(da.index)).values
            stat, p = newey_west_dm(d, lag=max(cfg.horizons))
            raw[f"{a} vs {b}"] = (stat, p, float(np.nanmean(d)))
    adj = holm({k: v[1] for k, v in raw.items() if not math.isnan(v[1])})
    report["ablation_tests"] = {k: {"dm_stat": s, "p": p, "p_holm": adj.get(k), "mean_loss_diff": md,
                                    "verdict": ("A yaxshiroq" if md < 0 else "A yomonroq") +
                                               (" (statistik ahamiyatli)" if adj.get(k, 1) < 0.05 else " (ahamiyatsiz)")}
                                for k, (s, p, md) in raw.items()}
    report["experiment_no"] = register_experiment(report, final)
    path = REPORTS_DIR / ("report_final.json" if final else "report_walkforward.json")
    path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    log.info("Hisobot: %s (tajriba #%d)", path, report["experiment_no"])
    return report


def register_experiment(report: dict, final: bool) -> int:
    """Har bir baholashni reyestrga yozadi. Ko'p tajriba = ko'p test: shu ro'yxat sizga nechta urinish
    qilinganini halol ko'rsatadi (p-qiymatlarni shunga qarab talqin qiling). Final holdout ishlatilgan
    bo'lsa, buni ham qayd etadi — holdoutni qayta-qayta ishlatish uni "test"likdan chiqaradi."""
    import datetime as dt
    import hashlib
    reg = REPORTS_DIR / "experiments.jsonl"
    prev = reg.read_text(encoding="utf-8", errors="replace").splitlines() if reg.exists() else []
    cfg_hash = hashlib.sha1(json.dumps(report["config"], sort_keys=True, default=str).encode()).hexdigest()[:10]
    holdout_uses = sum(1 for l in prev if json.loads(l).get("final")) + int(final)
    rec = {"no": len(prev) + 1, "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "config_hash": cfg_hash, "final": final, "holdout_uses_total": holdout_uses,
           "summary": {k: v.get("pinball") for k, v in report.get("walk_forward_summary", {}).items()},
           "tests": {k: v["p_holm"] for k, v in report.get("ablation_tests", {}).items()}}
    with reg.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    report["holdout_uses_total"] = holdout_uses
    if final and holdout_uses > 1:
        log.warning("DIQQAT: final holdout %d marta ishlatildi — u endi to'liq 'tegilmagan' emas!", holdout_uses)
    return rec["no"]
