"""Giperparametrlarni qidirish (Optuna, TPE + pruning).

Halollik qoidasi: qidiruv faqat walk-forward foldlarining BIRINCHI qismida baholanadi; oxirgi fold va holdout
qidiruvga umuman ko'rsatilmaydi. Tanlangan sozlama keyin oxirgi foldda tekshiriladi — agar u yerda ham
yaxshi bo'lsa, natija haqiqiy; faqat qidiruv foldlarida yaxshi bo'lsa — bu moslashib qolish (overfit).
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace

import numpy as np

from ..config import REPORTS_DIR, ModelConfig
from ..features.dataset import BatchMaker
from .evaluate import split_dates
from .train import apply_conformal, conformal_offsets, fit, pinball_np, predict

log = logging.getLogger(__name__)

SPACE = {
    "hidden": [64, 96, 128, 192],
    "layers": [1, 2, 3, 4],
    "heads": [4, 8],
    "lookback": [60, 90, 120, 180],
    "news_days": [5, 7, 10, 14],
    "dropout": (0.1, 0.4),
    "lr": (2e-4, 2e-3),
    "weight_decay": (1e-4, 1e-2),
}


def _score(cfg: ModelConfig, panel, cube, folds, device) -> list[float]:
    bm = BatchMaker(panel, cube, cfg, device)
    out = []
    for f in folds:
        m, _ = fit(cfg, bm, f["fit"], f["val"], verbose=False)
        off = conformal_offsets(predict(m, bm, f["val"]), panel.y[f["val"]])
        q = apply_conformal(predict(m, bm, f["test"]), off)
        out.append(pinball_np(q, panel.y[f["test"]], cfg.quantiles))
    return out


def tune(panel, cube, base: ModelConfig, device: str = "cpu", trials: int = 30, timeout: int | None = None) -> dict:
    import optuna

    folds_all, _ = split_dates(panel, base)
    if len(folds_all) < 2:
        raise SystemExit("Qidiruv uchun kamida 2 ta walk-forward fold kerak (wf_folds ni oshiring)")
    search, check = folds_all[:-1], folds_all[-1:]

    def objective(trial: "optuna.Trial") -> float:
        cfg = replace(base,
                      hidden=trial.suggest_categorical("hidden", SPACE["hidden"]),
                      layers=trial.suggest_categorical("layers", SPACE["layers"]),
                      heads=trial.suggest_categorical("heads", SPACE["heads"]),
                      lookback=trial.suggest_categorical("lookback", SPACE["lookback"]),
                      news_days=trial.suggest_categorical("news_days", SPACE["news_days"]),
                      dropout=trial.suggest_float("dropout", *SPACE["dropout"]),
                      lr=trial.suggest_float("lr", *SPACE["lr"], log=True),
                      weight_decay=trial.suggest_float("weight_decay", *SPACE["weight_decay"], log=True))
        # lookback o'zgarsa qatorlar ham o'zgaradi -> foldlar shu sozlama uchun qayta hisoblanadi
        fs, _ = split_dates(panel, cfg)
        fs = fs[:-1]
        losses = []
        bm = BatchMaker(panel, cube, cfg, device)
        for i, f in enumerate(fs):
            m, _ = fit(cfg, bm, f["fit"], f["val"], verbose=False)
            off = conformal_offsets(predict(m, bm, f["val"]), panel.y[f["val"]])
            losses.append(pinball_np(apply_conformal(predict(m, bm, f["test"]), off), panel.y[f["test"]], cfg.quantiles))
            trial.report(float(np.mean(losses)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=base.seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
    study.optimize(objective, n_trials=trials, timeout=timeout, show_progress_bar=False)
    best = study.best_params
    best_cfg = replace(base, **best)
    # tekshiruv: qidiruvda ishtirok etmagan oxirgi foldda — tanlangan vs boshlang'ich sozlama
    check_best = float(np.mean(_score(best_cfg, panel, cube, split_dates(panel, best_cfg)[0][-1:], device)))
    check_base = float(np.mean(_score(base, panel, cube, check, device)))
    rep = {"best_params": best, "best_search_loss": study.best_value, "trials": len(study.trials),
           "check_fold_loss_best": check_best, "check_fold_loss_base": check_base,
           "check_verdict": "tanlangan sozlama oxirgi foldda ham yaxshiroq" if check_best < check_base
           else "oxirgi foldda yaxshilanish yo'q — boshlang'ich sozlamada qolish tavsiya etiladi",
           "history": [{"n": t.number, "value": t.value, "params": t.params, "state": str(t.state)} for t in study.trials]}
    from ..config import MODELS_DIR
    (MODELS_DIR / "best_config.json").write_text(json.dumps(best, indent=1), encoding="utf-8")
    (REPORTS_DIR / "tuning.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    log.info("Eng yaxshi: %s (qidiruv %.4f) | oxirgi fold: tanlangan %.4f vs boshlang'ich %.4f",
             best, study.best_value, check_best, check_base)
    return rep
