"""Model o'lchami presetlari. CPU uchun kichik, Colab/Kaggle GPU uchun katta.

    cpu      — kompyuterda tez sinov (taxminan 1 soat, 2 yadro)
    fast     — Colab GPU'da ~20–30 daqiqalik tezkor to'liq ishga tushirish (2 fold, 2 modelli ansambl)
    gpu      — Colab T4/L4: 3 qatlamli Transformer, 120 kunlik oyna, 5 modelli ansambl
    gpu_xl   — Colab A100 / Kaggle P100: 4 qatlam, 180 kunlik oyna, 7 modelli ansambl
"""
from __future__ import annotations

from dataclasses import replace

from .config import ModelConfig

PRESETS: dict[str, dict] = {
    "cpu": dict(hidden=48, layers=1, heads=4, lookback=30, ensemble=2, epochs=6, batch_size=1024, patience=2,
                wf_folds=1, dropout=0.3, news_norm=True, news_dropout=0.2),
    "default": {},
    "fast": dict(hidden=96, layers=2, heads=4, lookback=60, news_days=7, ensemble=2, epochs=12, batch_size=1024,
                 patience=3, wf_folds=2, dropout=0.2, lr=1e-3, news_norm=True, news_dropout=0.2),
    "gpu": dict(hidden=128, layers=3, heads=8, lookback=120, news_days=10, ensemble=5, epochs=40, batch_size=512,
                patience=6, dropout=0.2, lr=7e-4, weight_decay=2e-3, asset_emb=24, warmup_epochs=2,
                news_norm=True, news_dropout=0.2),
    "gpu_xl": dict(hidden=192, layers=4, heads=8, lookback=180, news_days=14, ensemble=7, epochs=60, batch_size=512,
                   patience=8, dropout=0.2, lr=5e-4, weight_decay=3e-3, asset_emb=32, warmup_epochs=3,
                   news_norm=True, news_dropout=0.2),
}


def apply_preset(cfg: ModelConfig, name: str | None) -> ModelConfig:
    if not name:
        return cfg
    if name not in PRESETS:
        raise SystemExit(f"Noma'lum preset: {name}. Mavjud: {', '.join(PRESETS)}")
    return replace(cfg, **PRESETS[name])
