"""O'qitish, konformal kalibratsiya va bashorat."""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import asdict

import numpy as np
import torch

from ..config import ModelConfig
from ..features.dataset import PRICE_FEATS, BatchMaker
from ..features.seasonality import SEASON_DIM
from .net import AtlasNet, Ensemble, pinball_loss

log = logging.getLogger(__name__)


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)


def build_model(cfg: ModelConfig, bm: BatchMaker | None = None, n_assets: int | None = None,
                news_dim: int | None = None, device: str = "cpu") -> AtlasNet:
    if bm is not None:
        n_assets, news_dim, device = len(bm.p.symbols), bm.news_dim, bm.device
    return AtlasNet(cfg, n_assets=n_assets, price_dim=len(PRICE_FEATS),
                       season_dim=SEASON_DIM, news_dim=news_dim).to(device)


@torch.no_grad()
def predict(model: AtlasNet, bm: BatchMaker, rows: np.ndarray, bs: int = 4096, extra_news=None) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        out.append(model(bm(rows[i:i + bs], extra_news=extra_news)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(bm.cfg.horizons), 3), "float32")


def fit(cfg: ModelConfig, bm: BatchMaker, train_rows: np.ndarray, val_rows: np.ndarray,
        verbose: bool = True) -> tuple[AtlasNet, dict]:
    """Bitta model: AdamW + warmup + cosine, gradient clipping, GPU'da aralash aniqlik (AMP), early stopping."""
    set_seed(cfg.seed)
    model = build_model(cfg, bm)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_ep = max(1, int(np.ceil(len(train_rows) / cfg.batch_size)))
    total, warm = cfg.epochs * steps_per_ep, max(1, cfg.warmup_epochs * steps_per_ep)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, st / total))))
    use_amp = str(bm.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, best_state, bad, hist = np.inf, None, 0, []
    rng = np.random.default_rng(cfg.seed)
    for ep in range(cfg.epochs):
        t0 = time.time()
        model.train()
        perm = rng.permutation(train_rows)
        tl = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            b = bm(perm[i:i + cfg.batch_size])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = pinball_loss(model(b).float(), b["y"], cfg.quantiles)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(b["y"])
        tl /= max(len(perm), 1)
        q = predict(model, bm, val_rows)
        vl = pinball_np(q, bm.p.y[val_rows], cfg.quantiles)
        hist.append({"epoch": ep + 1, "train": tl, "val": vl})
        if verbose:
            log.info("seed %d  epoch %2d  train %.4f  val %.4f  (%.1fs)", cfg.seed, ep + 1, tl, vl, time.time() - t0)
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, {"history": hist, "best_val": best}


def fit_ensemble(cfg: ModelConfig, bm: BatchMaker, train_rows, val_rows, n: int | None = None,
                 verbose: bool = True) -> tuple[Ensemble, dict]:
    """n ta seed bilan o'qitib, ansambl qaytaradi. Ansambl bitta modeldan barqarorroq va yaxshi kalibrlangan."""
    from dataclasses import replace
    n = n or cfg.ensemble
    models, infos = [], []
    for k in range(n):
        m, info = fit(replace(cfg, seed=cfg.seed + 1000 * k), bm, train_rows, val_rows, verbose)
        models.append(m)
        infos.append(info)
    ens = Ensemble(models)
    vl = pinball_np(predict(ens, bm, val_rows), bm.p.y[val_rows], cfg.quantiles)
    if verbose and n > 1:
        log.info("Ansambl (%d model) val pinball: %.4f  (alohida: %s)", n, vl,
                 ", ".join(f"{i['best_val']:.4f}" for i in infos))
    return ens, {"members": infos, "best_val": vl}


# ---------- metrikalar ----------

def pinball_np(q: np.ndarray, y: np.ndarray, quantiles=(0.05, 0.5, 0.95)) -> float:
    tau = np.asarray(quantiles)
    e = y[..., None] - q
    return float(np.maximum(tau * e, (tau - 1) * e).mean())


def pinball_per_row(q: np.ndarray, y: np.ndarray, quantiles=(0.05, 0.5, 0.95)) -> np.ndarray:
    tau = np.asarray(quantiles)
    e = y[..., None] - q
    return np.maximum(tau * e, (tau - 1) * e).mean(axis=(1, 2))


def conformal_offsets(q: np.ndarray, y: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """CQR (Romano va boshq., 2019): har bir gorizont uchun intervalni kengaytirish/toraytirish miqdori."""
    E = np.maximum(q[..., 0] - y, y - q[..., 2])     # [N, H]
    n = len(E)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(E, level, axis=0).astype("float32")


def apply_conformal(q: np.ndarray, off: np.ndarray) -> np.ndarray:
    q = q.copy()
    q[..., 0] -= off
    q[..., 2] += off
    q[..., 0] = np.minimum(q[..., 0], q[..., 1])
    q[..., 2] = np.maximum(q[..., 2], q[..., 1])
    return q


def metrics(q: np.ndarray, y: np.ndarray, horizons, quantiles=(0.05, 0.5, 0.95)) -> dict:
    out = {"pinball": pinball_np(q, y, quantiles), "n": int(len(y))}
    for j, h in enumerate(horizons):
        cov = ((y[:, j] >= q[:, j, 0]) & (y[:, j] <= q[:, j, 2])).mean()
        out[f"h{h}"] = {
            "pinball": pinball_np(q[:, j:j + 1], y[:, j:j + 1], quantiles),
            "coverage90": float(cov),
            "width": float((q[:, j, 2] - q[:, j, 0]).mean()),
            "dir_acc": float((np.sign(q[:, j, 1]) == np.sign(y[:, j])).mean()),
        }
    return out


def save_bundle(path, model, cfg: ModelConfig, conformal: np.ndarray, news_dim: int,
                reducer_state: dict | None, report: dict | None = None, embedder: str = "hashing"):
    members = model.models if isinstance(model, Ensemble) else [model]
    torch.save({"states": [m.state_dict() for m in members], "cfg": asdict(cfg), "conformal": conformal,
                "news_dim": news_dim, "reducer": reducer_state, "report": report, "embedder": embedder,
                "version": 2}, path)


def load_bundle(path, device="cpu"):
    z = torch.load(path, map_location=device, weights_only=False)
    known = set(ModelConfig.__dataclass_fields__)
    cfg = ModelConfig(**{k: (tuple(v) if k in ("horizons", "quantiles") else v)
                         for k, v in z["cfg"].items() if k in known})
    return z, cfg


def load_model(path, n_assets: int, device="cpu"):
    """Saqlangan ansamblni tiklaydi."""
    z, cfg = load_bundle(path, device)
    states = z.get("states") or [z["state"]]
    models = []
    for st in states:
        m = build_model(cfg, n_assets=n_assets, news_dim=z["news_dim"], device=device)
        m.load_state_dict(st)
        m.eval()
        models.append(m)
    return Ensemble(models).eval(), z, cfg
