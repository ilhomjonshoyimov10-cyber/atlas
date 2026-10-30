"""SetupNet — strategiya signalining sifatini baholovchi chuqur model.

Kirish:
  * oxirgi 64 bar (OHLC + hajm, yo'nalishga normallashgan) -> 1D-CNN (dilated) + attention pooling
  * strategiya turi va aktiv guruhi -> embedding
  * kontekst: darajagacha masofa, tegishlar, RSI, trend bilan mosligi, SMA holati, MACD, volatillik...
Chiqish: P(maqsad stopdan oldin uriladi).

Baholash: vaqt bo'yicha bo'linish — oxirgi 365 kun model ko'rmagan holdout, oldidan 30 kun embargo
(belgilar 20 kun oldinga qaraydi). Natija tayanch darajaga (shu strategiyaning tarixiy yutish ulushi)
nisbatan o'lchanadi: AUC, Brier ko'nikmasi, va "eng ishonchli 30% signal" filtri.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .dataset import GROUPS, SEQ, TAB_NAMES
from .detect import SETUP_NAMES

log = logging.getLogger(__name__)
HOLDOUT_DAYS, EMBARGO_DAYS = 365, 30


class SetupNet(nn.Module):
    def __init__(self, n_tab: int, hidden: int = 48, dropout: float = 0.2, n_groups: int | None = None):
        super().__init__()
        c = 5
        self.conv = nn.Sequential(
            nn.Conv1d(c, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4), nn.GELU(),
        )
        self.att = nn.Linear(hidden, 1)
        self.setup_emb = nn.Embedding(len(SETUP_NAMES), 16)
        self.group_emb = nn.Embedding(n_groups or len(GROUPS), 8)   # aktiv guruhi (kunlik) yoki vaqt oralig'i (ichki kun)
        self.head = nn.Sequential(nn.Linear(hidden * 2 + 16 + 8 + n_tab, 64), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(64, 1))

    def forward(self, seq, tab, setup, group):
        h = self.conv(seq.transpose(1, 2)).transpose(1, 2)          # [B, T, H]
        w = torch.softmax(self.att(h).squeeze(-1), 1)
        pooled = (h * w.unsqueeze(-1)).sum(1)
        last = h[:, -1]
        z = torch.cat([pooled, last, self.setup_emb(setup), self.group_emb(group), tab], 1)
        return self.head(z).squeeze(-1)


@dataclass
class Data:
    seq: torch.Tensor
    tab: torch.Tensor
    setup: torch.Tensor
    group: torch.Tensor
    y: torch.Tensor


def to_tensors(ds: dict, idx: np.ndarray, groups: list[str] | None = None) -> Data:
    groups = groups or GROUPS
    m = ds["meta"].iloc[idx]
    return Data(torch.from_numpy(ds["seq"][idx]), torch.from_numpy(ds["tab"][idx]),
                torch.tensor([SETUP_NAMES.index(s) for s in m["setup"]]),
                torch.tensor([groups.index(g) for g in m["group"]]),
                torch.tensor(np.nan_to_num(m["win"].to_numpy(float)), dtype=torch.float32))


def _fit_one(tr: Data, va: Data, seed: int, epochs: int = 30, lr: float = 2e-3, n_groups: int | None = None) -> SetupNet:
    torch.manual_seed(seed)
    net = SetupNet(tr.tab.shape[1], n_groups=n_groups)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    best, best_state, bad = 1e9, None, 0
    n = len(tr.y)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            loss = nn.functional.binary_cross_entropy_with_logits(
                net(tr.seq[b], tr.tab[b], tr.setup[b], tr.group[b]), tr.y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(nn.functional.binary_cross_entropy_with_logits(
                net(va.seq, va.tab, va.setup, va.group), va.y))
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4:
                break
    net.load_state_dict(best_state)
    log.info("SetupNet seed=%d: val loss %.4f (epoch %d)", seed, best, ep + 1)
    return net


def predict(nets: list[SetupNet], d: Data) -> np.ndarray:
    with torch.no_grad():
        return np.mean([torch.sigmoid(n(d.seq, d.tab, d.setup, d.group)).numpy() for n in nets], 0)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    o = np.argsort(p)
    r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    pos = y == 1
    npos, nneg = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")


def split(meta: pd.DataFrame):
    labeled = meta["win"].notna().to_numpy()
    dates = pd.DatetimeIndex(meta["date"])
    end = dates[labeled].max()
    hold_start = end - pd.Timedelta(days=HOLDOUT_DAYS)
    train_end = hold_start - pd.Timedelta(days=EMBARGO_DAYS)
    tr_all = np.where(labeled & (dates < train_end))[0]
    ho = np.where(labeled & (dates >= hold_start))[0]
    td = dates[tr_all]
    val_cut = td.min() + (td.max() - td.min()) * 0.85
    tr = tr_all[td < val_cut - pd.Timedelta(days=EMBARGO_DAYS)]
    va = tr_all[td >= val_cut]
    return tr, va, ho, {"holdout_start": str(hold_start.date()), "holdout_end": str(end.date()),
                        "train_end": str(train_end.date())}


def train_and_evaluate(ds: dict, n_models: int = 3) -> tuple[list[SetupNet], dict]:
    from .dataset import setup_stats

    meta = ds["meta"]
    tr, va, ho, info = split(meta)
    Tr, Va, Ho = (to_tensors(ds, i) for i in (tr, va, ho))
    nets = [_fit_one(Tr, Va, s) for s in range(n_models)]
    p = predict(nets, Ho)
    y = Ho.y.numpy()
    # tayanch: shu strategiyaning train davridagi yutish ulushi
    base_rate = meta.iloc[tr].groupby("setup")["win"].mean()
    pb = meta.iloc[ho]["setup"].map(base_rate).fillna(meta.iloc[tr]["win"].mean()).to_numpy()
    brier, brier_b = float(np.mean((p - y) ** 2)), float(np.mean((pb - y) ** 2))
    hm = meta.iloc[ho].copy()
    hm["p"] = p
    hm["ex"] = hm["R"] - hm["base_nb"]
    # "Ishonchli signal" chegaralari validatsiyada aniqlanadi (holdoutga qarab tanlanmaydi)
    vm = meta.iloc[va].copy()
    vm["p"] = predict(nets, Va)
    hi_thr = vm.groupby("setup")["p"].quantile(0.7)
    lo_thr = vm.groupby("setup")["p"].quantile(0.3)
    gh, gl = float(vm["p"].quantile(0.7)), float(vm["p"].quantile(0.3))
    hm["hi"] = hm["setup"].map(hi_thr).fillna(gh)
    hm["lo"] = hm["setup"].map(lo_thr).fillna(gl)
    top = hm[hm["p"] >= hm["hi"]]
    bot = hm[hm["p"] <= hm["lo"]]
    report = {
        **info, "n_train": int(len(tr)), "n_val": int(len(va)), "n_holdout": int(len(ho)),
        "auc": _auc(y, p), "auc_baseline": _auc(y, pb),
        "brier": brier, "brier_baseline": brier_b, "brier_skill": float(1 - brier / brier_b),
        "holdout_win_rate": float(y.mean() * 100),
        "filter": {"all": {"n": int(len(hm)), "avg_R": float(hm["R"].mean()), "excess_R": float(hm["ex"].mean()),
                           "win_rate": float(hm["win"].mean() * 100)},
                   "top30": {"n": int(len(top)), "avg_R": float(top["R"].mean()), "excess_R": float(top["ex"].mean()),
                             "win_rate": float(top["win"].mean() * 100)},
                   "bottom30": {"n": int(len(bot)), "avg_R": float(bot["R"].mean()), "excess_R": float(bot["ex"].mean()),
                                "win_rate": float(bot["win"].mean() * 100)}},
        "stats_train": setup_stats(meta, np.isin(np.arange(len(meta)), np.r_[tr, va])),
        "stats_holdout": setup_stats(meta, np.isin(np.arange(len(meta)), ho)),
    }
    # holdoutdagi har bir strategiya uchun model filtri
    per = []
    for s, g in hm.groupby("setup"):
        if len(g) < 30:
            continue
        gt = g[g["p"] >= g["hi"]]
        per.append({"setup": s, "n": int(len(g)), "win_all": float(g["win"].mean() * 100),
                    "n_top": int(len(gt)), "win_top": float(gt["win"].mean() * 100) if len(gt) else None,
                    "auc": _auc(g["win"].to_numpy(), g["p"].to_numpy())})
    report["per_setup_holdout"] = per
    report["thresholds"] = {k: float(v) for k, v in hi_thr.items()}
    log.info("SetupNet holdout: AUC %.3f (tayanch %.3f), Brier skill %.3f", report["auc"], report["auc_baseline"],
             report["brier_skill"])
    return nets, report


def train_final(ds: dict, n_models: int = 3) -> list[SetupNet]:
    """Servis modeli: barcha belgilangan signallarda (oxirgi 15% — erta to'xtatish uchun)."""
    meta = ds["meta"]
    lab = np.where(meta["win"].notna().to_numpy())[0]
    d = pd.DatetimeIndex(meta["date"].iloc[lab])
    cut = d.min() + (d.max() - d.min()) * 0.85
    tr, va = lab[d < cut - pd.Timedelta(days=EMBARGO_DAYS)], lab[d >= cut]
    return [_fit_one(to_tensors(ds, tr), to_tensors(ds, va), 100 + s) for s in range(n_models)]


def save(nets: list[SetupNet], path, base_rates: dict, report: dict):
    torch.save({"states": [n.state_dict() for n in nets], "n_tab": len(TAB_NAMES),
                "base_rates": base_rates, "report": json.loads(json.dumps(report, default=float)),
                "setups": SETUP_NAMES, "groups": GROUPS}, path)


def load(path) -> tuple[list[SetupNet], dict]:
    z = torch.load(path, map_location="cpu", weights_only=False)
    nets = []
    for st in z["states"]:
        n = SetupNet(z["n_tab"]); n.load_state_dict(st); n.eval(); nets.append(n)
    return nets, z
