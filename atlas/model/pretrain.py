"""O'z-o'zini o'qitish (self-supervised) — narx Transformerini oldindan o'qitish.

G'oya: belgilangan (labelled) maqsad atigi 3 gorizontli kvantil — signal juda shovqinli. Lekin 37 aktivning
10+ yillik narx ketma-ketliklari ko'p. Shuning uchun avval kodlovchi (encoder) belgisiz vazifada o'qitiladi:

  * Masked reconstruction: 64–180 kunlik oynadagi kunlarning ~30%i yashiriladi (bloklar bilan),
    model ularni qolgan kunlardan tiklashi kerak -> bozor dinamikasini (volatillik klasterlari, trend,
    o'rtacha qaytish) o'rganadi.
  * Kontrastiv bo'lmagan, oddiy va barqaror: MSE faqat yashirilgan pozitsiyalarda.

Keyin AtlasNet'ning narx shoxi va aktiv embeddinglari shu og'irliklar bilan boshlanadi (fine-tuning).

Leakage: oldindan o'qitish FAQAT eng birinchi walk-forward test davridan oldingi qatorlarda bajariladi —
shuning uchun hech bir test yoki holdout davrining narx naqshlari kodlovchiga kirmaydi.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn as nn

from ..config import ModelConfig
from ..features.dataset import PRICE_FEATS, BatchMaker
from .net import PriceTransformer

log = logging.getLogger(__name__)


class MaskedPricePretrainer(nn.Module):
    def __init__(self, cfg: ModelConfig, n_assets: int):
        super().__init__()
        F_ = len(PRICE_FEATS)
        self.asset_emb = nn.Embedding(n_assets, cfg.asset_emb)
        self.price = PriceTransformer(F_, cfg.asset_emb, cfg.hidden, cfg.lookback, cfg.layers, cfg.heads, cfg.dropout)
        self.mask_tok = nn.Parameter(torch.zeros(F_))
        self.dec = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, F_))

    def tokens(self, x, a):
        p = self.price
        B, L, _ = x.shape
        h = p.inp(torch.cat([x, a[:, None].expand(B, L, -1)], -1))
        h = torch.cat([p.cls.expand(B, -1, -1), h], 1) + p.pos[: L + 1]
        return p.enc(h)[:, 1:]                                   # [B, L, H] (CLS siz)

    def forward(self, x, asset, mask):
        a = self.asset_emb(asset)
        xm = torch.where(mask[..., None], self.mask_tok.expand_as(x), x)
        return self.dec(self.tokens(xm, a))


def block_mask(B: int, L: int, ratio: float, device, gen: torch.Generator, block: int = 5) -> torch.Tensor:
    """Ketma-ket bloklar bilan yashirish (bitta kunni yashirish juda oson — qo'shnisidan nusxa olinadi)."""
    n_blocks = max(1, int(L * ratio / block))
    starts = torch.randint(0, max(1, L - block), (B, n_blocks), generator=gen, device="cpu").to(device)
    m = torch.zeros(B, L, dtype=torch.bool, device=device)
    idx = (starts[..., None] + torch.arange(block, device=device)).clamp(max=L - 1).view(B, -1)
    m.scatter_(1, idx, True)
    return m


def pretrain(cfg: ModelConfig, bm: BatchMaker, rows: np.ndarray, epochs: int = 20, ratio: float = 0.3,
             lr: float = 1e-3, verbose: bool = True) -> tuple[dict, dict]:
    """rows — faqat ruxsat etilgan (eng erta testdan oldingi) qatorlar. Natija: kodlovchi og'irliklari."""
    torch.manual_seed(cfg.seed)
    net = MaskedPricePretrainer(cfg, len(bm.p.symbols)).to(bm.device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    use_amp = str(bm.device).startswith("cuda")
    cut = int(len(rows) * 0.95)
    perm0 = rng.permutation(rows)
    tr, va = perm0[:cut], perm0[cut:]
    hist, best, best_state = [], np.inf, None
    for ep in range(epochs):
        t0 = time.time()
        net.train()
        tl, n = 0.0, 0
        trp = rng.permutation(tr)
        for i in range(0, len(trp), cfg.batch_size):
            b = bm(trp[i:i + cfg.batch_size])
            x = b["price"]
            m = block_mask(len(x), x.shape[1], ratio, x.device, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                rec = net(x, b["asset"], m).float()
            loss = ((rec - x) ** 2)[m].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(x); n += len(x)
        net.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            g2 = torch.Generator().manual_seed(123)
            for i in range(0, len(va), 4096):
                b = bm(va[i:i + 4096]); x = b["price"]
                m = block_mask(len(x), x.shape[1], ratio, x.device, g2)
                vl += (((net(x, b["asset"], m) - x) ** 2)[m].mean().item()) * len(x); vn += len(x)
        vl /= max(vn, 1)
        hist.append({"epoch": ep + 1, "train_mse": tl / max(n, 1), "val_mse": vl})
        if verbose:
            log.info("pretrain epoch %2d  train %.4f  val %.4f  (%.1fs)", ep + 1, tl / max(n, 1), vl, time.time() - t0)
        if vl < best:
            best = vl
            best_state = {"price": {k: v.detach().cpu().clone() for k, v in net.price.state_dict().items()},
                          "asset_emb": {k: v.detach().cpu().clone() for k, v in net.asset_emb.state_dict().items()}}
    # tayanch MSE (oynadagi o'rtacha bilan to'ldirish) — hisobot uchun
    with torch.no_grad():
        b = bm(va[:4096]); x = b["price"]
        m = block_mask(len(x), x.shape[1], ratio, x.device, torch.Generator().manual_seed(123))
        mean_fill = x.masked_fill(m[..., None], 0).sum(1, keepdim=True) / (~m).sum(1, keepdim=True)[..., None].clamp(min=1)
        base = (((mean_fill.expand_as(x) - x) ** 2)[m].mean().item())
    info = {"history": hist, "best_val_mse": best, "baseline_mean_fill_mse": base,
            "rows": int(len(rows)), "mask_ratio": ratio,
            "arch": {"hidden": cfg.hidden, "layers": cfg.layers, "heads": cfg.heads, "lookback": cfg.lookback,
                     "asset_emb": cfg.asset_emb}}
    log.info("Pretrain tugadi: val MSE %.4f (o'rtacha bilan to'ldirish tayanchi %.4f)", best, base)
    return best_state, info


def load_into(model: nn.Module, state: dict) -> bool:
    """AtlasNet'ga oldindan o'qitilgan narx kodlovchisi va aktiv embeddinglarini yuklaydi (o'lchamlar mos bo'lsa)."""
    try:
        model.price.load_state_dict(state["price"])
        model.asset_emb.load_state_dict(state["asset_emb"])
        return True
    except (RuntimeError, KeyError) as e:
        log.warning("Pretrain og'irliklari mos kelmadi (%s) — noldan o'qitiladi", e)
        return False
