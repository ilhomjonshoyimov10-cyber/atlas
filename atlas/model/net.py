"""AtlasNet v2 — ko'p shoxli, ko'p aktivli kvantil Transformer.

    narx shoxi        : Transformer encoder (oxirgi L kun) + o'rganiladigan pozitsiya + [CLS] token
    bozorlararo shox  : aktiv o'z embeddingi bilan BOSHQA aktivlar holatiga attention qiladi
                        (masalan: neft ↑ -> CAD, XOM; dollar ↑ -> oltin, EURUSD). 1 kun lag — leakage yo'q.
    yangilik shoxi    : oxirgi K kun yangilik tokenlari ustida attention (so'rov = aktiv embeddingi)
                        + to'g'ridan-to'g'ri past sig'imli yo'l "yangi xabar -> keyingi kunlar qaytimi"
    mavsumiylik shoxi : MLP(mavsum featurelari + aktiv embeddingi)
    birlashtirish     : gated fusion -> har bir gorizont uchun q05 <= q50 <= q95 (softplus => kesishmaydi)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class NewsAttention(nn.Module):
    def __init__(self, news_dim: int, hidden: int, asset_dim: int, K: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(news_dim), nn.Linear(news_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.pos = nn.Parameter(torch.zeros(K, hidden))   # yangilikning "yangiligi" (recency)
        self.q = nn.Linear(asset_dim, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.empty = nn.Parameter(torch.zeros(hidden))     # yangilik umuman yo'q bo'lganda
        self.scale = hidden ** -0.5

    def forward(self, tok, mask, a_emb):
        h = self.proj(tok) + self.pos
        score = (self.k(h) @ self.q(a_emb).unsqueeze(-1)).squeeze(-1) * self.scale   # [B, K]
        score = score.masked_fill(~mask, -1e9)
        w = torch.softmax(score, -1) * mask
        w = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
        out = (w.unsqueeze(-1) * self.v(h)).sum(1)
        has = mask.any(-1, keepdim=True).float()
        return has * out + (1 - has) * self.empty, w


class PriceTransformer(nn.Module):
    def __init__(self, in_dim: int, asset_dim: int, hidden: int, L: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.inp = nn.Linear(in_dim + asset_dim, hidden)
        self.pos = nn.Parameter(torch.randn(L + 1, hidden) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2 * hidden, dropout=dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, a):
        B, L, _ = x.shape
        h = self.inp(torch.cat([x, a[:, None].expand(B, L, -1)], -1))
        h = torch.cat([self.cls.expand(B, -1, -1), h], 1) + self.pos[: L + 1]
        return self.norm(self.enc(h)[:, 0])


class CrossAsset(nn.Module):
    """Aktiv -> boshqa aktivlar holati ustida ko'p boshli attention."""

    def __init__(self, n_assets: int, mkt_dim: int, asset_dim: int, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.ids = nn.Parameter(torch.arange(n_assets), requires_grad=False)
        self.tok = nn.Sequential(nn.Linear(mkt_dim + asset_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.q = nn.Linear(asset_dim, hidden)
        self.att = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.empty = nn.Parameter(torch.zeros(hidden))

    def forward(self, mkt, mask, a, asset_table: nn.Embedding):
        B, A, _ = mkt.shape
        e = asset_table(self.ids)[None].expand(B, -1, -1)
        kv = self.tok(torch.cat([mkt, e], -1))
        has_b = mask.any(-1)
        safe = mask.clone()
        safe[~has_b, 0] = True          # hammasi yopiq qator NaN bermasligi uchun
        out, w = self.att(self.q(a)[:, None], kv, kv, key_padding_mask=~safe, need_weights=True)
        has = has_b[:, None].float()
        return has * out[:, 0] + (1 - has) * self.empty, w[:, 0]


class AtlasNet(nn.Module):
    def __init__(self, cfg: ModelConfig, n_assets: int, price_dim: int, season_dim: int, news_dim: int,
                 market_dim: int = 4):
        super().__init__()
        H, A = cfg.hidden, cfg.asset_emb
        self.cfg = cfg
        self.nh = len(cfg.horizons)
        self.asset_emb = nn.Embedding(n_assets, A)
        self.price = PriceTransformer(price_dim, A, H, cfg.lookback, cfg.layers, cfg.heads, cfg.dropout)
        self.cross = CrossAsset(n_assets, market_dim, A, H, cfg.heads, cfg.dropout)
        self.news = NewsAttention(news_dim, H, A, cfg.news_days, cfg.dropout)
        self.season = nn.Sequential(nn.Linear(season_dim + A, H), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(H, H))
        # Gated fusion: har bir shox hissasini aktivga qarab moslashtiradi (talqin uchun ham foydali)
        self.gate = nn.Sequential(nn.Linear(A, 4), nn.Sigmoid())
        self.fuse = nn.Sequential(
            nn.LayerNorm(4 * H + A), nn.Linear(4 * H + A, 2 * H), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(2 * H, H), nn.GELU(),
        )
        self.head = nn.Linear(H, self.nh * 3)   # [median, pastki kenglik, yuqori kenglik]
        self.news_direct = nn.Sequential(nn.LayerNorm(news_dim), nn.Linear(news_dim, self.nh))
        self.direct_days = 3

    def forward(self, b: dict, return_attn: bool = False):
        a = self.asset_emb(b["asset"])
        hp = self.price(b["price"], a)
        zero = torch.zeros_like(hp)
        if self.cfg.use_market:
            hm, w_mkt = self.cross(b["market"], b["market_mask"], a, self.asset_emb)
        else:
            hm, w_mkt = zero, None
        if self.cfg.use_news:
            hn, attn = self.news(b["news"], b["news_mask"], a)
        else:
            hn, attn = zero, None
        hs = self.season(torch.cat([b["season"], a], -1)) if self.cfg.use_season else zero
        g = self.gate(a)
        z = self.fuse(torch.cat([hp * g[:, :1], hm * g[:, 1:2], hn * g[:, 2:3], hs * g[:, 3:4], a], -1))
        o = self.head(z).view(-1, self.nh, 3)
        med = o[..., 0]
        if self.cfg.use_news:
            k = self.direct_days
            m = b["news_mask"][:, -k:].float().unsqueeze(-1)
            recent = (b["news"][:, -k:] * m).sum(1) / m.sum(1).clamp(min=1)
            med = med + self.news_direct(recent) * (m.sum(1) > 0).float()
        lo = med - F.softplus(o[..., 1]) - 1e-3
        hi = med + F.softplus(o[..., 2]) + 1e-3
        q = torch.stack([lo, med, hi], -1)          # [B, H, 3]  (q05, q50, q95)
        if return_attn:
            return q, {"news": attn, "market": w_mkt, "gate": g}
        return q


class Ensemble(nn.Module):
    """Bir nechta seed bilan o'qitilgan modellar. Kvantillar o'rtachalanadi (tartib saqlanadi)."""

    def __init__(self, models: list[AtlasNet]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.cfg = models[0].cfg

    def forward(self, b: dict, return_attn: bool = False):
        outs = [m(b, return_attn=return_attn) for m in self.models]
        if return_attn:
            q = torch.stack([o[0] for o in outs]).mean(0)
            info = {}
            for k in outs[0][1]:
                vals = [o[1][k] for o in outs if o[1][k] is not None]
                info[k] = torch.stack(vals).mean(0) if vals else None
            return q, info
        return torch.stack(outs).mean(0)


def pinball_loss(q: torch.Tensor, y: torch.Tensor, quantiles=(0.05, 0.5, 0.95)) -> torch.Tensor:
    """q: [B, H, Q], y: [B, H]."""
    tau = torch.tensor(quantiles, device=q.device, dtype=q.dtype)
    e = y.unsqueeze(-1) - q
    return torch.maximum(tau * e, (tau - 1) * e).mean()
