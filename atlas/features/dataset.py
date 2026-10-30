"""Ko'p aktivli panel dataset: narx oynasi + yangilik oynasi + mavsumiylik + aktiv ID.

Bitta aktivda 10 yilda ~2500 qator bor — chuqur model uchun kam. Shuning uchun barcha aktivlar
BIRGA o'qitiladi (~37 aktiv x 2500 = ~90k namuna), aktiv identifikatori embedding sifatida beriladi.

Maqsad (target): h kunlik log-qaytim, volatillikka normallashtirilgan:
    y_h = (log C[t+h] - log C[t]) / (sigma_t * sqrt(h))
Bu barcha aktivlarni bir xil masshtabga keltiradi. Bashoratdan keyin narxga qayta aylantiriladi.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ASSET_BY_SYMBOL, ASSETS, EVENT_NAMES, ModelConfig
from .news_agg import SCOPE_IDX, NewsCube
from .seasonality import SEASON_DIM, season_features

PRICE_FEATS = ["ret_n", "vol_ratio", "range_n", "volume_z", "ma50_n", "ma200_n", "mom20_n", "log_vol"]
MARKET_FEATS = ["ret_n", "mom20_n", "vol_ratio", "ma50_n"]
WARMUP = 250
ASSET_IDS = {a.symbol: i for i, a in enumerate(ASSETS)}


def price_features(df: pd.DataFrame) -> pd.DataFrame:
    c = np.log(df["close"].astype(float).values)
    lr = np.diff(c, prepend=np.nan)
    s = pd.Series(lr)
    vol = s.ewm(span=20, min_periods=20).std().values
    vol = np.maximum(vol, 1e-4)
    vol_long = s.rolling(250, min_periods=60).std().values
    prev_vol = np.concatenate([[np.nan], vol[:-1]])
    f = pd.DataFrame(index=df.index)
    f["ret_n"] = lr / prev_vol
    f["vol_ratio"] = np.log(vol / np.maximum(vol_long, 1e-4))
    if {"high", "low"} <= set(df.columns):
        f["range_n"] = ((df["high"] - df["low"]) / df["close"]).values / vol
    else:
        f["range_n"] = 0.0
    if "volume" in df.columns and (df["volume"].fillna(0) > 0).mean() > 0.5:
        lv = np.log1p(df["volume"].fillna(0).astype(float))
        f["volume_z"] = ((lv - lv.rolling(60, min_periods=20).mean()) / (lv.rolling(60, min_periods=20).std() + 1e-6)).values
    else:
        f["volume_z"] = 0.0
    cs = pd.Series(c)
    f["ma50_n"] = (c - cs.rolling(50, min_periods=20).mean().values) / (vol * np.sqrt(50))
    f["ma200_n"] = (c - cs.rolling(200, min_periods=60).mean().values) / (vol * np.sqrt(200))
    f["mom20_n"] = (c - cs.shift(20).values) / (vol * np.sqrt(20))
    f["log_vol"] = np.log(vol) + 4.0
    f = f.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8, 8)
    f["_logc"] = c
    f["_vol"] = vol
    return f


@dataclass
class Panel:
    feats: np.ndarray       # [N, F]
    season: np.ndarray      # [N, S]
    asset: np.ndarray       # [N]
    date: np.ndarray        # [N] datetime64
    pos: np.ndarray         # [N] aktiv ichidagi indeks
    logc: np.ndarray        # [N]
    vol: np.ndarray         # [N]
    y: np.ndarray           # [N, H] (kelajak yo'q joyda NaN)
    horizons: tuple[int, ...]
    symbols: list[str]
    market: np.ndarray | None = None      # [T_kun, A, F_m] — bozorlararo kontekst (1 kun lag bilan)
    market_mask: np.ndarray | None = None  # [T_kun, A]
    market_start: np.datetime64 | None = None

    @classmethod
    def build(cls, prices: dict[str, pd.DataFrame], horizons=(1, 5, 20)) -> "Panel":
        parts = []
        # Tartib har doim ASSETS bo'yicha — qator indekslari jarayonlar orasida barqaror bo'lsin
        for sym in [a.symbol for a in ASSETS if a.symbol in prices]:
            df = prices[sym]
            if len(df) < WARMUP + 60:
                continue
            df = df.sort_values("date").reset_index(drop=True)
            f = price_features(df)
            n = len(df)
            y = np.full((n, len(horizons)), np.nan, "float32")
            for j, h in enumerate(horizons):
                fut = np.full(n, np.nan)
                fut[: n - h] = f["_logc"].values[h:]
                y[:, j] = (fut - f["_logc"].values) / (f["_vol"].values * np.sqrt(h))
            parts.append(dict(
                feats=f[PRICE_FEATS].values.astype("float32"),
                season=season_features(pd.DatetimeIndex(df["date"])).values,
                asset=np.full(n, ASSET_IDS[sym], "int64"),
                date=df["date"].values.astype("datetime64[ns]"),
                pos=np.arange(n),
                logc=f["_logc"].values.astype("float64"),
                vol=f["_vol"].values.astype("float32"),
                y=np.clip(y, -10, 10),
            ))
        if not parts:
            raise ValueError("Narx ma'lumotlari yo'q. Avval: python -m atlas.cli download")
        cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        syms = [a.symbol for a in ASSETS]
        panel = cls(**cat, horizons=tuple(horizons), symbols=syms)
        panel._build_market()
        return panel

    def _build_market(self):
        """Har bir kalendar kuni uchun barcha aktivlar holati: [ret_n, mom20_n, vol_ratio, ma50_n].

        LEAKAGE: turli bozorlar turli vaqtda yopiladi (forex/kripto ~00:00 UTC, NYSE ~20:00 UTC).
        Xavfsiz bo'lishi uchun boshqa aktivlar holati 1 KUN KECHIKTIRIB beriladi (t-1 yopilishi).
        """
        start = pd.Timestamp(self.date.min()).normalize()
        days = pd.date_range(start, pd.Timestamp(self.date.max()).normalize() + pd.Timedelta(days=1), freq="D")
        cols = [PRICE_FEATS.index(c) for c in MARKET_FEATS]
        A = len(self.symbols)
        M = np.zeros((len(days), A, len(cols)), "float32")
        mask = np.zeros((len(days), A), bool)
        for ai in range(A):
            rows = np.where(self.asset == ai)[0]
            if not len(rows):
                continue
            df = pd.DataFrame(self.feats[rows][:, cols], index=pd.DatetimeIndex(self.date[rows]).normalize())
            df = df[~df.index.duplicated(keep="last")].reindex(days).ffill(limit=4).shift(1)   # 1 kun lag
            ok = df.notna().all(1).values
            M[ok, ai] = df.values[ok]
            mask[:, ai] = ok
        self.market, self.market_mask, self.market_start = M, mask, np.datetime64(start)

    def market_day(self, rows) -> np.ndarray:
        return ((self.date[rows] - self.market_start) // np.timedelta64(1, "D")).astype("int64")

    def sample_rows(self, lookback: int, need_target: bool = True) -> np.ndarray:
        ok = self.pos >= max(lookback, WARMUP)
        if need_target:
            ok &= ~np.isnan(self.y).any(1)
        return np.where(ok)[0]

    def latest_rows(self) -> dict[str, int]:
        out = {}
        for sym, i in ASSET_IDS.items():
            rows = np.where(self.asset == i)[0]
            if len(rows):
                out[sym] = int(rows[-1])
        return out


class BatchMaker:
    """Indekslar bo'yicha tensorlar yig'adi (vektorlashtirilgan gather — tez)."""

    def __init__(self, panel: Panel, cube: NewsCube | None, cfg: ModelConfig, device="cpu"):
        import torch

        self.t = torch
        self.p = panel
        self.cfg = cfg
        self.device = device
        self.feats = torch.from_numpy(panel.feats).to(device)
        self.season = torch.from_numpy(panel.season).to(device)
        self.asset = torch.from_numpy(panel.asset).to(device)
        self.y = torch.from_numpy(np.nan_to_num(panel.y)).to(device)
        self.L = cfg.lookback
        self.K = cfg.news_days
        self.market = torch.from_numpy(panel.market).to(device)
        self.market_mask = torch.from_numpy(panel.market_mask).to(device)
        self.mday = torch.from_numpy(panel.market_day(np.arange(len(panel.date)))).to(device)
        if cube is not None:
            self.emb = torch.from_numpy(cube.emb).to(device)
            # CPU'da numpy bilan xotira umumiy: kollektor kubni yangilasa, tensorlar ham yangilanadi
            self.cnt = torch.from_numpy(cube.cnt).to(device)
            self.tone = torch.from_numpy(cube.tone).to(device)
            self.ev = torch.from_numpy(cube.ev).to(device)
            self.day = torch.from_numpy(cube.day_index(pd.DatetimeIndex(panel.date))).to(device)
            grp = np.array([SCOPE_IDX[ASSET_BY_SYMBOL[s].group] for s in panel.symbols])
            ast = np.array([SCOPE_IDX[s] for s in panel.symbols])
            self.grp_scope = torch.from_numpy(grp).to(device)
            self.ast_scope = torch.from_numpy(ast).to(device)
            self.news_dim = 3 * cube.emb.shape[2] + 6 + len(EVENT_NAMES)
        else:
            self.emb = None
            self.news_dim = 3 * cfg.emb_dim + 6 + len(EVENT_NAMES)

    def __call__(self, rows, extra_news=None, day_override=None) -> dict:
        t = self.t
        g = t.as_tensor(rows, device=self.device, dtype=t.long)
        offs = t.arange(-self.L + 1, 1, device=self.device)
        price = self.feats[g[:, None] + offs]                       # [B, L, F]
        a = self.asset[g]
        md = self.mday[g]
        out = {"price": price, "season": self.season[g], "asset": a, "y": self.y[g],
               "market": self.market[md], "market_mask": self.market_mask[md]}
        B = len(g)
        if self.emb is None:
            out["news"] = t.zeros(B, self.K, self.news_dim, device=self.device)
            out["news_mask"] = t.zeros(B, self.K, dtype=t.bool, device=self.device)
            return out
        day = self.day[g] if day_override is None else t.as_tensor(day_override, device=self.device).long().expand(len(g))
        d = day[:, None] + t.arange(-self.K + 1, 1, device=self.device)   # [B, K]
        valid = d >= 0
        dc = d.clamp(min=0)
        gs = self.grp_scope[a][:, None].expand_as(dc)
        as_ = self.ast_scope[a][:, None].expand_as(dc)
        zero = t.zeros_like(dc)
        tok = t.cat([
            self.emb[zero, dc], self.emb[gs, dc], self.emb[as_, dc],
            t.log1p(t.stack([self.cnt[zero, dc], self.cnt[gs, dc], self.cnt[as_, dc]], -1)),
            t.stack([self.tone[zero, dc], self.tone[gs, dc], self.tone[as_, dc]], -1),
            t.log1p(self.ev[dc]),
        ], -1) * valid[..., None]
        if extra_news is not None:   # "agar shu yangilik bo'lsa" ssenariysi (impact tahlili)
            tok = tok.clone()
            tok[:, -1] = tok[:, -1] + extra_news(a)
        mask = valid & (self.cnt[zero, dc] > 0)
        if extra_news is not None:
            mask[:, -1] = True
        out["news"] = tok
        out["news_mask"] = mask
        return out
