"""Yangiliklarni kunlik jamlash — leakage'dan himoyalangan holda.

QOIDA: D kuni uchun bashorat qarori `decision_time(D) = D + DECISION_HOUR_UTC` da qabul qilinadi.
Yangilik D kuniga faqat `available_at <= decision_time(D)` bo'lsa tushadi:
    use_date = ceil_day(available_at - DECISION_HOUR)

Natija (NewsCube) — har bir kalendar kuni va har bir "scope" uchun:
    scope = global | <guruh> (forex, crypto, ...) | <aktiv> (BTC, WHEAT, ...)
    emb   [S, T, d]  — o'rtacha embedding (PCA'dan keyin)
    cnt   [S, T]     — xabarlar soni
    tone  [S, T]     — o'rtacha ton (-1..1)
    ev    [T, E]     — global voqea turlari soni
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ASSET_BY_SYMBOL, ASSETS, DECISION_HOUR_UTC, EMB_DIR, EVENT_NAMES, GROUPS
from .tagging import detect_event, tag_assets

log = logging.getLogger(__name__)
SCOPES = ["global"] + GROUPS + [a.symbol for a in ASSETS]
SCOPE_IDX = {s: i for i, s in enumerate(SCOPES)}


def use_date(available_at: pd.Series) -> pd.Series:
    t = pd.to_datetime(available_at, utc=True) - pd.Timedelta(hours=DECISION_HOUR_UTC)
    return t.dt.ceil("D").dt.tz_localize(None)


@dataclass
class NewsCube:
    start: pd.Timestamp
    emb: np.ndarray   # [S, T, d]
    cnt: np.ndarray   # [S, T]
    tone: np.ndarray  # [S, T]
    ev: np.ndarray    # [T, E]
    built_until: str = ""   # kubga kirgan eng so'nggi xabarning available_at (servis shundan keyingisini qo'shadi)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, periods=self.emb.shape[1], freq="D")

    def day_index(self, dates) -> np.ndarray:
        d = (pd.DatetimeIndex(dates) - self.start).days.values
        return np.clip(d, -1, self.emb.shape[1] - 1)  # -1 => kub boshlanishidan oldin (yangilik yo'q)

    def save(self, path=EMB_DIR / "news_cube.npz"):
        np.savez_compressed(path, start=str(self.start.date()), emb=self.emb, cnt=self.cnt, tone=self.tone, ev=self.ev,
                            built_until=self.built_until)

    @classmethod
    def load(cls, path=EMB_DIR / "news_cube.npz") -> "NewsCube":
        z = np.load(path)
        bu = str(z["built_until"]) if "built_until" in z else ""
        return cls(pd.Timestamp(str(z["start"])), z["emb"], z["cnt"], z["tone"], z["ev"], bu)

    def pad_to(self, end) -> "NewsCube":
        T_new = (pd.Timestamp(end) - self.start).days + 1
        extra = T_new - self.emb.shape[1]
        if extra <= 0:
            return self
        S, _, d = self.emb.shape
        return NewsCube(self.start,
                        np.concatenate([self.emb, np.zeros((S, extra, d), "float32")], 1),
                        np.concatenate([self.cnt, np.zeros((S, extra), "float32")], 1),
                        np.concatenate([self.tone, np.zeros((S, extra), "float32")], 1),
                        np.concatenate([self.ev, np.zeros((extra, self.ev.shape[1]), "float32")], 0),
                        self.built_until)

    def add_item(self, day_i: int, scopes: list[int], e: np.ndarray, tone: float, ev_i: int):
        """Bitta yangi xabarni kubga qo'shadi (o'rtachalarni to'g'ri yangilaydi)."""
        for sc in scopes:
            n = self.cnt[sc, day_i]
            self.emb[sc, day_i] = (self.emb[sc, day_i] * n + e) / (n + 1)
            self.tone[sc, day_i] = (self.tone[sc, day_i] * n + tone) / (n + 1)
            self.cnt[sc, day_i] = n + 1
        self.ev[day_i, ev_i] += 1

    @classmethod
    def empty(cls, start, end, dim: int) -> "NewsCube":
        T = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
        S = len(SCOPES)
        return cls(pd.Timestamp(start), np.zeros((S, T, dim), "float32"), np.zeros((S, T), "float32"),
                   np.zeros((S, T), "float32"), np.zeros((T, len(EVENT_NAMES)), "float32"))


def scopes_for(asset_list: list[str]) -> list[int]:
    groups = {ASSET_BY_SYMBOL[s].group for s in asset_list if s in ASSET_BY_SYMBOL}
    return [SCOPE_IDX["global"]] + [SCOPE_IDX[g] for g in sorted(groups)] + \
        [SCOPE_IDX[s] for s in asset_list if s in SCOPE_IDX]


def annotate(news: pd.DataFrame) -> pd.DataFrame:
    """Aktiv teglari (agar datasetda bo'lmasa) va voqea turini qo'shadi."""
    news = news.copy()
    known = set(ASSET_BY_SYMBOL)
    txt = news["title"].astype(str)

    def _assets(row_assets, t):
        given = [s for s in str(row_assets or "").split(",") if s in known]
        return sorted(set(given) | set(tag_assets(t)))

    news["asset_list"] = [_assets(a, t) for a, t in zip(news.get("assets", ""), txt)]
    news["event"] = [detect_event(t) for t in txt]
    news["use_date"] = use_date(news["available_at"])
    return news


def build_cube(news: pd.DataFrame, emb: np.ndarray, tone: np.ndarray, start, end) -> NewsCube:
    """news: annotate() natijasi; emb: [N, d] (PCA'dan keyin), tone: [N]."""
    cube = NewsCube.empty(start, end, emb.shape[1])
    T = cube.emb.shape[1]
    di = cube.day_index(news["use_date"])
    ok = (di >= 0) & (di < T) & (news["use_date"] <= pd.Timestamp(end)).values
    ev_i = news["event"].map({e: i for i, e in enumerate(EVENT_NAMES)}).fillna(0).astype(int).values

    def _acc(scope_i: int, rows: np.ndarray):
        if len(rows) == 0:
            return
        np.add.at(cube.emb[scope_i], di[rows], emb[rows])
        np.add.at(cube.cnt[scope_i], di[rows], 1.0)
        np.add.at(cube.tone[scope_i], di[rows], tone[rows])

    all_rows = np.where(ok)[0]
    _acc(SCOPE_IDX["global"], all_rows)
    np.add.at(cube.ev, (di[all_rows], ev_i[all_rows]), 1.0)

    scope_rows: dict[int, list[int]] = {}
    for r in all_rows:
        lst = news["asset_list"].iat[r]
        groups = {ASSET_BY_SYMBOL[s].group for s in lst}
        for s in set(lst) | groups:
            scope_rows.setdefault(SCOPE_IDX[s], []).append(r)
    for si, rows in scope_rows.items():
        _acc(si, np.asarray(rows))

    n = np.maximum(cube.cnt, 1.0)
    cube.emb /= n[..., None]
    cube.tone /= n
    log.info("NewsCube: %d kun, %d scope, kuniga o'rtacha %.1f xabar", T, len(SCOPES), cube.cnt[0].mean())
    return cube


def build_from_news(news: pd.DataFrame, embedder, reducer_dim: int, fit_before: pd.Timestamp,
                    start, end):
    """To'liq quvur: annotate -> embed -> PCA (faqat fit_before dan oldingi xabarlarda fit) -> cube."""
    from .text_embed import PCAReducer

    news = annotate(news)
    texts = news["title"].astype(str).tolist()
    log.info("Embedding: %d xabar (%s)", len(texts), embedder.name)
    E, tone = embedder.encode(texts)
    # pandas 3 (copy-on-write) .values ni faqat-o'qish massiv qilib qaytaradi -> nusxa olamiz
    fit_mask = np.array(news["use_date"] < pd.Timestamp(fit_before), dtype=bool, copy=True)
    if fit_mask.sum() < reducer_dim * 4:
        log.warning("Holdoutdan oldin yangilik juda kam (%d) — PCA barcha xabarlarda fit qilinadi "
                    "(faqat o'lchamni kamaytirish uchun; ko'p yillik yangilik qo'shsangiz bu yo'qoladi)", fit_mask.sum())
        fit_mask[:] = True
    red = PCAReducer(reducer_dim).fit(E[fit_mask])
    Z = red.transform(E)
    cube = build_cube(news, Z, tone, start, end)
    cube.built_until = pd.Timestamp(news["available_at"].max()).isoformat()
    return cube, red, news
