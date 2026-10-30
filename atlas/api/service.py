"""Servis holati: narxlar, yangilik kubi, model — va ular ustidagi amallar."""
from __future__ import annotations

import logging
import pickle
import threading

import numpy as np
import pandas as pd
import torch

from ..config import ASSET_BY_SYMBOL, ASSETS, EVENT_NAMES, ModelConfig
from ..data.collector import NewsStore
from ..data.prices import load_prices
from ..features.dataset import ASSET_IDS, BatchMaker, Panel
from ..features.news_agg import NewsCube, scopes_for, use_date
from ..features.tagging import detect_event, tag_assets
from ..features.text_embed import PCAReducer, get_embedder
from ..model.train import apply_conformal, load_model
from ..pipeline import BUNDLE_PATH, REDUCER_PATH

log = logging.getLogger(__name__)


class Service:
    def __init__(self):
        self.lock = threading.RLock()
        self.store = NewsStore()
        self.ready = False
        self.error = None
        try:
            self._load()
            self.ready = True
        except Exception as e:  # ma'lumot hali tayyor bo'lmasa ham API ishlasin
            log.exception("Servis to'liq yuklanmadi")
            self.error = str(e)

    # ---------- yuklash ----------
    def _load(self):
        self.prices = load_prices()
        if not self.prices:
            raise RuntimeError("Narxlar yo'q. `python -m atlas.cli download` yoki `demo` ni ishga tushiring.")
        n_assets = len(ASSETS)
        self.model, z, cfg = load_model(BUNDLE_PATH, n_assets)
        self.cfg: ModelConfig = cfg
        self.conformal = np.asarray(z["conformal"], "float32")
        self.report = z.get("report") or {}
        self.panel = Panel.build(self.prices, cfg.horizons)
        self.cube = None
        if cfg.use_news:
            cube = NewsCube.load()
            self.cube = cube.pad_to(pd.Timestamp.utcnow().tz_localize(None).normalize() + pd.Timedelta(days=10))
            r = pickle.load(open(REDUCER_PATH, "rb"))
            self.reducer = PCAReducer.from_state(r["reducer"])
            self.embedder = get_embedder(r["embedder"] if r["embedder"] != "none" else "hashing")
            self.embedder_name = self.embedder.name
        self.bm = BatchMaker(self.panel, self.cube, cfg, "cpu")
        self.latest = self.panel.latest_rows()
        self._ingest_store_backlog()
        log.info("Servis tayyor: %d aktiv, yangilik=%s", len(self.latest), cfg.use_news)

    def _ingest_store_backlog(self, days: int = 14):
        """Kollektor bazasidagi so'nggi xabarlarni kubga qo'shadi (o'qitishda bo'lmagan yangi kunlar)."""
        if self.cube is None:
            return
        since = pd.Timestamp(self.cube.built_until) if self.cube.built_until else \
            pd.Timestamp(self.panel.date.max()).tz_localize("UTC")
        items = [it for it in self.store.recent(5000) if pd.Timestamp(it["first_seen"]) > since]
        log.info("Kubga qo'shiladigan yangi xabarlar: %d", len(items))
        self.add_news(items)

    # ---------- yangilik ----------
    def _encode(self, texts: list[str]):
        E, tone = self.embedder.encode(texts)
        return self.reducer.transform(E), tone

    def add_news(self, items: list[dict]):
        if self.cube is None or not items:
            return
        with self.lock:
            Z, tone = self._encode([f"{it['title']}" for it in items])
            ud = use_date(pd.Series([it["first_seen"] for it in items]))
            di = self.cube.day_index(ud)
            for k, it in enumerate(items):
                if not (0 <= di[k] < self.cube.emb.shape[1]):
                    continue
                assets = [s for s in (it.get("assets") or "").split(",") if s] or tag_assets(it["title"])
                ev = it.get("event") or detect_event(it["title"])
                self.cube.add_item(int(di[k]), scopes_for(assets), Z[k], float(tone[k]), EVENT_NAMES.index(ev))

    def live_day(self) -> int | None:
        if self.cube is None:
            return None
        now = pd.Series([pd.Timestamp.utcnow()])
        day = int(self.cube.day_index(use_date(now))[0])
        if self.cube.cnt[0, day] == 0:
            # Bugun hali xabar yo'q (masalan, kollektor o'chiq) — bo'sh kun modelga g'ayritabiiy ko'rinadi,
            # shuning uchun xabari bor eng so'nggi kunni olamiz.
            nz = np.nonzero(self.cube.cnt[0, : day + 1])[0]
            if len(nz):
                day = int(nz[-1])
        return day

    # ---------- bashorat ----------
    def _to_price(self, sym: str, qn: np.ndarray) -> list[dict]:
        row = self.latest[sym]
        logc, vol = self.panel.logc[row], float(self.panel.vol[row])
        out = []
        for j, h in enumerate(self.cfg.horizons):
            r = qn[j] * vol * np.sqrt(h)          # log-qaytim
            out.append({"h": h, "q05": float(np.exp(logc + r[0])), "q50": float(np.exp(logc + r[1])),
                        "q95": float(np.exp(logc + r[2])),
                        "ret_q05": float(np.expm1(r[0]) * 100), "ret_q50": float(np.expm1(r[1]) * 100),
                        "ret_q95": float(np.expm1(r[2]) * 100),
                        "p_up": float(self._p_up(qn[j]))})
        return out

    @staticmethod
    def _p_up(q: np.ndarray) -> float:
        """Kvantillardan o'sish ehtimolini chiziqli interpolyatsiya bilan taxminlash."""
        xs, ps = [q[0], q[1], q[2]], [0.05, 0.5, 0.95]
        if 0 <= xs[0]:
            return 0.97
        if 0 >= xs[2]:
            return 0.03
        cdf0 = float(np.interp(0.0, xs, ps))
        return 1 - cdf0

    @torch.no_grad()
    def forecast(self, symbols: list[str] | None = None, extra_news=None, explain=False):
        syms = [s for s in (symbols or list(self.latest)) if s in self.latest]
        rows = np.array([self.latest[s] for s in syms])
        with self.lock:
            b = self.bm(rows, extra_news=extra_news, day_override=self.live_day())
            q, attn = self.model(b, return_attn=True)
        q = apply_conformal(q.numpy(), self.conformal)
        res = {}
        for i, s in enumerate(syms):
            item = {"symbol": s, "as_of": str(pd.Timestamp(self.panel.date[self.latest[s]]).date()),
                    "last": float(np.exp(self.panel.logc[self.latest[s]])), "horizons": self._to_price(s, q[i])}
            if explain:
                if attn.get("news") is not None:
                    item["news_attention"] = attn["news"][i].numpy().round(4).tolist()
                if attn.get("market") is not None:
                    w = attn["market"][i].numpy()
                    top = np.argsort(-w)[:5]
                    item["market_attention"] = [{"symbol": self.panel.symbols[j], "name": ASSET_BY_SYMBOL[self.panel.symbols[j]].name,
                                                 "weight": float(w[j])} for j in top if self.panel.symbols[j] != s]
                if attn.get("gate") is not None:
                    g = attn["gate"][i].numpy()
                    tot = float(g.sum()) or 1.0
                    item["branch_weights"] = {"narx": float(g[0] / tot), "bozor": float(g[1] / tot),
                                              "yangilik": float(g[2] / tot), "mavsum": float(g[3] / tot)}
            res[s] = item
        return res

    PLACEBO = ("Markets await fresh economic data this week",
               "Analysts debate the outlook for the second half of the year",
               "Officials are scheduled to speak at an industry conference")

    def _delta(self, z: np.ndarray, tone: float, scopes: set, ev: str, day: int) -> torch.Tensor:
        """Bitta xabar kunlik tokenlarni qanchaga o'zgartirishi — har bir aktiv uchun [A, news_dim]."""
        from ..features.news_agg import SCOPE_IDX
        cube = self.cube
        dim = cube.emb.shape[2]
        delta = torch.zeros(len(self.panel.symbols), self.bm.news_dim)
        ei = EVENT_NAMES.index(ev)
        for sym, ai in ASSET_IDS.items():
            sc = [0, SCOPE_IDX[ASSET_BY_SYMBOL[sym].group], SCOPE_IDX[sym]]
            for slot, s_ in enumerate(sc):
                if s_ not in scopes:
                    continue
                n = cube.cnt[s_, day]
                delta[ai, slot * dim:(slot + 1) * dim] = torch.from_numpy((z - cube.emb[s_, day]) / (n + 1))
                delta[ai, 3 * dim + slot] = float(np.log1p(n + 1) - np.log1p(n))
                delta[ai, 3 * dim + 3 + slot] = float((tone - cube.tone[s_, day]) / (n + 1))
            delta[ai, 3 * dim + 6 + ei] = float(np.log1p(cube.ev[day, ei] + 1) - np.log1p(cube.ev[day, ei]))
        return delta

    def _q50_sigma(self, extra=None) -> dict[str, np.ndarray]:
        """Normallashtirilgan (sigma birligidagi) median bashorat, har bir aktiv uchun [H]."""
        syms = list(self.latest)
        rows = np.array([self.latest[s] for s in syms])
        with self.lock, torch.no_grad():
            q = self.model(self.bm(rows, extra_news=extra, day_override=self.live_day())).numpy()
        return {s: q[i, :, 1] for i, s in enumerate(syms)}

    def analyze_text(self, text: str) -> dict:
        """'Shu yangilik chiqsa nima bo'ladi?' — model orqali har bir aktivga ta'sirni o'lchash.

        Nazorat (placebo): xuddi shu aktivlarga teglangan, lekin mazmunan neytral xabarlar qo'shiladi.
        Ta'sir = (shu xabar bilan bashorat) - (neytral xabarlar bilan bashorat o'rtachasi).
        Shu tarzda "yana bitta xabar qo'shildi" degan hajm effekti ayiriladi va faqat MAZMUN ta'siri qoladi.
        """
        assets = tag_assets(text)
        ev = detect_event(text)
        out = {"text": text, "assets": assets, "event": ev, "impact": []}
        if self.cube is None:
            out["note"] = "Model yangiliksiz o'qitilgan — faqat teglar ko'rsatildi"
            return out
        Z, tone = self._encode([text, *self.PLACEBO])
        out["tone"] = float(tone[0])
        day = self.live_day()
        scopes = set(scopes_for(assets))
        d_test = self._delta(Z[0], float(tone[0]), scopes, ev, day)
        with_news = self._q50_sigma(lambda a: d_test[a])
        placebo = [self._q50_sigma(lambda a, d=self._delta(Z[k], 0.0, scopes, "none", day): d[a])
                   for k in range(1, len(Z))]
        hs = self.cfg.horizons
        sig = {s_: qn - np.mean([p[s_] for p in placebo], 0) for s_, qn in with_news.items()}   # sigma birligida
        # Barcha aktivlarga umumiy siljish (asosan "global yangilik foni" sezgirligi) — alohida ko'rsatiladi,
        # jadvalda esa aktivning o'ziga xos (kesim bo'yicha nisbiy) javobi qoladi.
        common = np.median(np.stack(list(sig.values())), 0)
        out["common_sigma"] = {f"h{h}": float(common[j]) for j, h in enumerate(hs)}
        for s_, sg in sig.items():
            rel = sg - common
            vol = float(self.panel.vol[self.latest[s_]])
            out["impact"].append({
                "symbol": s_, "name": ASSET_BY_SYMBOL[s_].name, "group": ASSET_BY_SYMBOL[s_].group,
                "tagged": s_ in assets,
                **{f"sigma_h{h}": float(rel[j]) for j, h in enumerate(hs)},
                # foizga: log-qaytim = sigma * vol * sqrt(h)
                **{f"s_ret_h{h}": float(np.expm1(rel[j] * vol * np.sqrt(h)) * 100) for j, h in enumerate(hs)},
                **{f"d_ret_h{h}": float(np.expm1(sg[j] * vol * np.sqrt(h)) * 100) for j, h in enumerate(hs)},
            })
        out["common"] = {f"h{h}": float(np.median([r[f"d_ret_h{h}"] for r in out["impact"]])) for h in hs}
        key = f"sigma_h{hs[0]}"
        out["impact"].sort(key=lambda r: (not r["tagged"], -abs(r[key])))   # tilga olingan aktivlar birinchi
        return out

    def quick_impact(self, title: str, top: int = 4) -> list[dict]:
        """Lenta uchun ixcham ta'sir: eng kuchli o'ziga xos javob bergan aktivlar."""
        res = self.analyze_text(title)
        if not res.get("impact"):
            return []
        h1, h5 = (f"s_ret_h{h}" for h in self.cfg.horizons[:2])
        imp = sorted(res["impact"], key=lambda r: -abs(r[f"sigma_h{self.cfg.horizons[0]}"]))[:top]
        return [{"symbol": r["symbol"], "name": r["name"], "tagged": r["tagged"],
                 "ret_h1": round(r[h1], 4), "ret_h5": round(r[h5], 4),
                 "sigma_h1": round(r[f"sigma_h{self.cfg.horizons[0]}"], 3)} for r in imp]

    # ---------- ko'rinishlar ----------
    def assets_overview(self) -> list[dict]:
        out = []
        for a in ASSETS:
            df = self.prices.get(a.symbol)
            if df is None or len(df) < 2:
                continue
            c = df["close"].values
            out.append({"symbol": a.symbol, "name": a.name, "group": a.group, "binance": a.binance,
                        "last": float(c[-1]), "change_pct": float((c[-1] / c[-2] - 1) * 100),
                        "date": str(df["date"].iloc[-1].date()), "spark": [float(x) for x in c[-60:]]})
        return out
