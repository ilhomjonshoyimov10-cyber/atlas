"""Asosiy kafolatlar: leakage yo'qligi, teglash, kalendar, kvantil tartibi."""
import numpy as np
import pandas as pd

from atlas.features.news_agg import SCOPE_IDX, annotate, build_cube, use_date
from atlas.features.tagging import detect_event, tag_assets


def test_use_date_respects_decision_time():
    s = pd.Series(pd.to_datetime([
        "2024-03-05 19:59", "2024-03-05 20:00", "2024-03-05 20:01", "2024-03-06 00:00"], utc=True))
    d = use_date(s)
    # 20:00 UTC gacha mavjud -> o'sha kun; undan keyin -> ertasi kun
    assert list(d.dt.strftime("%Y-%m-%d")) == ["2024-03-05", "2024-03-05", "2024-03-06", "2024-03-06"]


def test_date_only_news_is_used_next_day():
    # Kaggle: faqat sana bor -> available_at = ertasi 00:00 -> o'sha kun yopilishida ISHLATILMAYDI
    s = pd.Series(pd.to_datetime(["2016-06-24 00:00"], utc=True))  # 2016-06-23 sanali xabar
    assert use_date(s).iloc[0] == pd.Timestamp("2016-06-24")


def test_cube_never_contains_future_news():
    news = pd.DataFrame({
        "title": ["OPEC agrees output cut", "Bitcoin ETF approved by SEC"],
        "available_at": pd.to_datetime(["2024-01-10 21:00", "2024-01-10 10:00"], utc=True),
        "assets": ["", ""],
    })
    news = annotate(news)
    emb = np.ones((2, 4), "float32")
    cube = build_cube(news, emb, np.zeros(2, "float32"), "2024-01-01", "2024-01-20")
    d10 = cube.day_index([pd.Timestamp("2024-01-10")])[0]
    assert cube.cnt[SCOPE_IDX["global"], d10] == 1          # faqat BTC xabari (10:00)
    assert cube.cnt[SCOPE_IDX["CL"], d10] == 0              # OPEC 21:00 da -> ertaga
    assert cube.cnt[SCOPE_IDX["CL"], d10 + 1] == 1


def test_tagging_and_events():
    t = "Severe drought threatens wheat harvest, USDA says"
    assert "WHEAT" in tag_assets(t)
    assert detect_event("FOMC holds rates steady, Powell says") == "fomc"
    assert "ETH" not in tag_assets("Methane emissions rise")  # so'z chegarasi


def test_calendar_features_shape():
    from atlas.data.calendar import calendar_features
    f = calendar_features(pd.date_range("2024-01-01", "2024-02-01"))
    assert f.shape[1] == 9 and f.notna().all().all()
    assert f.loc["2024-01-30", "fomc_next1"] == 1.0   # FOMC 2024-01-31


def test_quantiles_never_cross():
    import torch
    from atlas.config import ModelConfig
    from atlas.model.net import AtlasNet
    cfg = ModelConfig(hidden=16, news_days=3, lookback=10, heads=2)
    m = AtlasNet(cfg, n_assets=3, price_dim=8, season_dim=24, news_dim=20)
    mm = torch.ones(5, 3).bool()
    mm[0] = False                      # bozor konteksti umuman yo'q qator — NaN bo'lmasligi kerak
    b = {"price": torch.randn(5, 10, 8), "season": torch.randn(5, 24), "asset": torch.tensor([0, 1, 2, 0, 1]),
         "news": torch.randn(5, 3, 20), "news_mask": torch.tensor([[1, 0, 1]] * 5).bool(),
         "market": torch.randn(5, 3, 4), "market_mask": mm}
    q = m(b)
    assert torch.isfinite(q).all()
    assert (q[..., 0] <= q[..., 1]).all() and (q[..., 1] <= q[..., 2]).all()


def test_market_context_is_lagged():
    """Bozorlararo kontekst t kunida faqat t-1 gacha bo'lgan ma'lumotni ko'rsatishi kerak."""
    from atlas.features.dataset import Panel, MARKET_FEATS, PRICE_FEATS, ASSET_IDS
    rng = np.random.default_rng(0)
    d = pd.bdate_range("2020-01-01", periods=400)
    mk = lambda: pd.DataFrame({"date": d, "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(d))))})
    p = Panel.build({"CL": mk(), "GOLD": mk()})
    ai = ASSET_IDS["CL"]
    rows = np.where(p.asset == ai)[0]
    r = rows[300]
    md = p.market_day(np.array([r]))[0]
    prev = rows[299]                                   # oldingi savdo kuni
    ci = PRICE_FEATS.index(MARKET_FEATS[0])
    assert np.isclose(p.market[md, ai, 0], p.feats[prev, ci])   # t kuni -> t-1 qiymati
    assert not np.isclose(p.market[md, ai, 0], p.feats[r, ci])


def _yf_frame(tickers, periods, freq, tz="UTC"):
    """yfinance group_by='ticker' formatidagi soxta jadval: ustunlar (Ticker, Price)."""
    idx = pd.date_range("2026-10-27", periods=periods, freq=freq, tz=tz)
    cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
    data = np.tile(np.arange(1, periods + 1, dtype=float)[:, None] * 10, (1, len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


def test_live_quotes_parsing(monkeypatch):
    import sys, types
    from atlas.data.live import LiveQuotes
    from atlas.config import ASSETS
    fake = types.SimpleNamespace(download=lambda tickers, interval="1d", **k:
                                 _yf_frame(tickers, 3 if interval == "1d" else 5, "D" if interval == "1d" else "min"))
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    class R:
        def raise_for_status(self): pass
        def json(self): return [{"symbol": a.binance.upper(), "lastPrice": "100.5", "priceChangePercent": "1.2"}
                                for a in ASSETS if a.binance]
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: R())
    lq = LiveQuotes()
    lq.poll_once()
    assert lq.error is None
    assert lq.quotes["BTC"]["price"] == 100.5 and lq.quotes["BTC"]["source"] == "Binance"
    assert lq.quotes["GOLD"]["price"] == 50.0 and "GC=F" in lq.quotes["GOLD"]["source"]
    assert len(lq.quotes) == len(ASSETS)


def test_price_clean_handles_yfinance_multiindex():
    from atlas.data.prices import _clean
    raw = _yf_frame(["GC=F"], 5, "D", tz=None).swaplevel(axis=1)   # yf.download(single): (Price, Ticker)
    raw.index.name = "Date"
    df = _clean(raw)
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"] and len(df) == 5


def test_build_from_news_only_recent_news():
    """Faqat so'nggi kunlar yangiliklari bo'lsa ham (real rejim, GDELT 30 kun) quvur ishlashi kerak."""
    from atlas.features.news_agg import build_from_news
    from atlas.features.text_embed import HashingEmbedder
    t = pd.date_range("2026-10-01", periods=300, freq="2h", tz="UTC")
    news = pd.DataFrame({"title": [f"Oil prices rise as OPEC cuts output {i}" for i in range(len(t))],
                         "available_at": t, "assets": ""})
    cube, red, _ = build_from_news(news, HashingEmbedder(), 16, fit_before=pd.Timestamp("2025-10-01"),
                                   start="2026-09-01", end="2026-11-01")
    assert cube.cnt[0].sum() == len(news)


# ---------------- bulut rejimi ----------------
def test_quotes_function_assets_match_config():
    """web/api/quotes.js dagi aktivlar ro'yxati atlas/config.py bilan bir xil bo'lishi shart."""
    import json
    import re
    from pathlib import Path

    from atlas.config import ASSETS

    js = (Path(__file__).resolve().parents[1] / "web" / "api" / "quotes.js").read_text()
    lst = json.loads(re.search(r"export const ASSETS = (\[.*?\]\]);", js, re.S).group(1))
    assert lst == [[a.symbol, a.yf, a.binance] for a in ASSETS]


def test_closed_bars_only_drops_unfinished_day():
    import pandas as pd

    from atlas.cloud import closed_bars_only

    df = pd.DataFrame({"date": pd.to_datetime(["2026-10-28", "2026-10-29", "2026-10-30"]), "close": [1.0, 2.0, 3.0]})
    morning = pd.Timestamp("2026-10-30 09:00", tz="UTC")
    evening = pd.Timestamp("2026-10-30 21:30", tz="UTC")
    assert closed_bars_only(df, morning)["date"].max() == pd.Timestamp("2026-10-29")
    assert closed_bars_only(df, evening)["date"].max() == pd.Timestamp("2026-10-30")


def test_track_record_scores_only_matured(tmp_path, monkeypatch):
    import json

    import pandas as pd

    import atlas.cloud as cloud

    monkeypatch.setattr(cloud, "LOG_PATH", tmp_path / "log.jsonl")
    dates = pd.bdate_range("2026-01-05", periods=10)
    prices = {"GOLD": pd.DataFrame({"date": dates, "close": [100.0 + i for i in range(10)]})}
    entry = {"symbol": "GOLD", "as_of": str(dates[7].date()), "last": 107.0,
             "h": [{"h": 1, "q05": 105, "q50": 108, "q95": 110, "p_up": 0.6},
                   {"h": 5, "q05": 100, "q50": 109, "q95": 120, "p_up": 0.6}]}
    cloud.LOG_PATH.write_text(json.dumps(entry) + "\n")
    tr = cloud.track_record(prices)
    assert tr["pending"] == 1                                  # 5 kunlik hali muddati kelmagan
    assert len(tr["recent"]) == 1 and tr["recent"][0]["real"] == 108.0
    assert tr["recent"][0]["inside"] and tr["recent"][0]["dir_ok"]
    assert tr["summary"][0]["coverage90"] == 100.0


def test_backtest_rows_remap_after_new_days():
    import numpy as np
    import pandas as pd

    from atlas.model.backtest import remap_rows

    class P:  # minimal panel: 2 aktiv, ikkinchisining oldiga yangi kun qo'shilgandek
        asset = np.array([0, 0, 0, 1, 1])
        date = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-01", "2026-01-02"]).values

    rows = remap_rows(P, np.array([1, 0, 1]), pd.to_datetime(["2026-01-02", "2026-01-03", "2026-02-01"]).values)
    assert rows.tolist() == [4, 2, -1]


# ---------------- texnik tahlil ----------------
def _ohlc(n=700, seed=3):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.01, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "open": o, "high": h, "low": l,
                         "close": c, "volume": rng.integers(1000, 5000, n)})


def test_ta_no_lookahead():
    """t kungi signallar va darajalar kelajakdagi barlar o'zgarsa ham o'zgarmasligi kerak."""
    from atlas.ta.detect import indicators, levels_at, pivots, scan

    df = _ohlc()
    cut = 550
    future = df.copy()
    future.loc[cut + 1:, ["open", "high", "low", "close"]] *= 1.3     # kelajakni butunlay o'zgartiramiz
    ev_a = [e for e in scan(df)[0] if e["t"] <= cut]
    ev_b = [e for e in scan(future)[0] if e["t"] <= cut]
    assert [(e["t"], e["setup"]) for e in ev_a] == [(e["t"], e["setup"]) for e in ev_b]
    atr = indicators(df)["atr"].to_numpy()
    la = levels_at(pivots(df), cut, atr[cut], df["close"].iat[cut])
    lb = levels_at(pivots(future), cut, atr[cut], df["close"].iat[cut])
    assert la == lb


def test_barrier_labels():
    import numpy as np
    import pandas as pd

    from atlas.ta.dataset import HORIZON, barrier_outcomes

    n = 40
    c = np.full(n, 100.0)
    c[3:] = 103.0                                   # 1-kundan keyin keskin o'sish
    df = pd.DataFrame({"open": c, "high": c + 0.1, "low": c - 0.1, "close": c})
    atr = np.full(n, 2.0)
    win_l, R_l = barrier_outcomes(df, atr, 1)
    win_s, R_s = barrier_outcomes(df, atr, -1)
    assert win_l[0] == 1 and R_l[0] == 1.5          # long: +1.5 ATR (103 >= 100+3) maqsadi urildi
    assert win_s[0] == 0 and R_s[0] == -1.0         # short: stop urildi
    assert np.isnan(win_l[n - HORIZON])             # muddat to'lmagan — noma'lum


def test_ta_analyze_runs():
    from atlas.ta.analyze import analyze

    a = analyze("GOLD", {"GOLD": _ohlc()})
    assert a["bias"] in ("ko'tarilish", "pasayish", "neytral") and a["text"]
    assert all(l["type"] in ("support", "resistance") for l in a["levels"])


# ---------------- jonli signallar ----------------
def test_signal_outcome_scoring(tmp_path, monkeypatch):
    import json

    import pandas as pd

    import atlas.live.engine as E

    monkeypatch.setattr(E, "SIGNALS_LOG", tmp_path / "s.jsonl")
    t0 = pd.Timestamp("2026-10-30 10:00")
    E.journal({"kind": "gold", "tf": "1h", "time": str(t0), "dir": 1, "tp": 103.0, "sl": 99.0, "status": "ochiq", "strong": True})
    E.journal({"kind": "gold", "tf": "1h", "time": str(t0), "dir": -1, "tp": 97.0, "sl": 101.0, "status": "ochiq", "strong": True})
    bars = pd.DataFrame({"date": [t0 + pd.Timedelta(hours=k) for k in range(1, 4)],
                         "open": [100, 101, 102.5], "high": [100.5, 102, 103.5], "low": [99.5, 100.5, 102], "close": [100, 101.5, 103]})
    assert E.score_open({"1h": bars}) == 2
    st = [r["status"] for r in E.read_journal()]
    assert st == ["maqsad", "stop"]                      # long maqsadga yetdi, short stop urildi
    assert E.summary()["live"] == {"strong": 2, "closed": 2, "tp": 1, "sl": 1}


def test_news_alert_only_when_strong(tmp_path, monkeypatch):
    import atlas.live.engine as E

    monkeypatch.setattr(E, "SIGNALS_LOG", tmp_path / "s.jsonl")
    weak = [{"symbol": "GOLD", "name": "Oltin", "tagged": True, "ret_h1": 0.001, "ret_h5": 0.002, "sigma_h1": 0.01}]
    assert E.news_alert({"title": "Company opens new office", "event": "none"}, weak, thr=0.1, push_enabled=False) is None
    e = E.news_alert({"title": "Fed raises rates", "event": "fomc"}, weak, thr=0.1, push_enabled=False)
    assert e is not None and e["event"] == "fomc"        # muhim makro voqea — kuchsiz bo'lsa ham xabar
