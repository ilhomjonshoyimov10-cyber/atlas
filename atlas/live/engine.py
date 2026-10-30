"""Jonli signal dvigateli: yangilik shoki va oltinning ichki kun signallari -> push xabarnoma + jurnal.

Ishlash joyi:
  * kompyuterda — FastAPI server ichida (har 60 soniyada), FinBERT bilan to'liq kuch;
  * bulutda — `python -m atlas.cli live --once` (GitHub Actions har 5 daqiqada, to'lov bloki hal bo'lgach).

Push kanali — ntfy.sh (bepul, kalitsiz). Sayt "Signallar" sahifasida brauzer xabarnomalarini yoqadi;
brauzer yopiq bo'lsa ham kelishi uchun ntfy veb-ilovasi yoki telefon ilovasida shu mavzuga obuna bo'linadi.

Halollik qoidalari:
  * signal faqat holdoutda tasodifiy kirishdan ishonchli yaxshi chiqqan vaqt oraliqlarida push qilinadi;
  * har bir xabarda o'lchangan aniqlik va 90% ishonch oralig'i yoziladi;
  * har bir signal jurnalga yoziladi va natijasi (maqsad/stop) keyin avtomatik belgilanadi — tanlab olinmaydi.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

import pandas as pd

from ..config import DATA_DIR
from .intraday import INTRA_DIR, TF_MINUTES, TIMEFRAMES, drop_unclosed, load_model, resample, scan_latest

log = logging.getLogger(__name__)
NTFY_TOPIC = os.environ.get("ATLAS_NTFY_TOPIC", "atlas-bozor-signal-uz7k3")
NTFY_URL = os.environ.get("ATLAS_NTFY_URL", "https://ntfy.sh")
SITE_URL = os.environ.get("ATLAS_SITE_URL", "https://atlas-bozor.vercel.app")
SIGNALS_LOG = DATA_DIR / "signals_log.jsonl"
GOLD_TICKER = "GC=F"
YF_PERIOD = {"5m": "5d", "15m": "10d", "30m": "20d", "1h": "60d"}
IMPORTANT_EVENTS = {"fomc", "cpi", "nfp", "gdp", "ecb", "opec", "geopolitics"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ push
def push(title: str, message: str, tags: list[str] | None = None, priority: int = 3, click: str | None = None) -> bool:
    import httpx
    try:
        r = httpx.post(NTFY_URL, json={"topic": NTFY_TOPIC, "title": title, "message": message, "tags": tags or [],
                                       "priority": priority, "click": click or SITE_URL}, timeout=10)
        return r.status_code < 300
    except Exception as e:   # tarmoq xatosi signal jurnalini to'xtatmasin
        log.warning("Push yuborilmadi: %s", e)
        return False


def journal(entry: dict) -> None:
    with SIGNALS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def read_journal() -> list[dict]:
    if not SIGNALS_LOG.exists():
        return []
    out = []
    for line in SIGNALS_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


# ------------------------------------------------------------------ narxlar
def fetch_intraday(ticker: str = GOLD_TICKER) -> dict[str, pd.DataFrame]:
    """yfinance orqali so'nggi ichki kun barlari; saqlangan tarix bilan birlashtiriladi (dataset o'sib boradi)."""
    import yfinance as yf

    INTRA_DIR.mkdir(parents=True, exist_ok=True)
    frames = {}
    for tf, per in YF_PERIOD.items():
        try:
            raw = yf.download(ticker, period=per, interval="60m" if tf == "1h" else tf, progress=False,
                              auto_adjust=False, threads=False)
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw.rename(columns=str.lower).reset_index()
            df = df.rename(columns={df.columns[0]: "date"})
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
            df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
            p = INTRA_DIR / f"GOLD_{tf}.parquet"
            if p.exists():
                old = pd.read_parquet(p)
                df = pd.concat([old[old["date"] < df["date"].min()], df])
            df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
            df.to_parquet(p, index=False)
            frames[tf] = df
        except Exception as e:
            log.warning("Oltin %s barlari olinmadi: %s", tf, e)
    if "1h" in frames:
        frames["4h"] = resample(frames["1h"], "4h")
    return frames


# ------------------------------------------------------------------ natijalarni belgilash
def score_open(frames: dict[str, pd.DataFrame]) -> int:
    """Ochiq signallar: keyingi barlarda maqsad yoki stop urildimi (bitta barda ikkalasi — stop)."""
    rows = read_journal()
    changed = 0
    for r in rows:
        if r.get("kind") != "gold" or r.get("status") not in (None, "ochiq"):
            continue
        df = frames.get(r["tf"])
        if df is None:
            continue
        after = df[df["date"] > pd.Timestamp(r["time"])].head(20)
        st = "ochiq"
        for _, b in after.iterrows():
            hit_sl = b["low"] <= r["sl"] if r["dir"] > 0 else b["high"] >= r["sl"]
            hit_tp = b["high"] >= r["tp"] if r["dir"] > 0 else b["low"] <= r["tp"]
            if hit_sl:
                st = "stop"; break
            if hit_tp:
                st = "maqsad"; break
        if st == "ochiq" and len(after) >= 20:
            st = "muddati o'tdi"
        if st != r.get("status", "ochiq"):
            r["status"], r["closed_at"] = st, _now()
            changed += 1
    if changed:
        SIGNALS_LOG.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows), encoding="utf-8")
    return changed


# ------------------------------------------------------------------ oltin signallari
def _fmt(v: float) -> str:
    return f"{v:,.2f}".replace(",", " ")


def gold_tick(frames: dict[str, pd.DataFrame] | None = None, news_ctx: str = "", push_enabled: bool = True) -> list[dict]:
    nets, z = load_model()
    if not nets:
        return []
    frames = frames or fetch_intraday()
    rows = [r for r in read_journal() if r.get("kind") == "gold"]
    seen = {(r.get("tf"), r.get("time"), r.get("setup")) for r in rows}
    last_push: dict = {}
    for r in rows:
        if r.get("strong") and not r.get("repeat"):
            k = (r["tf"], r["dir"])
            last_push[k] = max(last_push.get(k, pd.Timestamp.min), pd.Timestamp(r["time"]))
    fired = []
    for tf in TIMEFRAMES:
        df = frames.get(tf)
        if df is None or len(df) < 300:
            continue
        df = drop_unclosed(df, tf)
        for s in scan_latest(df, tf, nets, z, lookback_bars=2, frames=frames):
            key = (tf, s["time"], s["setup"])
            if key in seen:
                continue
            # faqat yangi yopilgan bar: kechikib topilgan signal (natijasi qisman ma'lum) jurnalga yozilmaydi
            closed_at = pd.Timestamp(s["time"]) + pd.Timedelta(minutes=TF_MINUTES[tf])
            if pd.Timestamp.now(tz="UTC").tz_localize(None) - closed_at > pd.Timedelta(minutes=1.5 * TF_MINUTES[tf] + 2):
                seen.add(key)
                continue
            seen.add(key)
            entry = {"kind": "gold", "symbol": "GOLD", "created": _now(), "status": "ochiq", "news": news_ctx, **s}
            # takroriy signal: shu TF va yo'nalishda so'nggi 8 bar ichida push bo'lgan bo'lsa — jurnalga yoziladi, push yo'q
            if s["strong"] and last_push.get((tf, s["dir"])) and \
                    pd.Timestamp(s["time"]) - last_push[(tf, s["dir"])] < pd.Timedelta(minutes=8 * TF_MINUTES[tf]):
                entry["repeat"] = True
            journal(entry)
            if s["strong"] and not entry.get("repeat"):
                last_push[(tf, s["dir"])] = pd.Timestamp(s["time"])
                fired.append(entry)
                if push_enabled:
                    ci = s.get("ci90") or [None, None]
                    side = "BUY ▲" if s["dir"] > 0 else "SELL ▼"
                    msg = (f"{s['label']} ({tf}). Kirish {_fmt(s['entry'])}, maqsad {_fmt(s['tp'])}, stop {_fmt(s['sl'])}.\n"
                           f"Model bahosi {s['p'] * 100:.0f}% (tasodifiy kirish ~{s['base_win'] * 100:.0f}%). "
                           f"Bu vaqt oralig'ida model ko'rmagan davrda shunday signallarning "
                           f"{s['measured_precision']:.0f}% maqsadga yetgan (90% oraliq {ci[0]:.0f}–{ci[1]:.0f}%)."
                           + (f"\nYangilik foni: {news_ctx}" if news_ctx else "")
                           + "\nInvestitsiya tavsiyasi emas.")
                    push(f"Oltin {side} · {tf}", msg, tags=["chart_with_upwards_trend" if s["dir"] > 0 else "chart_with_downwards_trend"],
                         priority=4, click=f"{SITE_URL}/?sel=GOLD")
    return fired


# ------------------------------------------------------------------ yangilik shoki
def news_threshold(store) -> float:
    """Kuchli ta'sir chegarasi: so'nggi xabarlar |sigma_h1| taqsimotining 95-persentili."""
    vals = []
    for it in store.recent(400):
        try:
            vals += [abs(x["sigma_h1"]) for x in json.loads(it.get("impact") or "[]")]
        except Exception:
            pass
    if len(vals) < 30:
        return float("inf")
    vals.sort()
    return vals[int(len(vals) * 0.95)]


def news_alert(item: dict, impact: list[dict], thr: float, push_enabled: bool = True) -> dict | None:
    """Yangi xabar: ta'sir kuchli bo'lsa yoki muhim makro voqea bo'lsa — push."""
    from ..features.tagging import detect_event
    ev = item.get("event") or detect_event(item["title"])
    strong = [x for x in impact if abs(x["sigma_h1"]) >= thr]
    if not strong and ev not in IMPORTANT_EVENTS:
        return None
    top = (strong or impact)[:4]
    lines = [f"{x['name']}: ertaga {'↑' if x['ret_h1'] >= 0 else '↓'}{abs(x['ret_h1']):.3f}%, 5 kunda "
             f"{'↑' if x['ret_h5'] >= 0 else '↓'}{abs(x['ret_h5']):.3f}%" for x in top]
    entry = {"kind": "news", "created": _now(), "id": item.get("id"), "title": item["title"], "source": item.get("source"),
             "event": ev, "first_seen": item.get("first_seen"), "impact": top, "strong": bool(strong)}
    journal(entry)
    if push_enabled:
        push(f"Yangilik: {item['title'][:90]}",
             "\n".join(lines) + "\nModel bahosi: shu xabarning o'ziga xos ta'siri (neytral xabarga nisbatan). Tavsiya emas.",
             tags=["newspaper"], priority=4 if ev in IMPORTANT_EVENTS else 3)
    return entry


def recent_gold_news(store, hours: int = 3) -> str:
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    out = []
    for it in store.recent(80, "GOLD"):
        if pd.Timestamp(it["first_seen"]) < since:
            continue
        try:
            imp = next((x for x in json.loads(it.get("impact") or "[]") if x["symbol"] == "GOLD"), None)
        except Exception:
            imp = None
        out.append(f"«{it['title'][:70]}»" + (f" (oltinga {imp['ret_h1']:+.3f}%)" if imp else ""))
    return "; ".join(out[:2])


# ------------------------------------------------------------------ fon jarayoni (server ichida)
class LiveEngine:
    def __init__(self, store=None, interval: int = 60):
        self.store = store
        self.interval = interval
        self._stop = threading.Event()
        self.last_run: str | None = None
        self.last_error: str | None = None
        self.fired = 0

    def run_once(self):
        frames = fetch_intraday()
        ctx = recent_gold_news(self.store) if self.store else ""
        self.fired += len(gold_tick(frames, ctx))
        score_open(frames)
        self.last_run = _now()

    def loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self.run_once()
                self.last_error = None
            except Exception as e:
                self.last_error = str(e)[:200]
                log.warning("Jonli signal xatosi: %s", e)
            self._stop.wait(max(15, self.interval - (time.time() - t0)))

    def start(self):
        threading.Thread(target=self.loop, daemon=True, name="live-signals").start()

    def stop(self):
        self._stop.set()


def summary(limit: int = 100) -> dict:
    """Sayt uchun: so'nggi signallar, natijalar statistikasi va har bir vaqt oralig'ining holati."""
    from .intraday import INTRA_REPORT
    rows = read_journal()
    gold = [r for r in rows if r.get("kind") == "gold"]
    strong = [r for r in gold if r.get("strong")]
    closed = [r for r in strong if r.get("status") in ("maqsad", "stop", "muddati o'tdi")]
    rep = json.loads(INTRA_REPORT.read_text(encoding="utf-8", errors="replace")) if INTRA_REPORT.exists() else {}
    return {
        "topic": NTFY_TOPIC, "ntfy_url": NTFY_URL,
        "timeframes": rep.get("timeframes", {}), "rules": rep.get("rules", {}),
        "live": {"strong": len(strong), "closed": len(closed),
                 "tp": sum(r["status"] == "maqsad" for r in closed), "sl": sum(r["status"] == "stop" for r in closed)},
        "recent": sorted([r for r in rows if r.get("kind") == "news" or r.get("strong")],
                         key=lambda r: r.get("created", ""), reverse=True)[:limit],
        "watch": sorted([r for r in gold if not r.get("strong")], key=lambda r: r.get("created", ""), reverse=True)[:20],
    }
