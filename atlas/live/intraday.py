"""Oltin (yoki istalgan aktiv) uchun ichki kun signallari: 5m, 15m, 30m, 1h, 4h.

Har bir vaqt oralig'i uchun:
  * kunlik texnik tahlildagi detektorlar (SNR, SND, BOS, klassik figuralar, RSI, MA, shamlar) xuddi shunday ishlaydi —
    ular ATR va barlar soniga asoslangan, vaqt o'lchoviga bog'liq emas;
  * belgi: maqsad 1.5·ATR, stop 1·ATR, muddat 20 bar (5m da ~1.7 soat, 4h da ~3.3 kun);
  * bitta umumiy SetupNet (vaqt oralig'i embedding sifatida) o'qitiladi;
  * vaqt bo'yicha bo'linish: 70% o'qitish, 15% validatsiya (chegara tanlash), 15% holdout (yakuniy o'lchov),
    oralarida embargo (belgi 20 bar oldinga qaraydi).

"Kuchli signal" chegarasi validatsiyada tanlanadi va holdoutda o'lchanadi. Natija — aniq raqam
(masalan 58% ± 6%), hech qachon "99%" emas: bunday aniqlik halol o'lchovda bozorda uchramaydi.
"""
from __future__ import annotations

import json
import logging
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import MODELS_DIR, REPORTS_DIR
from ..ta.dataset import _seq, _tab, barrier_outcomes
from ..ta.detect import SETUP_NAMES, SETUPS, indicators, pivots, sanitize_ohlc, scan
from ..ta.model import Data, _auc, _fit_one, predict

log = logging.getLogger(__name__)
TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h"]
TF_MINUTES = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
INTRA_MODEL = MODELS_DIR / "intraday.pt"
INTRA_REPORT = REPORTS_DIR / "intraday_report.json"
INTRA_DIR = Path(MODELS_DIR).parent / "intraday"
MIN_SIGNALS_VAL = 12        # chegarani tanlash uchun validatsiyada kamida shuncha signal bo'lsin


def bars_from_json(raw: list[list]) -> pd.DataFrame:
    """Vercel /api/bars javobi: [[unix_soniya, o, h, l, c, v], ...] -> DataFrame (UTC vaqt)."""
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"]).dropna(subset=["close"])
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_localize(None)
    df = df.drop(columns="ts").drop_duplicates("date").sort_values("date").reset_index(drop=True)
    for k in ("open", "high", "low"):
        df[k] = df[k].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0)
    return sanitize_ohlc(df)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = df.set_index("date").resample(rule, label="left", closed="left")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()}).dropna(subset=["close"])
    return out.reset_index()


def drop_unclosed(df: pd.DataFrame, tf: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Hali yopilmagan oxirgi barni olib tashlaydi (signal faqat yopilgan bardan)."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is not None:
        now = now.tz_convert("UTC").tz_localize(None)
    end = df["date"] + pd.Timedelta(minutes=TF_MINUTES[tf])
    return df[end <= now].reset_index(drop=True)


def load_frames(dirpath: Path = INTRA_DIR, symbol: str = "GOLD") -> dict[str, pd.DataFrame]:
    """Saqlangan ichki kun barlari: <symbol>_<tf>.parquet; 4h — 1h dan yig'iladi."""
    out = {}
    for tf in ("5m", "15m", "30m", "1h"):
        p = dirpath / f"{symbol}_{tf}.parquet"
        if p.exists():
            out[tf] = pd.read_parquet(p)
    if "1h" in out:
        out["4h"] = resample(out["1h"], "4h")
    return out


def build_dataset(frames: dict[str, pd.DataFrame], use_ctx: bool = True) -> dict:
    from .context import context_frame, ctx_vector
    rows, seqs, tabs = [], [], []
    for tf, df in frames.items():
        df = sanitize_ohlc(df.reset_index(drop=True))
        if len(df) < 400:
            continue
        ctx = context_frame(frames, tf, df) if use_ctx else None
        events, aux = scan(df)
        ind, pv = aux["ind"], aux["pv"]
        atr = ind["atr"].to_numpy()
        vol_z = ind["vol_z"].to_numpy() if df["volume"].sum() > 0 else np.zeros(len(df))
        outc = {d: barrier_outcomes(df, atr, d) for d in (1, -1)}
        for ev in events:
            t, d = ev["t"], ev["dir"]
            rows.append({"tf": tf, "group": tf, "date": df["date"].iat[t], "t": t, "setup": ev["setup"], "dir": d,
                         "win": outc[d][0][t], "R": outc[d][1][t]})
            seqs.append(_seq(df, t, ev["atr"], d, vol_z))
            tabs.append(_tab(ev, ind, pv, df) + (ctx_vector(ctx, t, d) if use_ctx else []))
        log.info("%s: %d bar, %d signal", tf, len(df), len(events))
    return {"meta": pd.DataFrame(rows), "seq": np.stack(seqs).astype(np.float32), "tab": np.asarray(tabs, np.float32)}


def _tensors(ds: dict, idx: np.ndarray) -> Data:
    m = ds["meta"].iloc[idx]
    return Data(torch.from_numpy(ds["seq"][idx]), torch.from_numpy(ds["tab"][idx]),
                torch.tensor([SETUP_NAMES.index(s) for s in m["setup"]]),
                torch.tensor([TIMEFRAMES.index(g) for g in m["group"]]),
                torch.tensor(np.nan_to_num(m["win"].to_numpy(float)), dtype=torch.float32))


def _split(meta: pd.DataFrame):
    """Har bir vaqt oralig'ida alohida xronologik bo'linish (70/15/15) + 20 barlik embargo."""
    tr, va, ho = [], [], []
    for tf, g in meta[meta["win"].notna()].groupby("tf"):
        g = g.sort_values("t")
        n = g["t"].max()
        a, b = int(n * 0.70), int(n * 0.85)
        tr += list(g.index[g["t"] < a - 20]); va += list(g.index[(g["t"] >= a) & (g["t"] < b - 20)])
        ho += list(g.index[g["t"] >= b])
    return np.array(tr), np.array(va), np.array(ho)


def wilson(k: int, n: int, z: float = 1.64) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def train(frames: dict[str, pd.DataFrame], n_models: int = 3) -> dict:
    ds = build_dataset(frames)
    meta = ds["meta"]
    tr, va, ho = _split(meta)
    Tr, Va, Ho = _tensors(ds, tr), _tensors(ds, va), _tensors(ds, ho)
    nets = [_fit_one(Tr, Va, s, n_groups=len(TIMEFRAMES)) for s in range(n_models)]
    pv, ph = predict(nets, Va), predict(nets, Ho)
    vm, hm = meta.loc[va].assign(p=pv), meta.loc[ho].assign(p=ph)
    report = {"timeframes": {}, "n_events": int(len(meta)), "rules": {"tp_atr": 1.5, "sl_atr": 1.0, "horizon_bars": 20}}
    thresholds = {}
    for tf in TIMEFRAMES:
        v, h = vm[vm["tf"] == tf], hm[hm["tf"] == tf]
        if len(v) < 50 or len(h) < 50:
            report["timeframes"][tf] = {"n_val": int(len(v)), "n_holdout": int(len(h)), "enabled": False,
                                        "reason": "ma'lumot yetarli emas"}
            continue
        # chegara: validatsiyada eng yuqori aniqlik (kamida MIN_SIGNALS_VAL signal bilan)
        best = None
        for q in np.arange(0.70, 0.99, 0.01):
            thr = float(v["p"].quantile(q))
            sel = v[v["p"] >= thr]
            if len(sel) < MIN_SIGNALS_VAL:
                break
            prec = sel["win"].mean()
            if best is None or prec > best[1]:
                best = (thr, prec, len(sel))
        sel_h = h[h["p"] >= best[0]] if best else h.iloc[:0]
        k, n = int(sel_h["win"].sum()), int(len(sel_h))
        lo, hi = wilson(k, n)
        base = float(h["win"].mean())
        enabled = bool(n >= 8 and lo > base)       # tasodifiy kirishdan ishonchli darajada yaxshi bo'lsagina yoqiladi
        thresholds[tf] = best[0] if best else 1.0
        report["timeframes"][tf] = {
            "n_val": int(len(v)), "n_holdout": int(len(h)), "auc_holdout": _auc(h["win"].to_numpy(), h["p"].to_numpy()),
            "base_win": base * 100, "threshold": best[0] if best else None,
            "val_precision": float(best[1] * 100) if best else None, "val_signals": int(best[2]) if best else 0,
            "holdout_signals": n, "holdout_precision": (k / n * 100) if n else None,
            "holdout_ci90": [lo * 100, hi * 100] if n else None,
            "holdout_avg_R": float(sel_h["R"].mean()) if n else None,
            "period_holdout": [str(h["date"].min()), str(h["date"].max())], "enabled": enabled,
        }
    lab = meta[meta["win"].notna()]
    report["base_rates"] = {f"{tf}|{s}": float(g["win"].mean()) for (tf, s), g in lab.groupby(["tf", "setup"]) if len(g) >= 20}
    torch.save({"states": [n.state_dict() for n in nets], "n_tab": int(Tr.tab.shape[1]), "thresholds": thresholds,
                "report": report, "timeframes": TIMEFRAMES}, INTRA_MODEL)
    INTRA_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return report


def load_model():
    from ..ta.model import SetupNet
    if not INTRA_MODEL.exists():
        return [], {}
    z = torch.load(INTRA_MODEL, map_location="cpu", weights_only=False)
    nets = []
    for st in z["states"]:
        n = SetupNet(z["n_tab"], n_groups=len(z["timeframes"])); n.load_state_dict(st); n.eval(); nets.append(n)
    return nets, z


def scan_latest(df: pd.DataFrame, tf: str, nets, z, lookback_bars: int = 1, frames: dict | None = None) -> list[dict]:
    """Oxirgi yopilgan bar(lar)da paydo bo'lgan signallar + model bahosi."""
    from .context import context_frame, ctx_vector
    df = sanitize_ohlc(df.tail(900).reset_index(drop=True))
    use_ctx = bool(z.get("use_ctx"))
    ctx = context_frame(frames or {tf: df}, tf, df) if use_ctx else None
    if len(df) < 260 or not nets:
        return []
    events, aux = scan(df, start=len(df) - lookback_bars)
    ind, pv = aux["ind"], aux["pv"]
    vol_z = ind["vol_z"].to_numpy() if df["volume"].sum() > 0 else np.zeros(len(df))
    rep = z["report"]["timeframes"].get(tf, {})
    out = []
    for ev in events:
        seq = torch.from_numpy(_seq(df, ev["t"], ev["atr"], ev["dir"], vol_z)[None])
        tab = torch.tensor([_tab(ev, ind, pv, df) + (ctx_vector(ctx, ev["t"], ev["dir"]) if use_ctx else [])],
                           dtype=torch.float32)
        d = Data(seq, tab, torch.tensor([SETUP_NAMES.index(ev["setup"])]), torch.tensor([TIMEFRAMES.index(tf)]),
                 torch.zeros(1))
        p = float(predict(nets, d)[0])
        c = float(df["close"].iat[ev["t"]])
        out.append({"tf": tf, "setup": ev["setup"], "label": SETUPS[ev["setup"]][1], "dir": ev["dir"],
                    "time": str(df["date"].iat[ev["t"]]), "entry": c, "atr": ev["atr"],
                    "tp": c + ev["dir"] * 1.5 * ev["atr"], "sl": c - ev["dir"] * 1.0 * ev["atr"], "p": p,
                    "threshold": z["thresholds"].get(tf), "strong": bool(rep.get("enabled") and p >= z["thresholds"].get(tf, 1.0)),
                    "measured_precision": rep.get("holdout_precision"), "ci90": rep.get("holdout_ci90"),
                    "base_win": z["report"]["base_rates"].get(f"{tf}|{ev['setup']}", rep.get("base_win", 40) / 100)})
    return out


def train_final(frames: dict[str, pd.DataFrame], wf: dict, n_models: int = 3, margin_pp: float = 3.0) -> dict:
    """Servis modeli: barcha ma'lumot (har TF da oxirgi 15% — erta to'xtatish va chegara uchun).

    Aniqlik raqamlari walk-forward natijasidan olinadi (bir nechta model ko'rmagan davr birlashmasi).
    Push faqat: kamida 20 ta sinov signali va 90% ishonch oralig'ining pastki chegarasi tasodifiy kirishdan
    kamida `margin_pp` foiz punkt yuqori bo'lsa (5 ta TF va 2 variant sinalgani uchun qo'shimcha zaxira).
    """
    ds = build_dataset(frames, use_ctx=True)
    meta = ds["meta"]
    lab = meta["win"].notna().to_numpy()
    rel = meta.groupby("tf")["t"].transform(lambda s: s / s.max()).to_numpy()
    tr, va = np.where(lab & (rel < 0.85 - 0.003))[0], np.where(lab & (rel >= 0.85))[0]
    Tr, Va = _tensors(ds, tr), _tensors(ds, va)
    nets = [_fit_one(Tr, Va, 100 + s, n_groups=len(TIMEFRAMES)) for s in range(n_models)]
    from .walkforward import _threshold
    vm = meta.iloc[va].assign(p=predict(nets, Va))
    thresholds, tfs = {}, {}
    for tf in TIMEFRAMES:
        thr = _threshold(vm[vm["tf"] == tf])
        thresholds[tf] = thr if thr is not None else 1.0
        w = wf.get(tf, {})
        ci = w.get("ci90")
        enabled = bool(w.get("strong_n", 0) >= 20 and ci and ci[0] >= w["base_win"] + margin_pp and thr is not None)
        tfs[tf] = {"enabled": enabled, "threshold": thr, "base_win": w.get("base_win"), "auc_holdout": w.get("auc"),
                   "holdout_signals": w.get("strong_n"), "holdout_precision": w.get("strong_win"), "holdout_ci90": ci,
                   "holdout_avg_R": w.get("strong_avg_R"), "n_holdout": w.get("n_test"), "method": "walk-forward, 4 blok",
                   **({} if enabled else {"reason": "ustunlik isbotlanmadi (walk-forward)" if w else "ma'lumot yo'q"})}
    lab_m = meta[lab]
    report = {"timeframes": tfs, "n_events": int(len(meta)),
              "rules": {"tp_atr": 1.5, "sl_atr": 1.0, "horizon_bars": 20},
              "features": "OHLC 64 bar + strategiya + kontekst (sessiya, katta TF trendi)",
              "base_rates": {f"{tf}|{s}": float(g["win"].mean()) for (tf, s), g in lab_m.groupby(["tf", "setup"]) if len(g) >= 20}}
    torch.save({"states": [n.state_dict() for n in nets], "n_tab": int(Tr.tab.shape[1]), "thresholds": thresholds,
                "report": report, "timeframes": TIMEFRAMES, "use_ctx": True}, INTRA_MODEL)
    INTRA_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return report
