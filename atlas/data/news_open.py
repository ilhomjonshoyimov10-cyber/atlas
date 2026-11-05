"""Ochiq yangilik datasetlarini yagona formatga keltirish.

Yagona sxema (har bir yangilik = bitta qator):
    id, origin, source, title, url, published_at (UTC), available_at (UTC), assets (ixtiyoriy)

`available_at` — LEAKAGE QOIDASI uchun asosiy ustun: yangilik faqat
`available_at <= decision_time` bo'lgandagina ishlatiladi.
    * Faqat sana ma'lum bo'lsa (Kaggle) -> keyingi kun 00:00 UTC (konservativ: o'sha kun yopilishida ishlatilmaydi)
    * Vaqt belgisi bor (FNSPID, GDELT) -> published_at + 5 daqiqa
    * O'z kollektorimiz -> first_seen (biz uni qachon ko'rganimiz)

Fayllarni qo'lda yuklab `data/news/raw/` ga qo'ying:
    Kaggle:  https://www.kaggle.com/datasets/aaron7sun/stocknews  (Combined_News_DJIA.csv, RedditNews.csv)
    FNSPID:  https://huggingface.co/datasets/Zihan1004/FNSPID  (nasdaq_exteral_data.csv)
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from ..config import NEWS_DIR

log = logging.getLogger(__name__)
RAW = NEWS_DIR / "raw"
RAW.mkdir(parents=True, exist_ok=True)
COLUMNS = ["id", "origin", "source", "title", "url", "published_at", "available_at", "assets"]
PUBLISH_LAG = pd.Timedelta(minutes=5)


def _hid(*parts: str) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _clean_title(s: pd.Series) -> pd.Series:
    # Kaggle sarlavhalarida b'...' artefaktlari bor
    s = s.astype(str).str.replace(r"^b['\"]", "", regex=True).str.replace(r"['\"]$", "", regex=True)
    return s.str.strip()


def load_kaggle_djia(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW / "Combined_News_DJIA.csv"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    tops = [c for c in df.columns if c.startswith("Top")]
    long = df.melt(id_vars=["Date"], value_vars=tops, value_name="title").dropna(subset=["title"])
    long["title"] = _clean_title(long["title"])
    day = pd.to_datetime(long["Date"]).dt.tz_localize("UTC")
    out = pd.DataFrame({
        "origin": "kaggle_djia",
        "source": "reddit_worldnews",
        "title": long["title"].values,
        "url": "",
        "published_at": day.values,
        "available_at": (day + pd.Timedelta(days=1)).values,  # faqat sana -> ertasiga
        "assets": "",
    })
    out["id"] = [_hid("k", t, d) for t, d in zip(out.title, out.published_at)]
    return out[COLUMNS].drop_duplicates("id")


def load_fnspid(path: Path | None = None, min_year: int = 2012, chunksize: int = 500_000,
                max_per_day: int | None = None, keep_symbols: set[str] | None = None) -> pd.DataFrame:
    """FNSPID (Nasdaq yangiliklari, 1999–2023, ~15 mln sarlavha).

    Hammasini FinBERT'dan o'tkazish shart emas: loyihadagi aktivlarga tegishli sarlavhalar to'liq olinadi,
    qolganlaridan har kuni `max_per_day` ta tasodifiy (lekin takrorlanuvchan) namuna olinadi.
    Vaqti 00:00:00 bo'lgan yozuvlar faqat sana deb hisoblanadi -> ertasi kundan ishlatiladi (leakage'dan himoya).
    """
    import os
    path = path or RAW / "nasdaq_exteral_data.csv"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    max_per_day = max_per_day if max_per_day is not None else int(os.environ.get("ATLAS_FNSPID_PER_DAY", "80"))
    if keep_symbols is None:
        from ..config import ASSETS
        keep_symbols = {a.yf for a in ASSETS if a.group == "equity"}
    parts = []
    for ch in pd.read_csv(path, usecols=["Date", "Article_title", "Stock_symbol", "Url"],
                          chunksize=chunksize, on_bad_lines="skip"):
        ch = ch.dropna(subset=["Date", "Article_title"])
        ts = pd.to_datetime(ch["Date"].astype(str).str.replace(" UTC", "", regex=False), errors="coerce", utc=True)
        ch = ch.assign(ts=ts).dropna(subset=["ts"])
        ch = ch[ch.ts.dt.year >= min_year]
        sym = ch["Stock_symbol"].fillna("").astype(str).str.upper()
        mine = sym.isin(keep_symbols)
        # boshqa sarlavhalardan kunlik namuna (sarlavha xeshi bo'yicha — har safar bir xil natija)
        rest = ch[~mine]
        if max_per_day and len(rest):
            key = rest["Article_title"].astype(str).map(lambda t: int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
            rest = rest.assign(k=key).sort_values("k").groupby(rest.ts.dt.date, group_keys=False).head(max_per_day)
        ch = pd.concat([ch[mine], rest.drop(columns="k", errors="ignore")])
        date_only = (ch.ts.dt.hour == 0) & (ch.ts.dt.minute == 0) & (ch.ts.dt.second == 0)
        avail = ch["ts"] + PUBLISH_LAG
        avail = avail.where(~date_only, ch["ts"].dt.normalize() + pd.Timedelta(days=1))
        parts.append(pd.DataFrame({
            "origin": "fnspid",
            "source": "nasdaq",
            "title": ch["Article_title"].astype(str).str.strip(),
            "url": ch["Url"].fillna(""),
            "published_at": ch["ts"],
            "available_at": avail,
            "assets": ch["Stock_symbol"].fillna("").astype(str).str.upper().where(
                ch["Stock_symbol"].fillna("").astype(str).str.upper().isin(keep_symbols), ""),
        }))
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    # kun bo'yicha namuna chunklar orasida qayta cheklanadi
    if max_per_day:
        own = out["assets"] != ""
        rest = out[~own]
        rest = rest.groupby(rest.published_at.dt.date, group_keys=False).head(max_per_day)
        out = pd.concat([out[own], rest], ignore_index=True)
    out["id"] = [_hid("f", t, d) for t, d in zip(out.title, out.published_at)]
    log.info("FNSPID: %d sarlavha (%s .. %s)", len(out), out.published_at.min(), out.published_at.max())
    return out[COLUMNS].drop_duplicates("id")


def fetch_gdelt_doc(query: str = "(economy OR inflation OR oil OR wheat OR bitcoin OR stocks)",
                    start: str | None = None, end: str | None = None, max_records: int = 250) -> pd.DataFrame:
    """GDELT DOC 2.0 API (bepul, kalitsiz). Oxirgi ~3 oyni qamrab oladi.

    Tarixiy (2015+) GDELT uchun BigQuery'dagi `gdelt-bq.gdeltv2.gkg_partitioned` jadvalidan
    `GDELT_BQ_SQL` so'rovini ishlating (bepul limit: 1 TB/oy) va natijani
    data/news/raw/gdelt_bq.csv ga saqlang -> load_gdelt_bq() uni o'qiydi.
    """
    import httpx

    params = {"query": query + " sourcelang:english", "mode": "artlist", "format": "json",
              "maxrecords": max_records, "sort": "datedesc"}
    if start:
        params["startdatetime"] = pd.Timestamp(start).strftime("%Y%m%d%H%M%S")
    if end:
        params["enddatetime"] = pd.Timestamp(end).strftime("%Y%m%d%H%M%S")
    r = httpx.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=30)
    r.raise_for_status()
    arts = r.json().get("articles", [])
    if not arts:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(arts)
    ts = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", utc=True)
    out = pd.DataFrame({
        "origin": "gdelt", "source": df.get("domain", ""), "title": df["title"].astype(str),
        "url": df["url"], "published_at": ts, "available_at": ts + PUBLISH_LAG, "assets": "",
    })
    out["id"] = [_hid("g", u) for u in out.url]
    return out[COLUMNS].drop_duplicates("id")


GDELT_BQ_SQL = """
-- BigQuery: 2015+ iqtisodiy mavzudagi yangiliklar (kuniga eng ko'p 400 ta)
SELECT DATE, DocumentIdentifier AS url, SourceCommonName AS source,
       REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP('2015-03-01')
  AND REGEXP_CONTAINS(V2Themes, r'ECON_|WB_1104_MACROECONOMIC|FOOD_SECURITY|ENV_OIL|ECON_STOCKMARKET|ECON_CURRENCY')
  AND SourceCommonName IN ('reuters.com','bloomberg.com','cnbc.com','wsj.com','ft.com','marketwatch.com',
                           'coindesk.com','oilprice.com','fxstreet.com','investing.com','apnews.com')
QUALIFY ROW_NUMBER() OVER (PARTITION BY SUBSTR(CAST(DATE AS STRING),1,8) ORDER BY RAND()) <= 400
"""


def load_gdelt_bq(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW / "gdelt_bq.csv"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path).dropna(subset=["title"])
    ts = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d%H%M%S", utc=True)
    out = pd.DataFrame({"origin": "gdelt", "source": df["source"], "title": df["title"].astype(str),
                        "url": df["url"], "published_at": ts, "available_at": ts + PUBLISH_LAG, "assets": ""})
    out["id"] = [_hid("g", u) for u in out.url]
    return out[COLUMNS].drop_duplicates("id")


def load_collected() -> pd.DataFrame:
    from .collector import NewsStore
    return NewsStore().to_frame()


def load_gdelt_hist(paths=None) -> pd.DataFrame:
    """GDELT DOC API arxividan yig'ilgan tarixiy sarlavhalar (2024+ bo'shliqni to'ldirish uchun).
    Fayl: data/news/raw/gdelt_hist*.jsonl(.gz), har qatorda {"t": sarlavha, "s": seendate, "d": domen, "u": url}.
    available_at = GDELT birinchi ko'rgan vaqt + nashr kechikishi (kelajak ishlatilmaydi)."""
    files = paths or sorted(RAW.glob("gdelt_hist*.jsonl*"))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    ts = pd.to_datetime(df["s"], format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
    out = pd.DataFrame({"origin": "gdelt", "source": df["d"].astype(str), "title": df["t"].astype(str),
                        "url": df["u"].astype(str), "published_at": ts, "available_at": ts + PUBLISH_LAG, "assets": ""})
    out = out.dropna(subset=["available_at"])
    out["id"] = [_hid("g", u) for u in out.url]
    return out[COLUMNS].drop_duplicates("id")


def load_all_news() -> pd.DataFrame:
    """Barcha manbalarni birlashtiradi. Natija available_at bo'yicha tartiblangan."""
    frames = [load_kaggle_djia(), load_fnspid(), load_gdelt_bq(), load_gdelt_hist(), load_collected()]
    extra = NEWS_DIR / "extra.parquet"
    if extra.exists():
        frames.append(pd.read_parquet(extra))
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    for c in ("published_at", "available_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df = df.dropna(subset=["available_at", "title"])
    df = df[df.title.str.len() > 10].drop_duplicates("id")
    log.info("Jami yangilik: %d (%s)", len(df), df.origin.value_counts().to_dict())
    return df.sort_values("available_at").reset_index(drop=True)
