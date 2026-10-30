"""Real-time yangilik kollektori.

Serverda doimiy ishlaydi: RSS manbalarini har N soniyada so'raydi va har bir yangi xabarga
`first_seen` (biz uni birinchi marta ko'rgan UTC vaqt) yozadi. O'qitish uchun aynan shu vaqt
`available_at` bo'ladi — sayt keyinchalik `published` ni o'zgartirsa ham leakage bo'lmaydi.

Ishga tushirish:  python -m atlas.cli collect --interval 60
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from ..config import NEWS_DB, NEWS_FEEDS
from ..features.tagging import tag_assets, detect_event

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id TEXT PRIMARY KEY,
    origin TEXT, source TEXT, title TEXT, summary TEXT, url TEXT,
    published_at TEXT, first_seen TEXT NOT NULL,
    assets TEXT, event TEXT, impact TEXT
);
CREATE INDEX IF NOT EXISTS ix_news_seen ON news(first_seen);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NewsStore:
    def __init__(self, path=NEWS_DB):
        self.path = str(path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(news)")}
            if "impact" not in cols:   # eski bazani yangilash
                c.execute("ALTER TABLE news ADD COLUMN impact TEXT")

    def set_impact(self, news_id: str, impact_json: str):
        with self._lock, self._conn() as c:
            c.execute("UPDATE news SET impact=? WHERE id=?", (impact_json, news_id))

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def insert(self, items: list[dict]) -> list[dict]:
        """Faqat YANGI xabarlarni qaytaradi (dedup id bo'yicha)."""
        new = []
        with self._lock, self._conn() as c:
            for it in items:
                cur = c.execute(
                    "INSERT OR IGNORE INTO news(id,origin,source,title,summary,url,published_at,first_seen,assets,event)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (it["id"], it.get("origin", "rss"), it["source"], it["title"], it.get("summary", ""),
                     it.get("url", ""), it.get("published_at"), it["first_seen"], it.get("assets", ""),
                     it.get("event", "none")))
                if cur.rowcount:
                    new.append(it)
        return new

    def recent(self, limit: int = 100, asset: str | None = None) -> list[dict]:
        q = "SELECT * FROM news"
        args: list = []
        if asset:
            q += " WHERE (',' || assets || ',') LIKE ?"
            args.append(f"%,{asset},%")
        q += " ORDER BY first_seen DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args)]

    def purge_demo(self) -> int:
        """Sintetik demo xabarlarini o'chiradi (haqiqiy rejimga o'tganda)."""
        with self._lock, self._conn() as c:
            return c.execute("DELETE FROM news WHERE origin='synthetic' OR source='demo'").rowcount

    def to_frame(self) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql_query("SELECT id,origin,source,title,url,published_at,first_seen,assets FROM news "
                                   "WHERE origin != 'synthetic'", c)
        if df.empty:
            return pd.DataFrame(columns=["id", "origin", "source", "title", "url", "published_at", "available_at", "assets"])
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df["available_at"] = pd.to_datetime(df["first_seen"], utc=True)  # <- leakage qoidasi
        return df.drop(columns=["first_seen"])


def _parse_feed(name: str, content: bytes, seen_at: str) -> list[dict]:
    import feedparser

    feed = feedparser.parse(content)
    out = []
    for e in feed.entries:
        title = (e.get("title") or "").strip()
        if len(title) < 8:
            continue
        link = e.get("link", "")
        pub = None
        for k in ("published_parsed", "updated_parsed"):
            if e.get(k):
                pub = datetime(*e[k][:6], tzinfo=timezone.utc).isoformat()
                break
        summary = (e.get("summary") or "")[:500]
        text = f"{title}. {summary}"
        out.append({
            "id": hashlib.sha1((link or title).encode()).hexdigest()[:16],
            "origin": "rss", "source": name, "title": title, "summary": summary, "url": link,
            "published_at": pub, "first_seen": seen_at,
            "assets": ",".join(tag_assets(text)), "event": detect_event(text),
        })
    return out


class Collector:
    def __init__(self, store: NewsStore | None = None, feeds: dict[str, str] | None = None,
                 interval: int = 60, on_new: Callable[[list[dict]], None] | None = None):
        self.store = store or NewsStore()
        self.feeds = feeds or NEWS_FEEDS
        self.interval = interval
        self.on_new = on_new
        self._etags: dict[str, str] = {}
        self.stats: dict[str, dict] = {}

    async def _poll_one(self, client, name: str, url: str) -> list[dict]:
        headers = {"User-Agent": "Mozilla/5.0 (Bashorat news collector)"}
        if name in self._etags:
            headers["If-None-Match"] = self._etags[name]
        try:
            r = await client.get(url, headers=headers, timeout=20, follow_redirects=True)
            if r.status_code == 304:
                return []
            r.raise_for_status()
            if "etag" in r.headers:
                self._etags[name] = r.headers["etag"]
            items = _parse_feed(name, r.content, _now())
            new = self.store.insert(items)
            self.stats[name] = {"ok": True, "last": _now(), "new": len(new)}
            return new
        except Exception as e:
            self.stats[name] = {"ok": False, "last": _now(), "error": str(e)[:200]}
            return []

    async def poll_once(self) -> list[dict]:
        import httpx

        async with httpx.AsyncClient() as client:
            res = await asyncio.gather(*(self._poll_one(client, n, u) for n, u in self.feeds.items()))
        new = [x for r in res for x in r]
        if new:
            log.info("Yangi xabarlar: %d", len(new))
            if self.on_new:
                self.on_new(new)
        return new

    async def run_forever(self):
        while True:
            await self.poll_once()
            await asyncio.sleep(self.interval)
