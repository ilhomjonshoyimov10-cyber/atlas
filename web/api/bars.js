// Kunlik barlar (oxirgi N kun) — zaxira ma'lumot yangilash yo'li uchun (GitHub Actions ishlamay qolsa).
// Javob ixcham: {symbol: [[sana, open, high, low, close, volume], ...]}; faqat yopilgan kunlar.
import { ASSETS } from "./quotes.js";

const UA = { "User-Agent": "Mozilla/5.0 (ATLAS market dashboard)" };

async function bars(ticker, range, interval = "1d") {
  for (const host of ["query1", "query2"]) {
    try {
      const url = `https://${host}.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?interval=${interval}&range=${range}&includePrePost=false`;
      const r = await fetch(url, { headers: UA, signal: AbortSignal.timeout(8000) });
      if (!r.ok) continue;
      const res = (await r.json())?.chart?.result?.[0];
      const q = res?.indicators?.quote?.[0];
      if (!res?.timestamp || !q) continue;
      const off = interval === "1d" ? (res.meta?.gmtoffset ?? 0) : 0;
      const fmt = (t) => interval === "1d" ? new Date((t + off) * 1000).toISOString().slice(0, 10) : t;   // ichki kun: UNIX soniya (UTC)
      return res.timestamp.map((t, i) => [fmt(t),
        q.open[i], q.high[i], q.low[i], q.close[i], q.volume?.[i] ?? 0]).filter((b) => b[4] != null);
    } catch { /* keyingi xost */ }
  }
  return null;
}

const INTERVALS = { "5m": "60d", "15m": "60d", "30m": "60d", "60m": "730d", "1h": "730d", "1d": "3mo" };

export default async function handler(req, res) {
  // Bitta aktiv, ichki kun barlari: /api/bars?ticker=GC=F&interval=5m&range=60d
  const iv = req.query?.interval;
  if (req.query?.ticker && INTERVALS[iv]) {
    const rng = /^\d+(d|mo|y)$/.test(req.query.range ?? "") ? req.query.range : INTERVALS[iv];
    const b = await bars(String(req.query.ticker), rng, iv === "1h" ? "60m" : iv);
    res.setHeader("Cache-Control", "public, s-maxage=60, stale-while-revalidate=120");
    res.setHeader("Access-Control-Allow-Origin", "*");
    return res.status(b ? 200 : 502).json({ ticker: req.query.ticker, interval: iv, range: rng, bars: b });
  }
  const range = ["5d", "1mo", "3mo"].includes(req.query?.range) ? req.query.range : "1mo";
  const out = {};
  await Promise.all(ASSETS.map(async ([s, t]) => { out[s] = await bars(t, range); }));
  res.setHeader("Cache-Control", "public, s-maxage=600, stale-while-revalidate=1200");
  res.status(200).json({ range, fetched_at: new Date().toISOString(), bars: out });
}
