// Kunlik barlar (oxirgi N kun) — zaxira ma'lumot yangilash yo'li uchun (GitHub Actions ishlamay qolsa).
// Javob ixcham: {symbol: [[sana, open, high, low, close, volume], ...]}; faqat yopilgan kunlar.
import { ASSETS } from "./quotes.js";

const UA = { "User-Agent": "Mozilla/5.0 (ATLAS market dashboard)" };

async function bars(ticker, range) {
  for (const host of ["query1", "query2"]) {
    try {
      const url = `https://${host}.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?interval=1d&range=${range}`;
      const r = await fetch(url, { headers: UA, signal: AbortSignal.timeout(8000) });
      if (!r.ok) continue;
      const res = (await r.json())?.chart?.result?.[0];
      const q = res?.indicators?.quote?.[0];
      if (!res?.timestamp || !q) continue;
      const off = res.meta?.gmtoffset ?? 0;
      return res.timestamp.map((t, i) => [new Date((t + off) * 1000).toISOString().slice(0, 10),
        q.open[i], q.high[i], q.low[i], q.close[i], q.volume?.[i] ?? 0]).filter((b) => b[4] != null);
    } catch { /* keyingi xost */ }
  }
  return null;
}

export default async function handler(req, res) {
  const range = ["5d", "1mo", "3mo"].includes(req.query?.range) ? req.query.range : "1mo";
  const out = {};
  await Promise.all(ASSETS.map(async ([s, t]) => { out[s] = await bars(t, range); }));
  res.setHeader("Cache-Control", "public, s-maxage=600, stale-while-revalidate=1200");
  res.status(200).json({ range, fetched_at: new Date().toISOString(), bars: out });
}
