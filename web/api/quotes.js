// Jonli narxlar (Vercel funksiyasi). Brauzer Yahoo'ga to'g'ridan-to'g'ri ulana olmaydi (CORS),
// shuning uchun narxlar shu yerda yig'iladi va CDN'da ~20 soniya keshlanadi — foydalanuvchi soni
// qancha bo'lishidan qat'i nazar manbalarga daqiqasiga bir necha so'rov ketadi.
// Ro'yxat atlas/config.py dagi ASSETS bilan bir xil (tests/test_core.py tekshiradi).
export const ASSETS = [["EURUSD", "EURUSD=X", null], ["GBPUSD", "GBPUSD=X", null], ["USDJPY", "USDJPY=X", null], ["AUDUSD", "AUDUSD=X", null], ["USDCHF", "USDCHF=X", null], ["USDCAD", "USDCAD=X", null], ["DXY", "DX-Y.NYB", null], ["BTC", "BTC-USD", "btcusdt"], ["ETH", "ETH-USD", "ethusdt"], ["BNB", "BNB-USD", "bnbusdt"], ["XRP", "XRP-USD", "xrpusdt"], ["SOL", "SOL-USD", "solusdt"], ["SPX", "^GSPC", null], ["NDX", "^IXIC", null], ["DJI", "^DJI", null], ["AAPL", "AAPL", null], ["MSFT", "MSFT", null], ["NVDA", "NVDA", null], ["AMZN", "AMZN", null], ["TSLA", "TSLA", null], ["GOOGL", "GOOGL", null], ["META", "META", null], ["JPM", "JPM", null], ["XOM", "XOM", null], ["CL", "CL=F", null], ["NG", "NG=F", null], ["WHEAT", "ZW=F", null], ["CORN", "ZC=F", null], ["SOY", "ZS=F", null], ["COFFEE", "KC=F", null], ["SUGAR", "SB=F", null], ["COCOA", "CC=F", null], ["COTTON", "CT=F", null], ["GOLD", "GC=F", null], ["SILVER", "SI=F", null], ["COPPER", "HG=F", null], ["US10Y", "^TNX", null]];

const UA = { "User-Agent": "Mozilla/5.0 (ATLAS market dashboard)" };

async function binance() {
  const syms = ASSETS.filter((a) => a[2]);
  const q = encodeURIComponent(JSON.stringify(syms.map((a) => a[2].toUpperCase())));
  for (const host of ["https://api.binance.com", "https://data-api.binance.vision"]) {
    try {
      const r = await fetch(`${host}/api/v3/ticker/24hr?symbols=${q}`, { headers: UA, signal: AbortSignal.timeout(6000) });
      if (!r.ok) continue;
      const by = Object.fromEntries((await r.json()).map((d) => [d.symbol, d]));
      const now = new Date().toISOString();
      const out = {};
      for (const [sym, , b] of syms) {
        const d = by[b.toUpperCase()];
        if (d) out[sym] = { symbol: sym, price: +d.lastPrice, change_pct: +d.priceChangePercent, time: now, source: "Binance" };
      }
      return out;
    } catch { /* keyingi manba */ }
  }
  return {};
}

async function yahooOne(sym, ticker) {
  for (const host of ["query1", "query2"]) {
    try {
      const url = `https://${host}.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?interval=1m&range=1d&includePrePost=true`;
      const r = await fetch(url, { headers: UA, signal: AbortSignal.timeout(6000) });
      if (!r.ok) continue;
      const m = (await r.json())?.chart?.result?.[0]?.meta;
      if (!m || !m.regularMarketPrice) continue;
      const prev = m.chartPreviousClose ?? m.previousClose;
      return { symbol: sym, price: m.regularMarketPrice, change_pct: prev ? (m.regularMarketPrice / prev - 1) * 100 : null,
        time: new Date((m.regularMarketTime ?? Date.now() / 1000) * 1000).toISOString(), source: `Yahoo ${ticker}` };
    } catch { /* keyingi xost */ }
  }
  return null;
}

export default async function handler(req, res) {
  const [bin, ...yh] = await Promise.all([
    binance(),
    ...ASSETS.map(([s, t]) => yahooOne(s, t)),   // kripto uchun ham — Binance javob bermasa zaxira
  ]);
  const quotes = {};
  for (const q of yh) if (q) quotes[q.symbol] = q;
  Object.assign(quotes, bin);   // kripto uchun Binance tezroq va aniqroq
  const missing = ASSETS.map((a) => a[0]).filter((s) => !quotes[s]);
  res.setHeader("Cache-Control", "public, s-maxage=20, stale-while-revalidate=60");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.status(200).json({ quotes, error: missing.length ? `Narx olinmadi: ${missing.join(", ")}` : null });
}
