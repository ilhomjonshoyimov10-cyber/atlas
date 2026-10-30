export type Group = "forex" | "crypto" | "index" | "equity" | "energy" | "agri" | "metal" | "rates";

export interface AssetRow {
  symbol: string; name: string; group: Group; binance: string | null;
  last: number; change_pct: number; date: string; spark: number[];
}
export interface HorizonFc {
  h: number; q05: number; q50: number; q95: number;
  ret_q05: number; ret_q50: number; ret_q95: number; p_up: number;
}
export interface Forecast {
  symbol: string; as_of: string; last: number; horizons: HorizonFc[]; news_attention?: number[];
  market_attention?: { symbol: string; name: string; weight: number }[];
  branch_weights?: Record<string, number>;
}
export interface Bar { t: string; open?: number; high?: number; low?: number; close: number; volume?: number }
export interface SeasonProfile {
  symbol: string; by: string;
  profile: { [k: string]: number }[];
  by_year: { years: number[]; keys: number[]; values: (number | null)[][] };
}
export interface NewsItem {
  id: string; source: string; title: string; summary?: string; url: string;
  published_at: string | null; first_seen: string; assets: string; event: string;
  impact?: string | null;   // JSON: QuickImpact[]
}
export interface QuickImpact { symbol: string; name: string; tagged: boolean; ret_h1: number; ret_h5: number; sigma_h1: number }
export interface Impact {
  symbol: string; name: string; group: Group; tagged: boolean; [k: string]: number | string | boolean;
}
export interface Analysis {
  text: string; assets: string[]; event: string; tone?: number; impact: Impact[]; note?: string;
  common?: Record<string, number>;
}
export interface Health { ready: boolean; error: string | null; news_model: boolean; demo?: boolean; mode?: string; updated_at?: string; data_until?: string; collector: Record<string, { ok: boolean; new?: number; error?: string }> }

/** Statik (bulut) rejim: ma'lumot GitHub'dagi `data` shoxchasidan JSON fayllar sifatida o'qiladi.
 *  Lokal rejimda (VITE_DATA_URL yo'q) hammasi FastAPI serveridan keladi. */
export const DATA_URL: string = (import.meta.env.VITE_DATA_URL ?? "").replace(/\/$/, "");
export const STATIC = DATA_URL !== "";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  const ct = r.headers.get("content-type") ?? "";
  if (r.ok && !ct.includes("json")) {
    throw new Error("Server eski versiyada ishlayapti — uni to'xtatib, qayta ishga tushiring (python -m atlas.cli serve)");
  }
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* matn emas */ }
    throw new Error(msg);
  }
  return r.json();
}

// raw.githubusercontent ~5 daqiqa keshlaydi; daqiqalik kalit eng yangi faylni olishga yordam beradi
async function getData<T>(rel: string): Promise<T> {
  const r = await fetch(`${DATA_URL}/${rel}?v=${Math.floor(Date.now() / 120000)}`);
  if (!r.ok) throw new Error(r.status === 404 ? "Ma'lumot hali tayyorlanmagan" : `${r.status}`);
  return JSON.parse(await r.text());
}

const cache = new Map<string, Promise<any>>();
function once<T>(rel: string, ttlMs = 120000): Promise<T> {
  const key = `${rel}@${Math.floor(Date.now() / ttlMs)}`;
  if (!cache.has(key)) cache.set(key, getData<T>(rel).catch((e) => { cache.delete(key); throw e; }));
  return cache.get(key)!;
}

export interface TrackRow {
  symbol: string; as_of: string; target: string; h: number; last: number;
  q05: number; q50: number; q95: number; real: number; inside: boolean; dir_ok: boolean | null; err_pct: number;
}
export interface Track {
  summary: { h: number; n: number; coverage90: number; direction: number | null; mae_pct: number }[];
  recent: TrackRow[]; pending: number; started: string | null;
}
export interface TALevel { price: number; type: "support" | "resistance"; touches: number; strength: number; last: string; dist_pct: number; dist_atr: number }
export interface TAZone { kind: "demand" | "supply"; lo: number; hi: number; born: string; from: string; tests: number; impulse_atr: number; dist_atr: number }
export interface TASetup {
  setup: string; label: string; family: string; dir: number; date: string; level: number | null;
  p_model: number | null; base_win: number | null; n_hist: number | null; hist_excess_R: number | null; hist_p_holm: number | null;
  holdout_win: number | null; strong: boolean; entry: number; tp: number; sl: number;
}
export interface TA {
  symbol: string; name: string; as_of: string; close: number; atr: number;
  structure: { trend: string; last_high: number | null; last_low: number | null; bos_up?: boolean; bos_down?: boolean };
  levels: TALevel[]; zones: TAZone[]; setups: TASetup[];
  patterns: { name: string; label: string; dir: number; neck: number; target: number; date: string }[];
  indicators: { rsi: number; macd_hist: number; sma50: number; sma200: number | null; bb_pctb: number | null; candle: string | null };
  factors: { name: string; score: number; weight: number; note: string }[];
  score: number; bias: string; confidence: string; text: string[]; disclaimer: string;
}
export interface StratRow {
  setup: string; name: string; family: string; dir: number; n: number; win_rate?: number; avg_R?: number;
  base_R?: number; excess_R?: number; p_holm?: number;
}
export interface Strategies {
  ready: boolean; holdout_start: string; holdout_end: string; n_train: number; n_holdout: number;
  auc: number; auc_baseline: number; brier_skill: number; holdout_win_rate: number;
  filter: Record<"all" | "top30" | "bottom30", { n: number; avg_R: number; excess_R: number; win_rate: number }>;
  stats_train: StratRow[]; stats_holdout: StratRow[];
  per_setup_holdout: { setup: string; n: number; win_all: number; n_top: number; win_top: number | null; auc: number }[];
  dataset: { n_events: number; n_labeled: number; assets: number; from: string; to: string };
  rules: { tp_atr: number; sl_atr: number; horizon: number };
  today?: { symbol: string; name: string; setup: string; label: string; dir: number; p_model: number | null; base_win: number | null; strong: boolean; entry: number; tp: number; sl: number }[];
}
export type Quotes = { quotes: Record<string, { symbol: string; price: number; change_pct: number | null; time: string; source: string }>; error: string | null };

export const api = {
  health: () => STATIC ? once<Health>("health.json", 60000) : get<Health>("/api/health"),
  quotes: () => get<Quotes>("/api/quotes"),
  assets: () => STATIC ? once<AssetRow[]>("assets.json") : get<AssetRow[]>("/api/assets"),
  history: async (s: string, years = 10): Promise<{ rows: Bar[] }> => {
    if (!STATIC) return get<{ rows: Bar[] }>(`/api/history/${s}?years=${years}`);
    const d = await once<{ t: string[]; c: number[] }>(`history/${s}.json`, 600000);
    return { rows: d.t.map((t, i) => ({ t, close: d.c[i] })) };
  },
  forecast: (s: string) => STATIC ? once<Forecast>(`forecast/${s}.json`) : get<Forecast>(`/api/forecast/${s}`),
  forecastAll: () => STATIC ? once<Record<string, Forecast>>("forecast.json") : get<Record<string, Forecast>>("/api/forecast"),
  seasonality: (s: string, by = "month") => STATIC ? once<SeasonProfile>(`seasonality/${s}_${by}.json`, 3600000)
    : get<SeasonProfile>(`/api/seasonality/${s}?by=${by}`),
  news: async (asset?: string): Promise<NewsItem[]> => {
    if (!STATIC) return get<NewsItem[]>(`/api/news?limit=80${asset ? `&asset=${asset}` : ""}`);
    const all = await once<NewsItem[]>("news.json", 60000);
    return (asset ? all.filter((n) => (n.assets || "").split(",").includes(asset)) : all).slice(0, 80);
  },
  report: () => STATIC ? once<any>("report.json", 3600000) : get<any>("/api/report"),
  track: () => STATIC ? once<Track>("track.json") : get<Track>("/api/track"),
  backtest: (thr: number, cost: number) => STATIC
    ? once<any>(`backtest/t${thr.toFixed(2)}_c${cost}.json`, 3600000)
    : get<any>(`/api/backtest?threshold=${thr}&cost_bps=${cost}`),
  ta: (s: string) => STATIC ? once<TA>(`ta/${s}.json`) : get<TA>(`/api/ta/${s}`),
  strategies: () => STATIC ? once<Strategies>("strategies.json") : get<Strategies>("/api/strategies"),
  /** Statik rejimda faqat lentadagi xabarlar uchun oldindan hisoblangan tahlil mavjud. */
  analyzeNews: (id: string) => once<Analysis>(`analysis/${id}.json`, 3600000),
  analyze: async (text: string): Promise<Analysis> => {
    const r = await fetch("/api/news/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }),
    });
    if (!r.ok) throw new Error((await r.json()).detail ?? r.status);
    return r.json();
  },
};

export const GROUP_NAMES: Record<Group, string> = {
  forex: "Valyuta", crypto: "Kripto", index: "Indekslar", equity: "Aksiyalar",
  energy: "Energiya", agri: "Qishloq xo'jaligi", metal: "Metallar", rates: "Obligatsiyalar",
};

export const EVENT_NAMES: Record<string, string> = {
  none: "Umumiy", fomc: "FOMC / Fed", cpi: "Inflyatsiya (CPI)", nfp: "Ish o'rinlari (NFP)", gdp: "YaIM",
  ecb: "ECB", earnings: "Hisobot", opec: "OPEC", usda: "USDA / hosil", geopolitics: "Geosiyosat",
  crypto_reg: "Kripto regulyatsiya",
};

export function fmtPrice(v: number): string {
  if (!isFinite(v)) return "—";
  const a = Math.abs(v);
  const d = a >= 1000 ? 2 : a >= 10 ? 2 : a >= 1 ? 4 : 5;
  return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }).replace(/,/g, " ");
}
export function fmtPct(v: number, digits = 2): string {
  if (!isFinite(v)) return "—";
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(digits)}%`;
}
export function timeAgo(iso: string): string {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "hozirgina";
  if (s < 3600) return `${Math.floor(s / 60)} daq oldin`;
  if (s < 86400) return `${Math.floor(s / 3600)} soat oldin`;
  return `${Math.floor(s / 86400)} kun oldin`;
}
