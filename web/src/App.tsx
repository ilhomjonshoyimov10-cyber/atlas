import { useEffect, useMemo, useRef, useState } from "react";
import { api, AssetRow, Bar, Forecast, fmtPct, fmtPrice, Group, GROUP_NAMES, Health, NewsItem, STATIC, TA, timeAgo } from "./api";
import PriceChart from "./components/PriceChart";
import Seasonality from "./components/Seasonality";
import NewsFeed from "./components/NewsFeed";
import WhatIf from "./components/WhatIf";
import ModelReport from "./components/ModelReport";
import MarketMap from "./components/MarketMap";
import Backtest from "./components/Backtest";
import TrackRecord from "./components/TrackRecord";
import TechAnalysis from "./components/TechAnalysis";
import Strategies from "./components/Strategies";

type Page = "asset" | "map" | "strategies" | "backtest" | "track";

type Tab = "ta" | "season" | "whatif" | "model";
const RANGES = [{ y: 0.5, l: "6O" }, { y: 1, l: "1Y" }, { y: 3, l: "3Y" }, { y: 5, l: "5Y" }, { y: 11, l: "Hammasi" }];
const ORDER: Group[] = ["crypto", "forex", "index", "equity", "energy", "agri", "metal", "rates"];

function Spark({ v, up }: { v: number[]; up: boolean }) {
  if (v.length < 2) return null;
  const lo = Math.min(...v), hi = Math.max(...v);
  const d = v.map((x, i) => `${i ? "L" : "M"}${(i / (v.length - 1)) * 60},${18 - ((x - lo) / (hi - lo || 1)) * 16}`).join("");
  return <svg viewBox="0 0 60 20" className={`spark ${up ? "up" : "down"}`} aria-hidden><path d={d} /></svg>;
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [assets, setAssets] = useState<AssetRow[]>([]);
  const [fcAll, setFcAll] = useState<Record<string, Forecast>>({});
  const [sel, setSel] = useState<string>(() => { try { return localStorage.getItem("sel") || "BTC"; } catch { return "BTC"; } });
  const [bars, setBars] = useState<Bar[]>([]);
  const [fc, setFc] = useState<Forecast | null>(null);
  const [years, setYears] = useState(1);
  const [tab, setTab] = useState<Tab>("ta");
  const [ta, setTa] = useState<TA | null>(null);
  const [page, setPage] = useState<Page>("asset");
  const [news, setNews] = useState<NewsItem[]>([]);
  const [fresh, setFresh] = useState<Set<string>>(new Set());
  const [onlySel, setOnlySel] = useState(false);
  const [q, setQ] = useState("");
  const [live, setLive] = useState<Record<string, number>>({});
  const [liveChg, setLiveChg] = useState<Record<string, number>>({});
  const [liveSrc, setLiveSrc] = useState<Record<string, string>>({});
  const [seed, setSeed] = useState<{ id?: string; text: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const tabsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api.assets().then(setAssets).catch((e) => setErr(e.message));
    api.forecastAll().then(setFcAll).catch(() => {});
    const t = setInterval(() => {
      api.health().then(setHealth).catch(() => {});
      if (STATIC) {   // bulut rejimida ma'lumot har ~30 daqiqada yangilanadi
        api.assets().then(setAssets).catch(() => {});
        api.forecastAll().then(setFcAll).catch(() => {});
      }
    }, STATIC ? 300000 : 30000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    try { localStorage.setItem("sel", sel); } catch { /* storage yo'q */ }
    setBars([]); setFc(null); setTa(null);
    api.ta(sel).then(setTa).catch(() => setTa(null));
    api.history(sel, 11).then((d) => setBars(d.rows)).catch((e) => setErr(e.message));
    api.forecast(sel).then(setFc).catch(() => setFc(null));
  }, [sel]);

  useEffect(() => {
    const load = () => api.news(onlySel ? sel : undefined).then(setNews).catch(() => {});
    load();
    const t = setInterval(load, 60000);   // fon tahlili tugagan ta'sirlarni ham olib keladi
    return () => clearInterval(t);
  }, [onlySel, sel]);

  // Yangiliklar oqimi (server WebSocket)
  useEffect(() => {
    if (STATIC) return;   // statik saytda server yo'q — lenta davriy yangilanadi
    let ws: WebSocket | null = null; let stop = false;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data);
        if (m.type === "news") {
          const items: NewsItem[] = m.items;
          setNews((old) => [...items, ...old].slice(0, 150));
          setFresh(new Set(items.map((i) => i.id)));
          api.forecastAll().then(setFcAll).catch(() => {});
        }
      };
      ws.onclose = () => { if (!stop) setTimeout(connect, 5000); };
    };
    connect();
    return () => { stop = true; ws?.close(); };
  }, []);

  // Kripto narxlari real vaqtda (Binance ochiq oqimi)
  useEffect(() => {
    const syms = assets.filter((a) => a.binance).map((a) => a.binance!);
    if (!syms.length) return;
    const map = Object.fromEntries(assets.filter((a) => a.binance).map((a) => [a.binance!.toUpperCase(), a.symbol]));
    const ws = new WebSocket(`wss://stream.binance.com:9443/stream?streams=${syms.map((s) => `${s}@miniTicker`).join("/")}`);
    let buf: Record<string, number> = {};
    let bufC: Record<string, number> = {};
    ws.onmessage = (e) => {
      const d = JSON.parse(e.data).data;
      const s = map[d.s];
      if (s) { buf[s] = parseFloat(d.c); bufC[s] = (parseFloat(d.c) / parseFloat(d.o) - 1) * 100; }
    };
    const t = setInterval(() => {
      if (Object.keys(buf).length) {
        setLive((o) => ({ ...o, ...buf })); setLiveChg((o) => ({ ...o, ...bufC }));
        setLiveSrc((o) => ({ ...o, ...Object.fromEntries(Object.keys(buf).map((k) => [k, "Binance, real vaqt"])) }));
        buf = {}; bufC = {};
      }
    }, 1000);
    return () => { ws.close(); clearInterval(t); };
  }, [assets]);

  // Qolgan barcha aktivlar (forex, oltin, neft, aksiyalar...) — server jonli narxlari
  useEffect(() => {
    const load = () => api.quotes().then((r) => {
      const p: Record<string, number> = {}, c: Record<string, number> = {}, src: Record<string, string> = {};
      Object.values(r.quotes).forEach((q) => {
        p[q.symbol] = q.price; if (q.change_pct !== null) c[q.symbol] = q.change_pct;
        src[q.symbol] = `${q.source}, ${new Date(q.time).toLocaleTimeString()}`;
      });
      // Binance oqimi tezroq — kripto uchun uni ustun qo'yamiz
      setLive((o) => ({ ...p, ...Object.fromEntries(Object.entries(o).filter(([k]) => assets.find((a) => a.symbol === k)?.binance)) }));
      setLiveChg((o) => ({ ...c, ...Object.fromEntries(Object.entries(o).filter(([k]) => assets.find((a) => a.symbol === k)?.binance)) }));
      setLiveSrc((o) => ({ ...src, ...Object.fromEntries(Object.entries(o).filter(([k]) => assets.find((a) => a.symbol === k)?.binance)) }));
    }).catch(() => {});
    load();
    const t = setInterval(load, 20000);
    return () => clearInterval(t);
  }, [assets]);

  const cur = assets.find((a) => a.symbol === sel);
  const grouped = useMemo(() => {
    const f = q.trim().toLowerCase();
    const list = assets.filter((a) => !f || a.symbol.toLowerCase().includes(f) || a.name.toLowerCase().includes(f));
    return ORDER.map((g) => [g, list.filter((a) => a.group === g)] as const).filter(([, l]) => l.length);
  }, [assets, q]);

  const price = live[sel] ?? cur?.last ?? 0;
  const chg = cur ? (liveChg[sel] ?? cur.change_pct) : 0;

  const pick = (s: string) => { setSel(s); setPage("asset"); window.scrollTo({ top: 0 }); };

  const analyzeAsset = (s: string) => {
    setSel(s); setPage("asset"); setTab("ta");
    setTimeout(() => tabsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  };

  const analyze = (n: NewsItem) => {
    setPage("asset");
    setSeed({ id: n.id, text: n.title }); setTab("whatif");
    tabsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <svg viewBox="0 0 32 32" aria-hidden><path d="M3 22 L12 14 L18 18 L29 6" /><path d="M18 18 L29 12 L29 26 Z" className="fanmark" /></svg>
          <span>ATLAS</span>
        </div>
        <nav className="pages" aria-label="Bo'limlar">
          {([["asset", "Aktiv"], ["map", "Bozor xaritasi"], ["strategies", "Strategiyalar"], ["track", "Prognoz jurnali"], ["backtest", "Backtest"]] as [Page, string][]).map(([k, l]) => (
            <button key={k} className={page === k ? "on" : ""} aria-current={page === k ? "page" : undefined} onClick={() => setPage(k)}>{l}</button>
          ))}
        </nav>
        <div className="status">
          {health?.ready ? (
            <>
              <span className="dot ok" />Model ishlayapti{health.news_model ? ", yangiliklar ulangan" : " (yangiliksiz)"}
              <span className="sep" />
              {STATIC && health.updated_at && <><span title={new Date(health.updated_at).toLocaleString()}>Yangilandi {timeAgo(health.updated_at)}</span><span className="sep" /></>}
              {Object.keys(health.collector).length
                ? `${Object.values(health.collector).filter((s) => s.ok).length}/${Object.keys(health.collector).length} manba faol`
                : "Kollektor o'chiq"}
            </>
          ) : (
            <><span className="dot warn" />{health?.error ?? (STATIC ? "Ma'lumot yuklanmoqda…" : "Server bilan aloqa yo'q")}</>
          )}
        </div>
      </header>

      {health?.demo && (
        <div className="banner demo" role="status">Demo rejim: grafik tarixi sintetik, haqiqiy bozor emas.
          Chap ro'yxat va sarlavhadagi jonli narxlar haqiqiy (Binance, Yahoo), lekin grafik va prognoz hali demo modelda. Haqiqiy ma'lumotga o'tish uchun
          bitta buyruq: <code>python -m atlas.cli real</code>, so'ng serverni qayta ishga tushiring.</div>
      )}
      {err && <div className="banner" role="alert">{err} — {STATIC ? "sahifani birozdan keyin yangilang." : "server ishga tushganini va ma'lumot yuklanganini tekshiring."}</div>}

      <div className="layout">
        <nav className="rail" aria-label="Aktivlar">
          <label className="search">
            <span className="sr-only">Aktiv qidirish</span>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Qidirish: bug'doy, BTC…" />
          </label>
          {grouped.map(([g, list]) => (
            <section key={g}>
              <h2>{GROUP_NAMES[g]}</h2>
              <ul>
                {list.map((a) => {
                  const f5 = fcAll[a.symbol]?.horizons.find((h) => h.h === 5);
                  const p = live[a.symbol] ?? a.last;
                  return (
                    <li key={a.symbol}>
                      <button className={`row ${a.symbol === sel && page === "asset" ? "active" : ""}`} onClick={() => pick(a.symbol)} aria-current={a.symbol === sel}>
                        <span className="nm"><strong>{a.name}</strong><small>{a.symbol}</small></span>
                        <Spark v={a.spark} up={a.change_pct >= 0} />
                        <span className="px"><span>{fmtPrice(p)}</span>
                          {f5 && <small className={f5.ret_q50 >= 0 ? "pos" : "neg"} title="5 savdo kunlik median prognoz">5k {fmtPct(f5.ret_q50, 1)}</small>}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
        </nav>

        <main className="main">
          {page === "map" ? (
            <MarketMap assets={assets} fc={fcAll} onPick={pick} />
          ) : page === "strategies" ? (
            <Strategies assets={assets} onAnalyze={analyzeAsset} />
          ) : page === "track" ? (
            <TrackRecord onPick={pick} />
          ) : page === "backtest" ? (
            <Backtest />
          ) : cur ? (
            <>
              <div className="asset-head">
                <div>
                  <h1>{cur.name}</h1>
                  <p className="sub"><span>{GROUP_NAMES[cur.group]}</span><span>{cur.symbol}</span>
                    <span>Oxirgi kunlik yopilish {cur.date}</span>{live[sel] && <span className="live" title={liveSrc[sel]}>● Jonli narx: {liveSrc[sel]}</span>}</p>
                </div>
                <button className="btn analyze-btn" onClick={() => analyzeAsset(sel)}>Analiz qil</button>
                <div className="bigprice">
                  <span className="num">{fmtPrice(price)}</span>
                  <span className={chg >= 0 ? "pos" : "neg"}>{fmtPct(chg)}</span>
                </div>
              </div>
              <div className="ranges" role="group" aria-label="Davr">
                {RANGES.map((r) => (
                  <button key={r.l} className={years === r.y ? "on" : ""} onClick={() => setYears(r.y)}>{r.l}</button>
                ))}
              </div>
              {bars.length ? <PriceChart bars={bars} forecast={fc} years={years} ta={tab === "ta" ? ta : null} /> : <div className="skeleton" style={{ height: 380 }} />}

              {fc && (
                <div className="horizons">
                  {fc.horizons.map((h) => (
                    <div key={h.h} className="hz">
                      <h3>{h.h === 1 ? "Ertaga" : `${h.h} savdo kunida`}</h3>
                      <p className={`hz-mid ${h.ret_q50 >= 0 ? "pos" : "neg"}`}>{fmtPct(h.ret_q50)}</p>
                      <p className="hz-range">90% oraliq: {fmtPrice(h.q05)} – {fmtPrice(h.q95)}</p>
                      <div className="pbar" aria-label={`O'sish ehtimoli ${Math.round(h.p_up * 100)}%`}>
                        <span style={{ width: `${h.p_up * 100}%` }} />
                      </div>
                      <p className="hz-p">O'sish ehtimoli {Math.round(h.p_up * 100)}%</p>
                    </div>
                  ))}
                </div>
              )}
              {fc?.news_attention && (
                <div className="attn">
                  <span>Model qaysi kun yangiliklariga qaradi:</span>
                  <div className="attn-bars">
                    {fc.news_attention.map((w, i, arr) => (
                      <div key={i} title={`${arr.length - 1 - i === 0 ? "bugun" : `${arr.length - 1 - i} kun oldin`}: ${(w * 100).toFixed(0)}%`}>
                        <span style={{ height: `${Math.max(6, (w / Math.max(...arr, 1e-6)) * 80)}%` }} />
                        <small>{arr.length - 1 - i === 0 ? "bugun" : `−${arr.length - 1 - i}`}</small>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {(fc?.market_attention?.length || fc?.branch_weights) && (
                <div className="explain">
                  {fc?.branch_weights && (
                    <div className="branches">
                      <span>Bashorat manbalari:</span>
                      <div className="branch-bar" role="img" aria-label="Model shoxlari ulushi">
                        {Object.entries(fc.branch_weights).map(([k, v]) => (
                          <span key={k} className={`br br-${k}`} style={{ flexGrow: v }} title={`${k}: ${(v * 100).toFixed(0)}%`}>{k} {(v * 100).toFixed(0)}%</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {fc?.market_attention && fc.market_attention.length > 0 && fc.market_attention[0].weight > 0.06 && (
                    <p className="mkt-att">Model ko'proq qaragan bozorlar: {fc.market_attention.slice(0, 4).map((m, i) => (
                      <span key={m.symbol}>{i ? ", " : ""}<button className="asset-link" onClick={() => pick(m.symbol)}>{m.name}</button> {(m.weight * 100).toFixed(0)}%</span>
                    ))}</p>
                  )}
                </div>
              )}

              <div className="tabs" ref={tabsRef} role="tablist">
                {([["ta", "Texnik tahlil"], ["season", "Mavsumiylik"], ["whatif", "Yangilik ta'siri"], ["model", "Model sifati"]] as [Tab, string][]).map(([k, l]) => (
                  <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>
                ))}
              </div>
              <div className="tab-body" role="tabpanel">
                {tab === "ta" && <TechAnalysis ta={ta} onPickLevel={() => window.scrollTo({ top: 0, behavior: "smooth" })} />}
                {tab === "season" && <Seasonality symbol={sel} name={cur.name} />}
                {tab === "whatif" && <WhatIf seed={seed} news={news} onPick={pick} />}
                {tab === "model" && <ModelReport />}
              </div>
              <p className="disclaimer">Bashoratlar ehtimoliy oraliqlardir va investitsiya tavsiyasi emas.</p>
            </>
          ) : (
            <div className="skeleton" style={{ height: 500 }} />
          )}
        </main>

        <aside className="newscol" aria-label="Yangiliklar">
          <div className="news-head">
            <h2>Yangiliklar</h2>
            <label className="toggle">
              <input type="checkbox" checked={onlySel} onChange={(e) => setOnlySel(e.target.checked)} />
              Faqat {cur?.name ?? sel}
            </label>
          </div>
          <NewsFeed items={news} fresh={fresh} onPick={pick} onAnalyze={analyze} />
        </aside>
      </div>
    </div>
  );
}
