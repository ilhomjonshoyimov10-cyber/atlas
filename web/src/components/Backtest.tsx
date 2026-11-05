import { useEffect, useMemo, useState } from "react";
import { api, fmtPct } from "../api";

/* Professional backtest: faqat out-of-sample bashoratlar, real xarajat, volatillik maqsadi,
   statistik ishonchlilik (bootstrap, PSR, Deflated Sharpe, tasodifiy signal testi) va barqarorlik xaritasi. */

type Pt = [string, number];
interface VStats {
  days: number; total_ret: number; cagr: number; ann_vol: number; sharpe: number; sortino: number | null;
  max_dd: number; dd_days: number; calmar: number; hit_rate: number; profit_factor: number | null; win_loss: number | null;
  skew: number; kurt: number; var95: number; cvar95: number; turnover_ann?: number; exposure?: number;
  beta?: number; corr?: number; alpha_ann?: number; info_ratio?: number; cost_drag?: number;
}
interface VTests {
  sharpe_ci90: [number, number]; p_sharpe_le_0: number; psr: number | null; dsr: number | null; sharpe_lag1: number | null;
  random?: { random_mean: number; random_p95: number; percentile: number; p_value: number; n: number };
}
interface Variant {
  name: string; stats: VStats; tests: VTests; equity: Pt[]; drawdown: Pt[]; rolling_sharpe: Pt[];
  monthly: [number, number, number][]; yearly: Record<string, number>;
  per_asset: { symbol: string; group: string; pnl: number; sharpe: number }[]; per_class: Record<string, number>;
}
interface Report {
  segment: string; strategy: string; strategy_name: string; threshold: number | null; cost: string | number;
  period: [string, string]; folds: [string, string, string][]; n_assets: number;
  deflated: { n_trials: number; sr0: number }; variants: Record<string, Variant>;
}

const SEG_NAMES: Record<string, string> = { wf: "Walk-forward (OOS)", holdout: "Final holdout" };
const COST_NAMES: Record<string, string> = { real: "Real (aktiv sinfi bo'yicha)", "0": "0 bps", "5": "5 bps", "10": "10 bps", "20": "20 bps" };
const GROUP_NAMES: Record<string, string> = {
  forex: "Valyuta", crypto: "Kripto", equity: "Aksiyalar", index: "Indekslar", energy: "Energiya",
  agri: "Qishloq xo'jaligi", metal: "Metallar", rates: "Obligatsiyalar",
};
const PALETTE = ["var(--fc)", "var(--news)", "var(--up)", "#b06ab3", "#4a9bb8", "var(--muted)"];
const MONTHS = ["Yan", "Fev", "Mar", "Apr", "May", "Iyn", "Iyl", "Avg", "Sen", "Okt", "Noy", "Dek"];

const f2 = (v: number | null | undefined, d = 2) => (v === null || v === undefined || !isFinite(v) ? "—" : v.toFixed(d));
const pct = (v: number | null | undefined, d = 1) => (v === null || v === undefined || !isFinite(v) ? "—" : fmtPct(v, d));

function colorOf(keys: string[], k: string) {
  if (k === "buy_hold") return "var(--ink)";
  return PALETTE[keys.filter((x) => x !== "buy_hold").indexOf(k) % PALETTE.length];
}

/* ---------- grafiklar ---------- */
function Lines({ series, focus, fmt, zero = true, height = 280, label }: {
  series: Record<string, Pt[]>; focus: string; fmt: (v: number) => string; zero?: boolean; height?: number; label: string;
}) {
  const W = 900, H = height, PL = 8, PR = 64, PT = 10, PB = 26;
  const keys = Object.keys(series).filter((k) => series[k]?.length);
  const dates = useMemo(() => Array.from(new Set(keys.flatMap((k) => series[k].map((p) => p[0])))).sort(), [series]);
  if (!keys.length || dates.length < 2) return null;
  const di = new Map(dates.map((d, i) => [d, i]));
  const vals = keys.flatMap((k) => series[k].map((p) => p[1]));
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  const pad = (hi - lo || 1) * 0.06; lo -= pad; hi += pad;
  const x = (d: string) => PL + ((di.get(d) ?? 0) / (dates.length - 1)) * (W - PL - PR);
  const y = (v: number) => PT + (1 - (v - lo) / (hi - lo)) * (H - PT - PB);
  const ticks = [lo + pad, (lo + hi) / 2, hi - pad];
  const order = [...keys.filter((k) => k !== focus), focus].filter((k) => keys.includes(k));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="bt-chart" role="img" aria-label={label}>
      {ticks.map((v, i) => (
        <g key={i}><line x1={PL} x2={W - PR} y1={y(v)} y2={y(v)} className="grid" />
          <text x={W - PR + 6} y={y(v) + 4} className="axis">{fmt(v)}</text></g>
      ))}
      {zero && lo < 0 && hi > 0 && <line x1={PL} x2={W - PR} y1={y(0)} y2={y(0)} className="zero" />}
      {[0, Math.floor((dates.length - 1) / 2), dates.length - 1].map((i) => (
        <text key={i} x={PL + (i / (dates.length - 1)) * (W - PL - PR)} y={H - 6} className="axis"
          textAnchor={i === 0 ? "start" : i === dates.length - 1 ? "end" : "middle"}>{dates[i]}</text>
      ))}
      {order.map((k) => (
        <path key={k} d={series[k].map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("")}
          fill="none" stroke={colorOf(keys, k)} strokeWidth={k === focus ? 2.4 : 1.2}
          strokeOpacity={k === focus || k === "buy_hold" ? 1 : 0.55} strokeDasharray={k === "buy_hold" ? "5 4" : undefined} />
      ))}
    </svg>
  );
}

function Underwater({ a, b }: { a: Pt[]; b?: Pt[] }) {
  const W = 900, H = 150, PL = 8, PR = 64, PT = 6, PB = 22;
  const all = [...a, ...(b ?? [])].map((p) => p[1]);
  if (a.length < 2) return null;
  const lo = Math.min(-1, ...all) * 1.08;
  const x = (i: number, n: number) => PL + (i / (n - 1)) * (W - PL - PR);
  const y = (v: number) => PT + (v / lo) * (H - PT - PB);
  const area = (s: Pt[]) => `M${x(0, s.length)},${y(0)} ` + s.map((p, i) => `L${x(i, s.length).toFixed(1)},${y(p[1]).toFixed(1)}`).join(" ") + ` L${x(s.length - 1, s.length)},${y(0)} Z`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="bt-chart" role="img" aria-label="Cho'qqidan tushish (drawdown)">
      <line x1={PL} x2={W - PR} y1={y(0)} y2={y(0)} className="grid" />
      <text x={W - PR + 6} y={y(lo / 1.08) + 4} className="axis">{(lo / 1.08).toFixed(0)}%</text>
      <text x={W - PR + 6} y={y(0) + 4} className="axis">0%</text>
      {b && <path d={b.map((p, i) => `${i ? "L" : "M"}${x(i, b.length).toFixed(1)},${y(p[1]).toFixed(1)}`).join("")} fill="none" stroke="var(--ink)" strokeDasharray="5 4" strokeWidth={1.2} />}
      <path d={area(a)} fill="color-mix(in oklab, var(--down) 28%, transparent)" stroke="var(--down)" strokeWidth={1.4} />
      <text x={PL} y={H - 6} className="axis">{a[0][0]}</text>
      <text x={W - PR} y={H - 6} className="axis" textAnchor="end">{a[a.length - 1][0]}</text>
    </svg>
  );
}

function heatColor(v: number, scale: number) {
  const t = Math.max(-1, Math.min(1, v / scale));
  const c = t >= 0 ? "var(--up)" : "var(--down)";
  return `color-mix(in oklab, ${c} ${Math.round(Math.abs(t) * 70)}%, var(--surface))`;
}

function MonthGrid({ monthly, yearly }: { monthly: [number, number, number][]; yearly: Record<string, number> }) {
  const years = Array.from(new Set(monthly.map((m) => m[0]))).sort();
  const get = (y: number, m: number) => monthly.find((r) => r[0] === y && r[1] === m)?.[2];
  const scale = Math.max(1, ...monthly.map((m) => Math.abs(m[2])));
  return (
    <div className="table-scroll">
      <table className="bt-months">
        <thead><tr><th scope="col">Yil</th>{MONTHS.map((m) => <th key={m} scope="col">{m}</th>)}<th scope="col">Yil</th></tr></thead>
        <tbody>
          {years.map((y) => (
            <tr key={y}><th scope="row">{y}</th>
              {MONTHS.map((_, i) => {
                const v = get(y, i + 1);
                return <td key={i} style={v === undefined ? undefined : { background: heatColor(v, scale) }}>{v === undefined ? "" : v.toFixed(1)}</td>;
              })}
              <td className={(yearly[String(y)] ?? 0) >= 0 ? "pos" : "neg"}><strong>{f2(yearly[String(y)], 1)}</strong></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------- ishonchlilik xulosasi ---------- */
function verdict(v: Variant, bh?: Variant) {
  const s = v.stats, t = v.tests;
  const parts: { cls: string; text: string }[] = [];
  if (s.sharpe <= 0) parts.push({ cls: "bad", text: `Strategiya bu davrda zarar ko'rgan (Sharpe ${f2(s.sharpe)}).` });
  else if ((t.dsr ?? 0) >= 0.95) parts.push({ cls: "good", text: `Statistik ishonchli: ko'p sozlama sinalganini hisobga olgandan keyin ham (Deflated Sharpe ${f2((t.dsr ?? 0) * 100, 0)}%) ustunlik saqlanadi.` });
  else if ((t.psr ?? 0) >= 0.95) parts.push({ cls: "warn", text: `Sharpe ${f2(s.sharpe)} va u 0 dan katta bo'lish ehtimoli ${f2((t.psr ?? 0) * 100, 0)}%. Lekin ${""}bir nechta sozlama sinalgan: tanlash xatosini tuzatgandan keyin (Deflated Sharpe ${f2((t.dsr ?? 0) * 100, 0)}%) ustunlik isbotlanmaydi.` });
  else parts.push({ cls: "warn", text: `Sharpe ${f2(s.sharpe)} ijobiy, lekin tasodifdan ajratib bo'lmaydi (90% oraliq ${f2(t.sharpe_ci90[0])} … ${f2(t.sharpe_ci90[1])}).` });
  if (t.random) parts.push({ cls: t.random.percentile >= 95 ? "good" : "", text: `Tasodifiy signallar bilan solishtirganda ${f2(t.random.percentile, 0)}-persentil (95+ bo'lsa — signalda haqiqiy ma'lumot bor).` });
  if (bh) parts.push({ cls: "", text: `Benchmark (sotib olib ushlash, xuddi shu risk bilan): Sharpe ${f2(bh.stats.sharpe)}, CAGR ${pct(bh.stats.cagr)}.${s.corr !== undefined ? ` Strategiya bilan korrelyatsiya ${f2(s.corr)}.` : ""}` });
  if (t.sharpe_lag1 !== null && t.sharpe_lag1 !== undefined)
    parts.push({ cls: t.sharpe_lag1 < s.sharpe * 0.5 ? "warn" : "", text: `Ijro bir kun kechiksa Sharpe ${f2(t.sharpe_lag1)} bo'ladi${t.sharpe_lag1 < s.sharpe * 0.5 ? " — signal tez eskiradi, ijro tezligi muhim." : "."}` });
  return parts;
}

/* ---------- sahifa ---------- */
export default function Backtest() {
  const [idx, setIdx] = useState<any>(null);
  const [seg, setSeg] = useState("wf");
  const [strategy, setStrategy] = useState("direction");
  const [thr, setThr] = useState(0.01);
  const [cost, setCost] = useState<string>("real");
  const [focus, setFocus] = useState("full");
  const [rep, setRep] = useState<Report | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { api.backtestIndex().then(setIdx).catch((e) => setErr(e.message)); }, []);
  useEffect(() => {
    if (idx && !idx.segments?.[seg]) { const s = Object.keys(idx.segments ?? {})[0]; if (s) setSeg(s); }
  }, [idx, seg]);
  useEffect(() => {
    if (!idx?.segments?.[seg]) return;
    setErr(null);
    const t = strategy === "cross_sectional" ? null : thr;
    api.backtest(seg, strategy, t, cost).then((r: Report) => {
      setRep(r);
      if (!r.variants[focus]) setFocus(Object.keys(r.variants).find((k) => k !== "buy_hold") ?? "buy_hold");
    }).catch((e) => setErr(e.message));
  }, [idx, seg, strategy, thr, cost]);

  const segInfo = idx?.segments?.[seg];
  const grid: any[] = segInfo?.grid ?? [];
  const keys = rep ? Object.keys(rep.variants) : [];
  const fv = rep?.variants[focus];
  const bh = rep?.variants.buy_hold;

  if (err) return (
    <div className="empty">
      <p><strong>Backtest hozircha ko'rsatilmaydi.</strong> {err}</p>
      <p className="muted">Backtest model ko'rmagan davr bashoratlaridan quriladi. Ular walk-forward baholash (Colab notebook yoki <code>python -m atlas.cli evaluate</code>) bajarilganda yaratiladi.</p>
    </div>
  );
  if (!idx || !rep || !fv) return <div className="skeleton" style={{ height: 420 }} />;

  const robust = (strategy === "cross_sectional" ? [null] : idx.thresholds as (number | null)[]);
  const cell = (t: number | null, c: string | number, lag = 0) =>
    grid.find((g) => g.variant === focus && g.strategy === strategy && (g.threshold ?? null) === t && String(g.cost) === String(c) && g.lag === lag);
  const trials = grid.filter((g) => g.cost === "real" && g.lag === 0);
  const better = trials.filter((g) => g.sharpe > (cell(strategy === "cross_sectional" ? null : thr, "real")?.sharpe ?? 0)).length;
  const s = fv.stats, t = fv.tests;

  return (
    <div className="bt">
      <h1 className="bt-title">Backtest</h1>
      <p className="lead">
        Model signallari bo'yicha savdo qilinganida nima bo'lardi — <strong>faqat model ko'rmagan davr</strong> ({rep.period[0]} — {rep.period[1]},
        {" "}{rep.folds.length > 1 ? `${rep.folds.length} ta walk-forward bo'lak` : "final holdout"}, {rep.n_assets} ta aktiv).
        Har bir aktiv teng risk oladi, portfel yillik 10% volatillikka moslanadi, xarajat har bir pozitsiya o'zgarishida hisoblanadi.
      </p>

      <div className="bt-controls">
        <div className="seg" role="group" aria-label="Davr">
          {Object.keys(idx.segments).map((k) => (
            <button key={k} className={k === seg ? "on" : ""} onClick={() => setSeg(k)}>{SEG_NAMES[k] ?? k}</button>
          ))}
        </div>
        <label>Strategiya
          <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
            {Object.entries(idx.strategies as Record<string, string>).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
        </label>
        <label>Signal chegarasi{strategy === "cross_sectional" ? " (bu strategiyada yo'q)" : <>: P(o'sish) &gt; <strong>{(50 + thr * 100).toFixed(1)}%</strong></>}
          <select value={thr} disabled={strategy === "cross_sectional"} onChange={(e) => setThr(+e.target.value)}>
            {(idx.thresholds as number[]).map((v) => <option key={v} value={v}>{(50 + v * 100).toFixed(1)}%</option>)}
          </select>
        </label>
        <label>Xarajat
          <select value={cost} onChange={(e) => setCost(e.target.value)}>
            {(idx.costs as (string | number)[]).map((c) => <option key={String(c)} value={String(c)}>{COST_NAMES[String(c)] ?? `${c} bps`}</option>)}
          </select>
        </label>
        <label>Model varianti
          <select value={focus} onChange={(e) => setFocus(e.target.value)}>
            {keys.map((k) => <option key={k} value={k}>{rep.variants[k].name}</option>)}
          </select>
        </label>
      </div>

      <div className="bt-kpis">
        <div><span>CAGR</span><strong className={s.cagr >= 0 ? "pos" : "neg"}>{pct(s.cagr)}</strong></div>
        <div><span>Sharpe</span><strong>{f2(s.sharpe)}</strong><small>90%: {f2(t.sharpe_ci90[0])} … {f2(t.sharpe_ci90[1])}</small></div>
        <div><span>Sortino</span><strong>{f2(s.sortino)}</strong></div>
        <div><span>Maks. tushish</span><strong className="neg">{pct(s.max_dd)}</strong><small>{s.dd_days} kun</small></div>
        <div><span>Calmar</span><strong>{f2(s.calmar)}</strong></div>
        <div><span>Foydali kunlar</span><strong>{f2(s.hit_rate, 1)}%</strong><small>PF {f2(s.profit_factor)}</small></div>
        <div><span>Aylanma (yillik)</span><strong>{f2(s.turnover_ann, 1)}×</strong><small>xarajat −{f2(s.cost_drag, 2)}%/yil</small></div>
        <div><span>Deflated Sharpe</span><strong className={(t.dsr ?? 0) >= 0.95 ? "pos" : ""}>{t.dsr === null ? "—" : `${f2(t.dsr * 100, 0)}%`}</strong><small>{rep.deflated.n_trials} sinov</small></div>
      </div>

      <ul className="bt-verdicts">
        {verdict(fv, focus === "buy_hold" ? undefined : bh).map((p, i) => <li key={i} className={p.cls}>{p.text}</li>)}
      </ul>

      <h3>Kapital egri chizig'i (jamlangan daromad, %)</h3>
      <Lines series={Object.fromEntries(keys.map((k) => [k, rep.variants[k].equity]))} focus={focus}
        fmt={(v) => `${v >= 0 ? "+" : ""}${v.toFixed(0)}%`} label="Jamlangan daromad egri chiziqlari" />
      <div className="legend">
        {keys.map((k) => (
          <button key={k} className={k === focus ? "on" : ""} onClick={() => setFocus(k)}>
            <i style={{ background: colorOf(keys, k) }} />{rep.variants[k].name}
          </button>
        ))}
      </div>
      {rep.folds.length > 1 && (
        <p className="hint">Bo'laklar: {rep.folds.map((f) => `${f[1]} — ${f[2]}`).join(" · ")}. Har birida model faqat o'sha davrgacha bo'lgan ma'lumotda o'qitilgan; bo'laklar orasidagi tanaffus — embargo.</p>
      )}

      <h3>Cho'qqidan tushish (drawdown)</h3>
      <Underwater a={fv.drawdown} b={focus !== "buy_hold" ? bh?.drawdown : undefined} />

      <h3>Barcha variantlar</h3>
      <div className="table-scroll">
        <table className="metrics">
          <thead><tr><th scope="col">Variant</th><th scope="col">CAGR</th><th scope="col">Sharpe</th><th scope="col">Sortino</th>
            <th scope="col">Maks. tushish</th><th scope="col">Volatillik</th><th scope="col">Beta</th><th scope="col">Aylanma</th>
            <th scope="col">PSR</th><th scope="col">Tasodifiydan</th></tr></thead>
          <tbody>
            {keys.map((k) => {
              const v = rep.variants[k];
              return (
                <tr key={k} className={k === focus ? "best" : ""} onClick={() => setFocus(k)} style={{ cursor: "pointer" }}>
                  <th scope="row">{v.name}</th>
                  <td className={v.stats.cagr >= 0 ? "pos" : "neg"}>{pct(v.stats.cagr)}</td><td>{f2(v.stats.sharpe)}</td><td>{f2(v.stats.sortino)}</td>
                  <td className="neg">{pct(v.stats.max_dd)}</td><td>{f2(v.stats.ann_vol, 1)}%</td><td>{f2(v.stats.beta)}</td>
                  <td>{v.stats.turnover_ann === undefined ? "—" : `${f2(v.stats.turnover_ann, 0)}×`}</td>
                  <td>{v.tests.psr === null ? "—" : `${f2(v.tests.psr * 100, 0)}%`}</td>
                  <td>{v.tests.random ? `${f2(v.tests.random.percentile, 0)}-pers.` : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h3>Statistik ishonchlilik — {fv.name}</h3>
      <div className="table-scroll">
        <table className="metrics bt-tests">
          <tbody>
            <tr><th scope="row">Sharpe, 90% ishonch oralig'i (block bootstrap)</th><td>{f2(t.sharpe_ci90[0])} … {f2(t.sharpe_ci90[1])}</td></tr>
            <tr><th scope="row">Haqiqiy Sharpe ≤ 0 bo'lish ehtimoli</th><td>{f2(t.p_sharpe_le_0 * 100, 1)}%</td></tr>
            <tr><th scope="row">Probabilistic Sharpe Ratio (qiyshiqlik va og'ir dumlar hisobga olingan)</th><td>{t.psr === null ? "—" : `${f2(t.psr * 100, 1)}%`}</td></tr>
            <tr><th scope="row">Deflated Sharpe ({rep.deflated.n_trials} ta sozlama sinalgan, tasodifiy "eng yaxshi" Sharpe ≈ {f2(rep.deflated.sr0)})</th><td>{t.dsr === null ? "—" : `${f2(t.dsr * 100, 1)}%`}</td></tr>
            {t.random && <tr><th scope="row">Tasodifiy signal testi ({t.random.n} marta aralashtirilgan, xarajatsiz)</th>
              <td>{f2(t.random.percentile, 0)}-persentil · tasodifiy o'rtacha {f2(t.random.random_mean)}, 95% {f2(t.random.random_p95)}</td></tr>}
            <tr><th scope="row">Ijro 1 kun kechiksa Sharpe</th><td>{f2(t.sharpe_lag1)}</td></tr>
            <tr><th scope="row">VaR 95% / CVaR 95% (kunlik)</th><td>{f2(s.var95)}% / {f2(s.cvar95)}%</td></tr>
            <tr><th scope="row">Qiyshiqlik / kurtoz</th><td>{f2(s.skew)} / {f2(s.kurt, 1)}</td></tr>
            {s.alpha_ann !== undefined && <tr><th scope="row">Alfa (benchmarkga nisbatan, yillik) / Information ratio</th><td>{pct(s.alpha_ann)} / {f2(s.info_ratio)}</td></tr>}
            <tr><th scope="row">Shu davrda sinalgan sozlamalar ichida bundan yuqori Sharpe</th><td>{better} / {trials.length}</td></tr>
          </tbody>
        </table>
      </div>

      <h3>Oylik daromad (%)</h3>
      <MonthGrid monthly={fv.monthly} yearly={fv.yearly} />

      {fv.rolling_sharpe.length > 2 && (
        <>
          <h3>Sirpanuvchi 6 oylik Sharpe</h3>
          <Lines series={{ [focus]: fv.rolling_sharpe, ...(bh && focus !== "buy_hold" ? { buy_hold: bh.rolling_sharpe } : {}) }}
            focus={focus} fmt={(v) => v.toFixed(1)} height={180} label="Sirpanuvchi Sharpe" />
        </>
      )}

      {focus !== "buy_hold" && (
        <>
          <h3>Barqarorlik: Sharpe har xil chegara va xarajatda</h3>
          <p className="hint">Natija bitta "omadli" sozlamaga bog'liq bo'lsa, qo'shni kataklar keskin farq qiladi. Yaxshi strategiyada butun xarita bir xil rangda bo'ladi.</p>
          <div className="table-scroll">
            <table className="bt-robust">
              <thead><tr><th scope="col">Chegara \ Xarajat</th>{(idx.costs as (string | number)[]).map((c) => <th key={String(c)} scope="col">{c === "real" ? "Real" : `${c} bps`}</th>)}<th scope="col">1 kun kechikish</th></tr></thead>
              <tbody>
                {robust.map((th) => (
                  <tr key={String(th)} className={(th === null || th === thr) ? "cur" : ""}>
                    <th scope="row">{th === null ? "Reyting" : `${(50 + th * 100).toFixed(1)}%`}</th>
                    {(idx.costs as (string | number)[]).map((c) => {
                      const g = cell(th, c);
                      return <td key={String(c)} style={g ? { background: heatColor(g.sharpe, 2) } : undefined}>{g ? f2(g.sharpe) : "—"}</td>;
                    })}
                    {(() => { const g = cell(th, "real", 1); return <td style={g ? { background: heatColor(g.sharpe, 2) } : undefined}>{g ? f2(g.sharpe) : "—"}</td>; })()}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3>Daromad manbalari</h3>
          <div className="bt-split">
            <div>
              <h4>Aktiv sinflari bo'yicha (jami hissa, % kapital)</h4>
              <ul className="bars">
                {Object.entries(fv.per_class).sort((a, b) => b[1] - a[1]).map(([g, v]) => {
                  const m = Math.max(...Object.values(fv.per_class).map(Math.abs), 0.01);
                  return (
                    <li key={g}><span>{GROUP_NAMES[g] ?? g}</span>
                      <span className="bar"><i className={v >= 0 ? "up" : "down"} style={{ width: `${(Math.abs(v) / m) * 100}%` }} /></span>
                      <b className={v >= 0 ? "pos" : "neg"}>{pct(v, 2)}</b></li>
                  );
                })}
              </ul>
            </div>
            <div>
              <h4>Eng ko'p foyda / zarar bergan aktivlar</h4>
              <div className="pa-grid">
                {[...fv.per_asset.slice(0, 6), ...fv.per_asset.slice(-6)].filter((r, i, a) => a.findIndex((x) => x.symbol === r.symbol) === i).map((r) => (
                  <div key={r.symbol} className="pa"><span>{r.symbol}</span><span className={r.pnl >= 0 ? "pos" : "neg"}>{pct(r.pnl, 2)}</span></div>
                ))}
              </div>
              <p className="hint">Eng yaxshi aktivlarni tanlab olib "shular bo'yicha savdo qilaman" deyish — klassik xato (cherry-picking): bu tanlov natijaga qarab qilingan.</p>
            </div>
          </div>
        </>
      )}

      <details className="bt-method">
        <summary>Metodologiya va cheklovlar</summary>
        <ul>
          <li><strong>Ma'lumot:</strong> faqat out-of-sample bashoratlar. Walk-forward rejimida har bo'lakda model faqat undan oldingi ma'lumotda o'qitilgan; holdout — butun tanlov jarayonida ishlatilmagan oxirgi yil.</li>
          <li><strong>Qaror vaqti:</strong> har kuni 20:00 UTC (yangiliklar shu vaqtgacha), pozitsiya keyingi kun yopilishigacha. "1 kun kechikish" ustuni ijro sekin bo'lganda natijani ko'rsatadi.</li>
          <li><strong>Strategiyalar:</strong> yo'nalish — P(o'sish) chegaradan yuqori bo'lsa long, past bo'lsa short; faqat long — short yo'q; kesishma reyting — har kuni kutilgan qaytim/noaniqlik nisbati bo'yicha eng yaxshi 20% long, eng yomon 20% short (bozor yo'nalishiga kam bog'liq).</li>
          <li><strong>Risk:</strong> har bir aktivga kunlik ~1% risk (vazn ∝ 1/volatillik), portfel yillik 10% volatillikka moslanadi (o'tgan 60 kun, 1 kun lag bilan), leverage ≤ 3×. Benchmark ham xuddi shu tarzda — solishtirish adolatli.</li>
          <li><strong>Xarajat:</strong> "Real" — bir tomonlama: valyuta 1, indeks va obligatsiya 1.5, aksiya va metall 2, energiya 3, qishloq xo'jaligi 4, kripto 8 bps (komissiya + yarim spread + slippage).</li>
          <li><strong>Statistika:</strong> Sharpe oralig'i — stationary block bootstrap (2000 marta); PSR va Deflated Sharpe — Bailey va López de Prado (2012, 2014); tasodifiy test — signal kunlari aralashtiriladi.</li>
          <li><strong>Hisobga olinmagan:</strong> soliq, moliyalashtirish (swap/funding) xarajati, likvidlik cheklovlari, qisqa sotish taqiqlari; kripto dam olish kunlari savdo qilinmaydi.</li>
          <li>Bu tadqiqot vositasi. O'tmishdagi natija kelajakni kafolatlamaydi va investitsiya tavsiyasi emas.</li>
        </ul>
      </details>
    </div>
  );
}
