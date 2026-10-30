import { useEffect, useMemo, useState } from "react";
import { api, fmtPct, STATIC } from "../api";

const COSTS = [0, 2, 5, 10, 20];   // statik rejimda oldindan hisoblangan xarajat qiymatlari

interface Stats { total_ret: number; ann_ret: number; ann_vol: number; sharpe: number; max_dd: number; hit_rate: number; days: number; turnover?: number }
interface BT {
  period: [string, string]; stats: Record<string, Stats>; series: Record<string, [string, number][]>;
  per_asset?: { symbol: string; sharpe: number; total_ret: number; hit_rate: number }[];
}
const NAMES: Record<string, string> = {
  full: "Model (yangilik bilan)", no_news: "Model (yangiliksiz)", price: "Faqat narx modeli", buy_hold: "Sotib olib ushlash",
};
const COLORS: Record<string, string> = { full: "var(--fc)", no_news: "var(--news)", price: "var(--muted)", buy_hold: "var(--ink)" };

function Curves({ series }: { series: BT["series"] }) {
  const W = 900, H = 300, P = 56;
  const keys = Object.keys(series);
  const all = keys.flatMap((k) => series[k].map((p) => p[1]));
  if (!all.length) return null;
  const lo = Math.min(0, ...all), hi = Math.max(0, ...all);
  const n = Math.max(...keys.map((k) => series[k].length));
  const x = (i: number) => P + (i / Math.max(1, n - 1)) * (W - 2 * P);
  const y = (v: number) => 12 + (1 - (v - lo) / (hi - lo || 1)) * (H - 40);
  const first = series[keys[0]];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="bt-chart" role="img" aria-label="Jamlangan daromad egri chiziqlari">
      <line x1={P} x2={W - P} y1={y(0)} y2={y(0)} className="grid" />
      {[lo, hi].map((v) => <text key={v} x={W - P + 4} y={y(v) + 4} className="axis">{fmtPct((Math.exp(v) - 1) * 100, 1)}</text>)}
      {[0, Math.floor((first.length - 1) / 2), first.length - 1].map((i) => (
        <text key={i} x={x(i)} y={H - 6} className="axis" textAnchor="middle">{first[i]?.[0]}</text>
      ))}
      {keys.map((k) => (
        <path key={k} d={series[k].map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[1]).toFixed(1)}`).join("")}
          fill="none" stroke={COLORS[k] ?? "var(--muted)"} strokeWidth={k === "full" ? 2.2 : 1.4}
          strokeDasharray={k === "buy_hold" ? "4 4" : undefined} />
      ))}
    </svg>
  );
}

export default function Backtest() {
  const [thr, setThr] = useState(STATIC ? 0.04 : 0.05);
  const [cost, setCost] = useState(5);
  const [d, setD] = useState<BT | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setErr(null);
    const t = setTimeout(() => {
      api.backtest(thr, cost).then(setD).catch((e) => setErr(e.message));
    }, 250);
    return () => clearTimeout(t);
  }, [thr, cost]);

  const order = useMemo(() => d ? ["full", "no_news", "price", "buy_hold"].filter((k) => d.stats[k]) : [], [d]);
  if (err) return (
    <div className="empty">
      <p><strong>Backtest hozircha ko'rsatilmaydi.</strong> {err}</p>
      <p className="muted">Backtest model ko'rmagan davr bashoratlaridan quriladi{STATIC ? " va har haftalik qayta o'qitishdan keyin yangilanadi." : <>. Ular <code>python -m atlas.cli real</code>
        (yoki <code>evaluate --final</code>) bajarilganda yaratiladi.</>}</p>
    </div>
  );
  if (!d) return <div className="skeleton" style={{ height: 360 }} />;
  const f = d.stats.full, nn = d.stats.no_news;
  return (
    <div className="bt">
      <p className="lead">Agar model signallari bo'yicha savdo qilinganida nima bo'lardi — faqat model <strong>ko'rmagan</strong> davrda
        ({d.period[0]} — {d.period[1]}). Har bir aktivda kunlik ~1% risk, komissiya hisobga olingan.</p>
      <div className="bt-controls">
        <label>Signal chegarasi: <strong>{Math.round((0.5 + thr) * 100)}%</strong> o'sish ehtimolidan yuqori → long
          <input type="range" min={0} max={0.2} step={STATIC ? 0.02 : 0.01} value={thr} onChange={(e) => setThr(Math.round(+e.target.value * 100) / 100)} />
        </label>
        <label>Xarajat (bps): <strong>{cost}</strong>
          {STATIC
            ? <input type="range" min={0} max={COSTS.length - 1} step={1} value={Math.max(0, COSTS.indexOf(cost))} onChange={(e) => setCost(COSTS[+e.target.value])} />
            : <input type="range" min={0} max={20} step={1} value={cost} onChange={(e) => setCost(+e.target.value)} />}
        </label>
      </div>
      {f && (
        <p className="bt-verdict">
          Model strategiyasi: Sharpe <strong>{f.sharpe.toFixed(2)}</strong>, jami {fmtPct(f.total_ret, 1)}, maksimal tushish {fmtPct(f.max_dd, 1)}.
          {nn && ` Yangiliksiz variant: Sharpe ${nn.sharpe.toFixed(2)} — yangilik ${f.sharpe > nn.sharpe ? "qo'shimcha foyda berdi" : "bu davrda foyda bermadi"}.`}
          {" "}Bitta qisqa davr natijasi tasodif bo'lishi mumkin — xulosani walk-forward bilan birga qiling.
        </p>
      )}
      <Curves series={Object.fromEntries(order.map((k) => [k, d.series[k]]))} />
      <div className="legend">
        {order.map((k) => <span key={k}><i style={{ background: COLORS[k] }} />{NAMES[k]}</span>)}
      </div>
      <div className="table-scroll">
        <table className="metrics">
          <thead><tr><th scope="col">Strategiya</th><th scope="col">Sharpe</th><th scope="col">Jami</th><th scope="col">Yillik</th>
            <th scope="col">Volatillik</th><th scope="col">Maks. tushish</th><th scope="col">Foydali kunlar</th><th scope="col">Aylanma</th></tr></thead>
          <tbody>
            {order.map((k) => {
              const s = d.stats[k];
              return (
                <tr key={k} className={k === "full" ? "best" : ""}>
                  <th scope="row">{NAMES[k]}</th><td>{s.sharpe.toFixed(2)}</td><td className={s.total_ret >= 0 ? "pos" : "neg"}>{fmtPct(s.total_ret, 1)}</td>
                  <td>{fmtPct(s.ann_ret, 1)}</td><td>{s.ann_vol.toFixed(1)}%</td><td className="neg">{fmtPct(s.max_dd, 1)}</td>
                  <td>{s.hit_rate.toFixed(1)}%</td><td>{s.turnover !== undefined ? s.turnover.toFixed(3) : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {d.per_asset && (
        <>
          <h3>Aktivlar bo'yicha (model, yangilik bilan)</h3>
          <div className="pa-grid">
            {d.per_asset.map((r) => (
              <div key={r.symbol} className="pa"><span>{r.symbol}</span>
                <span className={r.sharpe >= 0 ? "pos" : "neg"}>{r.sharpe.toFixed(2)}</span></div>
            ))}
          </div>
          <p className="hint">Raqam — aktivdagi Sharpe. Ko'p aktiv orasidan eng yaxshisini tanlab olish (cherry-picking) yolg'on natija beradi.</p>
        </>
      )}
      <p className="hint">Tadqiqot vositasi: slippage, likvidlik, soliq va qisqa sotish cheklovlari hisobga olinmagan. Investitsiya tavsiyasi emas.</p>
    </div>
  );
}
