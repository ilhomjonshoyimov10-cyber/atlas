import { useEffect, useState } from "react";
import { api, fmtPct, SeasonProfile } from "../api";

const MONTHS = ["Yan", "Fev", "Mar", "Apr", "May", "Iyn", "Iyl", "Avg", "Sen", "Okt", "Noy", "Dek"];

function cellColor(v: number | null, max: number) {
  if (v === null || !isFinite(v)) return "transparent";
  const a = Math.min(1, Math.abs(v) / max);
  return v >= 0 ? `color-mix(in oklab, var(--up) ${Math.round(a * 85)}%, var(--surface))`
    : `color-mix(in oklab, var(--down) ${Math.round(a * 85)}%, var(--surface))`;
}

export default function Seasonality({ symbol, name }: { symbol: string; name: string }) {
  const [d, setD] = useState<SeasonProfile | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setD(null); setErr(null);
    api.seasonality(symbol).then(setD).catch((e) => setErr(e.message));
  }, [symbol]);
  if (err) return <p className="muted">Mavsumiylik yuklanmadi: {err}</p>;
  if (!d) return <div className="skeleton" style={{ height: 320 }} />;

  const prof = d.profile;
  const maxAbs = Math.max(...prof.map((p) => Math.abs(p.mean)), 0.1);
  const flat = d.by_year.values.flat().filter((v): v is number => v !== null);
  const heatMax = Math.max(...flat.map(Math.abs), 1) * 0.7;
  const best = [...prof].sort((a, b) => b.mean - a.mean)[0];
  const worst = [...prof].sort((a, b) => a.mean - b.mean)[0];

  return (
    <div className="season">
      <p className="lead">
        {name} tarixan eng kuchli oy — <strong>{MONTHS[best.month - 1]}</strong> (o'rtacha {fmtPct(best.mean)},
        {" "}{Math.round(best.up_prob * 100)}% yillarda o'sgan), eng zaif — <strong>{MONTHS[worst.month - 1]}</strong> ({fmtPct(worst.mean)}).
        {Math.abs(best.t_stat) < 2 && " Farq statistik jihatdan zaif — buni tasodif deb hisoblash mumkin."}
      </p>
      <div className="season-bars" role="img" aria-label="Oylik o'rtacha qaytim">
        {prof.map((p) => {
          const h = (Math.abs(p.mean) / maxAbs) * 50;
          const strong = Math.abs(p.t_stat) >= 2;
          return (
            <div key={p.month} className="sb-col" title={`${MONTHS[p.month - 1]}: o'rtacha ${fmtPct(p.mean)}, o'sish ehtimoli ${Math.round(p.up_prob * 100)}%, t=${p.t_stat.toFixed(2)}, ${p.n} yil`}>
              <div className="sb-track">
                <div className={`sb-bar ${p.mean >= 0 ? "up" : "down"} ${strong ? "strong" : ""}`}
                  style={{ height: `${h}%`, [p.mean >= 0 ? "bottom" : "top"]: "50%" } as React.CSSProperties} />
              </div>
              <span className="sb-pct">{fmtPct(p.mean, 1)}</span>
              <span className="sb-m">{MONTHS[p.month - 1]}</span>
              <span className="sb-prob">{Math.round(p.up_prob * 100)}%</span>
            </div>
          );
        })}
      </div>
      <p className="hint">Ustun ostidagi foiz — o'sha oy necha foiz yillarda o'sish bilan yopilgan. To'q rang: |t| ≥ 2 (ishonchliroq).</p>
      <div className="heat-scroll">
        <table className="heat">
          <thead><tr><th scope="col">Yil</th>{MONTHS.map((m) => <th key={m} scope="col">{m}</th>)}</tr></thead>
          <tbody>
            {d.by_year.years.map((y, yi) => (
              <tr key={y}>
                <th scope="row">{y}</th>
                {Array.from({ length: 12 }, (_, m) => {
                  const ki = d.by_year.keys.indexOf(m + 1);
                  const v = ki >= 0 ? d.by_year.values[yi][ki] : null;
                  return <td key={m} style={{ background: cellColor(v, heatMax) }}>{v === null ? "" : v.toFixed(1)}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
