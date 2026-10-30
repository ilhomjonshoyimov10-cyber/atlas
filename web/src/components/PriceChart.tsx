import { useMemo, useRef, useState } from "react";
import { Bar, Forecast, fmtPct, fmtPrice } from "../api";

const W = 1000, H = 380, PAD_L = 8, PAD_R = 64, PAD_T = 16, PAD_B = 28;
const FUTURE_FRAC = 0.2; // grafikning o'ng 20% — prognoz zonasi

interface Props { bars: Bar[]; forecast: Forecast | null; years: number }

export default function PriceChart({ bars, forecast, years }: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  const data = useMemo(() => {
    const cut = new Date(); cut.setFullYear(cut.getFullYear() - years);
    const iso = cut.toISOString().slice(0, 10);
    const rows = bars.filter((b) => b.t >= iso);
    // Uzoq davrda nuqtalarni siyraklashtiramiz (har piksel ~1 nuqta)
    const step = Math.max(1, Math.floor(rows.length / 900));
    return rows.filter((_, i) => i % step === 0 || i === rows.length - 1);
  }, [bars, years]);

  const geo = useMemo(() => {
    if (data.length < 2) return null;
    const plotW = W - PAD_L - PAD_R;
    const histW = plotW * (forecast ? 1 - FUTURE_FRAC : 1);
    const hs = forecast?.horizons ?? [];
    const maxH = hs.length ? Math.max(...hs.map((h) => h.h)) : 1;
    let lo = Math.min(...data.map((d) => d.close));
    let hi = Math.max(...data.map((d) => d.close));
    hs.forEach((h) => { lo = Math.min(lo, h.q05); hi = Math.max(hi, h.q95); });
    const padY = (hi - lo) * 0.06;
    lo -= padY; hi += padY;
    const x = (i: number) => PAD_L + (i / (data.length - 1)) * histW;
    const xf = (h: number) => PAD_L + histW + (h / maxH) * (plotW - histW);
    const y = (v: number) => PAD_T + (1 - (v - lo) / (hi - lo)) * (H - PAD_T - PAD_B);
    const line = data.map((d, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(d.close).toFixed(1)}`).join("");
    const area = `${line}L${x(data.length - 1)},${H - PAD_B}L${x(0)},${H - PAD_B}Z`;
    const last = data[data.length - 1].close;
    const pts = [{ h: 0, q05: last, q50: last, q95: last }, ...hs];
    const fan = pts.length > 1
      ? pts.map((p, i) => `${i ? "L" : "M"}${xf(p.h)},${y(p.q95)}`).join("") +
        [...pts].reverse().map((p) => `L${xf(p.h)},${y(p.q05)}`).join("") + "Z"
      : "";
    const med = pts.map((p, i) => `${i ? "L" : "M"}${xf(p.h)},${y(p.q50)}`).join("");
    const ticks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * (i + 0.5)) / 5);
    const nX = 6;
    // Prognoz zonasi bo'lsa oxirgi tarixiy belgi "+1k" bilan ustma-ust tushmasin
    const xticks = Array.from({ length: forecast ? nX - 1 : nX }, (_, i) => Math.round((i / (nX - 1)) * (data.length - 1)));
    return { x, xf, y, line, area, fan, med, ticks, xticks, histW, plotW, maxH, hs, last };
  }, [data, forecast]);

  if (!geo) return <div className="chart-empty">Grafik uchun ma'lumot yetarli emas</div>;

  const onMove = (e: React.MouseEvent) => {
    const r = ref.current!.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W - PAD_L;
    if (px > geo.histW) { setHover(-1); return; }
    setHover(Math.max(0, Math.min(data.length - 1, Math.round((px / geo.histW) * (data.length - 1)))));
  };

  const up = (forecast?.horizons[0]?.ret_q50 ?? 0) >= 0;
  return (
    <div className="chart-wrap">
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} className="chart" onMouseMove={onMove} onMouseLeave={() => setHover(null)}
        role="img" aria-label="Narx grafigi va prognoz oralig'i">
        <defs>
          <linearGradient id="histFill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="var(--ink)" stopOpacity="0.10" />
            <stop offset="1" stopColor="var(--ink)" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="fanFill" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0" stopColor="var(--fc)" stopOpacity="0.45" />
            <stop offset="1" stopColor="var(--fc)" stopOpacity="0.14" />
          </linearGradient>
        </defs>
        {geo.ticks.map((t) => (
          <g key={t}>
            <line x1={PAD_L} x2={W - PAD_R} y1={geo.y(t)} y2={geo.y(t)} className="grid" />
            <text x={W - PAD_R + 6} y={geo.y(t) + 4} className="axis">{fmtPrice(t)}</text>
          </g>
        ))}
        {geo.xticks.map((i) => (
          <text key={i} x={geo.x(i)} y={H - 8} className="axis" textAnchor={i === 0 ? "start" : "middle"}>
            {data[i].t.slice(0, years > 2 ? 4 : 7)}
          </text>
        ))}
        <path d={geo.area} fill="url(#histFill)" />
        <path d={geo.line} className="price-line" />
        {forecast && (
          <g>
            <rect x={PAD_L + geo.histW} y={PAD_T} width={geo.plotW - geo.histW} height={H - PAD_T - PAD_B} className="future-zone" />
            <path d={geo.fan} fill="url(#fanFill)" className="fan" />
            <path d={geo.med} className={`median ${up ? "up" : "down"}`} />
            {geo.hs.map((h) => (
              <g key={h.h}>
                <line x1={geo.xf(h.h)} x2={geo.xf(h.h)} y1={geo.y(h.q95)} y2={geo.y(h.q05)} className="hz-tick" />
                <circle cx={geo.xf(h.h)} cy={geo.y(h.q50)} r={3.5} className={`median-dot ${h.ret_q50 >= 0 ? "up" : "down"}`} />
                <text x={geo.xf(h.h)} y={H - 8} className="axis" textAnchor="middle">+{h.h}k</text>
              </g>
            ))}
            <text x={PAD_L + geo.histW + 6} y={PAD_T + 14} className="zone-label">Prognoz</text>
          </g>
        )}
        <circle cx={geo.x(data.length - 1)} cy={geo.y(geo.last)} r={4} className="last-dot" />
        {hover !== null && hover >= 0 && (
          <g>
            <line x1={geo.x(hover)} x2={geo.x(hover)} y1={PAD_T} y2={H - PAD_B} className="cross" />
            <circle cx={geo.x(hover)} cy={geo.y(data[hover].close)} r={3.5} className="last-dot" />
          </g>
        )}
      </svg>
      {hover !== null && (
        <div className="chart-tip">
          {hover >= 0 ? (
            <><span>{data[hover].t}</span><strong>{fmtPrice(data[hover].close)}</strong></>
          ) : forecast ? (
            forecast.horizons.map((h) => (
              <span key={h.h} className="tip-row">
                <em>{h.h} kun</em> {fmtPrice(h.q05)} … <strong>{fmtPrice(h.q50)}</strong> … {fmtPrice(h.q95)}
                <i className={h.ret_q50 >= 0 ? "pos" : "neg"}>{fmtPct(h.ret_q50)}</i>
              </span>
            ))
          ) : null}
        </div>
      )}
    </div>
  );
}
