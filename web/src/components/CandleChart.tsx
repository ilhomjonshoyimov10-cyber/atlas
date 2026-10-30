import { useEffect, useRef } from "react";
import { createChart, CrosshairMode, IChartApi, IPriceLine, ISeriesApi, LineStyle, Time } from "lightweight-charts";
import { Bar, Forecast, TA } from "../api";

/** TradingView Lightweight Charts asosidagi sham grafigi:
 *  tarix (kunlik shamlar) + jonli narx (oxirgi sham real vaqtda) + ATLAS prognozi (q05/q50/q95)
 *  + texnik tahlil qatlamlari (support/resistance va supply/demand zonalari). */
interface Props { bars: Bar[]; forecast: Forecast | null; years: number; ta?: TA | null; live?: number | null }

const css = (v: string, fb: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || fb;

function addTradingDays(iso: string, n: number): string {
  const d = new Date(iso + "T00:00:00Z");
  let k = 0;
  while (k < n) { d.setUTCDate(d.getUTCDate() + 1); if (d.getUTCDay() % 6 !== 0) k++; }
  return d.toISOString().slice(0, 10);
}

export default function CandleChart({ bars, forecast, years, ta, live }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const candles = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const fcSeries = useRef<ISeriesApi<"Line">[]>([]);
  const lines = useRef<IPriceLine[]>([]);
  const liveBar = useRef<{ time: string; open: number; high: number; low: number; close: number } | null>(null);

  // grafikni bir marta yaratamiz
  useEffect(() => {
    if (!box.current) return;
    const ink = css("--ink", "#1b2429"), muted = css("--muted", "#5b696c"), line = css("--line", "#d3d9d1");
    const c = createChart(box.current, {
      autoSize: true,
      layout: { background: { color: "transparent" }, textColor: muted, fontFamily: "Onest, system-ui, sans-serif" },
      grid: { vertLines: { color: line }, horzLines: { color: line } },
      rightPriceScale: { borderColor: line },
      timeScale: { borderColor: line, rightOffset: 4 },
      crosshair: { mode: CrosshairMode.Normal },
    });
    const up = css("--up", "#1e7a57"), down = css("--down", "#b3452f");
    candles.current = c.addCandlestickSeries({ upColor: up, downColor: down, borderVisible: false, wickUpColor: up, wickDownColor: down });
    const fcCol = css("--fc", "#3b4cca");
    fcSeries.current = [
      c.addLineSeries({ color: fcCol, lineWidth: 1, lineStyle: LineStyle.Dotted, priceLineVisible: false, lastValueVisible: false, title: "q95" }),
      c.addLineSeries({ color: fcCol, lineWidth: 2, lineStyle: LineStyle.Dashed, priceLineVisible: false, lastValueVisible: true, title: "prognoz" }),
      c.addLineSeries({ color: fcCol, lineWidth: 1, lineStyle: LineStyle.Dotted, priceLineVisible: false, lastValueVisible: false, title: "q05" }),
    ];
    void ink;
    chart.current = c;
    return () => { c.remove(); chart.current = null; };
  }, []);

  // tarix va prognoz
  useEffect(() => {
    const s = candles.current, c = chart.current;
    if (!s || !c || !bars.length) return;
    s.setData(bars.map((b) => ({ time: b.t as Time, open: b.open ?? b.close, high: b.high ?? b.close, low: b.low ?? b.close, close: b.close })));
    liveBar.current = null;
    const last = bars[bars.length - 1];
    if (forecast) {
      const pts = [{ h: 0, q05: last.close, q50: last.close, q95: last.close }, ...forecast.horizons];
      const keys = ["q95", "q50", "q05"] as const;
      fcSeries.current.forEach((ser, i) => ser.setData(pts.map((p) => ({ time: addTradingDays(last.t, p.h) as Time, value: p[keys[i]] }))));
    } else fcSeries.current.forEach((ser) => ser.setData([]));
    const from = new Date(); from.setFullYear(from.getFullYear() - years);
    const end = forecast ? addTradingDays(last.t, Math.max(...forecast.horizons.map((h) => h.h))) : last.t;
    try { c.timeScale().setVisibleRange({ from: from.toISOString().slice(0, 10) as Time, to: end as Time }); }
    catch { c.timeScale().fitContent(); }
  }, [bars, forecast, years]);

  // texnik tahlil qatlamlari
  useEffect(() => {
    const s = candles.current;
    if (!s) return;
    lines.current.forEach((l) => s.removePriceLine(l));
    lines.current = [];
    if (!ta) return;
    const up = css("--up", "#1e7a57"), down = css("--down", "#b3452f");
    ta.levels.forEach((l) => lines.current.push(s.createPriceLine({
      price: l.price, color: l.type === "support" ? up : down, lineWidth: 1, lineStyle: LineStyle.Dashed,
      axisLabelVisible: true, title: `${l.type === "support" ? "S" : "R"} ×${l.touches}`,
    })));
    ta.zones.forEach((z) => [z.lo, z.hi].forEach((p, i) => lines.current.push(s.createPriceLine({
      price: p, color: z.kind === "demand" ? up : down, lineWidth: 2, lineStyle: LineStyle.Dotted,
      axisLabelVisible: false, title: i ? (z.kind === "demand" ? "Demand" : "Supply") : "",
    }))));
  }, [ta, bars]);

  // jonli narx: bugungi sham real vaqtda yangilanadi
  useEffect(() => {
    const s = candles.current;
    if (!s || !bars.length || live == null || !isFinite(live)) return;
    const last = bars[bars.length - 1];
    const today = new Date().toISOString().slice(0, 10);
    if (today < last.t) return;
    if (today === last.t) {
      const b = { time: last.t, open: last.open ?? last.close, high: Math.max(last.high ?? last.close, live), low: Math.min(last.low ?? last.close, live), close: live };
      s.update({ ...b, time: b.time as Time });
      return;
    }
    const prev = liveBar.current && liveBar.current.time === today ? liveBar.current
      : { time: today, open: last.close, high: last.close, low: last.close, close: last.close };
    const b = { ...prev, high: Math.max(prev.high, live), low: Math.min(prev.low, live), close: live };
    liveBar.current = b;
    s.update({ ...b, time: b.time as Time });
  }, [live, bars]);

  return (
    <div className="candle-wrap">
      <div ref={box} className="candle-chart" role="img" aria-label="Sham grafigi, prognoz va texnik darajalar" />
      <span className="candle-legend">
        <i className="lg-fc" /> ATLAS prognozi (q05–q50–q95)
        {ta && <><i className="lg-s" /> support <i className="lg-r" /> resistance</>}
        {live != null && <><i className="lg-live" /> oxirgi sham jonli</>}
      </span>
    </div>
  );
}
