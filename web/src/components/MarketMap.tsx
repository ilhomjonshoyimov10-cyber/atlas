import { useMemo, useState } from "react";
import { AssetRow, Forecast, fmtPct, fmtPrice, Group, GROUP_NAMES } from "../api";

const ORDER: Group[] = ["crypto", "forex", "index", "equity", "energy", "agri", "metal", "rates"];

interface Props { assets: AssetRow[]; fc: Record<string, Forecast>; onPick: (s: string) => void }

export default function MarketMap({ assets, fc, onPick }: Props) {
  const [h, setH] = useState(5);
  const [sort, setSort] = useState<"group" | "signal">("group");

  const tiles = useMemo(() => assets.map((a) => {
    const f = fc[a.symbol]?.horizons.find((x) => x.h === h);
    // Signal kuchi: median / (oraliq kengligi) — noaniqlikka nisbatan qanchalik aniq yo'nalish bor
    const snr = f ? f.ret_q50 / Math.max(1e-6, (f.ret_q95 - f.ret_q05) / 3.29) : 0;
    return { a, f, snr };
  }), [assets, fc, h]);

  const groups = sort === "group"
    ? ORDER.map((g) => [GROUP_NAMES[g], tiles.filter((t) => t.a.group === g)] as const).filter(([, l]) => l.length)
    : [["Signal kuchi bo'yicha", [...tiles].sort((x, y) => Math.abs(y.snr) - Math.abs(x.snr))] as const];

  const best = [...tiles].filter((t) => t.f).sort((x, y) => y.snr - x.snr);
  return (
    <div className="mmap">
      <div className="mmap-head">
        <p className="lead">Barcha aktivlar uchun model prognozi bitta ekranda. Rang yo'nalishni, to'qligi esa
          signalning noaniqlikka nisbatan kuchini bildiradi.</p>
        <div className="seg" role="group" aria-label="Gorizont">
          {[1, 5, 20].map((x) => (
            <button key={x} className={h === x ? "on" : ""} onClick={() => setH(x)}>{x === 1 ? "Ertaga" : `${x} kun`}</button>
          ))}
        </div>
        <div className="seg" role="group" aria-label="Tartib">
          <button className={sort === "group" ? "on" : ""} onClick={() => setSort("group")}>Guruhlar</button>
          <button className={sort === "signal" ? "on" : ""} onClick={() => setSort("signal")}>Signal kuchi</button>
        </div>
      </div>
      {best.length > 2 && (
        <p className="mmap-summary">
          Eng kuchli o'sish signali: <button className="asset-link" onClick={() => onPick(best[0].a.symbol)}>{best[0].a.name}</button> ({fmtPct(best[0].f!.ret_q50)}),
          {" "}eng kuchli pasayish: <button className="asset-link" onClick={() => onPick(best[best.length - 1].a.symbol)}>{best[best.length - 1].a.name}</button> ({fmtPct(best[best.length - 1].f!.ret_q50)}).
        </p>
      )}
      {groups.map(([title, list]) => (
        <section key={title} className="mmap-group">
          <h3>{title}</h3>
          <div className="tiles">
            {list.map(({ a, f, snr }) => {
              const k = Math.min(1, Math.abs(snr) / 0.6);
              const bg = !f ? "var(--surface)" : snr >= 0
                ? `color-mix(in oklab, var(--up) ${Math.round(8 + k * 60)}%, var(--surface))`
                : `color-mix(in oklab, var(--down) ${Math.round(8 + k * 60)}%, var(--surface))`;
              const strong = k > 0.55;
              return (
                <button key={a.symbol} className={`tile ${strong ? "strong" : ""}`} style={{ background: bg }} onClick={() => onPick(a.symbol)}
                  title={f ? `${a.name}: median ${fmtPct(f.ret_q50)}, 90% oraliq ${fmtPct(f.ret_q05)} … ${fmtPct(f.ret_q95)}` : a.name}>
                  <span className="t-name">{a.name}</span>
                  <span className="t-ret">{f ? fmtPct(f.ret_q50) : "—"}</span>
                  <span className="t-sub">{f ? `${Math.round(f.p_up * 100)}% ehtimol, ${fmtPrice(f.q50)}` : ""}</span>
                </button>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}
