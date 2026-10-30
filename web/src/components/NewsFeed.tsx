import { EVENT_NAMES, fmtPct, NewsItem, QuickImpact, timeAgo } from "../api";

function parseImpact(s?: string | null): QuickImpact[] {
  if (!s) return [];
  try { return JSON.parse(s); } catch { return []; }
}

interface Props {
  items: NewsItem[]; fresh: Set<string>; onPick: (sym: string) => void; onAnalyze: (n: NewsItem) => void;
}

export default function NewsFeed({ items, fresh, onPick, onAnalyze }: Props) {
  if (!items.length) {
    return (
      <div className="empty">
        <p>Hozircha yangilik yo'q.</p>
        <p className="muted">Kollektor manbalarni har daqiqada tekshiradi — birinchi xabar kelishi bilan shu yerda paydo bo'ladi.</p>
      </div>
    );
  }
  return (
    <ol className="news">
      {items.map((n) => {
        const assets = (n.assets || "").split(",").filter(Boolean);
        const imp = parseImpact(n.impact).filter((r) => Math.abs(r.sigma_h1) >= 0.1).slice(0, 3);
        return (
          <li key={n.id} className={fresh.has(n.id) ? "fresh" : ""}>
            <div className="news-meta">
              <span className="src">{n.source.replace(/_/g, " ")}</span>
              <time dateTime={n.first_seen} title={`Birinchi ko'rilgan: ${new Date(n.first_seen).toLocaleString()}`}>{timeAgo(n.first_seen)}</time>
              {n.event && n.event !== "none" && <span className={`ev ev-${n.event}`}>{EVENT_NAMES[n.event] ?? n.event}</span>}
            </div>
            <a href={n.url || undefined} target="_blank" rel="noreferrer" className="news-title">{n.title}</a>
            {imp.length > 0 && (
              <div className="news-impact" aria-label="Model bahosi: ertangi o'ziga xos ta'sir">
                {imp.map((r) => (
                  <button key={r.symbol} className={`imp ${r.ret_h1 >= 0 ? "up" : "down"}`} onClick={() => onPick(r.symbol)}
                    title={`${r.name}: ertaga ${fmtPct(r.ret_h1, 3)}, 5 kunda ${fmtPct(r.ret_h5, 3)} (${r.sigma_h1 >= 0 ? "+" : ""}${r.sigma_h1}σ)`}>
                    {r.name} {r.ret_h1 >= 0 ? "↑" : "↓"}{Math.abs(r.ret_h1).toFixed(2)}%
                  </button>
                ))}
              </div>
            )}
            <div className="news-foot">
              {assets.slice(0, 6).map((a) => (
                <button key={a} className="chip" onClick={() => onPick(a)}>{a}</button>
              ))}
              <button className="link-btn" onClick={() => onAnalyze(n)}>Ta'sirini hisoblash</button>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
