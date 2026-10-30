import { useEffect, useMemo, useState } from "react";
import { api, AssetRow, fmtPrice, Strategies as S } from "../api";

// O'zbekcha nomlar va sinonimlar -> aktiv belgisi ("menga oltinni analiz qil")
const ALIASES: Record<string, string> = {
  oltin: "GOLD", tilla: "GOLD", gold: "GOLD", kumush: "SILVER", mis: "COPPER", neft: "CL", oil: "CL", gaz: "NG",
  "bug'doy": "WHEAT", bugdoy: "WHEAT", "makkajo'xori": "CORN", soya: "SOY", qahva: "COFFEE", shakar: "SUGAR",
  kakao: "COCOA", paxta: "COTTON", bitkoin: "BTC", bitcoin: "BTC", efir: "ETH", ethereum: "ETH", yevro: "EURUSD",
  euro: "EURUSD", funt: "GBPUSD", iyena: "USDJPY", dollar: "DXY", apple: "AAPL", tesla: "TSLA", nvidia: "NVDA",
};

export function resolveAsset(q: string, assets: AssetRow[]): string | null {
  const t = q.toLowerCase().replace(/[^a-z0-9'ʻ’ ]/g, " ").replace(/[ʻ’]/g, "'");
  for (const w of t.split(/\s+/)) {
    if (!w) continue;
    const a = assets.find((x) => x.symbol.toLowerCase() === w);
    if (a) return a.symbol;
    for (const [k, v] of Object.entries(ALIASES)) if (w.startsWith(k)) return v;
    const byName = assets.find((x) => x.name.toLowerCase().startsWith(w) && w.length >= 3);
    if (byName) return byName.symbol;
  }
  return null;
}

const pct = (v?: number | null, d = 1) => (v == null || !isFinite(v) ? "—" : `${v.toFixed(d)}%`);
const r2 = (v?: number | null) => (v == null || !isFinite(v) ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`);

export default function Strategies({ assets, onAnalyze }: { assets: AssetRow[]; onAnalyze: (sym: string) => void }) {
  const [d, setD] = useState<S | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => { api.strategies().then(setD).catch((e) => setErr(e.message)); }, []);
  const hold = useMemo(() => Object.fromEntries((d?.stats_holdout ?? []).map((r) => [r.setup, r])), [d]);
  const per = useMemo(() => Object.fromEntries((d?.per_setup_holdout ?? []).map((r) => [r.setup, r])), [d]);

  const go = (e: React.FormEvent) => {
    e.preventDefault();
    const s = resolveAsset(q, assets);
    if (s) { setMsg(null); onAnalyze(s); } else setMsg("Aktiv topilmadi. Masalan: «oltin», «BTC», «bug'doy», «EURUSD».");
  };

  return (
    <div className="strat">
      <h1>Strategiyalar va texnik tahlil</h1>
      <form className="ask" onSubmit={go}>
        <label htmlFor="ask" className="sr-only">Qaysi aktivni tahlil qilay?</label>
        <input id="ask" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Menga analiz qilib ber: oltin, BTC, bug'doy…" />
        <button className="btn" type="submit">Analiz qil</button>
      </form>
      {msg && <p className="error" role="alert">{msg}</p>}
      {err && <p className="muted">Hisobot yuklanmadi: {err}</p>}
      {!d && !err && <div className="skeleton" style={{ height: 300 }} />}
      {d && d.ready && (
        <>
          <p className="lead">{d.dataset.assets} ta aktivning {d.dataset.from} — {d.dataset.to} tarixidan <strong>{d.dataset.n_events.toLocaleString()}</strong> ta
            strategiya signali yig'ildi: SNR (support/resistance), SND zonalari, bozor tuzilmasi (BOS), klassik figuralar, indikatorlar va
            sham figuralari. Har bir signal bir xil qoida bilan baholandi: maqsad {d.rules.tp_atr}·ATR, stop {d.rules.sl_atr}·ATR,
            muddat {d.rules.horizon} kun. Signal natijasi o'sha aktivda shu davrda <em>har kuni kirish</em> natijasi bilan solishtiriladi.</p>

          <div className="track-cards">
            <div className="hz">
              <h3>SetupNet · AUC (model ko'rmagan yil)</h3>
              <p className="hz-mid">{d.auc.toFixed(3)}</p>
              <p className="hz-range">faqat strategiya turi bo'yicha tayanch: {d.auc_baseline.toFixed(3)} · 0.5 = tasodif</p>
            </div>
            <div className="hz">
              <h3>Model eng ishonchli deb topgan 30%</h3>
              <p className="hz-mid pos">{pct(d.filter.top30.win_rate)}</p>
              <p className="hz-range">maqsadga yetdi (barcha signallar: {pct(d.filter.all.win_rate)}, eng kuchsiz 30%: {pct(d.filter.bottom30.win_rate)})</p>
            </div>
            <div className="hz">
              <h3>O'rtacha natija, R</h3>
              <p className="hz-mid">{r2(d.filter.top30.avg_R)}</p>
              <p className="hz-range">ishonchli 30% · hammasi {r2(d.filter.all.avg_R)} · kuchsiz 30% {r2(d.filter.bottom30.avg_R)}</p>
            </div>
          </div>
          <p className="hint">Holdout: {d.holdout_start} — {d.holdout_end} ({d.n_holdout.toLocaleString()} signal). Model bu davrni o'qitishda ko'rmagan;
            "ishonchli" chegarasi ham oldingi davrda tanlangan.</p>

          {d.today && d.today.length > 0 && (
            <>
              <h3>Bugungi signallar</h3>
              <div className="table-scroll">
                <table className="metrics">
                  <thead><tr><th scope="col">Aktiv</th><th scope="col">Strategiya</th><th scope="col">Yo'nalish</th><th scope="col">Tarixiy</th>
                    <th scope="col">Model</th><th scope="col">Kirish</th><th scope="col">Maqsad</th><th scope="col">Stop</th></tr></thead>
                  <tbody>
                    {d.today.slice(0, 15).map((r, i) => (
                      <tr key={i} className={r.strong ? "best" : ""}>
                        <th scope="row"><button className="asset-link" onClick={() => onAnalyze(r.symbol)}>{r.name}</button></th>
                        <td>{r.label}</td><td className={r.dir > 0 ? "pos" : "neg"}>{r.dir > 0 ? "Long ▲" : "Short ▼"}</td>
                        <td>{r.base_win != null ? `${(r.base_win * 100).toFixed(0)}%` : "—"}</td>
                        <td><strong>{r.p_model != null ? `${(r.p_model * 100).toFixed(0)}%` : "—"}</strong></td>
                        <td>{fmtPrice(r.entry)}</td><td>{fmtPrice(r.tp)}</td><td className="muted">{fmtPrice(r.sl)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          <h3>Har bir strategiya natijasi</h3>
          <div className="table-scroll">
            <table className="metrics">
              <thead><tr><th scope="col">Strategiya</th><th scope="col">Signallar</th><th scope="col">Yutish</th>
                <th scope="col" title="Signal R − shu davrda har kuni kirish R">Ustunlik, R</th>
                <th scope="col" title="Holm tuzatishi bilan: 0.05 dan kichik bo'lsa tasodif emas">p (Holm)</th>
                <th scope="col">Oxirgi yil: yutish</th><th scope="col">Oxirgi yil: model ishonchli deganlari</th></tr></thead>
              <tbody>
                {[...d.stats_train].filter((r) => r.n >= 20).sort((a, b) => (b.excess_R ?? -9) - (a.excess_R ?? -9)).map((r) => {
                  const h = hold[r.setup], p = per[r.setup];
                  const sig = (r.p_holm ?? 1) < 0.05;
                  return (
                    <tr key={r.setup}>
                      <th scope="row">{r.name}<span className="grp">{r.family} · {r.dir > 0 ? "long" : "short"}</span></th>
                      <td>{r.n.toLocaleString()}</td><td>{pct(r.win_rate)}</td>
                      <td className={sig ? ((r.excess_R ?? 0) > 0 ? "pos" : "neg") : "muted"}>{r2(r.excess_R)}</td>
                      <td className={sig ? "" : "muted"}>{r.p_holm != null ? r.p_holm.toFixed(3) : "—"}</td>
                      <td>{pct(h?.win_rate)}</td>
                      <td>{p?.win_top != null ? `${pct(p.win_top)} (${p.n_top})` : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="hint">Rangli "Ustunlik" — statistik ahamiyatli (ko'p strategiya sinalgani uchun Holm tuzatishi bilan). Kulrang — tasodifdan
            farqlab bo'lmaydi. Ko'p mashhur strategiyalar o'z-o'zidan ustunlik bermaydi; foyda modelning signallarni saralashidan keladi.
            Bu tadqiqot natijasi, investitsiya tavsiyasi emas.</p>
        </>
      )}
    </div>
  );
}
