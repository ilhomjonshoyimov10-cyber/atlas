import { useEffect, useState } from "react";
import { Analysis, api, EVENT_NAMES, fmtPct, GROUP_NAMES, NewsItem, STATIC, timeAgo } from "../api";

const EXAMPLES = [
  "OPEC agrees surprise output cut as Saudi pushes for higher crude prices",
  "Severe drought threatens wheat harvest in Black Sea region",
  "Fed signals rate cut as inflation cools, Powell says",
  "SEC approves spot bitcoin ETF in landmark crypto decision",
];

interface Props { seed: { id?: string; text: string } | null; news: NewsItem[]; onPick: (s: string) => void }

export default function WhatIf({ seed, news, onPick }: Props) {
  const [text, setText] = useState(seed?.text ?? "");
  const [picked, setPicked] = useState<string | null>(null);
  const [res, setRes] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async (t = text) => {
    if (t.trim().length < 8) { setErr("Kamida bitta to'liq sarlavha yozing (inglizcha)."); return; }
    setBusy(true); setErr(null);
    try { setRes(await api.analyze(t)); } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  const runNews = async (n: { id?: string; text: string }) => {
    if (!n.id) return;
    setBusy(true); setErr(null); setPicked(n.id); setText(n.text);
    try { setRes(await api.analyzeNews(n.id)); }
    catch { setErr("Bu xabar uchun tahlil hali tayyor emas — u keyingi yangilanishda (≈30 daqiqa) hisoblanadi."); setRes(null); }
    finally { setBusy(false); }
  };
  useEffect(() => {
    if (!seed) return;
    if (STATIC) runNews(seed); else { setText(seed.text); run(seed.text); }
  }, [seed]); // eslint-disable-line

  const top = res?.impact.slice(0, 12) ?? [];
  const maxAbs = Math.max(0.001, ...top.map((r) => Math.abs(r.s_ret_h1 as number)));
  return (
    <div className="whatif">
      {STATIC ? (
        <>
          <p className="lead">Lentadagi xabarni tanlang — model uni bugungi yangiliklar oqimiga qo'shib, har bir aktivning
            bashorati qanchaga siljishini ko'rsatadi. Har bir yangi xabar avtomatik baholanadi.</p>
          <ul className="wi-pick">
            {news.slice(0, 12).map((n) => (
              <li key={n.id}>
                <button className={`wi-item ${picked === n.id ? "on" : ""}`} onClick={() => runNews({ id: n.id, text: n.title })}>
                  <span>{n.title}</span><small>{n.source.replace(/_/g, " ")} · {timeAgo(n.first_seen)}</small>
                </button>
              </li>
            ))}
          </ul>
          <p className="hint">O'zingiz yozgan matnni tahlil qilish lokal versiyada ishlaydi: <code>python -m atlas.cli serve</code>.</p>
        </>
      ) : (
      <form onSubmit={(e) => { e.preventDefault(); run(); }} className="wi-form">
        <label htmlFor="wi-text" className="sr-only">Yangilik matni</label>
        <textarea id="wi-text" rows={3} value={text} onChange={(e) => setText(e.target.value)}
          placeholder="Masalan: Fed raises rates and warns of further hikes" />
        <div className="wi-actions">
          <button type="submit" className="btn" disabled={busy}>{busy ? "Hisoblanmoqda…" : "Ta'sirini hisoblash"}</button>
          <div className="examples">
            {EXAMPLES.map((ex) => (
              <button type="button" key={ex} className="chip ghost" onClick={() => { setText(ex); run(ex); }}>{ex.split(" ").slice(0, 4).join(" ")}…</button>
            ))}
          </div>
        </div>
      </form>
      )}
      {busy && STATIC && <div className="skeleton" style={{ height: 200 }} />}
      {err && <p className="error" role="alert">{err}</p>}
      {res && (
        <div className="wi-result">
          {STATIC && <p className="wi-headline">“{res.text}”</p>}
          <div className="wi-tags">
            <span>Voqea turi: <strong>{EVENT_NAMES[res.event] ?? res.event}</strong></span>
            {res.tone !== undefined && <span>Ton: <strong className={Math.abs(res.tone) < 0.05 ? "" : res.tone > 0 ? "pos" : "neg"}>
              {Math.abs(res.tone) < 0.05 ? "neytral" : res.tone > 0 ? "ijobiy" : "salbiy"} ({res.tone.toFixed(2)})</strong></span>}
            <span>Tegishli aktivlar: <strong>{res.assets.length ? res.assets.join(", ") : "aniqlanmadi — umumiy fon sifatida"}</strong></span>
          </div>
          {res.common && (
            <p className="hint">Ta'sir neytral xabar bilan solishtirib o'lchanadi, so'ng barcha aktivlarga umumiy siljish
              (ertaga {fmtPct(res.common.h1, 3)}) ayiriladi — jadvalda aktivning aynan shu xabarga o'ziga xos javobi.</p>
          )}
          {res.note ? <p className="muted">{res.note}</p> : (
            <table className="impact">
              <thead><tr><th scope="col">Aktiv</th><th scope="col">Ertangi ta'sir</th><th scope="col">5 kun</th><th scope="col">20 kun</th><th scope="col" title="Volatillik birligida (σ)">Kuchi, σ</th></tr></thead>
              <tbody>
                {top.map((r) => {
                  const v = r.s_ret_h1 as number;
                  return (
                    <tr key={r.symbol}>
                      <th scope="row"><button className="asset-link" onClick={() => onPick(r.symbol)}>{r.name}</button>
                        <span className="grp">{GROUP_NAMES[r.group]}{r.tagged ? " · tilga olingan" : ""}</span></th>
                      <td>
                        <div className="impact-bar"><span className={v >= 0 ? "up" : "down"}
                          style={{ width: `${(Math.abs(v) / maxAbs) * 50}%`, [v >= 0 ? "left" : "right"]: "50%" } as React.CSSProperties} /></div>
                        <span className={v >= 0 ? "pos" : "neg"}>{fmtPct(v, 3)}</span>
                      </td>
                      <td className={(r.s_ret_h5 as number) >= 0 ? "pos" : "neg"}>{fmtPct(r.s_ret_h5 as number, 3)}</td>
                      <td className={(r.s_ret_h20 as number) >= 0 ? "pos" : "neg"}>{fmtPct(r.s_ret_h20 as number, 3)}</td>
                      <td className="muted">{(r.sigma_h1 as number) >= 0 ? "+" : "−"}{Math.abs(r.sigma_h1 as number).toFixed(2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
          <p className="hint">Bu model sezgirligi: bitta xabar odatda kichik siljish beradi. Natija investitsiya tavsiyasi emas.</p>
        </div>
      )}
    </div>
  );
}
