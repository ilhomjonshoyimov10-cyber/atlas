import { useEffect, useRef, useState } from "react";
import { api, fmtPrice, GoldSignal, NewsSignal, Signals, timeAgo } from "../api";

const TF_ORDER = ["5m", "15m", "30m", "1h", "4h"];
const STATUS: Record<string, { t: string; c: string }> = {
  maqsad: { t: "✓ maqsadga yetdi", c: "pos" }, stop: { t: "✗ stop", c: "neg" },
  "muddati o'tdi": { t: "muddati o'tdi", c: "muted" }, ochiq: { t: "ochiq", c: "" },
};

interface Live { title: string; message: string; time: number }

function useNtfy(topic: string | undefined, url: string | undefined, enabled: boolean, onMsg: (m: Live) => void) {
  useEffect(() => {
    if (!topic || !url || !enabled) return;
    const es = new EventSource(`${url}/${topic}/sse`);
    es.onmessage = (e) => {
      try {
        const d = JSON.parse(e.data);
        if (d.event !== "message") return;
        const m = { title: d.title ?? "ATLAS", message: d.message ?? "", time: (d.time ?? Date.now() / 1000) * 1000 };
        onMsg(m);
        if ("Notification" in window && Notification.permission === "granted") {
          new Notification(m.title, { body: m.message, tag: d.id });
        }
      } catch { /* noto'g'ri xabar */ }
    };
    return () => es.close();
  }, [topic, url, enabled]); // eslint-disable-line
}

export default function SignalsPage({ onPick }: { onPick: (s: string) => void }) {
  const [d, setD] = useState<Signals | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [perm, setPerm] = useState<string>(() => ("Notification" in window ? Notification.permission : "unsupported"));
  const [listen, setListen] = useState<boolean>(() => { try { return localStorage.getItem("ntfy") === "1"; } catch { return false; } });
  const [live, setLive] = useState<Live[]>([]);
  const t = useRef<number>();

  useEffect(() => {
    const load = () => api.signals().then(setD).catch((e) => setErr(e.message));
    load();
    t.current = window.setInterval(load, 60000);
    return () => window.clearInterval(t.current);
  }, []);
  useNtfy(d?.topic, d?.ntfy_url, listen, (m) => setLive((o) => [m, ...o].slice(0, 20)));

  const enable = async () => {
    let p = perm;
    if ("Notification" in window && Notification.permission === "default") p = await Notification.requestPermission();
    setPerm(p);
    setListen(true);
    try { localStorage.setItem("ntfy", "1"); } catch { /* yo'q */ }
  };
  const disable = () => { setListen(false); try { localStorage.setItem("ntfy", "0"); } catch { /* yo'q */ } };

  if (err) return <p className="muted">Signallar yuklanmadi: {err}</p>;
  if (!d) return <div className="skeleton" style={{ height: 360 }} />;
  const sub = `${d.ntfy_url}/${d.topic}`;
  return (
    <div className="sig">
      <h1>Jonli signallar</h1>
      <p className="lead">Yangi xabar chiqishi bilan model uning har bir aktivga ta'sirini hisoblaydi; ta'sir kuchli bo'lsa yoki muhim makro voqea
        (Fed, CPI, NFP, OPEC…) bo'lsa — xabarnoma keladi. Oltin uchun 5m, 15m, 30m, 1h va 4h grafiklarda strategiya signallari tekshiriladi va
        faqat model ko'rmagan davrda tasodifiy kirishdan <strong>ishonchli yaxshi</strong> chiqqan vaqt oraliqlarida push yuboriladi.</p>

      <div className="sig-sub">
        <div>
          <h3>Xabarnomalar</h3>
          {listen ? (
            <p><span className="dot ok" /> Yoqilgan — sayt ochiq turganda brauzer xabarnomasi keladi.
              {perm === "denied" && " Brauzer xabarnomaga ruxsat bermagan: manzil satridagi qulf belgisidan ruxsat bering."}
              {" "}<button className="link-btn" onClick={disable}>O'chirish</button></p>
          ) : (
            <button className="btn" onClick={enable}>Brauzer xabarnomalarini yoqish</button>
          )}
          <p className="hint">Sayt yopiq bo'lsa ham kelishi uchun: <a href={sub} target="_blank" rel="noreferrer">{sub.replace("https://", "")}</a> ni
            oching → <em>Subscribe</em> → brauzer xabarnomalariga ruxsat bering. Telefonda: <strong>ntfy</strong> ilovasini o'rnatib (Android / iOS,
            bepul), <code>{d.topic}</code> mavzusiga obuna bo'ling.</p>
        </div>
        <div>
          <h3>Jonli natija</h3>
          <p className="sig-big">{d.live.closed ? `${Math.round((d.live.tp / d.live.closed) * 100)}%` : "—"}</p>
          <p className="hint">{d.live.strong} ta kuchli signal berildi, {d.live.closed} tasi yopildi: {d.live.tp} maqsad, {d.live.sl} stop.
            Jurnal o'zgartirilmaydi va tanlab olinmaydi.</p>
          {d.engine && <p className="hint">{d.engine.running ? `Dvigatel ishlayapti${d.engine.last_run ? `, oxirgi tekshiruv ${timeAgo(d.engine.last_run)}` : ""}` : "Dvigatel hozir ishlamayapti"}</p>}
        </div>
      </div>

      <h3>Oltin: vaqt oraliqlari bo'yicha model holati</h3>
      <div className="table-scroll">
        <table className="metrics">
          <thead><tr><th scope="col">Oraliq</th><th scope="col">Holat</th><th scope="col">Kuchli signallar aniqligi</th>
            <th scope="col">90% ishonch oralig'i</th><th scope="col">Tasodifiy kirish</th><th scope="col">Sinovdagi signallar</th></tr></thead>
          <tbody>
            {TF_ORDER.map((tf) => {
              const s = d.timeframes[tf];
              if (!s) return null;
              return (
                <tr key={tf} className={s.enabled ? "best" : ""}>
                  <th scope="row">{tf}</th>
                  <td className={s.enabled ? "pos" : "muted"}>{s.enabled ? "Push yoqilgan" : s.reason ?? "Kuzatuvda (ustunlik isbotlanmadi)"}</td>
                  <td>{s.holdout_precision != null ? `${s.holdout_precision.toFixed(0)}%` : "—"}</td>
                  <td className="muted">{s.holdout_ci90 ? `${s.holdout_ci90[0].toFixed(0)}–${s.holdout_ci90[1].toFixed(0)}%` : "—"}</td>
                  <td>{s.base_win != null ? `${s.base_win.toFixed(0)}%` : "—"}</td>
                  <td>{s.holdout_signals ?? 0}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="hint">Qoida: maqsad {d.rules.tp_atr}·ATR, stop {d.rules.sl_atr}·ATR, muddat {d.rules.horizon_bars} bar. "Push yoqilgan" faqat
        model ko'rmagan davrda kuchli signallar aniqligining pastki chegarasi tasodifiy kirishdan yuqori bo'lgan oraliqlarda. 99% aniqlik halol
        o'lchovda bozorda uchramaydi — shuning uchun bu yerda faqat o'lchangan raqamlar ko'rsatiladi. Investitsiya tavsiyasi emas.</p>

      {live.length > 0 && (
        <>
          <h3>Hozirgina kelgan</h3>
          <ul className="sig-list">
            {live.map((m, i) => <li key={i} className="fresh"><strong>{m.title}</strong><span>{m.message}</span><small>{timeAgo(new Date(m.time).toISOString())}</small></li>)}
          </ul>
        </>
      )}

      <h3>So'nggi signallar</h3>
      {d.recent.length === 0 ? <p className="muted">Hozircha kuchli signal yo'q — bu normal: kuchli holat kamdan-kam bo'ladi.</p> : (
        <ul className="sig-list">
          {d.recent.map((r, i) => r.kind === "gold" ? <GoldRow key={i} s={r} onPick={onPick} /> : <NewsRow key={i} n={r} onPick={onPick} />)}
        </ul>
      )}

      {d.watch.length > 0 && (
        <>
          <h3>Kuzatuvdagi (push yuborilmagan) oltin signallari</h3>
          <div className="table-scroll">
            <table className="metrics compact">
              <thead><tr><th scope="col">Vaqt (UTC)</th><th scope="col">Oraliq</th><th scope="col">Strategiya</th><th scope="col">Yo'nalish</th><th scope="col">Model</th><th scope="col">Natija</th></tr></thead>
              <tbody>
                {d.watch.map((s, i) => (
                  <tr key={i}><td>{s.time.slice(5, 16)}</td><td>{s.tf}</td><td>{s.label}</td>
                    <td className={s.dir > 0 ? "pos" : "neg"}>{s.dir > 0 ? "Buy" : "Sell"}</td><td>{(s.p * 100).toFixed(0)}%</td>
                    <td className={STATUS[s.status ?? "ochiq"]?.c}>{STATUS[s.status ?? "ochiq"]?.t}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function GoldRow({ s, onPick }: { s: GoldSignal; onPick: (x: string) => void }) {
  const st = STATUS[s.status ?? "ochiq"] ?? STATUS.ochiq;
  return (
    <li className={`sig-gold ${s.dir > 0 ? "up" : "down"}`}>
      <strong><button className="asset-link" onClick={() => onPick("GOLD")}>Oltin</button> {s.dir > 0 ? "BUY ▲" : "SELL ▼"} · {s.tf} · {s.label}</strong>
      <span>Kirish {fmtPrice(s.entry)} · maqsad {fmtPrice(s.tp)} · stop {fmtPrice(s.sl)} · model {(s.p * 100).toFixed(0)}%
        {s.measured_precision != null && ` · o'lchangan aniqlik ${s.measured_precision.toFixed(0)}%`}</span>
      {s.news && <span className="muted">Yangilik foni: {s.news}</span>}
      <small>{timeAgo(s.created)} · <span className={st.c}>{st.t}</span></small>
    </li>
  );
}

function NewsRow({ n, onPick }: { n: NewsSignal; onPick: (x: string) => void }) {
  return (
    <li className="sig-news">
      <strong>{n.title}</strong>
      <span>{n.impact.map((x, i) => (
        <button key={i} className={`imp ${x.ret_h1 >= 0 ? "up" : "down"}`} onClick={() => onPick(x.symbol)}>
          {x.name} {x.ret_h1 >= 0 ? "↑" : "↓"}{Math.abs(x.ret_h1).toFixed(3)}%
        </button>
      ))}</span>
      <small>{n.source?.replace(/_/g, " ")} · {timeAgo(n.first_seen ?? n.created)}{n.event !== "none" ? ` · ${n.event.toUpperCase()}` : ""}</small>
    </li>
  );
}
