import { fmtPct, fmtPrice, TA } from "../api";

const BIAS: Record<string, { t: string; c: string }> = {
  "ko'tarilish": { t: "Ko'tarilish ustun", c: "pos" },
  pasayish: { t: "Pasayish ustun", c: "neg" },
  neytral: { t: "Aniq ustunlik yo'q", c: "" },
};

export default function TechAnalysis({ ta, onPickLevel }: { ta: TA | null; onPickLevel?: () => void }) {
  if (!ta) return <div className="skeleton" style={{ height: 300 }} />;
  const b = BIAS[ta.bias] ?? BIAS.neytral;
  const sup = ta.levels.filter((l) => l.type === "support");
  const res = ta.levels.filter((l) => l.type === "resistance");
  return (
    <div className="ta">
      <div className="ta-verdict">
        <div>
          <span className="muted">Umumiy xulosa · {ta.as_of}</span>
          <p className={`ta-bias ${b.c}`}>{b.t}</p>
          <span className="muted">ishonch: {ta.confidence} · ball {ta.score >= 0 ? "+" : "−"}{Math.abs(ta.score).toFixed(2)}</span>
        </div>
        <div className="ta-gauge" role="img" aria-label={`Ball ${ta.score.toFixed(2)}, −1 dan +1 gacha`}>
          <span style={{ left: `${(ta.score + 1) * 50}%` }} />
        </div>
      </div>

      <ol className="ta-text">
        {ta.text.map((t, i) => <li key={i}>{t}</li>)}
      </ol>

      <div className="ta-grid">
        <section>
          <h3>Darajalar (SNR)</h3>
          <table className="metrics compact">
            <thead><tr><th scope="col">Daraja</th><th scope="col">Narx</th><th scope="col">Masofa</th><th scope="col">Tegish</th></tr></thead>
            <tbody>
              {[...res].reverse().map((l) => (
                <tr key={`r${l.price}`}><th scope="row" className="neg">Resistance</th><td>{fmtPrice(l.price)}</td><td>{fmtPct(l.dist_pct, 1)}</td><td>{l.touches}</td></tr>
              ))}
              <tr className="ta-now"><th scope="row">Hozir</th><td>{fmtPrice(ta.close)}</td><td>—</td><td /></tr>
              {sup.map((l) => (
                <tr key={`s${l.price}`}><th scope="row" className="pos">Support</th><td>{fmtPrice(l.price)}</td><td>{fmtPct(l.dist_pct, 1)}</td><td>{l.touches}</td></tr>
              ))}
            </tbody>
          </table>
          {onPickLevel && <button className="link-btn" onClick={onPickLevel}>Grafikda ko'rish ↑</button>}
        </section>
        <section>
          <h3>Supply / Demand zonalari</h3>
          {ta.zones.length ? (
            <ul className="ta-zones">
              {ta.zones.map((z, i) => (
                <li key={i} className={z.kind}>
                  <strong>{z.kind === "demand" ? "Demand" : "Supply"}</strong> {fmtPrice(z.lo)} – {fmtPrice(z.hi)}
                  <small>{z.born} · impuls {z.impulse_atr.toFixed(1)}·ATR · {z.tests ? `${z.tests} marta test` : "test qilinmagan"}</small>
                </li>
              ))}
            </ul>
          ) : <p className="muted">Narx yaqinida buzilmagan zona yo'q.</p>}
          <h3>Omillar</h3>
          <ul className="ta-factors">
            {ta.factors.map((f, i) => (
              <li key={i}><span className={f.score > 0.05 ? "pos" : f.score < -0.05 ? "neg" : "muted"}>{f.score > 0.05 ? "▲" : f.score < -0.05 ? "▼" : "●"}</span>
                <strong>{f.name}</strong> <small>{f.note}</small></li>
            ))}
          </ul>
        </section>
      </div>

      <h3>Faol signallar (oxirgi 5 kun)</h3>
      {ta.setups.length ? (
        <div className="table-scroll">
          <table className="metrics">
            <thead><tr><th scope="col">Strategiya</th><th scope="col">Yo'nalish</th><th scope="col">Sana</th>
              <th scope="col" title="Shu strategiya + aktiv guruhi bo'yicha 10+ yillik datasetdagi ulush">Tarixiy yutish</th>
              <th scope="col" title="SetupNet: shu aniq holat uchun maqsad stopdan oldin urilish ehtimoli">Model bahosi</th>
              <th scope="col">Kirish</th><th scope="col">Maqsad</th><th scope="col">Stop</th></tr></thead>
            <tbody>
              {ta.setups.map((s) => (
                <tr key={s.setup} className={s.strong ? "best" : ""}>
                  <th scope="row">{s.label}<span className="grp">{s.family}{s.strong ? " · model ishonchli deb baholadi" : ""}</span></th>
                  <td className={s.dir > 0 ? "pos" : "neg"}>{s.dir > 0 ? "Long ▲" : "Short ▼"}</td>
                  <td>{s.date}</td>
                  <td>{s.base_win != null ? `${(s.base_win * 100).toFixed(0)}%` : "—"}{s.n_hist ? <small className="muted"> ({s.n_hist})</small> : null}</td>
                  <td><strong>{s.p_model != null ? `${(s.p_model * 100).toFixed(0)}%` : "—"}</strong></td>
                  <td>{fmtPrice(s.entry)}</td><td className={s.dir > 0 ? "pos" : "neg"}>{fmtPrice(s.tp)}</td><td className="muted">{fmtPrice(s.sl)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <p className="muted">Oxirgi 5 kunda signal yo'q.</p>}
      <p className="hint">Har bir signal qoidasi: maqsad 1.5·ATR, stop 1·ATR, muddat 20 kun. Bunday nisbatda tasodifiy kirish taxminan
        40% holatda maqsadga yetadi — shuning uchun "tarixiy yutish" va "model bahosi"ni shu tayanch bilan solishtiring. {ta.disclaimer}</p>
    </div>
  );
}
