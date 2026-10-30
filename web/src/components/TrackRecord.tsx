import { useEffect, useMemo, useState } from "react";
import { api, fmtPct, fmtPrice, Track } from "../api";

const H_NAMES: Record<number, string> = { 1: "Ertangi", 5: "5 kunlik", 20: "20 kunlik" };

export default function TrackRecord({ onPick }: { onPick: (s: string) => void }) {
  const [d, setD] = useState<Track | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [h, setH] = useState(1);
  useEffect(() => { api.track().then(setD).catch((e) => setErr(e.message)); }, []);
  const rows = useMemo(() => (d?.recent ?? []).filter((r) => r.h === h).slice(0, 80), [d, h]);

  if (err) return <p className="muted">Jurnal yuklanmadi: {err}</p>;
  if (!d) return <div className="skeleton" style={{ height: 360 }} />;
  return (
    <div className="track">
      <h1>Prognoz jurnali</h1>
      <p className="lead">Model har kuni barcha aktivlar uchun prognoz beradi va u shu yerga yozib qo'yiladi. Muddati kelgach
        prognoz haqiqiy narx bilan solishtiriladi. Natijalar tanlab olinmaydi va keyin o'zgartirilmaydi — bu modelning
        ochiq imtihoni.{d.started && <> Jurnal {d.started} dan beri yuritilmoqda.</>}</p>

      {d.summary.length === 0 ? (
        <div className="empty">
          <p><strong>Hali muddati kelgan prognoz yo'q.</strong></p>
          <p className="muted">{d.pending} ta prognoz kutilmoqda. Ertangi prognozlar natijasi keyingi savdo kuni yopilgandan so'ng
            shu yerda paydo bo'ladi.</p>
        </div>
      ) : (
        <div className="track-cards">
          {d.summary.map((s) => (
            <div key={s.h} className="hz">
              <h3>{H_NAMES[s.h] ?? `${s.h} kunlik`} · {s.n} ta</h3>
              <p className="hz-mid">{s.coverage90.toFixed(0)}%</p>
              <p className="hz-range">haqiqiy narx 90% oraliqqa tushdi (maqsad ≈90%)</p>
              <p className="hz-p">Yo'nalish to'g'ri: <strong>{s.direction === null ? "—" : `${s.direction.toFixed(0)}%`}</strong>
                {" "}(tanga tashlash = 50%) · o'rtacha xato {s.mae_pct.toFixed(2)}%</p>
            </div>
          ))}
        </div>
      )}

      {d.recent.length > 0 && (
        <>
          <div className="ranges" role="group" aria-label="Gorizont">
            {[1, 5, 20].map((x) => (
              <button key={x} className={h === x ? "on" : ""} onClick={() => setH(x)}>{H_NAMES[x]}</button>
            ))}
          </div>
          <div className="table-scroll">
            <table className="metrics track-table">
              <thead><tr><th scope="col">Natija sanasi</th><th scope="col">Aktiv</th><th scope="col">Prognoz</th>
                <th scope="col">90% oraliq</th><th scope="col">Haqiqiy</th><th scope="col">Oraliqda</th><th scope="col">Yo'nalish</th></tr></thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={`${r.symbol}-${r.as_of}-${r.h}`}>
                    <td>{r.target}</td>
                    <th scope="row"><button className="asset-link" onClick={() => onPick(r.symbol)}>{r.symbol}</button></th>
                    <td>{fmtPrice(r.q50)} <small className="muted">{fmtPct((r.q50 / r.last - 1) * 100, 2)}</small></td>
                    <td className="muted">{fmtPrice(r.q05)} – {fmtPrice(r.q95)}</td>
                    <td>{fmtPrice(r.real)} <small className={r.real >= r.last ? "pos" : "neg"}>{fmtPct((r.real / r.last - 1) * 100, 2)}</small></td>
                    <td className={r.inside ? "pos" : "neg"}>{r.inside ? "✓ ha" : "✗ yo'q"}</td>
                    <td className={r.dir_ok === null ? "muted" : r.dir_ok ? "pos" : "neg"}>{r.dir_ok === null ? "—" : r.dir_ok ? "✓ to'g'ri" : "✗ xato"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      <p className="hint">Bitta kun yoki bitta aktiv natijasi tasodif bo'lishi mumkin — xulosani ko'p kunlik statistikadan qiling. Investitsiya tavsiyasi emas.</p>
    </div>
  );
}
