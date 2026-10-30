import { useEffect, useState } from "react";
import { api } from "../api";

const VNAME: Record<string, string> = {
  climatology: "Baza (shartsiz)", price: "Faqat narx", "price+season": "Narx + mavsum",
  no_news: "Narx + mavsum + bozorlar", full: "To'liq (+ yangilik)",
};

export default function ModelReport() {
  const [rep, setRep] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.report().then(setRep).catch((e) => setErr(e.message)); }, []);
  if (err) return <p className="muted">Hisobot yuklanmadi: {err}</p>;
  if (!rep) return <div className="skeleton" style={{ height: 240 }} />;
  const r = rep["report_final.json"] ?? rep["report_walkforward.json"];
  if (!r) return <p className="muted">Hali baholash o'tkazilmagan. <code>python -m atlas.cli evaluate</code> ni ishga tushiring.</p>;
  const hz: number[] = r.config.horizons;
  const variants: string[] = r.variants;
  const hold = r.folds.find((f: any) => f.fold === "HOLDOUT");
  const best = Object.entries(r.walk_forward_summary as Record<string, any>).sort((a, b) => a[1].pinball - b[1].pinball)[0]?.[0];

  return (
    <div className="report">
      <p className="lead">
        Model {r.folds.filter((f: any) => f.fold !== "HOLDOUT").length} ta walk-forward davrda sinaldi
        (har birida faqat o'tmish bilan o'qitilgan, kelajak ko'rsatilmagan). Pinball loss qancha kichik bo'lsa,
        bashorat oralig'i shuncha aniq. 90% oraliq haqiqatda ~90% holatni qamrab olishi kerak.
      </p>
      {r.experiment_no && (
        <p className="hint">Bu tajriba #{r.experiment_no}. Final holdout jami {r.holdout_uses_total ?? 0} marta ishlatilgan
          {(r.holdout_uses_total ?? 0) > 1 ? " — diqqat: u endi to'liq tegilmagan emas." : "."}</p>
      )}
      <h3>Walk-forward o'rtachasi</h3>
      <div className="table-scroll">
        <table className="metrics">
          <thead>
            <tr><th scope="col">Variant</th><th scope="col">Pinball</th>
              {hz.map((h) => <th key={h} scope="col">Qamrov {h}k</th>)}
              {hz.map((h) => <th key={`d${h}`} scope="col">Yo'nalish {h}k</th>)}</tr>
          </thead>
          <tbody>
            {variants.map((v) => {
              const s = r.walk_forward_summary[v];
              if (!s) return null;
              return (
                <tr key={v} className={v === best ? "best" : ""}>
                  <th scope="row">{VNAME[v] ?? v}</th>
                  <td>{s.pinball.toFixed(4)}</td>
                  {hz.map((h) => <td key={h} className={Math.abs(s[`h${h}_coverage90`] - 0.9) <= 0.03 ? "pos" : ""}>{(s[`h${h}_coverage90`] * 100).toFixed(1)}%</td>)}
                  {hz.map((h) => <td key={`d${h}`}>{(s[`h${h}_dir_acc`] * 100).toFixed(1)}%</td>)}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <h3>Ablation: yangilik va mavsumiylik foyda beryaptimi?</h3>
      <p className="hint">Diebold–Mariano testi (Newey–West), p-qiymatlar Holm usulida ko'p taqqoslash uchun tuzatilgan.</p>
      <ul className="tests">
        {Object.entries(r.ablation_tests as Record<string, any>).map(([k, t]) => {
          const [a, b] = k.split(" vs ");
          const good = t.mean_loss_diff < 0 && t.p_holm < 0.05;
          return (
            <li key={k} className={good ? "good" : t.p_holm < 0.05 ? "bad" : ""}>
              <strong>{VNAME[a] ?? a}</strong> vs {VNAME[b] ?? b}:{" "}
              {t.mean_loss_diff < 0 ? "yaxshiroq" : "yomonroq"}, p (Holm) = {t.p_holm?.toFixed(4) ?? "—"}
              {t.p_holm < 0.05 ? " — statistik ahamiyatli" : " — ahamiyatsiz"}
            </li>
          );
        })}
      </ul>
      {hold && (
        <>
          <h3>Final holdout ({hold.test_start} — {hold.test_end})</h3>
          <p className="hint">Bu davr o'qitish va tanlov jarayonida umuman ishlatilmagan.</p>
          <div className="table-scroll">
            <table className="metrics">
              <thead><tr><th scope="col">Variant</th><th scope="col">Pinball</th>{hz.map((h) => <th key={h} scope="col">Qamrov {h}k</th>)}</tr></thead>
              <tbody>
                {variants.map((v) => (
                  <tr key={v}><th scope="row">{VNAME[v] ?? v}</th><td>{hold.metrics[v].pinball.toFixed(4)}</td>
                    {hz.map((h) => <td key={h}>{(hold.metrics[v][`h${h}`].coverage90 * 100).toFixed(1)}%</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
