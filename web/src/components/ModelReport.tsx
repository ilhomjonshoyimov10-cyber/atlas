import { useEffect, useState } from "react";
import { api } from "../api";

const VNAME: Record<string, string> = {
  climatology: "Baza (shartsiz)", price: "Faqat narx", "price+season": "Narx + mavsum",
  no_news: "Narx + mavsum + bozorlar", full: "To'liq (+ yangilik)", full_scratch: "To'liq, oldindan o'qitishsiz",
};

function DeepLearning({ dl }: { dl: any }) {
  const wf = dl["report_walkforward.json"];
  const pt = dl["pretrain.json"];
  if (!wf && !pt) return null;
  const c = wf?.config ?? {};
  const rows = wf ? Object.entries(wf.walk_forward_summary as Record<string, any>).sort((a, b) => a[1].pinball - b[1].pinball) : [];
  const folds = wf ? wf.folds.filter((f: any) => f.fold !== "HOLDOUT") : [];
  return (
    <div className="report">
      <h3>Deep learning tajribasi (Colab GPU)</h3>
      {wf && (
        <p className="lead">
          Transformer: {c.layers} qatlam, {c.hidden} o'lcham, {c.heads} attention boshi, {c.lookback} kunlik oyna,
          {" "}{c.ensemble} modelli ansambl. {folds.length} ta walk-forward davr
          {folds.length ? ` (${folds[0].test_start} — ${folds[folds.length - 1].test_end})` : ""}.
        </p>
      )}
      {pt && (
        <p className="hint">
          Self-supervised oldindan o'qitish (masked reconstruction, {Math.round((pt.mask_ratio ?? 0.3) * 100)}% kun yashirilgan,
          {" "}{pt.cutoff} gacha {pt.rows?.toLocaleString()} oyna): tiklash xatosi <strong>{pt.best_val_mse?.toFixed(3)}</strong>,
          oddiy o'rtacha bilan to'ldirish {pt.baseline_mean_fill_mse?.toFixed(3)}
          {pt.baseline_mean_fill_mse ? ` (${Math.round((1 - pt.best_val_mse / pt.baseline_mean_fill_mse) * 100)}% yaxshiroq)` : ""}.
        </p>
      )}
      {wf && (
        <>
          <div className="table-scroll">
            <table className="metrics">
              <thead><tr><th scope="col">Variant</th><th scope="col">Pinball</th>
                {(c.horizons ?? []).map((h: number) => <th key={h} scope="col">Qamrov {h}k</th>)}</tr></thead>
              <tbody>
                {rows.map(([v, m], i) => (
                  <tr key={v} className={i === 0 ? "best" : ""}><th scope="row">{VNAME[v] ?? v}</th><td>{m.pinball.toFixed(4)}</td>
                    {(c.horizons ?? []).map((h: number) => <td key={h}>{(m[`h${h}_coverage90`] * 100).toFixed(1)}%</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
          <ul className="tests">
            {Object.entries(wf.ablation_tests as Record<string, any>).map(([k, t]) => {
              const [a, b] = k.split(" vs ");
              return (
                <li key={k} className={t.mean_loss_diff < 0 && t.p_holm < 0.05 ? "good" : ""}>
                  <strong>{VNAME[a] ?? a}</strong> vs {VNAME[b] ?? b}: {t.mean_loss_diff < 0 ? "yaxshiroq" : "yomonroq"},
                  {" "}p (Holm) = {t.p_holm?.toFixed(3) ?? "—"}{t.p_holm < 0.05 ? " — statistik ahamiyatli" : " — ahamiyatsiz"}
                </li>
              );
            })}
          </ul>
        </>
      )}
      {dl && <DeepLearning dl={dl} />}
    </div>
  );
}

export default function ModelReport() {
  const [rep, setRep] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.report().then(setRep).catch((e) => setErr(e.message)); }, []);
  if (err) return <p className="muted">Hisobot yuklanmadi: {err}</p>;
  if (!rep) return <div className="skeleton" style={{ height: 240 }} />;
  const r = rep["report_final.json"] ?? rep["report_walkforward.json"];
  const dl = rep.deep_learning;
  if (!r && dl) return <DeepLearning dl={dl} />;
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
      {dl && <DeepLearning dl={dl} />}
    </div>
  );
}
