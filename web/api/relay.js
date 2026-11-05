// VAQTINCHALIK: Vercel sandbox'dagi yangilik yangilanishi natijasini olish uchun. Ish tugagach o'chiriladi.
const HOST = "https://sb-56jn1vgfug8a.vercel.run/";
export default async function handler(req, res) {
  const f = String(req.query?.f ?? "");
  if (!/^(part\d\d|sha\.txt)$/.test(f)) return res.status(400).send("bad");
  const r = await fetch(HOST + f, { signal: AbortSignal.timeout(20000) });
  res.setHeader("Cache-Control", "no-store");
  res.status(r.status).send(await r.text());
}
