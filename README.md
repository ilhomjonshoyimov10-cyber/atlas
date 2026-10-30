# ATLAS — yangilik va mavsumiylikka asoslangan bozor prognozi

Forex, kripto, aksiyalar, indekslar va birja tovarlari (bug'doy, neft, oltin, kakao...) bir joyda.
Deep learning modeli narx tarixi, mavsumiylik va **birinchi chiqadigan yangiliklarni** birga o'qib,
har bir aktiv uchun **1, 5 va 20 savdo kunidan keyingi narx oralig'ini** bashorat qiladi:
q05 (pessimistik), q50 (median), q95 (optimistik).

```
 Yangilik manbalari (Fed, ECB, BLS, EIA, USDA, SEC, CNBC, CoinDesk, ...)
        │  har 60 soniyada, first_seen yoziladi
        ▼
 Kollektor (SQLite) ──► FinBERT (muzlatilgan) ──► PCA 64 ──► kunlik "yangilik kubi" (global / guruh / aktiv)
        │                                                              │
        └──► har bir yangi xabar uchun model ta'sir tahlili ──► lenta  │
                                                                       ▼
 Yahoo Finance 12 yil ─► narx featurelari ─┬──────────────► ┌───── AtlasNet v2 (ansambl) ─────┐
                                           └─► bozor holati │ narx: Transformer encoder           │
                                               (1 kun lag)  │ bozorlararo: aktiv → 37 aktiv attn. │
 Kalendar: FOMC, CPI, NFP, bayram ─► mavsumiylik ─────────► │ yangilik: attention + to'g'ri yo'l  │
                                                            │ mavsum: MLP   · gated fusion        │
                                                            └─► q05/q50/q95 × (1, 5, 20 kun) ─────┘
                                                                       │ konformal kalibratsiya
                                     FastAPI + WebSocket ──► Web: aktiv · bozor xaritasi · backtest
```

## Onlayn versiya — doim ishlaydi, bepul

Sayt: **Vercel** (statik sahifa + jonli narx funksiyasi). Hisob-kitob: **GitHub Actions**. Doimiy server yo'q, shuning uchun
hech narsa "uxlab qolmaydi" va pul to'lanmaydi.

```
 GitHub Actions (har 30 daqiqada)                       GitHub Actions (har yakshanba)
   narx yangilash → RSS yangiliklar → FinBERT              to'liq qayta o'qitish: walk-forward,
   → ta'sir bahosi → prognoz → prognoz jurnali             holdout, backtest, servis modeli
              │                                                     │
              └──────────────► `data` shoxchasi (JSON) ◄─────────────┘
                                        │  raw.githubusercontent.com
                                        ▼
 Vercel: React sayt  ◄──  /api/quotes (Binance + Yahoo jonli narx, 20 s CDN kesh)
```

| Fayl | Vazifasi |
|---|---|
| `atlas/cloud.py` | `python -m atlas.cli cloud-update` — barcha API javoblarini `site/*.json` qilib yozadi |
| `.github/workflows/update.yml` | har 30 daqiqalik yangilash (`data` shoxchasiga tarixsiz bitta commit) |
| `.github/workflows/retrain.yml` | haftalik qayta o'qitish; qo'lda ham ishga tushirsa bo'ladi (Actions → Run workflow) |
| `web/api/quotes.js` | jonli narxlar uchun Vercel funksiyasi |
| `web/.env.production` | sayt ma'lumotni qayerdan o'qishi (`VITE_DATA_URL`) |

Onlayn versiyada "Yangilik ta'siri" bo'limi lentadagi xabarlar uchun oldindan hisoblangan tahlilni ko'rsatadi;
o'zingiz yozgan matnni tahlil qilish lokal serverda ishlaydi. Yangi bo'lim: **Prognoz jurnali**. Unda har kungi prognoz
yozib boriladi va muddati kelgach haqiqiy narx bilan solishtiriladi. Natijalar tanlab olinmaydi.

Eslatma: GitHub ochiq repoda 60 kun hech qanday commit bo'lmasa rejalashtirilgan ishlarni to'xtatadi. Shunday bo'lsa,
Actions sahifasida "Enable workflow" tugmasini bosing.

## Texnik tahlil va strategiyalar (SNR, SND, klassika) + SetupNet

"Analiz qil" tugmasi (yoki `python -m atlas.cli ta GOLD`) tanlangan aktiv uchun yozma tahlil beradi: bozor tuzilmasi
(HH/HL, BOS), support/resistance darajalari, supply/demand zonalari, klassik figuralar, indikatorlar, faol signallar,
har bir signalning tarixiy natijasi, deep learning bahosi va ATLAS prognozi bilan birga umumiy xulosa.

| Qism | Fayl |
|---|---|
| Detektorlar: pivotlar (K bar keyin tasdiqlanadi), SNR klasterlari, SND zonalari, tuzilma, qo'sh cho'qqi/tub, bosh-yelka, uchburchaklar, RSI, MA kesishishi, sham figuralari | `atlas/ta/detect.py` |
| Dataset: 37 aktiv × 10+ yil, har bir signal = bitta yozuv; triple-barrier belgi (maqsad 1.5·ATR, stop 1·ATR, 20 kun); taqqoslash — o'sha aktivda ±6 oyda har kuni kirish; Holm tuzatishi | `atlas/ta/dataset.py` |
| SetupNet: 64 barlik 1D-CNN (dilated) + attention pooling + strategiya/guruh embedding + kontekst → P(maqsad stopdan oldin) | `atlas/ta/model.py` |
| Yozma tahlil | `atlas/ta/analyze.py` |

O'qitish: `python -m atlas.cli ta-train` (CPU'da bir necha daqiqa; haftalik `retrain.yml` ichida ham bor).
Baholash: oxirgi 365 kun model ko'rmagan holdout, oldidan 30 kunlik embargo. "Ishonchli signal" chegarasi
validatsiyada tanlanadi. Natijalar saytdagi **Strategiyalar** sahifasida chiqadi. Asosiy xulosa: mashhur strategiyalarning
ko'pi o'z-o'zidan tasodifiy kirishdan yaxshi emas; foyda modelning signallarni saralashidan keladi.

## Haqiqiy bozorga o'tish — bitta buyruq

```powershell
cd Desktop\ATLAS
.venv\Scripts\activate
python -m atlas.cli real          # ~30–60 daqiqa CPU'da (tezkor rejim)
python -m atlas.cli serve         # http://localhost:8000
```

`real` buyrug'i: 12 yillik haqiqiy narxlarni yuklaydi (Yahoo Finance) → demo yangiliklarini o'chiradi →
GDELT'dan so'nggi 30 kun yangiliklarini oladi → embedding (FinBERT bo'lsa FinBERT, aks holda tezkor zaxira) →
walk-forward + holdout baholash (backtest uchun) → servis modelini o'qitadi.
Variantlar: `--full` (to'liq rejim, GPU tavsiya), `--skip-eval` (tezroq, backtestsiz), `--gdelt-days 90`.

**Jonli narxlar** server ishga tushishi bilan avtomatik keladi (demo rejimda ham):
kripto — Binance (real vaqt), qolganlari — Yahoo Finance 1 daqiqalik barlar (har 30 s).
Diqqat: oltin, kumush, mis, neft, gaz va qishloq xo'jaligi mahsulotlari — **fyuchers** narxi (GC=F, CL=F, ZW=F...),
shuning uchun TradingView'dagi spot XAUUSD (OANDA) dan bir necha dollar farq qilishi normal. Aksiyalar va
indekslar narxi birja qoidasiga ko'ra ~15 daqiqagacha kechikishi mumkin.

Yangilik shoxi tarixda o'rganishi uchun ko'p yillik yangilik kerak: FNSPID va Kaggle datasetlarini
`data/news/raw/` ga qo'ying yoki `notebooks/train_colab.ipynb` ni ishlating. Faqat 30 kunlik GDELT bilan
model yangilikdan deyarli hech narsa o'rgana olmaydi — bu natijalarda ko'rinadi.

## Tez boshlash (internet va GPU'siz, sintetik ma'lumotda)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m atlas.cli demo --quick      # sintetik bozor + yangiliklar → o'qitish → baholash → model
cd web && npm install && npm run build && cd ..
ATLAS_COLLECT=0 python -m atlas.cli serve           # http://localhost:8000
```

Sintetik ma'lumotga **ma'lum** signallar joylangan (masalan, "OPEC cuts output" → neft o'sadi).
Quvur to'g'ri ishlasa, ablationda `full` modeli `price` modelidan statistik ahamiyatli yaxshiroq chiqadi —
bu yangilik shoxi va leakage himoyasi ishlayotganini tekshiradi.

### Windows uchun

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m atlas.cli serve          # tayyor demo model bilan: http://localhost:8000
# (kollektorsiz: $env:ATLAS_COLLECT="0"; python -m atlas.cli serve)
```

Loyiha ichida `data/` papkasida sintetik demo uchun o'qitilgan model va `web/dist` da tayyor web bor —
`serve` darhol ishlaydi. Web kodini o'zgartirsangiz: `cd web && npm install && npm run build`.

### Demo natijalari (v2, sintetik bozor, 2 yadroli CPU, tezkor rejim: 2 fold, 5 epoch, 1 qatlam)

| Variant | Pinball (walk-forward) | 90% qamrov, 1 kun | Pinball (holdout) |
|---|---|---|---|
| Baza (shartsiz) | 0.2203 | 89.8% | 0.2239 |
| Faqat narx | 0.2201 | 90.5% | 0.2223 |
| Narx + mavsum | 0.2193 | 90.1% | 0.2222 |
| Narx + mavsum + bozorlar | 0.2195 | 90.3% | 0.2223 |
| To'liq (+ yangilik) | 0.2198 | 90.0% | 0.2229 |

Holdout backtest (2025-10 — 2026-10, 5 bps xarajat): yangilikli model Sharpe **1.12** (maks. tushish −1.7%),
yangiliksiz model −0.44, sotib olib ushlash 0.92.

Halol xulosa:
* Kalibratsiya ishlaydi (90% oraliq ~90% qamraydi). Model yangilik signalini o'rganadi: o'qitishda
  ishlatilmagan davrda bashorat va yangilik yo'nalishi korrelyatsiyasi 0.31–0.50.
* Pinball bo'yicha variantlar farqi **statistik ahamiyatsiz** (Holm p > 0.6) — tezkor CPU rejimida Transformer
  hali to'liq o'qimagan (validatsiya loss 5-epochda ham pasayayotgan edi).
* Backtestdagi ustunlik — bitta yillik davr; tasodif bo'lishi mumkin. Haqiqiy xulosani faqat haqiqiy ma'lumotda,
  to'liq rejimda (5 fold, 30 epoch, 2 qatlam, 3 seedli ansambl, GPU, FinBERT) chiqaring.

## Haqiqiy ma'lumot bilan

| Qadam | Buyruq | Izoh |
|---|---|---|
| 1. Narxlar | `python -m atlas.cli download` | 37 aktiv, 12 yil, kunlik (Yahoo Finance) |
| 2. Yangiliklar | fayllarni `data/news/raw/` ga qo'ying | Kaggle `Combined_News_DJIA.csv`, FNSPID `nasdaq_exteral_data.csv` |
| 3. GDELT (ixtiyoriy) | `python -m atlas.cli gdelt --days 90` | tarixiy uchun BigQuery SQL: `news_open.GDELT_BQ_SQL` |
| 4. Kollektor | `python -m atlas.cli collect` | serverda doimiy; o'z `first_seen` tarixingiz to'planadi |
| 5. Embedding | `python -m atlas.cli features --embedder finbert` | GPU'da tez (Colab) |
| 6. Baholash | `python -m atlas.cli evaluate` | walk-forward + ablation, holdoutga tegmaydi |
| 7. Final | `python -m atlas.cli evaluate --final` | **faqat bir marta, oxirida** |
| 8. Servis modeli | `python -m atlas.cli train` | `data/models/atlas.pt` |
| 9. Ishga tushirish | `python -m atlas.cli serve` yoki `docker compose up` | API + web + kollektor |

O'qitish uchun eng qulay yo'l: `notebooks/train_colab.ipynb` (Colab/Kaggle bepul GPU yetadi).

## Leakage qoidalari (eng muhim qism)

* Qaror vaqti: `decision_time(D) = D kuni 20:00 UTC` (NYSE yopilishiga yaqin, konservativ).
* Yangilik D kuniga **faqat** `available_at ≤ decision_time(D)` bo'lsa tushadi (`features/news_agg.use_date`).
* `available_at`:
  * Kaggle (faqat sana) → **ertasi kun** 00:00 UTC (o'sha kun yopilishida ishlatilmaydi);
  * FNSPID, GDELT → e'lon vaqti + 5 daqiqa;
  * o'z kollektorimiz → `first_seen` (sayt keyin `published` ni o'zgartirsa ham ta'sir qilmaydi).
* Eski xabarni bugun topib "o'sha paytda ham bor edi" deb olish taqiqlangan — shuning uchun kollektor
  faqat o'zi ko'rgan vaqtni yozadi.
* PCA faqat holdoutdan oldingi xabarlarda fit qilinadi.
* Walk-forward: train va test orasida embargo (≈32 kalendar kun) — 20 kunlik target oynalari testga kirmaydi.
* Rejadagi voqealar (FOMC/CPI/NFP) oldindan e'lon qilinadi → "ertaga FOMC" belgisi leakage emas.
  Favqulodda yig'ilishlar kalendarga kiritilmagan.
* `tests/test_core.py` bu qoidalarni avtomatik tekshiradi: `pytest -q`.

## Model (v2)

* **Narx shoxi — Transformer encoder** (pre-norm, [CLS] token, o'rganiladigan pozitsiya): oxirgi 60 kun.
* **Bozorlararo shox (cross-asset attention)**: har bir aktiv qolgan 36 aktivning holatiga (qaytim, momentum,
  volatillik rejimi, MA50 masofasi) attention qiladi — masalan neft → CAD/XOM, dollar → oltin/EURUSD
  aloqalarini o'zi o'rganadi. Turli bozorlar turli vaqtda yopilgani uchun bu ma'lumot **1 kun kechiktirib**
  beriladi (leakage yo'q, `tests/test_core.py` tekshiradi).
* **Gated fusion**: har bir aktiv uchun shoxlar (narx/bozor/yangilik/mavsum) og'irligi o'rganiladi va UI'da ko'rsatiladi.
* **Ansambl**: servis modeli 3 ta seed bilan o'qitiladi, kvantillar o'rtachalanadi — barqarorroq va yaxshiroq kalibrlangan.
* **O'qitish**: AdamW, warmup + cosine LR, gradient clipping, GPU'da bfloat16 AMP, early stopping.
* **Tajribalar reyestri**: har bir `evaluate` `data/reports/experiments.jsonl` ga yoziladi (konfiguratsiya xeshi,
  natijalar, p-qiymatlar, holdout necha marta ishlatilgani). Ko'p tajriba = ko'p test — shu ro'yxat buni halol ko'rsatadi.

Asosiy qismlar:

* **Ko'p aktivli o'qitish**: 1 aktivda 10 yilda ~2500 qator — kam. 37 aktiv birga ~90 ming namuna beradi;
  aktiv ID embedding sifatida beriladi.
* **Target**: `y_h = log(C[t+h]/C[t]) / (σ_t·√h)` — volatillikka normallashtirilgan, barcha aktivlar bir masshtabda.
* **Narx shoxi**: oxirgi 60 kun, 8 ta feature (normallashtirilgan qaytim, volatillik rejimi, MA50/200 masofasi,
  momentum, hajm z-score...) → 2 qatlamli GRU.
* **Yangilik shoxi**: oxirgi 7 kun; har kun = [global, aktiv guruhi, aktivning o'zi] embeddinglari + soni + ton +
  voqea turi (FOMC, CPI, NFP, OPEC, USDA, earnings, geosiyosat, kripto-regulyatsiya...).
  Attention so'rovi = aktiv embeddingi → model har bir aktiv uchun qaysi kun xabarlari muhimligini o'rganadi.
  FinBERT **muzlatilgan** — o'qitiladigan parametrlar kam.
* **Mavsumiylik shoxi**: oy/hafta kuni/yil kuni sin-cos, oy va chorak oxiri, hisobot mavsumi, AQSh bayramlari,
  ekish/o'sish/hosil davrlari, FOMC/CPI/NFP gacha kunlar. Aktiv embeddingi bilan birga → bug'doy va bitcoin
  uchun turli mavsumiylik.
* **Chiqish**: q05 ≤ q50 ≤ q95 (softplus orqali — kvantillar hech qachon kesishmaydi), pinball loss.
* **Kalibratsiya**: konformal (CQR) — validatsiyada 90% oraliq haqiqatda ~90% qamrashi uchun kengaytiriladi.

## Baholash

`data/reports/report_walkforward.json`:
* variantlar: `climatology` (shartsiz baza), `price`, `price+season`, `no_news` (narx+mavsum+bozorlar), `full`;
* **asosiy savol** — `full vs no_news`: boshqa hamma narsa bir xil, faqat yangilik qo'shilgan;
* metrikalar: pinball loss, 90% oraliq qamrovi, kenglik, yo'nalish aniqligi — har bir gorizont uchun;
* Diebold–Mariano testi (Newey–West HAC) + **Holm tuzatishi** (ko'p taqqoslash).

Qoidalar: test natijasiga qarab feature yoki giperparametr tanlamang; holdoutni faqat oxirida bir marta ishlating.
Ko'p tajriba qilsangiz, hammasini yozib boring va p-qiymatlarni tuzating.

## Backtest

`python -m atlas.cli backtest` yoki web'dagi **Backtest** bo'limi. Faqat holdout davrining tashqi
(out-of-sample) bashoratlari ishlatiladi: p_up chegarasidan yuqori → long, pastida → short, volatillikka teskari hajm,
pozitsiya o'zgarishiga komissiya. Model (yangilik bilan), yangiliksiz model, faqat narx modeli va
"sotib olib ushlash" solishtiriladi: Sharpe, jami/yillik daromad, maksimal tushish, foydali kunlar, aylanma.
Slippage, likvidlik va soliq hisobga olinmagan — bu tadqiqot vositasi.

## Web interfeys

* Chapda — barcha aktivlar guruhlar bo'yicha, sparkline va 5 kunlik median prognoz; kripto narxlari Binance oqimidan jonli.
* Markazda — 6 oydan 11 yilgacha narx grafigi va unga ulangan **prognoz yelpig'ichi** (q05–q95), 1/5/20 kunlik
  oraliqlar, o'sish ehtimoli, model qaysi kun yangiliklariga qaraganini ko'rsatuvchi attention.
* **Mavsumiylik** — oylik o'rtacha qaytim, o'sish ehtimoli, t-statistika va yil×oy issiqlik xaritasi.
* **Yangilik ta'siri** — istalgan sarlavhani kiriting: model uni bugungi oqimga qo'shib, har bir aktiv
  bashorati qanchaga siljishini ko'rsatadi.
* **Model sifati** — walk-forward natijalari, ablation testlari, final holdout.
* O'ngda — real-time yangiliklar lentasi (WebSocket). Har bir yangi xabarni model **avtomatik** tahlil qiladi
  va eng ko'p ta'sirlangan aktivlarni ko'rsatadi (masalan, "Neft ↑0.16%").
* **Bozor xaritasi** — 37 aktivning prognozi bitta ekranda; rang — yo'nalish, to'qlik — signalning noaniqlikka nisbatan kuchi.
* **Backtest** — yuqoridagi bo'limga qarang.
* Aktiv sahifasida: bashorat manbalari (shoxlar ulushi) va model ko'proq qaragan boshqa bozorlar.

Ishlab chiqish rejimi: `python -m atlas.cli serve` + `cd web && npm run dev` (http://localhost:5173).

## API

| Endpoint | Tavsif |
|---|---|
| `GET /api/assets` | aktivlar, oxirgi narx, sparkline |
| `GET /api/history/{sym}?years=10` | OHLCV tarix |
| `GET /api/forecast`, `/api/forecast/{sym}` | q05/q50/q95, o'sish ehtimoli, attention |
| `GET /api/seasonality/{sym}?by=month\|week\|dow` | mavsumiy profil |
| `GET /api/news?asset=BTC` | kollektor yig'gan yangiliklar |
| `POST /api/news/analyze {"text": ...}` | yangilikning aktivlarga ta'siri |
| `GET /api/report` | baholash hisoboti |
| `GET /api/backtest?threshold=0.05&cost_bps=5` | holdout backtest |
| `GET /api/experiments` | tajribalar reyestri |
| `WS /ws` | yangi xabarlar oqimi |

## Fayl tuzilmasi

```
atlas/
  config.py            aktivlar, manbalar, giperparametrlar
  data/                prices.py, news_open.py, collector.py, calendar.py, synthetic.py
  features/            tagging.py, text_embed.py, news_agg.py, seasonality.py, dataset.py
  model/               net.py (v2), train.py, evaluate.py, backtest.py
  api/                 service.py, main.py
  pipeline.py, cli.py
web/                   React + TypeScript dashboard
notebooks/             train_colab.ipynb
tests/                 leakage va model testlari
```

## Cheklovlar

* Yahoo Finance narxlari ~15 daqiqa kechikadi (kriptodan tashqari); tijorat uchun pullik feed ulang.
* Ba'zi RSS manbalari vaqti-vaqti bilan manzilini o'zgartiradi — `config.NEWS_FEEDS` ni yangilab turing;
  holati `/api/health` da ko'rinadi.
* Yangilik ta'siri odatda kichik va shovqinli. Ablation statistik ahamiyatli foyda ko'rsatmasa, yangilik shoxini
  majburlab qo'shmang — buni halol tan olish ham natija.
* Bashoratlar ehtimoliy va investitsiya tavsiyasi emas.
