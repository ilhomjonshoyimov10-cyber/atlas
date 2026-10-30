---
name: atlas-ops
description: Operate, update, retrain, debug and redeploy the ATLAS market-forecast project (repo ilhomjonshoyimov10-cyber/atlas, site atlas-bozor.vercel.app). Use whenever the user mentions ATLAS, the market/forecast site, stale prices or news on the site, retraining the model, GitHub Actions for atlas, the data branch, Vercel deploys of atlas, or adding an asset or news source, even if they don't name the skill.
---

# ATLAS operations

ATLAS has three moving parts. Know which one a problem lives in before touching anything.

| Part | Where | What it does |
|---|---|---|
| Code | `main` branch | Python package `atlas/`, React site `web/`, workflows `.github/workflows/` |
| Data | `data` branch (single orphan commit, force-pushed) | prices parquet, `news.sqlite`, `embeddings/`, `models/atlas.pt`, `reports/`, `forecast_log.jsonl`, and `site/*.json` that the website reads |
| Site | Vercel project `atlas-bozor` (root dir `web`, region fra1) | static React app plus `web/api/quotes.js` live-price function; it auto-deploys on every push to `main` |

The site reads data from `https://raw.githubusercontent.com/ilhomjonshoyimov10-cyber/atlas/data/site/` (set in `web/.env.production`). raw.githubusercontent caches for about 5 minutes, so wait before calling a fresh data push "not working".

## Routine jobs

- `update.yml` runs every 30 min (`cron 7,37`): `python -m atlas.cli cloud-update` with `ATLAS_DATA=state`. It updates prices for closed bars only (today's bar is kept after 21:00 UTC), collects RSS, scores new headlines with FinBERT, exports `site/`, appends the forecast log, and force-pushes `data`.
- `retrain.yml` runs Sundays at 02:23 UTC, or manually: `atlas.cli real --embedder finbert` (walk-forward, holdout, backtest OOS files, serving model), then `cloud-update --skip-news`.
- Trigger a run manually: `POST /repos/ilhomjonshoyimov10-cyber/atlas/actions/workflows/<file>/dispatches` with `{"ref":"main"}`.

## Diagnosing "the site is stale"

1. Read `site/health.json` on the data branch. Check `updated_at`, `data_until`, `collector` and `prices.failed`.
2. List recent workflow runs. If a job failed in a few seconds with no steps, read the check-run annotations. The message "account is locked due to a billing issue" means GitHub billing must be fixed by the user. Never touch payment settings yourself; tell the user what to do.
3. A Yahoo or RSS failure for one asset or feed is tolerated by design and does not fail the job. Fix feeds in `atlas/config.py` `NEWS_FEEDS`.

## Changing things safely

- **.gitignore** ignores only the root `/data/`. A bare `data/` pattern would also hide the `atlas/data/` package.
- **Asset list**: edit `ASSETS` in `atlas/config.py` AND the `ASSETS` array in `web/api/quotes.js`. `tests/test_core.py::test_quotes_function_assets_match_config` enforces the match. Adding an asset changes the model's input size, so run retrain afterwards.
- **Leakage rules are the core of the project.** Do not weaken them: news is used only if `available_at` falls before the decision time (20:00 UTC), the market context is lagged 1 day, PCA is fit before the holdout, and there is a 32-day embargo. Run `pytest -q` before every push.
- **Backtest** row indices are remapped by (asset, date) in `model/backtest.remap_rows`, because new price rows shift panel indices.
- **Forecast journal** (`forecast_log.jsonl`) is append-only and keyed by (symbol, as_of). Never rewrite or filter it: its value is that results are not cherry-picked.
- **Frontend static mode** is on when `VITE_DATA_URL` is set. Every new API endpoint needs a static file in `cloud.export_site` and a branch in `web/src/api.ts`.

## Technical analysis (atlas/ta)

- `atlas.cli ta-train` builds the strategy dataset: about 82k signals from SNR, SND, structure/BOS, classic patterns, indicators and candles. Labels use a triple barrier: TP 1.5·ATR, SL 1·ATR, 20 bars. It trains SetupNet and writes `models/ta_setup.pt` and `reports/ta_report.json`. It runs weekly in `retrain.yml`.
- `atlas.cli ta GOLD` prints the written analysis. The site reads `site/ta/{SYM}.json` and `site/strategies.json` (exported by `cloud.export_site`).
- Detectors must stay causal. A pivot counts only after `confirm = idx + K`, zigzag cleaning runs on confirmed pivots, and a zone exists only after its impulse bar closes. `tests/test_core.py::test_ta_no_lookahead` guards this.
- Compare every strategy claim against the same-asset ±6-month random-entry baseline with Holm correction. Report honestly: most classic setups show no edge, and the value comes from SetupNet ranking setups.
- Run every price frame through `sanitize_ohlc` before TA. Yahoo sometimes returns an open outside [low, high].

## Local run (user's Windows PC)

`C:\Users\user\Desktop\ATLAS_AI\start_atlas.bat` starts the FastAPI server at http://localhost:8000, with live WebSocket news and free-text "what-if" analysis. The online site only shows precomputed analyses for feed headlines.

## Verifying a change

1. `pytest -q` passes.
2. `cd web && npm run build` passes.
3. After pushing, check that the Vercel deployment reached `READY`.
4. Open the site and check: prices appear in the left rail, the gold price is close to COMEX, the forecast fan shows, and the Backtest and Prognoz jurnali pages load. Check a phone-width page for no horizontal scroll.
