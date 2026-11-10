# halalkronos-forecast

Real Kronos inference service for HalalKronos. Deployed on Render.

## Auth

`POST /predict` requires `Authorization: Bearer $KRONOS_SERVICE_TOKEN`. Missing or wrong token returns 401. A missing env var returns 503.

`GET /health` is public and returns only phase, model name, tokenizer, uptime, and a short error.

## CORS

Allowlist from `KRONOS_CORS_ORIGINS` (default `https://hdashpluskronos.vercel.app,http://localhost:3000`).

## Health phases

- `loading` — model download or load in progress
- `ready` — predictor usable
- `unavailable` — load failed

`ready: true` is set only after `KronosPredictor` is constructed.

## Bands

`upper` / `lower` are the 90th and 10th percentiles of sampled close paths (`band_method: sample_quantiles_10_90`). They are not calibrated prediction intervals.

## Verdict

Heuristic only: `accumulate` if expected move > 1.5% and sample agreement score >= 50; `avoid` if move < -1.5%; otherwise `hold`. Not a validated buy/sell signal.
