---
title: HalalKronos Forecast
sdk: docker
app_port: 7860
---
# HalalKronos Kronos service
Real Kronos-small (CPU) forecast API. Endpoints: GET /health, GET /ohlcv?symbol=&interval=&range=, POST /predict {symbol, interval, lookback, pred_len, T, top_p, sample_count}.
Returns mean candles, upper/lower (p90/p10 close) bands, expectedMovePct, verdict (accumulate/hold/avoid), confidence (0-100).

## Deploy to Hugging Face Spaces (free)
1. huggingface.co -> New Space -> SDK: Docker (Blank), CPU basic, Public.
2. Create a write token at huggingface.co/settings/tokens.
3. From this folder:
   pip install huggingface_hub
   hf auth login            # paste token
   hf upload <user>/halalkronos-forecast . . --repo-type=space --exclude ".venv/*"
4. Wait for build; check https://<user>-halalkronos-forecast.hf.space/health shows ready:true.
5. Vercel -> project -> Settings -> Environment Variables: KRONOS_API_URL=https://<user>-halalkronos-forecast.hf.space (Production), redeploy.
Render/Railway: deploy this Dockerfile; set PORT if needed.
