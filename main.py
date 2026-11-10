"""HalalKronos forecast service: real Kronos-small on CPU. Contract matches frontend/src/lib/types.ts ForecastResult."""
import os, sys, time, threading
import torch
torch.set_num_threads(1)
import numpy as np, pandas as pd, yfinance as yf
import hmac
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kronos_src"))
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa

MODEL_ID = os.getenv("KRONOS_MODEL", "NeoQuasar/Kronos-small")
TOK_ID = os.getenv("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
SERVICE_TOKEN = os.getenv("KRONOS_SERVICE_TOKEN", "")
ALLOWED = [o.strip() for o in os.getenv("KRONOS_CORS_ORIGINS", "https://hdashpluskronos.vercel.app,http://localhost:3000").split(",") if o.strip()]
MAX_CONCURRENT = int(os.getenv("KRONOS_MAX_CONCURRENT", "1"))
STARTED = time.time()

app = FastAPI(title="HalalKronos forecast service")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Authorization", "Content-Type"])
_state = {"predictor": None, "error": None, "loaded_at": None, "phase": "loading"}
_lock = threading.Lock()
_load_lock = threading.Lock()
_inflight = 0
_inflight_lock = threading.Lock()

def require_token(authorization: str | None) -> None:
    if not SERVICE_TOKEN:
        raise HTTPException(503, "KRONOS_SERVICE_TOKEN is not set")
    presented = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
    if not presented or not hmac.compare_digest(presented, SERVICE_TOKEN):
        raise HTTPException(401, "unauthorized")

def load():
    if not _load_lock.acquire(blocking=False):
        return
    try:
        _state["phase"] = "loading"
        tok = KronosTokenizer.from_pretrained(TOK_ID)
        mdl = Kronos.from_pretrained(MODEL_ID)
        _state["predictor"] = KronosPredictor(mdl, tok, device="cpu", max_context=int(os.getenv("KRONOS_MAX_CONTEXT","512")))
        _state["loaded_at"] = time.time()
        _state["error"] = None
        _state["phase"] = "ready"
    except Exception as e:
        _state["error"] = repr(e)[:300]
        _state["phase"] = "unavailable"
    finally:
        _load_lock.release()

threading.Thread(target=load, daemon=True).start()

RANGE_DEFAULT = {"1d": "2y", "1h": "730d", "60m": "730d", "15m": "60d", "5m": "60d"}
_cache = {}

def fetch(symbol, interval, period):
    key = (symbol, interval, period)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 300:
        return hit[1]
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise HTTPException(404, f"No data for {symbol} {interval}")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["amount"] = df["close"] * df["volume"]
    _cache[key] = (time.time(), df)
    return df

class PredictIn(BaseModel):
    symbol: str
    interval: str = "1d"
    lookback: int = Field(400, ge=50, le=512)
    pred_len: int = Field(20, ge=1, le=120)
    T: float = 1.0
    top_p: float = 0.9
    sample_count: int = Field(5, ge=1, le=10)

@app.get("/")
@app.get("/health")
def health():
    return {
        "ok": _state["phase"] != "unavailable",
        "ready": _state["predictor"] is not None,
        "phase": _state["phase"],
        "model": MODEL_ID,
        "tokenizer": TOK_ID,
        "uptime_s": int(time.time() - STARTED),
        "error": _state["error"],
    }

@app.get("/ohlcv")
def ohlcv(symbol: str, interval: str = "1d", range: str = "6mo"):
    df = fetch(symbol.upper(), interval, range)
    candles = [{"time": t.isoformat(), "open": float(r.open), "high": float(r.high), "low": float(r.low),
                "close": float(r.close), "volume": float(r.volume)} for t, r in df.iterrows()]
    return {"symbol": symbol.upper(), "interval": interval, "range": range, "candles": candles}

def future_index(idx, n, interval):
    step = idx.to_series().diff().dropna().median()
    if interval == "1d":
        return pd.bdate_range(idx[-1] + pd.Timedelta(days=1), periods=n)
    return pd.DatetimeIndex([idx[-1] + step * (i + 1) for i in range(n)])

@app.post("/predict")
def predict(b: PredictIn, authorization: str | None = Header(default=None)):
    require_token(authorization)
    global _inflight
    p = _state["predictor"]
    if p is None:
        raise HTTPException(503, detail={"phase": _state["phase"], "error": _state["error"] or "Model is loading"})
    with _inflight_lock:
        if _inflight >= MAX_CONCURRENT:
            raise HTTPException(429, "inference busy")
        _inflight += 1
    started = time.time()
    try:
        sym = b.symbol.upper()
        df = fetch(sym, b.interval, RANGE_DEFAULT.get(b.interval, "2y")).tail(b.lookback)
        if len(df) < 60:
            raise HTTPException(400, f"need at least 60 candles, got {len(df)}")
        x_ts = pd.Series(df.index)
        y_idx = future_index(df.index, b.pred_len, b.interval)
        y_ts = pd.Series(y_idx)
        x = df[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
        samples = []
        with _lock:
            for _ in range(b.sample_count):
                out = p.predict(df=x, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=b.pred_len,
                                T=b.T, top_p=b.top_p, sample_count=1, verbose=False)
                samples.append(out[["open", "high", "low", "close", "volume"]].to_numpy())
        arr = np.stack(samples)
        mean = arr.mean(0)
        closes = arr[:, :, 3]
        upper = np.percentile(closes, 90, axis=0) if b.sample_count > 1 else closes.max(0)
        lower = np.percentile(closes, 10, axis=0) if b.sample_count > 1 else closes.min(0)
        last = float(df["close"].iloc[-1])
        move = (mean[-1, 3] / last - 1) * 100
        band = float((upper[-1] - lower[-1]) / last * 100)
        agree = float(np.mean(np.sign(closes[:, -1] - last) == np.sign(mean[-1, 3] - last)))
        conf = float(np.clip(100 * (0.5 * agree + 0.5 * (1 / (1 + band / max(abs(move), 0.5)))), 0, 100))
        verdict = "accumulate" if move > 1.5 and conf >= 50 else ("avoid" if move < -1.5 else "hold")
        return {
            "symbol": sym, "interval": b.interval, "exchange": "NSE", "lastClose": last, "offline": False,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "inference_ms": int((time.time() - started) * 1000),
            "input_count": len(df), "horizon": b.pred_len,
            "band_method": "sample_quantiles_10_90",
            "mean": [{"time": t.isoformat(), "open": float(r[0]), "high": float(r[1]), "low": float(r[2]),
                      "close": float(r[3]), "volume": float(max(r[4], 0))} for t, r in zip(y_idx, mean)],
            "upper": [float(v) for v in upper], "lower": [float(v) for v in lower],
            "expectedMovePct": round(float(move), 2), "bandWidthPct": round(band, 2),
            "verdict": verdict, "confidence": round(conf, 1), "sampleCount": b.sample_count,
            "model": MODEL_ID, "tokenizer": TOK_ID,
        }
    finally:
        with _inflight_lock:
            _inflight -= 1
