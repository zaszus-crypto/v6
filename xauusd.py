#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD BOT v28.1 — INSTITUTIONAL HARDENED EDITION (Multi-Source)
=================================================================
Provider chain: OANDA v20 → TwelveData (2 keys) → Finnhub → Yahoo Futures
Validation: Gold-API spot (non-blocking sanity cross-check)
Fixes preserved from v28.0: P0-1..P0-5, P1-1..P1-8, P2-1, P2-3, P2-7, P2-10.
New in v28.1: multi-provider fallback, schema validation, circuit breaker.
"""

import os, sys, json, logging, time as py_time, hashlib, fcntl, pickle
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, roc_auc_score

# ===========================================================================
# ENV HELPERS
# ===========================================================================
def _env_float(k, d):
    v = (os.getenv(k) or "").strip()
    try: return float(v) if v else d
    except: return d
def _env_int(k, d):
    v = (os.getenv(k) or "").strip()
    try: return int(v) if v else d
    except: return d
def _env_bool(k, d=False):
    v = (os.getenv(k) or "").strip().lower()
    return v in ("true","1","yes","on") if v else d
def _env_str(k, d=""):
    return (os.getenv(k) or "").strip() or d

# ===========================================================================
# CONFIG
# ===========================================================================
@dataclass(frozen=True)
class Config:
    # ── Notifications & state ──────────────────────────────
    telegram_token: str = _env_str("TELEGRAM_TOKEN")
    telegram_chat_id: str = _env_str("TELEGRAM_CHAT_ID")
    gist_token: str = _env_str("GH_GIST_TOKEN")
    gist_id: str = _env_str("GH_GIST_ID")
    backtest_mode: bool = _env_bool("BACKTEST_MODE", False)

    # ── Data providers ─────────────────────────────────────
    oanda_api_key: str = _env_str("OANDA_API_KEY")
    oanda_env: str = _env_str("OANDA_ENV", "practice")
    twelvedata_key: str = _env_str("TWELVE_DATA_KEY")
    twelvedata_key_2: str = _env_str("TWELVE_DATA_KEY_2")
    finnhub_key: str = _env_str("FINNHUB_KEY")
    goldapi_key: str = _env_str("GOLD_API_KEY")
    price_deviation_max_pct: float = _env_float("PRICE_DEVIATION_MAX_PCT", 1.0)

    # ── Time-series / model ────────────────────────────────
    timeframe_minutes: int = 15
    min_candles: int = 300
    forward_bars: int = 8
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    risk_reward: float = 2.0
    spread_usd: float = 0.30
    slippage_usd: float = 0.10

    base_min_signal_probability: float = 0.38
    max_threshold: float = 0.50
    min_threshold: float = 0.35
    min_oos_auc: float = 0.45
    min_oos_precision: float = 0.45

    max_entropy_threshold: float = 0.99
    min_hurst_threshold: float = 0.50

    cooldown_minutes: int = 120
    max_daily_signals: int = 6
    max_daily_loss_pct: float = 4.0
    max_weekly_drawdown_pct: float = 8.0
    max_consecutive_losses: int = 4
    extreme_atr_percentile: float = 0.995

    account_equity: float = _env_float("ACCOUNT_EQUITY", 1000.0)
    risk_percent: float = _env_float("RISK_PERCENT", 1.0)
    contract_size: float = 100.0
    max_lot: float = 1.0
    min_lot: float = 0.01

    hurst_window: int = 150
    hurst_stride: int = 5
    entropy_window: int = 150
    entropy_stride: int = 5

    model_cache_path: str = str(Path.home() / ".xauusd_model.pkl")
    model_max_age_minutes: int = 60 * 12
    state_lock_path: str = str(Path.home() / ".xauusd_state.lock")
    state_lock_timeout_sec: int = 300
    state_path: str = str(Path.home() / ".xauusd_state.json")
    kill_switch_env: str = "KILL_SWITCH"

CONFIG = Config()

FEATURES = [
    "rsi","adx","di_difference","ema_distance","ema_slope",
    "trend_strength","atr_pct","atr_regime","realized_vol","vol_zscore",
    "bb_position","keltner_position","hurst","shannon_entropy",
    "body_ratio","upper_wick_ratio","lower_wick_ratio","bos_score",
    "liquidity_sweep_score","return_1","return_3","return_5",
    "volume_zscore","volume_ratio","hour_sin","hour_cos",
    "dayofweek_sin","dayofweek_cos",
    "session_london","session_newyork","session_overlap",
]

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stderr, force=True)
log = logging.getLogger("xauusd_v281")

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

# ===========================================================================
# FILE LOCK
# ===========================================================================
class FileLock:
    def __init__(self, path: str, timeout: int = 300):
        self.path = path; self.timeout = timeout; self._fh = None
    def __enter__(self):
        self._fh = open(self.path, "a+")
        deadline = py_time.time() + self.timeout
        while py_time.time() < deadline:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                py_time.sleep(0.5)
        self._fh.close()
        raise TimeoutError(f"State lock timeout after {self.timeout}s")
    def __exit__(self, *a):
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
        except Exception: pass

# ===========================================================================
# STATE
# ===========================================================================
def _default_state() -> Dict[str, Any]:
    return {
        "version": 281,
        "_revision": 0,
        "last_signal_time": None,
        "daily_signals_count": 0,
        "daily_pnl_usd": 0.0,
        "last_reset_date": now_utc().strftime("%Y-%m-%d"),
        "active_signals": [],
        "performance_history": {"wins": 0, "losses": 0},
        "adaptive_threshold": CONFIG.base_min_signal_probability,
        "recent_outcomes": [],
        "consecutive_losses": 0,
        "week_start_equity": CONFIG.account_equity,
        "week_start_date": now_utc().strftime("%Y-%W"),
        "last_data_source": None,
        "open_signal_ids": [],
    }

def _merge_defaults(s: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in _default_state().items():
        if k not in s: s[k] = v
    return s

def load_state() -> Dict[str, Any]:
    if os.path.exists(CONFIG.state_path):
        try:
            with open(CONFIG.state_path, "r", encoding="utf-8") as f:
                s = _merge_defaults(json.load(f))
            if s.get("adaptive_threshold", 0.60) > 0.50:
                s["adaptive_threshold"] = CONFIG.base_min_signal_probability
            log.info("✅ State loaded from local (revision=%s)", s.get("_revision"))
            return s
        except Exception as e:
            log.warning("Local state corrupt: %s", e)
    if CONFIG.gist_token and CONFIG.gist_id:
        for attempt in range(1, 4):
            try:
                r = requests.get(f"https://api.github.com/gists/{CONFIG.gist_id}",
                    headers={"Authorization": f"token {CONFIG.gist_token}"}, timeout=15)
                if r.status_code == 200:
                    for fn, fc in r.json().get("files", {}).items():
                        if fn.endswith("bot_state.json"):
                            s = _merge_defaults(json.loads(fc["content"]))
                            log.info("✅ State loaded from Gist (cold start)")
                            return s
            except Exception as e:
                log.warning("Gist load attempt %d: %s", attempt, e)
            py_time.sleep(2 ** attempt)
    return _default_state()

def save_state(state: Dict[str, Any]) -> bool:
    if CONFIG.backtest_mode:
        log.info("🧪 BACKTEST_MODE — state write skipped")
        return True
    state["_revision"] = int(state.get("_revision", 0)) + 1
    try:
        tmp = CONFIG.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, CONFIG.state_path)
    except Exception as e:
        log.error("Local save failed: %s", e); return False
    if CONFIG.gist_token and CONFIG.gist_id:
        for attempt in range(1, 3):
            try:
                r = requests.patch(f"https://api.github.com/gists/{CONFIG.gist_id}",
                    json={"files": {"bot_state.json": {"content": json.dumps(state, indent=2)}}},
                    headers={"Authorization": f"token {CONFIG.gist_token}"}, timeout=15)
                if r.status_code == 200:
                    log.info("✅ State synced to Gist (revision=%s)", state["_revision"])
                    break
            except Exception as e:
                log.warning("Gist sync attempt %d: %s", attempt, e)
            py_time.sleep(2 ** attempt)
    return True

# ===========================================================================
# TELEGRAM
# ===========================================================================
def send_telegram(message: str) -> bool:
    if CONFIG.backtest_mode:
        log.info("🧪 BACKTEST_MODE — Telegram send skipped")
        return True
    if not CONFIG.telegram_token or not CONFIG.telegram_chat_id: return False
    url = f"https://api.telegram.org/bot{CONFIG.telegram_token}/sendMessage"
    payload = {"chat_id": CONFIG.telegram_chat_id, "text": message,
               "parse_mode": "HTML", "disable_web_page_preview": True}
    for attempt in range(1, 4):
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 200: return True
            if r.status_code == 429:
                py_time.sleep(int(r.json().get("parameters", {}).get("retry_after", 5)) + 1)
                continue
            if 500 <= r.status_code < 600:
                py_time.sleep(2 * attempt); continue
        except Exception:
            py_time.sleep(2 * attempt)
    return False

# ===========================================================================
# DATA — Multi-provider with validation
# ===========================================================================
REQUIRED_COLS = ["datetime", "open", "high", "low", "close", "volume"]

def _validate_ohlc(df: Optional[pd.DataFrame], name: str) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    if len(df) < CONFIG.min_candles:
        log.warning("Schema %s: only %d bars < %d", name, len(df), CONFIG.min_candles)
        return None
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        log.warning("Schema %s: missing %s", name, missing)
        return None
    if not df["datetime"].is_monotonic_increasing:
        df = df.sort_values("datetime").reset_index(drop=True)
    if df[["open","high","low","close"]].isna().any().any():
        log.warning("Schema %s: NaN in OHLC", name)
        return None
    if (df["close"] <= 0).any():
        log.warning("Schema %s: non-positive close", name)
        return None
    return df

# ── Provider 1: OANDA v20 (broker-grade) ────────────────────────────
def _fetch_oanda(interval: str, bars: int) -> Optional[pd.DataFrame]:
    if not CONFIG.oanda_api_key:
        return None
    gran_map = {"15min": "M15", "1h": "H1", "5min": "M5", "1min": "M1"}
    gran = gran_map.get(interval, "M15")
    host = ("https://api-fxpractice.oanda.com" if CONFIG.oanda_env == "practice"
            else "https://api-fxtrade.oanda.com")
    url = f"{host}/v3/instruments/XAU_USD/candles"
    params = {"granularity": gran, "count": min(bars, 5000), "price": "M"}
    headers = {"Authorization": f"Bearer {CONFIG.oanda_api_key}"}
    for attempt in range(1, 3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            if r.status_code == 401:
                log.error("OANDA: auth failed (check OANDA_API_KEY)"); return None
            if r.status_code == 429:
                py_time.sleep(20); continue
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            candles = r.json().get("candles", [])
            if not candles:
                raise RuntimeError("empty candles")
            rows = []
            for c in candles:
                if not c.get("complete", False):
                    continue  # skip forming bar
                m = c["mid"]
                rows.append({
                    "datetime": pd.to_datetime(c["time"], utc=True),
                    "open": float(m["o"]), "high": float(m["h"]),
                    "low": float(m["l"]), "close": float(m["c"]),
                    "volume": int(c.get("volume", 0)),
                })
            if len(rows) < CONFIG.min_candles:
                raise RuntimeError(f"only {len(rows)} complete candles")
            return pd.DataFrame(rows)
        except Exception as e:
            log.warning("OANDA attempt %d: %s", attempt, e)
            py_time.sleep(2 ** attempt)
    return None

# ── Provider 2: TwelveData (2-key rotation) ─────────────────────────
def _fetch_twelvedata_multi(interval: str, bars: int) -> Optional[pd.DataFrame]:
    keys = [k for k in [CONFIG.twelvedata_key, CONFIG.twelvedata_key_2] if k]
    if not keys: return None
    for i, key in enumerate(keys, 1):
        df = _fetch_twelvedata_single(key, interval, bars)
        if df is not None and len(df) >= CONFIG.min_candles:
            return df
        if i < len(keys):
            log.info("TwelveData key #%d failed, trying key #%d", i, i + 1)
    return None

def _fetch_twelvedata_single(api_key: str, interval: str, bars: int) -> Optional[pd.DataFrame]:
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": "XAU/USD", "interval": interval, "outputsize": min(bars, 5000),
              "apikey": api_key, "format": "JSON", "timezone": "UTC"}
    for attempt in range(1, 3):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                py_time.sleep(20); continue
            data = r.json()
            if "code" in data and data.get("code") not in (200, "200"):
                raise RuntimeError(f"API: {data.get('message', data['code'])}")
            values = data.get("values", [])
            if not values:
                raise RuntimeError("empty values")
            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            # ✅ FIX: missing volume column safe
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
            else:
                df["volume"] = 0.0
            df = df.sort_values("datetime").reset_index(drop=True)
            return df[["datetime","open","high","low","close","volume"]]
        except Exception as e:
            log.warning("TwelveData attempt %d: %s", attempt, e)
            py_time.sleep(2 ** attempt)
    return None

# ── Provider 3: Finnhub ─────────────────────────────────────────────
def _fetch_finnhub(interval: str, bars: int) -> Optional[pd.DataFrame]:
    if not CONFIG.finnhub_key: return None
    res_map = {"15min": "15", "1h": "60", "5min": "5", "1min": "1"}
    resolution = res_map.get(interval, "15")
    now = int(now_utc().timestamp())
    start = now - bars * CONFIG.timeframe_minutes * 60 * 2
    url = "https://finnhub.io/api/v1/forex/candle"
    params = {"symbol": "OANDA:XAU_USD", "resolution": resolution,
              "from": start, "to": now, "token": CONFIG.finnhub_key}
    for attempt in range(1, 3):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                py_time.sleep(60); continue
            d = r.json()
            if d.get("s") != "ok":
                raise RuntimeError(f"status={d.get('s')}")
            if not d.get("c"):
                raise RuntimeError("empty c")
            df = pd.DataFrame({
                "datetime": pd.to_datetime(d["t"], unit="s", utc=True),
                "open": d["o"], "high": d["h"], "low": d["l"], "close": d["c"],
                "volume": d.get("v", [0]*len(d["c"])),
            })
            return df.tail(bars).reset_index(drop=True)
        except Exception as e:
            log.warning("Finnhub attempt %d: %s", attempt, e)
            py_time.sleep(2 ** attempt)
    return None

# ── Provider 4: Yahoo Futures (last resort) ─────────────────────────
def _fetch_yahoo_futures(interval: str, bars: int) -> Optional[pd.DataFrame]:
    headers = {"User-Agent": "Mozilla/5.0"}
    yf_int = "15m" if interval == "15min" else interval
    for host in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
        try:
            url = f"https://{host}/v8/finance/chart/GC=F?interval={yf_int}&range=60d"
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code != 200: continue
            res = r.json().get("chart", {}).get("result", [])
            if not res: continue
            res = res[0]
            q = res.get("indicators", {}).get("quote", [{}])[0]
            df = pd.DataFrame({
                "datetime": pd.to_datetime(res["timestamp"], unit="s", utc=True),
                "open": q.get("open"), "high": q.get("high"),
                "low": q.get("low"), "close": q.get("close"),
                "volume": q.get("volume", [0]*len(res["timestamp"])),
            }).dropna()
            if df.empty: continue
            return df.tail(bars).reset_index(drop=True)
        except Exception as e:
            log.warning("Yahoo %s: %s", host, e)
    return None

# ── Validation: external spot cross-check ───────────────────────────
def _fetch_external_spot() -> Optional[float]:
    if not CONFIG.goldapi_key: return None
    try:
        r = requests.get("https://www.goldapi.io/api/XAU/USD",
                         headers={"x-access-token": CONFIG.goldapi_key}, timeout=10)
        if r.status_code == 200:
            return float(r.json().get("price", 0)) or None
    except Exception:
        pass
    return None

def _price_matches(df: pd.DataFrame, ext_spot: Optional[float], name: str) -> bool:
    if ext_spot is None: return True
    if name == "YAHOO_FUTURES": return True  # basis differs; skip
    last = float(df["close"].iloc[-1])
    dev = abs(last - ext_spot) / ext_spot * 100
    if dev > CONFIG.price_deviation_max_pct:
        log.warning("⚠️ %s dev %.2f%% > %.2f%% vs ext %.2f — REJECT",
                    name, dev, CONFIG.price_deviation_max_pct, ext_spot)
        return False
    log.info("✅ %s validated: dev=%.3f%% vs ext %.2f", name, dev, ext_spot)
    return True

# ── Orchestrator ────────────────────────────────────────────────────
def fetch_xauusd_data(interval: str = "15min", bars: int = 5000) -> Tuple[pd.DataFrame, str]:
    ext_spot = _fetch_external_spot()
    if ext_spot:
        log.info("🔎 External spot: %.2f USD (Gold-API)", ext_spot)

    providers = [
        ("OANDA_V20",        _fetch_oanda),
        ("TWELVE_DATA_SPOT", _fetch_twelvedata_multi),
        ("FINNHUB_SPOT",     _fetch_finnhub),
        ("YAHOO_FUTURES",    _fetch_yahoo_futures),
    ]
    errors = []
    for name, fn in providers:
        try:
            df = fn(interval, bars)
            df = _validate_ohlc(df, name)
            if df is None:
                errors.append(f"{name}: invalid"); continue
            if not _price_matches(df, ext_spot, name):
                errors.append(f"{name}: deviation"); continue
            log.info("✅ Data: %s | %d bars | last=%.2f",
                     name, len(df), float(df["close"].iloc[-1]))
            return df, name
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}")
            log.warning("Provider %s failed: %s", name, e)
    raise RuntimeError(f"All sources failed: {'; '.join(errors)}")

def sanitize_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 2: return None
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    age_min = (now_utc() - df["datetime"].iloc[-1]).total_seconds() / 60
    if age_min < -1:
        log.warning("Future timestamp — dropping last bar")
        df = df.iloc[:-1].reset_index(drop=True)
    elif age_min < CONFIG.timeframe_minutes:
        df = df.iloc[:-1].reset_index(drop=True)
    return df if len(df) >= CONFIG.min_candles else None

# ===========================================================================
# FEATURES
# ===========================================================================
def wilder_atr(df, period=14):
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"], (df["high"]-prev).abs(), (df["low"]-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    s = series[np.isfinite(series)]
    if len(s) < max_lag * 2: return 0.5
    lags = np.arange(2, max_lag + 1)
    tau = np.array([np.std(s[lag:] - s[:-lag]) for lag in lags])
    valid = tau > 0
    if valid.sum() < 3: return 0.5
    try:
        return float(np.clip(np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0], 0.0, 1.0))
    except Exception:
        return 0.5

def shannon_entropy(returns: np.ndarray, bins: int = 10) -> float:
    r = returns[np.isfinite(returns)]
    if len(r) < 20: return 1.0
    hist, _ = np.histogram(r, bins=bins)
    p = hist / (hist.sum() + 1e-12); p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(bins))

def rolling_hurst(close: np.ndarray, window: int, stride: int) -> np.ndarray:
    n = len(close); log_s = np.log(np.maximum(close, 1e-10)); out = np.full(n, np.nan)
    for i in range(window - 1, n, stride):
        out[i] = hurst_exponent(log_s[i - window + 1 : i + 1])
    return pd.Series(out).ffill().values  # NO bfill (P0-2)

def rolling_entropy(returns: np.ndarray, window: int, stride: int) -> np.ndarray:
    n = len(returns); out = np.full(n, np.nan)
    for i in range(window - 1, n, stride):
        out[i] = shannon_entropy(returns[i - window + 1 : i + 1])
    return pd.Series(out).ffill().values

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["close"].diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + g / (l + 1e-12))
    df["atr"] = wilder_atr(df, CONFIG.atr_period)
    df["atr_pct"] = df["atr"] / (df["close"] + 1e-12)
    df["atr_regime"] = df["atr"] / (df["atr"].rolling(100).mean() + 1e-12)
    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_trend"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_distance"] = (df["ema_fast"] - df["ema_slow"]) / (df["atr"] + 1e-12)
    df["ema_slope"] = (df["ema_fast"] - df["ema_fast"].shift(3)) / (df["atr"] + 1e-12)
    df["trend_strength"] = (df["close"] - df["ema_trend"]) / (df["atr"] + 1e-12)

    up, dn = df["high"].diff(), -df["low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (df["atr"] + 1e-12)
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (df["atr"] + 1e-12)
    df["adx"] = (100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)).ewm(alpha=1/14, adjust=False).mean()
    df["di_difference"] = pdi - mdi

    m, s = df["close"].rolling(20).mean(), df["close"].rolling(20).std()
    df["bb_position"] = (df["close"] - m) / (2 * s + 1e-12)
    km = df["close"].ewm(span=20, adjust=False).mean()
    kb = df["atr"].ewm(span=20, adjust=False).mean() * 2
    df["keltner_position"] = (df["close"] - km) / (kb + 1e-12)

    ret = df["close"].pct_change()
    df["realized_vol"] = ret.rolling(20).std() * np.sqrt(24 * 60 / CONFIG.timeframe_minutes)
    df["vol_zscore"] = (df["realized_vol"] - df["realized_vol"].rolling(100).mean()) / (df["realized_vol"].rolling(100).std() + 1e-12)
    df["hurst"] = rolling_hurst(df["close"].values, CONFIG.hurst_window, CONFIG.hurst_stride)
    df["shannon_entropy"] = rolling_entropy(ret.fillna(0).values, CONFIG.entropy_window, CONFIG.entropy_stride)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (rng + 1e-12)
    df["upper_wick_ratio"] = (df["high"] - df[["open","close"]].max(axis=1)) / (rng + 1e-12)
    df["lower_wick_ratio"] = (df[["open","close"]].min(axis=1) - df["low"]) / (rng + 1e-12)

    sw = 5
    df["prior_high"] = df["high"].rolling(sw).max().shift(1)
    df["prior_low"]  = df["low"].rolling(sw).min().shift(1)
    df["bos_score"] = np.where((df["high"] > df["prior_high"]) & (df["close"] > df["prior_high"]), 1.0,
                        np.where((df["low"] < df["prior_low"]) & (df["close"] < df["prior_low"]), -1.0, 0.0))
    df["liquidity_sweep_score"] = np.where((df["low"] < df["prior_low"]) & (df["close"] > df["prior_low"]), 1.0,
                                    np.where((df["high"] > df["prior_high"]) & (df["close"] < df["prior_high"]), -1.0, 0.0))
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    if "volume" in df.columns and (df["volume"] > 0).sum() > 20:
        v = df["volume"].fillna(0); vm, vs = v.rolling(20).mean(), v.rolling(20).std()
        df["volume_zscore"] = ((v - vm) / (vs + 1e-12)).clip(-5, 5)
        df["volume_ratio"]  = (v / (vm + 1e-12)).clip(0, 10)
    else:
        df["volume_zscore"], df["volume_ratio"] = 0.0, 1.0

    hr = df["datetime"].dt.hour; dow = df["datetime"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * dow / 7)
    df["session_london"]  = hr.between(7, 15).astype(int)
    df["session_newyork"] = hr.between(13, 21).astype(int)
    df["session_overlap"] = hr.between(13, 16).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df if len(df) >= CONFIG.min_candles else df.iloc[0:0]

# ===========================================================================
# TRIPLE BARRIER (vectorized)
# ===========================================================================
def triple_barrier_labels(df: pd.DataFrame, fwd: int, atr_mult: float) -> pd.DataFrame:
    df = df.copy()
    c, h, l, a = df["close"].values, df["high"].values, df["low"].values, df["atr"].values
    n = len(df); labels = np.zeros(n, dtype=int)
    for i in range(n - fwd):
        if not np.isfinite(a[i]) or a[i] <= 0: continue
        up = c[i] + a[i] * atr_mult; dn = c[i] - a[i] * atr_mult
        hh = h[i+1:i+fwd+1]; ll = l[i+1:i+fwd+1]
        hit_up = hh >= up; hit_dn = ll <= dn
        trig = hit_up | hit_dn
        if not trig.any(): labels[i] = 0; continue
        j = int(np.argmax(trig))
        if hit_up[j] and hit_dn[j]: labels[i] = 0
        elif hit_up[j]: labels[i] = 1
        elif hit_dn[j]: labels[i] = -1
    df["label"] = labels
    return df

# ===========================================================================
# ML
# ===========================================================================
def build_model():
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=20,
                                class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.03,
                                    subsample=0.8, random_state=42)
    return VotingClassifier([("rf", rf), ("gb", gb)], voting="soft")

def purged_embargoed_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 4) -> Dict[str, float]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs, precs = [], []
    for tr_idx, va_idx in tscv.split(X):
        purge_end = va_idx[0] - CONFIG.forward_bars
        tr_purged = tr_idx[tr_idx < purge_end]
        if len(tr_purged) < 100 or len(np.unique(y[tr_purged])) < 2: continue
        try:
            m = build_model()
            w = np.exp(-(len(tr_purged) - 1 - np.arange(len(tr_purged))) / 500.0)
            m.fit(X[tr_purged], y[tr_purged], sample_weight=w)
            proba = m.predict_proba(X[va_idx]); preds = m.predict(X[va_idx])
            dmask = y[va_idx] != 0
            if dmask.sum() < 10: continue
            y_dir, p_dir = y[va_idx][dmask], preds[dmask]
            mask_dir = p_dir != 0
            if mask_dir.sum() >= 10 and len(np.unique(y_dir[mask_dir])) >= 2:
                precs.append(precision_score(y_dir[mask_dir], p_dir[mask_dir], average="weighted", zero_division=0))
            classes = list(m.classes_)
            if 1 in classes and -1 in classes:
                bi = classes.index(1); y_bin = (y_dir == 1).astype(int)
                if len(np.unique(y_bin)) >= 2:
                    aucs.append(roc_auc_score(y_bin, proba[dmask][:, bi]))
        except Exception:
            continue
    if not aucs: return {"valid": False, "auc_mean": 0.5, "prec_mean": 0.0}
    return {"valid": True, "auc_mean": float(np.mean(aucs)),
            "prec_mean": float(np.mean(precs)) if precs else 0.0}

def _load_cached_model() -> Optional[Dict[str, Any]]:
    if not os.path.exists(CONFIG.model_cache_path): return None
    try:
        age_min = (py_time.time() - os.path.getmtime(CONFIG.model_cache_path)) / 60
        if age_min > CONFIG.model_max_age_minutes: return None
        with open(CONFIG.model_cache_path, "rb") as f: obj = pickle.load(f)
        log.info("📦 Loaded cached model (age %.1fh)", age_min / 60)
        return obj
    except Exception as e:
        log.warning("Cache load failed: %s", e); return None

def _save_cached_model(obj: Dict[str, Any]) -> None:
    if CONFIG.backtest_mode: return
    try:
        tmp = CONFIG.model_cache_path + ".tmp"
        with open(tmp, "wb") as f: pickle.dump(obj, f)
        os.replace(tmp, CONFIG.model_cache_path)
    except Exception as e:
        log.warning("Cache save failed: %s", e)

def train_model(df_feat: pd.DataFrame, force: bool = False) -> Dict[str, Any]:
    if not force:
        cached = _load_cached_model()
        if cached and cached.get("valid"): return cached
    df_lab = triple_barrier_labels(df_feat, CONFIG.forward_bars, CONFIG.atr_sl_multiplier)
    train_df = df_lab.iloc[: -CONFIG.forward_bars]
    X, y = train_df[FEATURES].values, train_df["label"].values
    if len(X) < 400 or len(np.unique(y)) < 2:
        return {"valid": False, "reason": "insufficient_data"}
    wf = purged_embargoed_cv(X, y)
    if not wf["valid"] or wf["auc_mean"] < CONFIG.min_oos_auc or wf["prec_mean"] < CONFIG.min_oos_precision:
        return {"valid": False, "reason": f"wf_gate AUC:{wf['auc_mean']:.2f} PREC:{wf['prec_mean']:.2f}"}
    n = len(y)
    weights = np.exp(-(n - 1 - np.arange(n)) / 500.0)  # P0-1: newer = higher
    model = build_model(); model.fit(X, y, sample_weight=weights)
    out = {"valid": True, "model": model, "auc": wf["auc_mean"], "prec": wf["prec_mean"]}
    _save_cached_model(out)
    log.info("🎓 Model trained | AUC=%.3f PREC=%.3f n=%d", wf["auc_mean"], wf["prec_mean"], len(X))
    return out

# ===========================================================================
# SIGNAL RESOLUTION
# ===========================================================================
def resolve_active_signals(state: Dict[str, Any], df: pd.DataFrame) -> None:
    if not state.get("active_signals"): return
    now, still_active, resolved = now_utc(), [], []
    for sig in state["active_signals"]:
        try:
            sig_time = datetime.fromisoformat(sig["time"])
            expires  = datetime.fromisoformat(sig["expires_at"])
        except Exception: continue
        sig_df = df[df["datetime"] > sig_time]
        outcome, exit_price = None, None
        for _, bar in sig_df.iterrows():
            if sig["type"] == "BUY":
                if bar["low"]  <= sig["sl"]: outcome, exit_price = "loss", sig["sl"]; break
                if bar["high"] >= sig["tp"]: outcome, exit_price = "win",  sig["tp"]; break
            else:
                if bar["high"] >= sig["sl"]: outcome, exit_price = "loss", sig["sl"]; break
                if bar["low"]  <= sig["tp"]: outcome, exit_price = "win",  sig["tp"]; break
        if outcome is None:
            if now >= expires and len(sig_df):
                exit_price = float(sig_df["close"].iloc[-1])
                exit_price -= CONFIG.spread_usd / 2 if sig["type"] == "BUY" else -CONFIG.spread_usd / 2
                pnl = ((exit_price - sig["entry"]) if sig["type"] == "BUY"
                       else (sig["entry"] - exit_price)) * sig["lot"] * CONFIG.contract_size
                resolved.append((sig, "timeout", exit_price, pnl))
            else:
                still_active.append(sig)
        else:
            cost = (CONFIG.spread_usd + CONFIG.slippage_usd) * sig["lot"] * CONFIG.contract_size
            pnl = (sig["risk_usd"] * CONFIG.risk_reward if outcome == "win" else -sig["risk_usd"]) - cost
            resolved.append((sig, outcome, exit_price, pnl))

    for sig, outcome, exit_price, pnl in resolved:
        if outcome == "win":
            state["performance_history"]["wins"] += 1
            state["consecutive_losses"] = 0
        elif outcome == "loss":
            state["performance_history"]["losses"] += 1
            state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1
        state["daily_pnl_usd"] += pnl
        state["recent_outcomes"].append({"outcome": outcome, "prob": sig.get("prob", 0.0),
                                         "signal_id": sig.get("signal_id")})
        log.info("📊 Resolved %s → %s (PnL $%.2f)", sig["type"], outcome, pnl)
    state["recent_outcomes"] = state["recent_outcomes"][-50:]
    state["active_signals"] = still_active
    state["open_signal_ids"] = [s.get("signal_id") for s in still_active if s.get("signal_id")]

    recent = [o for o in state["recent_outcomes"] if o["outcome"] in ("win","loss")]
    if len(recent) >= 15:
        win_rate = sum(1 for o in recent[-20:] if o["outcome"] == "win") / len(recent[-20:])
        new_th = float(np.clip(CONFIG.base_min_signal_probability + (0.55 - win_rate) * 0.5,
                                CONFIG.min_threshold, CONFIG.max_threshold))
        if abs(new_th - state["adaptive_threshold"]) > 0.005:
            state["adaptive_threshold"] = new_th

# ===========================================================================
# TRADE SIZING
# ===========================================================================
def compute_trade(sig_type: str, df_feat: pd.DataFrame, prob: float,
                  source: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    last = df_feat.iloc[-1]
    price, atr = float(last["close"]), float(last["atr"])
    spread_half = CONFIG.spread_usd / 2
    sl_dist_raw = atr * CONFIG.atr_sl_multiplier
    tp_dist_raw = sl_dist_raw * CONFIG.risk_reward
    sl_dist_net = sl_dist_raw + CONFIG.spread_usd
    tp_dist_net = tp_dist_raw - CONFIG.spread_usd
    if tp_dist_net <= 0: return None

    if sig_type == "BUY":
        entry = price + spread_half; sl = entry - sl_dist_net; tp = entry + tp_dist_net
    else:
        entry = price - spread_half; sl = entry + sl_dist_net; tp = entry - tp_dist_net

    risk_target = CONFIG.account_equity * (CONFIG.risk_percent / 100.0)
    sl_ticks = abs(entry - sl)
    if sl_ticks <= 0: return None
    lot = round(risk_target / (sl_ticks * CONFIG.contract_size), 2)
    if lot < CONFIG.min_lot:
        log.info("⛔ Computed lot %.4f < min_lot; risk-adjusted skip.", lot); return None
    lot = min(lot, CONFIG.max_lot)
    actual_risk = lot * sl_ticks * CONFIG.contract_size
    if actual_risk > risk_target * 1.05:
        log.warning("⛔ Actual risk $%.2f > target $%.2f", actual_risk, risk_target); return None

    signal_id = hashlib.sha256(
        f"{sig_type}:{entry:.2f}:{now_utc().isoformat()}".encode()
    ).hexdigest()[:16]

    return {
        "signal_id": signal_id, "type": sig_type,
        "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
        "lot": lot, "risk_usd": round(actual_risk, 2), "prob": prob, "source": source,
        "time": now_utc().isoformat(),
        "expires_at": (now_utc() + timedelta(minutes=CONFIG.timeframe_minutes * CONFIG.forward_bars * 2)).isoformat(),
    }

def format_message(t: Dict, source: str) -> str:
    return (
        f"<b>{'🟢' if t['type'] == 'BUY' else '🔴'} XAUUSD {t['type']}</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕒 {t['time'][:16].replace('T',' ')} UTC\n📡 Source: {source}\n"
        f"🔑 ID: {t['signal_id']}\n🎯 Prob: <b>{t['prob']*100:.1f}%</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 Entry: <b>{t['entry']:.2f}</b>\n🛑 SL: <b>{t['sl']:.2f}</b>\n🎯 TP: <b>{t['tp']:.2f}</b>\n"
        f"⚖️ Lot: <b>{t['lot']}</b> (Risk ${t['risk_usd']:.2f})\n"
        f"<i>Expires: {t['expires_at'][:16].replace('T',' ')} UTC</i>"
    )

# ===========================================================================
# CIRCUIT BREAKERS
# ===========================================================================
def weekly_reset_if_needed(state: Dict[str, Any]) -> None:
    wk = now_utc().strftime("%Y-%W")
    if state.get("week_start_date") != wk:
        state["week_start_date"] = wk
        state["week_start_equity"] = CONFIG.account_equity + state.get("daily_pnl_usd", 0.0)

def circuit_breakers_ok(state: Dict[str, Any], df_feat: pd.DataFrame) -> Tuple[bool, str]:
    daily_limit = -CONFIG.max_daily_loss_pct * CONFIG.account_equity / 100.0
    if state["daily_pnl_usd"] <= daily_limit:
        return False, f"Daily loss limit ({state['daily_pnl_usd']:.2f})"
    week_dd = -CONFIG.max_weekly_drawdown_pct * CONFIG.account_equity / 100.0
    if state["daily_pnl_usd"] <= week_dd:
        return False, f"Weekly DD ({state['daily_pnl_usd']:.2f})"
    if state.get("consecutive_losses", 0) >= CONFIG.max_consecutive_losses:
        return False, f"{state['consecutive_losses']} consecutive losses"
    atr_s = df_feat["atr"].tail(500)
    if len(atr_s) > 100 and float(df_feat.iloc[-1]["atr"]) >= atr_s.quantile(CONFIG.extreme_atr_percentile):
        return False, f"ATR > {CONFIG.extreme_atr_percentile*100:.1f}pct tail"
    return True, ""

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    log.info("=" * 60)
    log.info("XAUUSD v28.1 — Institutional Hardened (Multi-Source)")
    log.info("=" * 60)

    with FileLock(CONFIG.state_lock_path, CONFIG.state_lock_timeout_sec):
        state = load_state()
        weekly_reset_if_needed(state)
        if state.get("last_reset_date") != now_utc().strftime("%Y-%m-%d"):
            state["daily_signals_count"] = 0
            state["daily_pnl_usd"] = 0.0
            state["last_reset_date"] = now_utc().strftime("%Y-%m-%d")

        if (_env_bool(CONFIG.kill_switch_env, False) or now_utc().weekday() >= 5
                or (now_utc().weekday() == 4 and now_utc().hour >= 21)):
            log.info("📅 Market closed or Kill Switch active.")
            save_state(state); return

        try:
            df, source = fetch_xauusd_data()
            df = sanitize_df(df)
            if df is None:
                log.warning("⚠️ Insufficient sanitized data."); return
        except Exception as e:
            log.error("❌ All providers failed: %s", e); return

        state["last_data_source"] = source
        resolve_active_signals(state, df)
        save_state(state)

        if state["daily_signals_count"] >= CONFIG.max_daily_signals:
            log.info("⛔ Max daily signals."); return
        if state["daily_pnl_usd"] <= -(CONFIG.max_daily_loss_pct * CONFIG.account_equity / 100.0):
            log.info("⛔ Daily loss limit."); return
        if state.get("last_signal_time"):
            try:
                mins = (now_utc() - datetime.fromisoformat(state["last_signal_time"])).total_seconds() / 60
                if mins < CONFIG.cooldown_minutes:
                    log.info("⏸ Cooldown (%.0fm).", mins); return
            except Exception: pass

        df_feat = calculate_features(df)
        if len(df_feat) < CONFIG.min_candles:
            log.warning("⚠️ Insufficient features after NaN purge."); return
        last = df_feat.iloc[-1]

        h_val, e_val = float(last["hurst"]), float(last["shannon_entropy"])
        log.info("📊 Regime: hurst=%.3f (min=%.2f) | entropy=%.3f (max=%.2f)",
                 h_val, CONFIG.min_hurst_threshold, e_val, CONFIG.max_entropy_threshold)
        if e_val > CONFIG.max_entropy_threshold:
            log.info("🌊 SKIP: entropy %.3f > %.2f", e_val, CONFIG.max_entropy_threshold); return
        if h_val < CONFIG.min_hurst_threshold:
            log.info("🌊 SKIP: hurst %.3f < %.2f", h_val, CONFIG.min_hurst_threshold); return

        cb_ok, cb_reason = circuit_breakers_ok(state, df_feat)
        if not cb_ok:
            log.warning("⛔ Circuit breaker: %s", cb_reason); return

        models = train_model(df_feat)
        if not models.get("valid"):
            log.info("⚠️ Invalid model: %s", models.get("reason")); return

        X_live = last[FEATURES].values.reshape(1, -1)
        model = models["model"]
        pred = int(model.predict(X_live)[0])
        proba = model.predict_proba(X_live)[0]
        prob_dict = {int(c): float(proba[i]) for i, c in enumerate(model.classes_)}

        if pred == 0:
            log.info("ℹ️ Neutral prediction."); return

        sig_type = "BUY" if pred == 1 else "SELL"
        prob = prob_dict.get(pred, 0.0)
        if prob < float(state["adaptive_threshold"]):
            log.info("📉 Prob %.2f < threshold %.2f.", prob, state["adaptive_threshold"]); return

        close_price, ema_trend = float(last["close"]), float(last["ema_trend"])
        adx, rsi = float(last["adx"]), float(last["rsi"])
        if sig_type == "BUY" and close_price < ema_trend:
            log.info("🛡️ Price < EMA50. Skip."); return
        if sig_type == "SELL" and close_price > ema_trend:
            log.info("🛡️ Price > EMA50. Skip."); return
        if adx < 20.0:
            log.info("🛡️ ADX %.1f < 20.", adx); return
        if sig_type == "BUY" and rsi > 75:
            log.info("🛡️ RSI overbought."); return
        if sig_type == "SELL" and rsi < 25:
            log.info("🛡️ RSI oversold."); return

        log.info("🔥 Signal confirmed | %s (Prob %.2f)", sig_type, prob)

        trade = compute_trade(sig_type, df_feat, prob, source, state)
        if trade is None:
            log.info("⛔ Trade sizing rejected."); return
        if trade["signal_id"] in state.get("open_signal_ids", []):
            log.warning("⚠️ Duplicate signal_id, skip."); return
        if not send_telegram(format_message(trade, source)):
            log.error("❌ Telegram failed; aborting state update."); return

        state["active_signals"].append(trade)
        state["open_signal_ids"].append(trade["signal_id"])
        state["daily_signals_count"] += 1
        state["last_signal_time"] = now_utc().isoformat()
        save_state(state)
        log.info("✅ %s sent | Entry=%.2f Lot=%.2f Risk=$%.2f",
                 sig_type, trade["entry"], trade["lot"], trade["risk_usd"])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.critical("💥 Fatal: %s", e, exc_info=True)
        sys.exit(1)
