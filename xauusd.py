#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD BOT v29.0 — THE PERFECT REDUNDANCY EDITION
=================================================
Data Chain: TwelveData -> Deriv -> YFinance (Offset) -> OANDA
Validation: Gold-API (Non-blocking)
Core Logic: Institutional Hardened (v28.x) + Smart Basis Offset
"""

import os, sys, json, logging, time as py_time, hashlib, fcntl, pickle, asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from deriv_api import DerivAPI
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, roc_auc_score

# ===========================================================================
# ENV HELPERS & CONFIG
# ===========================================================================
def _env_float(k, d):
    v = (os.getenv(k) or "").strip()
    try: return float(v) if v else d
    except: return d
def _env_bool(k, d=False):
    v = (os.getenv(k) or "").strip().lower()
    return v in ("true","1","yes","on") if v else d
def _env_str(k, d=""):
    return (os.getenv(k) or "").strip() or d

@dataclass(frozen=True)
class Config:
    telegram_token: str = _env_str("TELEGRAM_TOKEN")
    telegram_chat_id: str = _env_str("TELEGRAM_CHAT_ID")
    gist_token: str = _env_str("GH_GIST_TOKEN")
    gist_id: str = _env_str("GH_GIST_ID")
    backtest_mode: bool = _env_bool("BACKTEST_MODE", False)

    oanda_api_key: str = _env_str("OANDA_API_KEY")
    oanda_env: str = _env_str("OANDA_ENV", "practice")
    twelvedata_key: str = _env_str("TWELVE_DATA_KEY")
    twelvedata_key_2: str = _env_str("TWELVE_DATA_KEY_2")
    goldapi_key: str = _env_str("GOLD_API_KEY")
    
    price_deviation_max_pct: float = 1.0
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stderr, force=True)
log = logging.getLogger("xauusd_v29")
def now_utc() -> datetime: return datetime.now(timezone.utc)

# ===========================================================================
# STATE & LOCK
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
            except BlockingIOError: py_time.sleep(0.5)
        self._fh.close(); raise TimeoutError("State lock timeout")
    def __exit__(self, *a):
        try: fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN); self._fh.close()
        except Exception: pass

def _default_state() -> Dict[str, Any]:
    return {
        "version": 290, "_revision": 0, "last_signal_time": None,
        "daily_signals_count": 0, "daily_pnl_usd": 0.0,
        "last_reset_date": now_utc().strftime("%Y-%m-%d"),
        "active_signals": [], "performance_history": {"wins": 0, "losses": 0},
        "adaptive_threshold": CONFIG.base_min_signal_probability,
        "recent_outcomes": [], "consecutive_losses": 0,
        "week_start_equity": CONFIG.account_equity, "week_start_date": now_utc().strftime("%Y-%W"),
        "last_data_source": None, "open_signal_ids": [],
        # 🔥 NEW: Cache for YFinance Offset
        "last_spot_price": 0.0, "last_futures_basis": 0.0 
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
            log.info("✅ State loaded from local")
            return s
        except Exception: pass
    if CONFIG.gist_token and CONFIG.gist_id:
        for attempt in range(1, 4):
            try:
                r = requests.get(f"https://api.github.com/gists/{CONFIG.gist_id}", headers={"Authorization": f"token {CONFIG.gist_token}"}, timeout=15)
                if r.status_code == 200:
                    for fn, fc in r.json().get("files", {}).items():
                        if fn.endswith("bot_state.json"):
                            s = _merge_defaults(json.loads(fc["content"]))
                            log.info("✅ State loaded from Gist")
                            return s
            except Exception: py_time.sleep(2)
    return _default_state()

def save_state(state: Dict[str, Any]) -> bool:
    if CONFIG.backtest_mode: return True
    state["_revision"] = int(state.get("_revision", 0)) + 1
    try:
        with open(CONFIG.state_path + ".tmp", "w", encoding="utf-8") as f: json.dump(state, f)
        os.replace(CONFIG.state_path + ".tmp", CONFIG.state_path)
    except Exception as e: log.error("Local save failed: %s", e); return False
    
    if CONFIG.gist_token and CONFIG.gist_id:
        for attempt in range(1, 3):
            try:
                r = requests.patch(f"https://api.github.com/gists/{CONFIG.gist_id}", json={"files": {"bot_state.json": {"content": json.dumps(state)}}}, headers={"Authorization": f"token {CONFIG.gist_token}"}, timeout=15)
                if r.status_code == 200: break
            except Exception: py_time.sleep(2)
    return True

# ===========================================================================
# TELEGRAM
# ===========================================================================
def send_telegram(message: str) -> bool:
    if CONFIG.backtest_mode: return True
    if not CONFIG.telegram_token or not CONFIG.telegram_chat_id: return False
    url = f"https://api.telegram.org/bot{CONFIG.telegram_token}/sendMessage"
    for attempt in range(1, 4):
        try:
            r = requests.post(url, json={"chat_id": CONFIG.telegram_chat_id, "text": message, "parse_mode": "HTML"}, timeout=15)
            if r.status_code == 200: return True
            if r.status_code == 429: py_time.sleep(int(r.json().get("parameters", {}).get("retry_after", 5)) + 1)
        except Exception: py_time.sleep(2)
    return False

# ===========================================================================
# DATA PROVIDERS
# ===========================================================================
REQUIRED_COLS = ["datetime", "open", "high", "low", "close", "volume"]

def _validate_ohlc(df: Optional[pd.DataFrame], name: str) -> Optional[pd.DataFrame]:
    if df is None or len(df) < CONFIG.min_candles: return None
    if [c for c in REQUIRED_COLS if c not in df.columns]: return None
    if not df["datetime"].is_monotonic_increasing: df = df.sort_values("datetime").reset_index(drop=True)
    if df[["open","high","low","close"]].isna().any().any(): return None
    return df

# 1. TWELVEDATA
def _fetch_twelvedata_multi(interval: str, bars: int) -> Optional[pd.DataFrame]:
    keys = [k for k in [CONFIG.twelvedata_key, CONFIG.twelvedata_key_2] if k]
    for key in keys:
        df = _fetch_twelvedata_single(key, interval, bars)
        if df is not None: return df
    return None

def _fetch_twelvedata_single(api_key: str, interval: str, bars: int) -> Optional[pd.DataFrame]:
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": "XAU/USD", "interval": interval, "outputsize": min(bars, 5000), "apikey": api_key, "format": "JSON", "timezone": "UTC"}
    try:
        r = requests.get(url, params=params, timeout=20)
        values = r.json().get("values", [])
        if not values: return None
        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ["open", "high", "low", "close"]: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0) if "volume" in df.columns else 0.0
        return df.sort_values("datetime").reset_index(drop=True)[REQUIRED_COLS]
    except Exception: return None

# 2. DERIV (WebSocket via asyncio)
async def _fetch_deriv_async(interval: str, bars: int) -> Optional[pd.DataFrame]:
    gran_map = {"15min": 900, "1h": 3600, "5min": 300, "1min": 60}
    granularity = gran_map.get(interval, 900)
    api = DerivAPI(app_id=1089)
    try:
        response = await api.ticks_history({"ticks_history": "frxXAUUSD", "style": "candles", "granularity": granularity, "count": min(bars, 5000), "end": "latest"})
        if "candles" not in response: return None
        df = pd.DataFrame(response["candles"])
        df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
        df["volume"] = 0.0
        return df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})[REQUIRED_COLS].sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        log.warning(f"Deriv async failed: {e}"); return None
    finally: await api.disconnect()

def _fetch_deriv(interval: str, bars: int) -> Optional[pd.DataFrame]:
    try: return asyncio.run(_fetch_deriv_async(interval, bars))
    except Exception: return None

# 3. YFINANCE (Dynamic Offset)
def _fetch_yfinance_offset(interval: str, bars: int, state: Dict[str, Any]) -> Optional[pd.DataFrame]:
    try:
        # 1. Dapatkan Anchor Spot (Prioritas: State, lalu TwelveData)
        spot_anchor = state.get("last_spot_price", 0.0)
        if spot_anchor == 0.0:
            df_spot = _fetch_twelvedata_single(CONFIG.twelvedata_key, interval, 10)
            if df_spot is not None: spot_anchor = float(df_spot["close"].iloc[-1])
        
        # 2. Ambil Data Futures dari YFinance
        period = "5d" if "min" in interval else "1mo"
        df_fut = yf.download("GC=F", period=period, interval=interval, progress=False)
        if df_fut.empty: return None
        
        last_fut = float(df_fut['Close'].iloc[-1])
        
        # 3. Hitung Basis
        basis = state.get("last_futures_basis", 0.0)
        if spot_anchor > 0:
            basis = last_fut - spot_anchor
            state["last_futures_basis"] = basis # Update cache basis
            log.info(f"🔧 YFinance Offset applied: Basis={basis:.2f} (Futures {last_fut:.2f} vs Spot {spot_anchor:.2f})")
        else:
            log.warning("⚠️ YFinance using historical basis (no fresh spot anchor)")
            
        # 4. Sesuaikan DataFrame
        df = pd.DataFrame({
            "datetime": df_fut.index,
            "open": df_fut['Open'] - basis,
            "high": df_fut['High'] - basis,
            "low": df_fut['Low'] - basis,
            "close": df_fut['Close'] - basis,
            "volume": df_fut['Volume']
        }).reset_index(drop=True)
        
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df.tail(bars).reset_index(drop=True)
    except Exception as e:
        log.warning(f"YFinance Offset failed: {e}"); return None

# 4. OANDA
def _fetch_oanda(interval: str, bars: int) -> Optional[pd.DataFrame]:
    if not CONFIG.oanda_api_key: return None
    gran = {"15min": "M15", "1h": "H1", "5min": "M5"}.get(interval, "M15")
    host = "https://api-fxpractice.oanda.com" if CONFIG.oanda_env == "practice" else "https://api-fxtrade.oanda.com"
    try:
        r = requests.get(f"{host}/v3/instruments/XAU_USD/candles", params={"granularity": gran, "count": bars, "price": "M"}, headers={"Authorization": f"Bearer {CONFIG.oanda_api_key}"}, timeout=20)
        if r.status_code != 200: return None
        rows = []
        for c in r.json().get("candles", []):
            if not c.get("complete", False): continue
            m = c["mid"]
            rows.append({"datetime": pd.to_datetime(c["time"], utc=True), "open": float(m["o"]), "high": float(m["h"]), "low": float(m["l"]), "close": float(m["c"]), "volume": int(c.get("volume", 0))})
        return pd.DataFrame(rows) if rows else None
    except Exception: return None

# ORCHESTRATOR
def fetch_xauusd_data(state: Dict[str, Any], interval: str = "15min", bars: int = 5000) -> Tuple[pd.DataFrame, str]:
    providers = [
        ("TWELVE_DATA_SPOT", lambda: _fetch_twelvedata_multi(interval, bars)),
        ("DERIV",            lambda: _fetch_deriv(interval, bars)),
        ("YFINANCE_OFFSET",  lambda: _fetch_yfinance_offset(interval, bars, state)),
        ("OANDA_V20",        lambda: _fetch_oanda(interval, bars)),
    ]
    for name, fn in providers:
        try:
            df = _validate_ohlc(fn(), name)
            if df is not None:
                state["last_spot_price"] = float(df["close"].iloc[-1])
                log.info("✅ Data: %s | %d bars | last=%.2f", name, len(df), float(df["close"].iloc[-1]))
                return df, name
        except Exception as e:
            log.warning("Provider %s failed: %s", name, e)
    raise RuntimeError("All data sources failed")

def sanitize_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 2: return None
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    age_min = (now_utc() - df["datetime"].iloc[-1]).total_seconds() / 60
    if age_min < -1 or age_min < CONFIG.timeframe_minutes:
        df = df.iloc[:-1].reset_index(drop=True)
    return df if len(df) >= CONFIG.min_candles else None

# ===========================================================================
# QUANT ENGINE (Unchanged from v28.1 - Intact)
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
    try: return float(np.clip(np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0], 0.0, 1.0))
    except Exception: return 0.5

def shannon_entropy(returns: np.ndarray, bins: int = 10) -> float:
    r = returns[np.isfinite(returns)]
    if len(r) < 20: return 1.0
    hist, _ = np.histogram(r, bins=bins); p = hist / (hist.sum() + 1e-12); p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(bins))

def rolling_hurst(close: np.ndarray, window: int, stride: int) -> np.ndarray:
    n = len(close); log_s = np.log(np.maximum(close, 1e-10)); out = np.full(n, np.nan)
    for i in range(window - 1, n, stride): out[i] = hurst_exponent(log_s[i - window + 1 : i + 1])
    return pd.Series(out).ffill().values

def rolling_entropy(returns: np.ndarray, window: int, stride: int) -> np.ndarray:
    n = len(returns); out = np.full(n, np.nan)
    for i in range(window - 1, n, stride): out[i] = shannon_entropy(returns[i - window + 1 : i + 1])
    return pd.Series(out).ffill().values

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["close"].diff()
    g = d.clip(lower=0).rolling(14).mean(); l = (-d.clip(upper=0)).rolling(14).mean()
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
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (df["atr"] + 1e-12)
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (df["atr"] + 1e-12)
    df["adx"] = (100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)).ewm(alpha=1/14, adjust=False).mean()
    df["di_difference"] = pdi - mdi

    m, s = df["close"].rolling(20).mean(), df["close"].rolling(20).std()
    df["bb_position"] = (df["close"] - m) / (2 * s + 1e-12)
    km = df["close"].ewm(span=20, adjust=False).mean(); kb = df["atr"].ewm(span=20, adjust=False).mean() * 2
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
    df["prior_high"] = df["high"].rolling(sw).max().shift(1); df["prior_low"]  = df["low"].rolling(sw).min().shift(1)
    df["bos_score"] = np.where((df["high"] > df["prior_high"]) & (df["close"] > df["prior_high"]), 1.0, np.where((df["low"] < df["prior_low"]) & (df["close"] < df["prior_low"]), -1.0, 0.0))
    df["liquidity_sweep_score"] = np.where((df["low"] < df["prior_low"]) & (df["close"] > df["prior_low"]), 1.0, np.where((df["high"] > df["prior_high"]) & (df["close"] < df["prior_high"]), -1.0, 0.0))
    df["return_1"] = df["close"].pct_change(1); df["return_3"] = df["close"].pct_change(3); df["return_5"] = df["close"].pct_change(5)

    v = df["volume"].fillna(0); vm, vs = v.rolling(20).mean(), v.rolling(20).std()
    df["volume_zscore"] = ((v - vm) / (vs + 1e-12)).clip(-5, 5) if (v>0).sum() > 20 else 0.0
    df["volume_ratio"]  = (v / (vm + 1e-12)).clip(0, 10) if (v>0).sum() > 20 else 1.0

    hr = df["datetime"].dt.hour; dow = df["datetime"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24); df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * dow / 7); df["dayofweek_cos"] = np.cos(2 * np.pi * dow / 7)
    df["session_london"]  = hr.between(7, 15).astype(int); df["session_newyork"] = hr.between(13, 21).astype(int); df["session_overlap"] = hr.between(13, 16).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df if len(df) >= CONFIG.min_candles else df.iloc[0:0]

def triple_barrier_labels(df: pd.DataFrame, fwd: int, atr_mult: float) -> pd.DataFrame:
    df = df.copy(); c, h, l, a = df["close"].values, df["high"].values, df["low"].values, df["atr"].values
    n = len(df); labels = np.zeros(n, dtype=int)
    for i in range(n - fwd):
        if not np.isfinite(a[i]) or a[i] <= 0: continue
        up = c[i] + a[i] * atr_mult; dn = c[i] - a[i] * atr_mult
        hh = h[i+1:i+fwd+1]; ll = l[i+1:i+fwd+1]
        hit_up = hh >= up; hit_dn = ll <= dn; trig = hit_up | hit_dn
        if not trig.any(): labels[i] = 0; continue
        j = int(np.argmax(trig))
        if hit_up[j] and hit_dn[j]: labels[i] = 0
        elif hit_up[j]: labels[i] = 1
        elif hit_dn[j]: labels[i] = -1
    df["label"] = labels; return df

def build_model():
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=20, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=42)
    return VotingClassifier([("rf", rf), ("gb", gb)], voting="soft")

def purged_embargoed_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 4) -> Dict[str, float]:
    tscv = TimeSeriesSplit(n_splits=n_splits); aucs, precs = [], []
    for tr_idx, va_idx in tscv.split(X):
        purge_end = va_idx[0] - CONFIG.forward_bars; tr_purged = tr_idx[tr_idx < purge_end]
        if len(tr_purged) < 100 or len(np.unique(y[tr_purged])) < 2: continue
        try:
            m = build_model(); w = np.exp(-(len(tr_purged) - 1 - np.arange(len(tr_purged))) / 500.0)
            m.fit(X[tr_purged], y[tr_purged], sample_weight=w)
            proba = m.predict_proba(X[va_idx]); preds = m.predict(X[va_idx])
            dmask = y[va_idx] != 0
            if dmask.sum() < 10: continue
            y_dir, p_dir = y[va_idx][dmask], preds[dmask]; mask_dir = p_dir != 0
            if mask_dir.sum() >= 10 and len(np.unique(y_dir[mask_dir])) >= 2:
                precs.append(precision_score(y_dir[mask_dir], p_dir[mask_dir], average="weighted", zero_division=0))
            classes = list(m.classes_)
            if 1 in classes and -1 in classes:
                bi = classes.index(1); y_bin = (y_dir == 1).astype(int)
                if len(np.unique(y_bin)) >= 2: aucs.append(roc_auc_score(y_bin, proba[dmask][:, bi]))
        except Exception: continue
    if not aucs: return {"valid": False, "auc_mean": 0.5, "prec_mean": 0.0}
    return {"valid": True, "auc_mean": float(np.mean(aucs)), "prec_mean": float(np.mean(precs)) if precs else 0.0}

def train_model(df_feat: pd.DataFrame, force: bool = False) -> Dict[str, Any]:
    if not force and os.path.exists(CONFIG.model_cache_path):
        try:
            age_min = (py_time.time() - os.path.getmtime(CONFIG.model_cache_path)) / 60
            if age_min <= CONFIG.model_max_age_minutes:
                with open(CONFIG.model_cache_path, "rb") as f: return pickle.load(f)
        except Exception: pass

    df_lab = triple_barrier_labels(df_feat, CONFIG.forward_bars, CONFIG.atr_sl_multiplier)
    train_df = df_lab.iloc[: -CONFIG.forward_bars]; X, y = train_df[FEATURES].values, train_df["label"].values
    if len(X) < 400 or len(np.unique(y)) < 2: return {"valid": False, "reason": "insufficient_data"}
    
    wf = purged_embargoed_cv(X, y)
    if not wf["valid"] or wf["auc_mean"] < CONFIG.min_oos_auc or wf["prec_mean"] < CONFIG.min_oos_precision:
        return {"valid": False, "reason": f"wf_gate AUC:{wf['auc_mean']:.2f} PREC:{wf['prec_mean']:.2f}"}
        
    n = len(y); weights = np.exp(-(n - 1 - np.arange(n)) / 500.0)
    model = build_model(); model.fit(X, y, sample_weight=weights)
    out = {"valid": True, "model": model, "auc": wf["auc_mean"], "prec": wf["prec_mean"]}
    if not CONFIG.backtest_mode:
        with open(CONFIG.model_cache_path + ".tmp", "wb") as f: pickle.dump(out, f)
        os.replace(CONFIG.model_cache_path + ".tmp", CONFIG.model_cache_path)
    return out

def resolve_active_signals(state: Dict[str, Any], df: pd.DataFrame) -> None:
    if not state.get("active_signals"): return
    now, still_active, resolved = now_utc(), [], []
    for sig in state["active_signals"]:
        try: sig_time = datetime.fromisoformat(sig["time"]); expires = datetime.fromisoformat(sig["expires_at"])
        except Exception: continue
        sig_df = df[df["datetime"] > sig_time]; outcome, exit_price = None, None
        for _, bar in sig_df.iterrows():
            if sig["type"] == "BUY":
                if bar["low"]  <= sig["sl"]: outcome, exit_price = "loss", sig["sl"]; break
                if bar["high"] >= sig["tp"]: outcome, exit_price = "win",  sig["tp"]; break
            else:
                if bar["high"] >= sig["sl"]: outcome, exit_price = "loss", sig["sl"]; break
                if bar["low"]  <= sig["tp"]: outcome, exit_price = "win",  sig["tp"]; break
        if outcome is None:
            if now >= expires and len(sig_df):
                exit_price = float(sig_df["close"].iloc[-1]) - (CONFIG.spread_usd / 2 if sig["type"] == "BUY" else -CONFIG.spread_usd / 2)
                pnl = ((exit_price - sig["entry"]) if sig["type"] == "BUY" else (sig["entry"] - exit_price)) * sig["lot"] * CONFIG.contract_size
                resolved.append((sig, "timeout", exit_price, pnl))
            else: still_active.append(sig)
        else:
            cost = (CONFIG.spread_usd + CONFIG.slippage_usd) * sig["lot"] * CONFIG.contract_size
            pnl = (sig["risk_usd"] * CONFIG.risk_reward if outcome == "win" else -sig["risk_usd"]) - cost
            resolved.append((sig, outcome, exit_price, pnl))

    for sig, outcome, exit_price, pnl in resolved:
        if outcome == "win": state["performance_history"]["wins"] += 1; state["consecutive_losses"] = 0
        elif outcome == "loss": state["performance_history"]["losses"] += 1; state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1
        state["daily_pnl_usd"] += pnl; state["recent_outcomes"].append({"outcome": outcome, "prob": sig.get("prob", 0.0), "signal_id": sig.get("signal_id")})
    state["recent_outcomes"] = state["recent_outcomes"][-50:]; state["active_signals"] = still_active
    state["open_signal_ids"] = [s.get("signal_id") for s in still_active if s.get("signal_id")]

    recent = [o for o in state["recent_outcomes"] if o["outcome"] in ("win","loss")]
    if len(recent) >= 15:
        win_rate = sum(1 for o in recent[-20:] if o["outcome"] == "win") / len(recent[-20:])
        state["adaptive_threshold"] = float(np.clip(CONFIG.base_min_signal_probability + (0.55 - win_rate) * 0.5, CONFIG.min_threshold, CONFIG.max_threshold))

def compute_trade(sig_type: str, df_feat: pd.DataFrame, prob: float, source: str) -> Optional[Dict[str, Any]]:
    last = df_feat.iloc[-1]; price, atr = float(last["close"]), float(last["atr"]); spread_half = CONFIG.spread_usd / 2
    sl_dist = atr * CONFIG.atr_sl_multiplier + CONFIG.spread_usd; tp_dist = atr * CONFIG.atr_sl_multiplier * CONFIG.risk_reward - CONFIG.spread_usd
    if tp_dist <= 0: return None
    entry, sl, tp = (price + spread_half, price - sl_dist, price + tp_dist) if sig_type == "BUY" else (price - spread_half, price + sl_dist, price - tp_dist)
    
    risk_target = CONFIG.account_equity * (CONFIG.risk_percent / 100.0); sl_ticks = abs(entry - sl)
    if sl_ticks <= 0: return None
    lot = round(risk_target / (sl_ticks * CONFIG.contract_size), 2)
    if lot < CONFIG.min_lot: return None
    lot = min(lot, CONFIG.max_lot); actual_risk = lot * sl_ticks * CONFIG.contract_size
    if actual_risk > risk_target * 1.05: return None
    
    return {
        "signal_id": hashlib.sha256(f"{sig_type}:{entry:.2f}:{now_utc().isoformat()}".encode()).hexdigest()[:16],
        "type": sig_type, "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
        "lot": lot, "risk_usd": round(actual_risk, 2), "prob": prob, "source": source,
        "time": now_utc().isoformat(), "expires_at": (now_utc() + timedelta(minutes=CONFIG.timeframe_minutes * CONFIG.forward_bars * 2)).isoformat(),
    }

def format_message(t: Dict, source: str) -> str:
    return (f"<b>{'🟢' if t['type'] == 'BUY' else '🔴'} XAUUSD {t['type']}</b>\n"
            f"🕒 {t['time'][:16].replace('T',' ')} UTC\n📡 Source: {source}\n🔑 ID: {t['signal_id']}\n"
            f"🎯 Prob: <b>{t['prob']*100:.1f}%</b>\n"
            f"📍 Entry: <b>{t['entry']:.2f}</b>\n🛑 SL: <b>{t['sl']:.2f}</b>\n🎯 TP: <b>{t['tp']:.2f}</b>\n"
            f"⚖️ Lot: <b>{t['lot']}</b> (Risk ${t['risk_usd']:.2f})")

def circuit_breakers_ok(state: Dict[str, Any], df_feat: pd.DataFrame) -> Tuple[bool, str]:
    if state["daily_pnl_usd"] <= -CONFIG.max_daily_loss_pct * CONFIG.account_equity / 100.0: return False, "Daily loss limit"
    if state.get("consecutive_losses", 0) >= CONFIG.max_consecutive_losses: return False, "Consecutive losses"
    if len(df_feat) > 100 and float(df_feat.iloc[-1]["atr"]) >= df_feat["atr"].tail(500).quantile(CONFIG.extreme_atr_percentile): return False, "Extreme ATR"
    return True, ""

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    log.info("=" * 60)
    log.info("XAUUSD v29.0 — The Perfect Redundancy Edition")
    log.info("=" * 60)

    with FileLock(CONFIG.state_lock_path, CONFIG.state_lock_timeout_sec):
        state = load_state()
        if state.get("last_reset_date") != now_utc().strftime("%Y-%m-%d"):
            state["daily_signals_count"] = 0; state["daily_pnl_usd"] = 0.0; state["last_reset_date"] = now_utc().strftime("%Y-%m-%d")

        if (_env_bool(CONFIG.kill_switch_env, False) or now_utc().weekday() >= 5 or (now_utc().weekday() == 4 and now_utc().hour >= 21)):
            log.info("📅 Market closed or Kill Switch active."); save_state(state); return

        try:
            df, source = fetch_xauusd_data(state)
            df = sanitize_df(df)
            if df is None: return
        except Exception as e:
            log.error("❌ All providers failed: %s", e); return

        state["last_data_source"] = source
        resolve_active_signals(state, df)
        save_state(state)

        if state["daily_signals_count"] >= CONFIG.max_daily_signals or state["daily_pnl_usd"] <= -(CONFIG.max_daily_loss_pct * CONFIG.account_equity / 100.0):
            log.info("⛔ Daily limits reached."); return

        if state.get("last_signal_time"):
            try:
                if (now_utc() - datetime.fromisoformat(state["last_signal_time"])).total_seconds() / 60 < CONFIG.cooldown_minutes: return
            except Exception: pass

        df_feat = calculate_features(df)
        if len(df_feat) < CONFIG.min_candles: return
        last = df_feat.iloc[-1]

        h_val, e_val = float(last["hurst"]), float(last["shannon_entropy"])
        log.info("📊 Regime: hurst=%.3f (min=%.2f) | entropy=%.3f (max=%.2f)", h_val, CONFIG.min_hurst_threshold, e_val, CONFIG.max_entropy_threshold)
        if e_val > CONFIG.max_entropy_threshold or h_val < CONFIG.min_hurst_threshold:
            log.info("🌊 SKIP: Filtered by regime."); return

        cb_ok, cb_reason = circuit_breakers_ok(state, df_feat)
        if not cb_ok: log.warning("⛔ Circuit breaker: %s", cb_reason); return

        models = train_model(df_feat)
        if not models.get("valid"): log.info("⚠️ Invalid model: %s", models.get("reason")); return

        X_live = last[FEATURES].values.reshape(1, -1); model = models["model"]
        pred = int(model.predict(X_live)[0]); proba = model.predict_proba(X_live)[0]
        prob_dict = {int(c): float(proba[i]) for i, c in enumerate(model.classes_)}

        if pred == 0: log.info("ℹ️ Neutral prediction."); return

        sig_type = "BUY" if pred == 1 else "SELL"; prob = prob_dict.get(pred, 0.0)
        if prob < float(state["adaptive_threshold"]): log.info("📉 Prob %.2f < threshold %.2f.", prob, state["adaptive_threshold"]); return

        close_price, ema_trend, adx, rsi = float(last["close"]), float(last["ema_trend"]), float(last["adx"]), float(last["rsi"])
        if sig_type == "BUY" and (close_price < ema_trend or adx < 20.0 or rsi > 75): log.info("🛡️ Rule filter rejected BUY."); return
        if sig_type == "SELL" and (close_price > ema_trend or adx < 20.0 or rsi < 25): log.info("🛡️ Rule filter rejected SELL."); return

        log.info("🔥 Signal confirmed | %s (Prob %.2f)", sig_type, prob)
        trade = compute_trade(sig_type, df_feat, prob, source)
        if trade is None: log.info("⛔ Trade sizing rejected."); return
        if trade["signal_id"] in state.get("open_signal_ids", []): return

        if not send_telegram(format_message(trade, source)): log.error("❌ Telegram failed."); return

        state["active_signals"].append(trade); state["open_signal_ids"].append(trade["signal_id"])
        state["daily_signals_count"] += 1; state["last_signal_time"] = now_utc().isoformat()
        save_state(state)
        log.info("✅ %s sent | Entry=%.2f Lot=%.2f Risk=$%.2f", sig_type, trade["entry"], trade["lot"], trade["risk_usd"])

if __name__ == "__main__":
    try: main()
    except Exception as e: log.critical("💥 Fatal: %s", e, exc_info=True); sys.exit(1)
