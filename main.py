#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD BOT v27.1 — Ultimate Optimized Edition
Filename: main.py
"""

import os, sys, json, logging, time as py_time
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
# CONFIG (OPTIMIZED FOR XAUUSD 15M)
# ===========================================================================
@dataclass(frozen=True)
class Config:
    telegram_token: str = _env_str("TELEGRAM_TOKEN")
    telegram_chat_id: str = _env_str("TELEGRAM_CHAT_ID")
    gist_token: str = _env_str("GH_GIST_TOKEN")
    gist_id: str = _env_str("GH_GIST_ID")
    backtest_mode: bool = _env_bool("BACKTEST_MODE", False)

    timeframe_minutes: int = 15
    min_candles: int = 300
    forward_bars: int = 12            # ✅ OPTIMASI: 3 jam ke depan (sebelumnya 4)
    atr_period: int = 14
    atr_sl_multiplier: float = 2.0    # ✅ OPTIMASI: SL 2.0 ATR (sebelumnya 1.5)
    risk_reward: float = 1.5          # ✅ OPTIMASI: TP 1.5 RR (sebelumnya 2.5)
    spread_usd: float = 0.30

    base_min_signal_probability: float = 0.60
    max_threshold: float = 0.75
    min_threshold: float = 0.50
    min_oos_auc: float = 0.54         # ✅ OPTIMASI: Gate AUC realistis (sebelumnya 0.58)
    min_oos_precision: float = 0.50   # ✅ OPTIMASI: Gate Precision realistis (sebelumnya 0.55)

    max_entropy_threshold: float = 0.99
    min_hurst_threshold: float = 0.30
    max_wick_ratio_threshold: float = 0.75

    cooldown_minutes: int = 120
    max_daily_signals: int = 6
    max_daily_loss_pct: float = 4.0
    account_equity: float = _env_float("ACCOUNT_EQUITY", 1000.0)
    risk_percent: float = _env_float("RISK_PERCENT", 1.0)
    contract_size: float = 100.0
    max_lot: float = 1.0
    min_lot: float = 0.01

    hurst_window: int = 150
    hurst_stride: int = 5
    entropy_window: int = 150
    entropy_stride: int = 5

    state_path: str = str(Path.home() / ".xauusd_state.json")
    kill_switch_env: str = "KILL_SWITCH"

CONFIG = Config()

FEATURES = [
    "rsi","adx","di_difference","ema_distance","ema_slope",
    "trend_strength","atr_pct","atr_regime","realized_vol","vol_zscore",
    "bb_position","keltner_position","hurst","shannon_entropy",
    "body_ratio","upper_wick_ratio","lower_wick_ratio","bos_score",
    "liquidity_sweep_score","return_1","return_3","return_5",
    "volume_zscore","volume_ratio","hour_sin","hour_cos","dayofweek",
    "session_london","session_newyork","session_overlap",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stderr, force=True,
)
log = logging.getLogger("xauusd_v27")

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def validate_config():
    if CONFIG.account_equity <= 0:
        raise ValueError(f"ACCOUNT_EQUITY must be > 0")
    if not (0 < CONFIG.risk_percent <= 100):
        raise ValueError(f"RISK_PERCENT out of range")
    if CONFIG.forward_bars < 1 or CONFIG.forward_bars > 20:
        raise ValueError("forward_bars out of sane range")

# ===========================================================================
# STATE MANAGEMENT
# ===========================================================================
def _default_state() -> Dict[str, Any]:
    return {
        "version": 271,
        "last_signal_time": None,
        "last_choppy_time": None,
        "daily_signals_count": 0,
        "daily_pnl_usd": 0.0,
        "last_reset_date": now_utc().strftime("%Y-%m-%d"),
        "active_signals": [],
        "performance_history": {"wins": 0, "losses": 0},
        "adaptive_threshold": CONFIG.base_min_signal_probability,
        "recent_outcomes": [],
        "last_data_source": None,
    }

def load_state() -> Dict[str, Any]:
    default = _default_state()
    gist_configured = bool(CONFIG.gist_token and CONFIG.gist_id)

    if gist_configured:
        for attempt in range(1, 4):
            try:
                r = requests.get(
                    f"https://api.github.com/gists/{CONFIG.gist_id}",
                    headers={"Authorization": f"token {CONFIG.gist_token}"},
                    timeout=15,
                )
                if r.status_code == 200:
                    for fn, fc in r.json().get("files", {}).items():
                        if fn.endswith("bot_state.json"):
                            state = json.loads(fc["content"])
                            for k, v in default.items():
                                if k not in state: state[k] = v
                            log.info("✅ State loaded from Gist")
                            return state
                else:
                    log.warning("Gist GET status %d (attempt %d)", r.status_code, attempt)
            except Exception as e:
                log.warning("Gist load attempt %d: %s", attempt, e)
            py_time.sleep(2 ** attempt)

    if os.path.exists(CONFIG.state_path):
        try:
            with open(CONFIG.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            for k, v in default.items():
                if k not in state: state[k] = v
            log.info("✅ State loaded from local file")
            return state
        except Exception as e:
            log.warning("Local state corrupt: %s", e)

    if gist_configured:
        log.error("❌ Gist configured but unreachable. Starting fresh.")
    else:
        log.info("ℹ️ State default (first run, no persistence configured)")
    return default

def save_state(state: Dict[str, Any]) -> bool:
    saved_gist = False
    gist_configured = bool(CONFIG.gist_token and CONFIG.gist_id)

    if gist_configured:
        for attempt in range(1, 4):
            try:
                r = requests.patch(
                    f"https://api.github.com/gists/{CONFIG.gist_id}",
                    json={"files": {"bot_state.json": {"content": json.dumps(state, indent=2)}}},
                    headers={"Authorization": f"token {CONFIG.gist_token}"},
                    timeout=15,
                )
                if r.status_code == 200:
                    saved_gist = True
                    log.info("✅ State saved to Gist")
                    break
                elif r.status_code == 429:
                    py_time.sleep(2 ** attempt)
            except Exception as e:
                log.warning("Gist save attempt %d: %s", attempt, e)
                py_time.sleep(2 ** attempt)

    try:
        tmp = CONFIG.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, CONFIG.state_path)
        if not saved_gist:
            log.info("✅ State saved to local file (fallback)")
    except Exception as e:
        log.warning("Local state save failed: %s", e)

    if gist_configured and not saved_gist:
        log.error("❌ CRITICAL: Gist save failed. Next run may lose state!")
    return True

# ===========================================================================
# TELEGRAM
# ===========================================================================
def send_telegram(message: str) -> bool:
    if CONFIG.backtest_mode: return True
    if not CONFIG.telegram_token or not CONFIG.telegram_chat_id:
        log.warning("⚠️ Telegram not configured.")
        return False
    url = f"https://api.telegram.org/bot{CONFIG.telegram_token}/sendMessage"
    payload = {
        "chat_id": CONFIG.telegram_chat_id, "text": message,
        "parse_mode": "HTML", "disable_web_page_preview": True,
    }
    for attempt in range(1, 4):
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 200: return True
            if r.status_code == 429:
                ra = int(r.json().get("parameters", {}).get("retry_after", 5))
                py_time.sleep(ra + 1)
                continue
            if 500 <= r.status_code < 600:
                py_time.sleep(2 ** attempt); continue
            return False
        except Exception as e:
            log.warning("Telegram attempt %d: %s", attempt, e)
            py_time.sleep(2 * attempt)
    return False

# ===========================================================================
# DATA FETCHING (TWELVE DATA)
# ===========================================================================
def fetch_xauusd_data(interval: str = "15min", bars: int = 5000) -> Tuple[pd.DataFrame, str]:
    api_key = _env_str("TWELVE_DATA_KEY")
    if not api_key:
        log.warning("⚠️ TWELVE_DATA_KEY kosong, fallback ke Yahoo GC=F")
        return _fetch_yahoo_futures(interval, bars)

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD",
        "interval": interval,
        "outputsize": min(bars, 5000),
        "apikey": api_key,
        "format": "JSON",
        "timezone": "UTC",
    }
    for attempt in range(1, 4):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "error":
                raise RuntimeError(f"TwelveData: {data.get('message')}")
            values = data.get("values", [])
            if not values:
                raise RuntimeError("TwelveData: empty values")
            
            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
            else:
                df["volume"] = 0.0
                
            df = df.sort_values("datetime").reset_index(drop=True)
            df = df[["datetime", "open", "high", "low", "close", "volume"]]
            
            if len(df) < CONFIG.min_candles:
                raise RuntimeError(f"TwelveData: only {len(df)} bars")
            
            log.info("✅ Data: XAU/USD spot (TWELVE_DATA) | %d bars", len(df))
            return df, "TWELVE_DATA_SPOT"
        except Exception as e:
            log.warning("TwelveData attempt %d: %s", attempt, e)
            py_time.sleep(2 ** attempt)
            
    log.error("❌ TwelveData gagal total, fallback ke Yahoo futures")
    return _fetch_yahoo_futures(interval, bars)


def _fetch_yahoo_futures(interval: str = "15m", bars: int = 5000) -> Tuple[pd.DataFrame, str]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    days = max(60, bars // (24 * 4))
    yf_interval = "15m" if interval == "15min" else interval
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval={yf_interval}&range={days}d"
    
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    res = r.json().get("chart", {}).get("result", [])
    if not res or not res[0].get("timestamp"):
        raise RuntimeError("Yahoo GC=F: no data")
        
    res = res[0]
    q = res.get("indicators", {}).get("quote", [{}])[0]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(res["timestamp"], unit="s", utc=True),
        "open": q.get("open"), "high": q.get("high"),
        "low": q.get("low"), "close": q.get("close"),
        "volume": q.get("volume", 0),
    }).dropna(subset=["open", "high", "low", "close"])
    
    log.warning("⚠️ Menggunakan GC=F FUTURES — harga bukan spot!")
    return df, "YAHOO_FUTURES"

def sanitize_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 2: return None
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0) & (df["open"] > 0)]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open","high","low","close"])
    if len(df) < 2: return None

    age = (now_utc() - df["datetime"].iloc[-1]).total_seconds() / 60
    if age > CONFIG.timeframe_minutes * 3:
        log.warning("⚠️ Data stale (%.0f min)", age); return None
    if age < CONFIG.timeframe_minutes:
        log.info("🕒 Dropping unclosed bar (age=%.1f min)", age)
        df = df.iloc[:-1].reset_index(drop=True)
    return df if len(df) >= CONFIG.min_candles else None

# ===========================================================================
# FEATURE ENGINEERING
# ===========================================================================
def wilder_atr(df, period=14):
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-prev).abs(),
                    (df["low"]-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < max_lag * 2: return 0.5
    lags = np.arange(2, max_lag + 1)
    tau = np.array([np.std(s[lag:] - s[:-lag]) for lag in lags], dtype=float)
    valid = tau > 0
    if valid.sum() < 3: return 0.5
    try:
        return float(np.clip(np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0], 0.0, 1.0))
    except Exception:
        return 0.5

def shannon_entropy(returns: np.ndarray, bins: int = 10) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 20: return 1.0
    hist, _ = np.histogram(r, bins=bins)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    mx = np.log2(bins)
    return float(-np.sum(p * np.log2(p)) / mx) if mx > 0 else 1.0

def rolling_hurst(close: np.ndarray, window: int, stride: int, max_lag: int = 20) -> np.ndarray:
    n = len(close)
    log_s = np.log(np.maximum(close, 1e-10))
    out = np.full(n, np.nan)
    for i in range(window - 1, n, stride):
        win = log_s[i - window + 1 : i + 1]
        out[i] = hurst_exponent(win, max_lag)
    s = pd.Series(out)
    return s.ffill().bfill().fillna(0.5).values

def rolling_entropy(returns: np.ndarray, window: int, stride: int, bins: int = 10) -> np.ndarray:
    n = len(returns)
    out = np.full(n, np.nan)
    for i in range(window - 1, n, stride):
        win = returns[i - window + 1 : i + 1]
        out[i] = shannon_entropy(win, bins)
    s = pd.Series(out)
    return s.ffill().bfill().fillna(1.0).values

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

    up = df["high"].diff()
    dn = -df["low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/CONFIG.atr_period, adjust=False).mean() / (df["atr"] + 1e-12)
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1/CONFIG.atr_period, adjust=False).mean() / (df["atr"] + 1e-12)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
    df["adx"] = dx.ewm(alpha=1/CONFIG.atr_period, adjust=False).mean()
    df["di_difference"] = pdi - mdi

    m = df["close"].rolling(20).mean()
    s = df["close"].rolling(20).std()
    df["bb_position"] = (df["close"] - m) / (2 * s + 1e-12)
    km = df["close"].ewm(span=20, adjust=False).mean()
    kb = df["atr"].ewm(span=20, adjust=False).mean() * 2
    df["keltner_position"] = (df["close"] - km) / (kb + 1e-12)

    ret = df["close"].pct_change()
    df["realized_vol"] = ret.rolling(20).std() * np.sqrt(24 * 60 / CONFIG.timeframe_minutes)
    df["vol_zscore"] = (df["realized_vol"] - df["realized_vol"].rolling(100).mean()) / (
        df["realized_vol"].rolling(100).std() + 1e-12)

    df["hurst"] = rolling_hurst(df["close"].values, CONFIG.hurst_window, CONFIG.hurst_stride, 20)
    df["shannon_entropy"] = rolling_entropy(ret.fillna(0).values, CONFIG.entropy_window, CONFIG.entropy_stride, 10)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (rng + 1e-12)
    df["upper_wick_ratio"] = (df["high"] - df[["open","close"]].max(axis=1)) / (rng + 1e-12)
    df["lower_wick_ratio"] = (df[["open","close"]].min(axis=1) - df["low"]) / (rng + 1e-12)

    sw = 5
    df["prior_high"] = df["high"].rolling(sw).max().shift(1)
    df["prior_low"]  = df["low"].rolling(sw).min().shift(1)
    df["bos_score"] = np.where((df["high"] > df["prior_high"]) & (df["close"] > df["prior_high"]), 1.0,
                        np.where((df["low"] < df["prior_low"]) & (df["close"] < df["prior_low"]), -1.0, 0.0))
    df["liquidity_sweep_score"] = np.where(
        (df["low"] < df["prior_low"]) & (df["close"] > df["prior_low"]), 1.0,
        np.where((df["high"] > df["prior_high"]) & (df["close"] < df["prior_high"]), -1.0, 0.0))

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    if "volume" in df.columns and df["volume"].notna().sum() > 20 and (df["volume"] > 0).sum() > 20:
        v = df["volume"].fillna(0)
        vm, vs = v.rolling(20).mean(), v.rolling(20).std()
        df["volume_zscore"] = ((v - vm) / (vs + 1e-12)).clip(-5, 5)
        df["volume_ratio"]  = (v / (vm + 1e-12)).clip(0, 10)
    else:
        df["volume_zscore"] = 0.0
        df["volume_ratio"]  = 1.0

    hr = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["session_london"]  = hr.between(7, 15).astype(int)
    df["session_newyork"] = hr.between(12, 20).astype(int)
    df["session_overlap"] = hr.between(12, 15).astype(int)

    df[["volume_zscore", "volume_ratio"]] = df[["volume_zscore","volume_ratio"]].ffill().fillna(0)
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

def triple_barrier_labels(df: pd.DataFrame, fwd: int, atr_mult: float) -> pd.DataFrame:
    df = df.copy()
    c = df["close"].values; h = df["high"].values
    l = df["low"].values;  a = df["atr"].values
    n = len(df)
    labels = np.zeros(n, dtype=int)
    for i in range(n - fwd):
        if not np.isfinite(a[i]) or a[i] <= 0: continue
        up = c[i] + a[i] * atr_mult
        dn = c[i] - a[i] * atr_mult
        for j in range(i + 1, i + fwd + 1):
            hit_up = h[j] >= up
            hit_dn = l[j] <= dn
            if hit_up and hit_dn: labels[i] = 0; break
            if hit_up: labels[i] = 1; break
            if hit_dn: labels[i] = -1; break
    df["label"] = labels
    return df

# ===========================================================================
# ML ENGINE
# ===========================================================================
def build_model():
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=6, min_samples_leaf=15,
        max_features="sqrt", class_weight="balanced_subsample",
        random_state=42, n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=42,
    )
    return VotingClassifier([("rf", rf), ("gb", gb)], voting="soft")

def purged_cv(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    tscv = TimeSeriesSplit(n_splits=3)
    aucs, precs = [], []
    for tr_idx, va_idx in tscv.split(X):
        tr_end = tr_idx[-1] - CONFIG.forward_bars
        if tr_end < 100: continue
        tr_slice = np.arange(tr_idx[0], tr_end + 1)
        if len(np.unique(y[tr_slice])) < 2: continue
        try:
            m = build_model()
            m.fit(X[tr_slice], y[tr_slice])
            proba = m.predict_proba(X[va_idx])
            preds = m.predict(X[va_idx])
            dmask = y[va_idx] != 0
            if dmask.sum() < 10: continue
            y_dir = y[va_idx][dmask]
            p_dir = preds[dmask]
            if len(np.unique(y_dir)) >= 2:
                precs.append(precision_score(y_dir, p_dir, average="weighted", zero_division=0))
            classes = list(m.classes_)
            if 1 in classes and -1 in classes:
                bi = classes.index(1)
                y_bin = (y_dir == 1).astype(int)
                if len(np.unique(y_bin)) >= 2:
                    aucs.append(roc_auc_score(y_bin, proba[dmask][:, bi]))
        except Exception:
            continue
    if not aucs:
        return {"valid": False, "auc_mean": 0.5, "prec_mean": 0.0}
    return {"valid": True,
            "auc_mean": float(np.mean(aucs)),
            "prec_mean": float(np.mean(precs)) if precs else 0.0}

def train_model(df_feat: pd.DataFrame) -> Dict[str, Any]:
    df_lab = triple_barrier_labels(df_feat, CONFIG.forward_bars, CONFIG.atr_sl_multiplier)
    train_df = df_lab.iloc[: -CONFIG.forward_bars]
    X = train_df[FEATURES].values
    y = train_df["label"].values
    if len(X) < 400 or len(np.unique(y)) < 2:
        return {"valid": False, "reason": f"insufficient_data (n={len(X)})"}
    wf = purged_cv(X, y)
    if not wf["valid"] or wf["auc_mean"] < CONFIG.min_oos_auc or wf["prec_mean"] < CONFIG.min_oos_precision:
        return {"valid": False,
                "reason": f"wf_gate AUC:{wf['auc_mean']:.2f} PREC:{wf['prec_mean']:.2f}"}
    weights = np.exp(-np.arange(len(y))[::-1] / 500.0)
    model = build_model()
    model.fit(X, y, sample_weight=weights)
    log.info("🎓 Model trained | AUC=%.3f PREC=%.3f n=%d", wf["auc_mean"], wf["prec_mean"], len(X))
    return {"valid": True, "model": model, "auc": wf["auc_mean"]}

# ===========================================================================
# SIGNAL RESOLUTION
# ===========================================================================
def resolve_active_signals(state: Dict[str, Any], df: pd.DataFrame) -> None:
    if not state.get("active_signals"): return
    now = now_utc()
    still_active, resolved = [], []

    for sig in state["active_signals"]:
        try:
            sig_time = datetime.fromisoformat(sig["time"])
            expires  = datetime.fromisoformat(sig["expires_at"])
        except Exception:
            continue

        sig_df = df[df["datetime"] > sig_time]
        outcome = None; exit_price = None

        for _, bar in sig_df.iterrows():
            if sig["type"] == "BUY":
                if bar["low"]  <= sig["sl"]: outcome, exit_price = "loss", sig["sl"]; break
                if bar["high"] >= sig["tp"]: outcome, exit_price = "win",  sig["tp"]; break
            else:
                if bar["high"] >= sig["sl"]: outcome, exit_price = "loss", sig["sl"]; break
                if bar["low"]  <= sig["tp"]: outcome, exit_price = "win",  sig["tp"]; break

        if outcome is None:
            if now >= expires:
                exit_price = float(sig_df["close"].iloc[-1]) if len(sig_df) else sig["entry"]
                pnl_per_unit = ((exit_price - sig["entry"]) if sig["type"] == "BUY"
                                else (sig["entry"] - exit_price))
                pnl_usd = pnl_per_unit * sig["lot"] * CONFIG.contract_size
                outcome = "timeout"
                resolved.append((sig, outcome, exit_price, pnl_usd))
            else:
                still_active.append(sig)
        else:
            pnl = (sig["risk_usd"] * CONFIG.risk_reward) if outcome == "win" else (-sig["risk_usd"])
            resolved.append((sig, outcome, exit_price, pnl))

    for sig, outcome, exit_price, pnl in resolved:
        if outcome == "win": state["performance_history"]["wins"] += 1
        elif outcome == "loss": state["performance_history"]["losses"] += 1
        state["daily_pnl_usd"] = float(state.get("daily_pnl_usd", 0.0)) + pnl
        state["recent_outcomes"].append({
            "time": sig["time"], "type": sig["type"],
            "outcome": outcome, "prob": sig.get("prob", 0.0),
            "pnl": round(pnl, 2),
        })
        log.info("📊 Resolved %s @%.2f → %s (PnL $%.2f | daily $%.2f)",
                 sig["type"], exit_price, outcome, pnl, state["daily_pnl_usd"])

    state["recent_outcomes"] = state["recent_outcomes"][-50:]
    state["active_signals"] = still_active

    recent = [o for o in state["recent_outcomes"] if o["outcome"] in ("win","loss")]
    if len(recent) >= 10:
        win_rate = sum(1 for o in recent[-20:] if o["outcome"] == "win") / len(recent[-20:])
        base = CONFIG.base_min_signal_probability
        adjustment = (0.55 - win_rate) * 0.5
        new_th = float(np.clip(base + adjustment, CONFIG.min_threshold, CONFIG.max_threshold))
        if abs(new_th - state["adaptive_threshold"]) > 0.005:
            log.info("🎛️ Threshold %.3f → %.3f (win_rate=%.1f%%)",
                     state["adaptive_threshold"], new_th, win_rate * 100)
            state["adaptive_threshold"] = new_th

# ===========================================================================
# TRADE COMPUTE
# ===========================================================================
def compute_trade(sig_type: str, df_feat: pd.DataFrame, prob: float, source: str) -> Dict[str, Any]:
    last = df_feat.iloc[-1]
    price = float(last["close"]); atr = float(last["atr"])
    sl_dist = atr * CONFIG.atr_sl_multiplier
    tp_dist = sl_dist * CONFIG.risk_reward
    half_spread = CONFIG.spread_usd / 2.0

    if sig_type == "BUY":
        entry = price + half_spread
        sl    = price - sl_dist
        tp    = price + tp_dist
    else:
        entry = price - half_spread
        sl    = price + sl_dist
        tp    = price - tp_dist

    risk_usd = CONFIG.account_equity * (CONFIG.risk_percent / 100.0)
    sl_abs = abs(entry - sl)
    if sl_abs <= 0: lot = CONFIG.min_lot
    else:
        raw = risk_usd / (sl_abs * CONFIG.contract_size)
        lot = max(CONFIG.min_lot, min(round(raw, 2), CONFIG.max_lot))

    expires = now_utc() + timedelta(minutes=CONFIG.timeframe_minutes * CONFIG.forward_bars * 2)
    return {
        "type": sig_type, "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
        "lot": lot, "risk_usd": round(risk_usd, 2), "prob": prob, "source": source,
        "time": now_utc().isoformat(), "expires_at": expires.isoformat(),
    }

def format_message(t: Dict, source: str) -> str:
    emoji = "🟢" if t["type"] == "BUY" else "🔴"
    return (
        f"<b>{emoji} XAUUSD {t['type']}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕒 {t['time'][:16].replace('T',' ')} UTC\n"
        f"📡 Source: {source}\n"
        f"🎯 Probability: <b>{t['prob']*100:.1f}%</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 Entry: <b>{t['entry']:.2f}</b>\n"
        f"🛑 SL: <b>{t['sl']:.2f}</b>\n"
        f"🎯 TP: <b>{t['tp']:.2f}</b>\n"
        f"⚖️ Lot: <b>{t['lot']}</b> (Risk ${t['risk_usd']:.2f})\n"
        f"<i>Expires: {t['expires_at'][:16].replace('T',' ')} UTC</i>"
    )

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    log.info("=" * 60)
    log.info("XAUUSD v27.1 — Ultimate Optimized GitHub Actions Edition")
    log.info("=" * 60)
    log.info("ACCOUNT_EQUITY: $%.2f | RISK: %.2f%%", CONFIG.account_equity, CONFIG.risk_percent)

    try:
        validate_config()
    except ValueError as e:
        log.critical("🛑 Invalid config: %s", e); return

    state = load_state()
    today = now_utc().strftime("%Y-%m-%d")
    if state.get("last_reset_date") != today:
        state["daily_signals_count"] = 0
        state["daily_pnl_usd"] = 0.0
        state["last_reset_date"] = today
        log.info("🔄 Daily state reset")

    if _env_bool(CONFIG.kill_switch_env, False) or os.path.exists(str(Path.home()/".xauusd_kill")):
        log.critical("🛑 Kill switch active. Stop."); return

    now = now_utc()
    if now.weekday() >= 5:
        log.info("📅 Weekend. Skip."); return
    if now.weekday() == 4 and now.hour >= 21:
        log.info("📅 Friday evening — market closing. Skip."); return

    try:
        df, source = fetch_xauusd_data()
        df = sanitize_df(df)
        if df is None:
            log.warning("⚠️ Invalid or stale data."); save_state(state); return
    except Exception as e:
        log.error("❌ Fetch failed: %s", e); save_state(state); return

    state["last_data_source"] = source

    try:
        resolve_active_signals(state, df)
    except Exception as e:
        log.error("❌ Resolution failed: %s", e, exc_info=True)
    save_state(state)

    if state["daily_signals_count"] >= CONFIG.max_daily_signals:
        log.info("⛔ Max daily signals reached (%d).", CONFIG.max_daily_signals); return

    max_loss = CONFIG.max_daily_loss_pct * CONFIG.account_equity / 100.0
    if state.get("daily_pnl_usd", 0.0) <= -max_loss:
        log.critical("🛑 Daily loss limit reached ($%.2f <= -$%.2f).",
                     state["daily_pnl_usd"], max_loss); return

    if state.get("last_signal_time"):
        try:
            mins = (now_utc() - datetime.fromisoformat(state["last_signal_time"])).total_seconds() / 60
            if mins < CONFIG.cooldown_minutes:
                log.info("⏳ Cooldown active (%.0f/%d min).", mins, CONFIG.cooldown_minutes); return
        except Exception:
            pass

    try:
        df_feat = calculate_features(df)
    except Exception as e:
        log.error("❌ Feature engineering failed: %s", e, exc_info=True); return

    if len(df_feat) < 100:
        log.warning("⚠️ Too few feature rows (%d).", len(df_feat)); return

    last = df_feat.iloc[-1]
    entropy = float(last["shannon_entropy"]); hurst = float(last["hurst"])
    if entropy > CONFIG.max_entropy_threshold or hurst < CONFIG.min_hurst_threshold:
        log.info("🌊 Choppy (Entropy=%.2f, Hurst=%.2f < %.2f). Skip.",
                 entropy, hurst, CONFIG.min_hurst_threshold)
        state["last_choppy_time"] = now_utc().isoformat(); save_state(state); return

    rng_ = float(last["high"] - last["low"])
    body = abs(float(last["close"] - last["open"]))
    wick = (rng_ - body) / (rng_ + 1e-12) if rng_ > 0 else 1.0
    if wick > CONFIG.max_wick_ratio_threshold:
        log.info("🕯️ Excessive wick (%.2f > %.2f). Skip.", wick, CONFIG.max_wick_ratio_threshold)
        state["last_choppy_time"] = now_utc().isoformat(); save_state(state); return

    try:
        models = train_model(df_feat)
    except Exception as e:
        log.error("❌ Training failed: %s", e, exc_info=True); return

    if not models.get("valid"):
        log.info("⚠️ Invalid model: %s", models.get("reason")); save_state(state); return

    X_live = last[FEATURES].values.reshape(1, -1)
    model = models["model"]
    pred = int(model.predict(X_live)[0])
    proba = model.predict_proba(X_live)[0]
    classes = list(model.classes_)
    prob_dict = {int(c): float(proba[i]) for i, c in enumerate(classes)}

    if pred == 0:
        log.info("ℹ️ Neutral prediction."); save_state(state); return

    sig_type = "BUY" if pred == 1 else "SELL"
    prob = prob_dict.get(pred, 0.0)
    th = float(state["adaptive_threshold"])

    if prob < th:
        log.info("📉 Prob %.2f < threshold %.2f. Skip.", prob, th); save_state(state); return

    try:
        trade = compute_trade(sig_type, df_feat, prob, source)
    except Exception as e:
        log.error("❌ Trade compute failed: %s", e, exc_info=True); return

    if not send_telegram(format_message(trade, source)):
        log.error("❌ Telegram failed. State NOT incremented."); return

    state["active_signals"].append(trade)
    state["daily_signals_count"] += 1
    state["last_signal_time"] = now_utc().isoformat()
    state["last_choppy_time"] = None
    save_state(state)
    log.info("✅ %s sent | Entry=%.2f Lot=%.2f Prob=%.2f Src=%s",
             sig_type, trade["entry"], trade["lot"], prob, source)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("⚠️ Interrupted.")
    except Exception as e:
        log.critical("💥 Fatal: %s", e, exc_info=True)
        sys.exit(1)
