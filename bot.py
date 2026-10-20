#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD BOT v28.1 — FINAL DEFINITIVE EDITION
=============================================
Satu file, siap deploy, tidak perlu edit lagi.

Fitur:
  ✅ TwelveData spot + Yahoo fallback berlapis
  ✅ 3-class ML (BUY/SELL/NEUTRAL) + rule filter (EMA50, ADX, RSI)
  ✅ Rolling Hurst/Entropy per bar
  ✅ Model caching (TTL 6 jam)
  ✅ Signal lifecycle resolution SEBELUM cooldown (PnL akurat)
  ✅ Adaptive threshold berbasis win-rate
  ✅ DIAGNOSTIC_MODE built-in (bypass semua gate, tidak kirim Telegram)
  ✅ Migrasi state otomatis dari versi lama
  ✅ PnL aktual (lot × Δprice × contract_size)
  ✅ Kill switch: env var + file
  ✅ Market hours akurat
  ✅ Stale data protection
"""

import os
import sys
import json
import pickle
import logging
import time as py_time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, roc_auc_score


# ===========================================================================
# ENV HELPERS
# ===========================================================================
def _env_float(k, d):
    v = (os.getenv(k) or "").strip()
    try:
        return float(v) if v else d
    except (ValueError, TypeError):
        return d


def _env_int(k, d):
    v = (os.getenv(k) or "").strip()
    try:
        return int(v) if v else d
    except (ValueError, TypeError):
        return d


def _env_bool(k, d=False):
    v = (os.getenv(k) or "").strip().lower()
    return v in ("true", "1", "yes", "on") if v else d


def _env_str(k, d=""):
    return (os.getenv(k) or "").strip() or d


# ===========================================================================
# CONFIG
# ===========================================================================
@dataclass(frozen=True)
class Config:
    telegram_token: str = _env_str("TELEGRAM_TOKEN")
    telegram_chat_id: str = _env_str("TELEGRAM_CHAT_ID")
    gist_token: str = _env_str("GH_GIST_TOKEN")
    gist_id: str = _env_str("GH_GIST_ID")
    backtest_mode: bool = _env_bool("BACKTEST_MODE", False)
    diagnostic_mode: bool = _env_bool("DIAGNOSTIC_MODE", False)

    timeframe_minutes: int = 15
    min_candles: int = 600
    forward_bars: int = 8
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    risk_reward: float = 2.0
    spread_usd: float = 0.30
    slippage_usd: float = 0.10

    base_min_signal_probability: float = 0.38
    max_threshold: float = 0.50
    min_threshold: float = 0.35

    min_oos_auc: float = 0.50
    min_oos_precision: float = 0.45

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
    model_cache_path: str = str(Path.home() / ".xauusd_model_v28.pkl")
    model_cache_ttl_sec: int = 6 * 3600
    kill_switch_path: str = str(Path.home() / ".xauusd_kill")
    kill_switch_env: str = "KILL_SWITCH"

    max_active_signals: int = 20


CONFIG = Config()

FEATURES = [
    "rsi", "adx", "di_difference", "ema_distance", "ema_slope",
    "trend_strength", "atr_pct", "atr_regime", "realized_vol", "vol_zscore",
    "bb_position", "keltner_position", "hurst", "shannon_entropy",
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "bos_score",
    "liquidity_sweep_score", "return_1", "return_3", "return_5",
    "volume_zscore", "volume_ratio", "hour_sin", "hour_cos", "dayofweek",
    "session_london", "session_newyork", "session_overlap",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stderr,
    force=True,
)
log = logging.getLogger("xauusd_v28")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ===========================================================================
# MARKET HOURS
# ===========================================================================
def is_market_open() -> bool:
    now = now_utc()
    wd, hr = now.weekday(), now.hour
    if wd == 5:
        return False
    if wd == 6 and hr < 22:
        return False
    if wd == 4 and hr >= 21:
        return False
    return True


# ===========================================================================
# STATE MANAGEMENT
# ===========================================================================
def _default_state() -> Dict[str, Any]:
    return {
        "version": 281,
        "last_signal_time": None,
        "last_choppy_time": None,
        "last_run_time": None,
        "daily_signals_count": 0,
        "daily_pnl_usd": 0.0,
        "last_reset_date": now_utc().strftime("%Y-%m-%d"),
        "active_signals": [],
        "performance_history": {"wins": 0, "losses": 0, "timeouts": 0},
        "adaptive_threshold": CONFIG.base_min_signal_probability,
        "recent_outcomes": [],
        "last_data_source": None,
    }


def _migrate_state(state: Dict[str, Any]) -> Dict[str, Any]:
    default = _default_state()
    for k, v in default.items():
        if k not in state:
            state[k] = v

    ro = state.get("recent_outcomes", [])
    if ro and isinstance(ro[0], str):
        log.info("🔄 Migrasi recent_outcomes string → dict")
        state["recent_outcomes"] = [{"outcome": o, "prob": 0.0} for o in ro]

    ph = state.get("performance_history", {})
    if "timeouts" not in ph:
        ph["timeouts"] = 0
        state["performance_history"] = ph

    if not isinstance(state.get("active_signals"), list):
        state["active_signals"] = []

    th = state.get("adaptive_threshold", CONFIG.base_min_signal_probability)
    if not isinstance(th, (int, float)) or not (
        CONFIG.min_threshold <= float(th) <= CONFIG.max_threshold
    ):
        state["adaptive_threshold"] = CONFIG.base_min_signal_probability

    state["version"] = 281
    return state


def load_state() -> Dict[str, Any]:
    if CONFIG.gist_token and CONFIG.gist_id:
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
                            try:
                                state = json.loads(fc["content"])
                                state = _migrate_state(state)
                                log.info("✅ State loaded from Gist")
                                return state
                            except json.JSONDecodeError:
                                log.warning("Gist state corrupt")
            except Exception as e:
                log.warning("Gist load attempt %d: %s", attempt, e)
                py_time.sleep(2 ** attempt)

    if os.path.exists(CONFIG.state_path):
        try:
            with open(CONFIG.state_path, "r", encoding="utf-8") as f:
                state = _migrate_state(json.load(f))
            log.info("✅ State loaded from local file")
            return state
        except Exception as e:
            log.warning("Local state corrupt: %s", e)

    log.info("ℹ️ State default (first run)")
    return _default_state()


def save_state(state: Dict[str, Any]) -> bool:
    saved_gist = False
    if CONFIG.gist_token and CONFIG.gist_id:
        for attempt in range(1, 4):
            try:
                r = requests.patch(
                    f"https://api.github.com/gists/{CONFIG.gist_id}",
                    json={"files": {"bot_state.json": {
                        "content": json.dumps(state, indent=2)}}},
                    headers={"Authorization": f"token {CONFIG.gist_token}"},
                    timeout=15,
                )
                if r.status_code == 200:
                    saved_gist = True
                    log.info("✅ State saved to Gist")
                    break
                if r.status_code == 429:
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
            log.info("✅ State saved to local (fallback)")
    except Exception as e:
        log.warning("Local save failed: %s", e)
    return True


# ===========================================================================
# TELEGRAM
# ===========================================================================
def send_telegram(message: str) -> bool:
    if CONFIG.backtest_mode:
        log.info("📨 [BACKTEST] Telegram suppressed")
        return True
    if not CONFIG.telegram_token or not CONFIG.telegram_chat_id:
        log.warning("⚠️ Telegram not configured")
        return False
    url = f"https://api.telegram.org/bot{CONFIG.telegram_token}/sendMessage"
    payload = {
        "chat_id": CONFIG.telegram_chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    for attempt in range(1, 4):
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 200:
                return True
            if r.status_code == 429:
                py_time.sleep(
                    int(r.json().get("parameters", {}).get("retry_after", 5)) + 1
                )
                continue
        except Exception as e:
            log.warning("Telegram attempt %d: %s", attempt, e)
            py_time.sleep(2 * attempt)
    return False


# ===========================================================================
# DATA FETCHING
# ===========================================================================
def fetch_xauusd_data(interval: str = "15min", bars: int = 5000) -> Tuple[pd.DataFrame, str]:
    api_key = _env_str("TWELVE_DATA_KEY")
    if api_key:
        try:
            return _fetch_twelvedata(api_key, interval, bars)
        except Exception as e:
            log.warning("TwelveData failed: %s. Fallback Yahoo.", e)
    else:
        log.info("ℹ️ No TWELVE_DATA_KEY, using Yahoo")
    return _fetch_yahoo_futures(interval, bars)


def _fetch_twelvedata(api_key: str, interval: str, bars: int) -> Tuple[pd.DataFrame, str]:
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
            data = r.json()
            if data.get("status") == "error":
                raise RuntimeError(data.get("message", "TwelveData error"))
            values = data.get("values", [])
            if not values:
                raise RuntimeError("Empty values")

            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["volume"] = (
                pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
                if "volume" in df.columns else 0.0
            )
            df = df.sort_values("datetime").reset_index(drop=True)
            df = df[["datetime", "open", "high", "low", "close", "volume"]]
            log.info("✅ Data: XAU/USD spot (TWELVE_DATA) | %d bars", len(df))
            return df, "TWELVE_DATA_SPOT"
        except Exception as e:
            log.warning("TwelveData attempt %d: %s", attempt, e)
            py_time.sleep(2 ** attempt)
    raise RuntimeError("TwelveData failed after 3 attempts")


def _fetch_yahoo_futures(interval: str, bars: int) -> Tuple[pd.DataFrame, str]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    yf_interval = "15m" if interval == "15min" else interval
    tickers = [("XAUUSD=X", "YAHOO_SPOT"), ("GC=F", "YAHOO_FUTURES")]
    for ticker, src in tickers:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{ticker}?interval={yf_interval}&range=60d"
        )
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            chart = r.json().get("chart", {}).get("result", [])
            if not chart or not chart[0].get("timestamp"):
                continue
            res = chart[0]
            q = res.get("indicators", {}).get("quote", [{}])[0]
            df = pd.DataFrame({
                "datetime": pd.to_datetime(res["timestamp"], unit="s", utc=True),
                "open": q.get("open"),
                "high": q.get("high"),
                "low": q.get("low"),
                "close": q.get("close"),
                "volume": q.get("volume", 0),
            }).dropna(subset=["open", "high", "low", "close"])
            if len(df) >= CONFIG.min_candles:
                log.info("✅ Data: %s (%s) | %d bars", ticker, src, len(df))
                return df, src
        except Exception as e:
            log.warning("%s failed: %s", ticker, e)
    raise RuntimeError("All data sources failed")


def sanitize_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 2:
        return None
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0) & (df["open"] > 0)]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])
    if len(df) == 0:
        return None
    age = (now_utc() - df["datetime"].iloc[-1]).total_seconds() / 60
    if age > CONFIG.timeframe_minutes * 3:
        log.warning("⚠️ Data stale (%.0f min)", age)
        return None
    if age < CONFIG.timeframe_minutes:
        df = df.iloc[:-1].reset_index(drop=True)
    if len(df) < CONFIG.min_candles:
        log.warning("sanitize: %d bars < %d", len(df), CONFIG.min_candles)
        return None
    return df


# ===========================================================================
# FEATURE ENGINEERING
# ===========================================================================
def wilder_atr(df, period=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    s = series[np.isfinite(series)]
    if len(s) < max_lag * 2:
        return 0.5
    lags = np.arange(2, max_lag + 1)
    tau = np.array([np.std(s[lag:] - s[:-lag]) for lag in lags])
    valid = tau > 0
    if valid.sum() < 3:
        return 0.5
    try:
        return float(np.clip(
            np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0], 0.0, 1.0
        ))
    except Exception:
        return 0.5


def shannon_entropy(returns: np.ndarray, bins: int = 10) -> float:
    r = returns[np.isfinite(returns)]
    if len(r) < 20:
        return 1.0
    hist, _ = np.histogram(r, bins=bins)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(bins))


def rolling_hurst(close: np.ndarray, window: int, stride: int) -> np.ndarray:
    n = len(close)
    log_s = np.log(np.maximum(close, 1e-10))
    out = np.full(n, np.nan)
    for i in range(window - 1, n, stride):
        out[i] = hurst_exponent(log_s[i - window + 1: i + 1])
    return pd.Series(out).ffill().bfill().fillna(0.5).values


def rolling_entropy(returns: np.ndarray, window: int, stride: int) -> np.ndarray:
    n = len(returns)
    out = np.full(n, np.nan)
    for i in range(window - 1, n, stride):
        out[i] = shannon_entropy(returns[i - window + 1: i + 1])
    return pd.Series(out).ffill().bfill().fillna(1.0).values


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
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(
        alpha=1 / 14, adjust=False).mean() / (df["atr"] + 1e-12)
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(
        alpha=1 / 14, adjust=False).mean() / (df["atr"] + 1e-12)
    df["adx"] = (100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)).ewm(
        alpha=1 / 14, adjust=False).mean()
    df["di_difference"] = pdi - mdi

    m, s = df["close"].rolling(20).mean(), df["close"].rolling(20).std()
    df["bb_position"] = (df["close"] - m) / (2 * s + 1e-12)
    km = df["close"].ewm(span=20, adjust=False).mean()
    kb = df["atr"].ewm(span=20, adjust=False).mean() * 2
    df["keltner_position"] = (df["close"] - km) / (kb + 1e-12)

    ret = df["close"].pct_change()
    df["realized_vol"] = ret.rolling(20).std() * np.sqrt(24 * 60 / CONFIG.timeframe_minutes)
    df["vol_zscore"] = (
        (df["realized_vol"] - df["realized_vol"].rolling(100).mean())
        / (df["realized_vol"].rolling(100).std() + 1e-12)
    )
    df["hurst"] = rolling_hurst(df["close"].values, CONFIG.hurst_window, CONFIG.hurst_stride)
    df["shannon_entropy"] = rolling_entropy(
        ret.fillna(0).values, CONFIG.entropy_window, CONFIG.entropy_stride)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (rng + 1e-12)
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (rng + 1e-12)
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (rng + 1e-12)

    sw = 5
    df["prior_high"] = df["high"].rolling(sw).max().shift(1)
    df["prior_low"] = df["low"].rolling(sw).min().shift(1)
    df["bos_score"] = np.where(
        (df["high"] > df["prior_high"]) & (df["close"] > df["prior_high"]), 1.0,
        np.where(
            (df["low"] < df["prior_low"]) & (df["close"] < df["prior_low"]), -1.0, 0.0))
    df["liquidity_sweep_score"] = np.where(
        (df["low"] < df["prior_low"]) & (df["close"] > df["prior_low"]), 1.0,
        np.where(
            (df["high"] > df["prior_high"]) & (df["close"] < df["prior_high"]), -1.0, 0.0))
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    if "volume" in df.columns and (df["volume"] > 0).sum() > 20:
        v = df["volume"].fillna(0)
        vm, vs = v.rolling(20).mean(), v.rolling(20).std()
        df["volume_zscore"] = ((v - vm) / (vs + 1e-12)).clip(-5, 5)
        df["volume_ratio"] = (v / (vm + 1e-12)).clip(0, 10)
    else:
        df["volume_zscore"], df["volume_ratio"] = 0.0, 1.0

    hr = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["session_london"] = hr.between(7, 15).astype(int)
    df["session_newyork"] = hr.between(12, 20).astype(int)
    df["session_overlap"] = hr.between(12, 15).astype(int)

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def triple_barrier_labels(df: pd.DataFrame, fwd: int, atr_mult: float) -> pd.DataFrame:
    df = df.copy()
    c, h, l, a = df["close"].values, df["high"].values, df["low"].values, df["atr"].values
    labels = np.zeros(len(df), dtype=int)
    for i in range(len(df) - fwd):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        up, dn = c[i] + a[i] * atr_mult, c[i] - a[i] * atr_mult
        for j in range(i + 1, i + fwd + 1):
            if h[j] >= up and l[j] <= dn:
                labels[i] = 0
                break
            if h[j] >= up:
                labels[i] = 1
                break
            if l[j] <= dn:
                labels[i] = -1
                break
    df["label"] = labels
    return df


# ===========================================================================
# ML ENGINE
# ===========================================================================
def build_model() -> VotingClassifier:
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=5, min_samples_leaf=20,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=42)
    return VotingClassifier([("rf", rf), ("gb", gb)], voting="soft")


def purged_cv(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    tscv = TimeSeriesSplit(n_splits=3)
    aucs, precs = [], []
    for tr_idx, va_idx in tscv.split(X):
        tr_end = tr_idx[-1] - CONFIG.forward_bars
        if tr_end < 100:
            continue
        tr_slice = np.arange(tr_idx[0], tr_end + 1)
        if len(np.unique(y[tr_slice])) < 2:
            continue
        try:
            m = build_model()
            m.fit(X[tr_slice], y[tr_slice])
            proba, preds = m.predict_proba(X[va_idx]), m.predict(X[va_idx])
            dmask = y[va_idx] != 0
            if dmask.sum() < 10:
                continue
            y_dir, p_dir = y[va_idx][dmask], preds[dmask]
            if len(np.unique(y_dir)) >= 2:
                precs.append(precision_score(
                    y_dir, p_dir, average="weighted", zero_division=0))
            classes = list(m.classes_)
            if 1 in classes and -1 in classes:
                bi = classes.index(1)
                y_bin = (y_dir == 1).astype(int)
                if len(np.unique(y_bin)) >= 2:
                    aucs.append(roc_auc_score(y_bin, proba[dmask][:, bi]))
        except Exception as e:
            log.debug("CV fold error: %s", e)
            continue
    if not aucs:
        return {"valid": False, "auc_mean": 0.5, "prec_mean": 0.0}
    return {
        "valid": True,
        "auc_mean": float(np.mean(aucs)),
        "prec_mean": float(np.mean(precs)) if precs else 0.0,
    }


def _model_cache_valid() -> bool:
    if CONFIG.model_cache_ttl_sec <= 0 or not os.path.exists(CONFIG.model_cache_path):
        return False
    return (py_time.time() - os.path.getmtime(CONFIG.model_cache_path)) < CONFIG.model_cache_ttl_sec


def _load_cached_model() -> Optional[VotingClassifier]:
    if not _model_cache_valid():
        return None
    try:
        with open(CONFIG.model_cache_path, "rb") as f:
            model = pickle.load(f)
        log.info("✅ Model loaded from cache")
        return model
    except Exception as e:
        log.warning("Cache load failed: %s", e)
        return None


def _save_cached_model(model: VotingClassifier) -> None:
    try:
        with open(CONFIG.model_cache_path, "wb") as f:
            pickle.dump(model, f)
        log.info("✅ Model cached")
    except Exception as e:
        log.warning("Cache save failed: %s", e)


def train_model(df_feat: pd.DataFrame) -> Dict[str, Any]:
    cached = _load_cached_model()
    if cached is not None:
        return {"valid": True, "model": cached, "cached": True}

    df_lab = triple_barrier_labels(df_feat, CONFIG.forward_bars, CONFIG.atr_sl_multiplier)
    train_df = df_lab.iloc[:-CONFIG.forward_bars]
    X, y = train_df[FEATURES].values, train_df["label"].values
    if len(X) < 400 or len(np.unique(y)) < 2:
        return {"valid": False,
                "reason": f"insufficient_data n={len(X)} cls={len(np.unique(y))}"}

    wf = purged_cv(X, y)
    if not wf["valid"]:
        return {"valid": False, "reason": "cv_invalid"}
    if wf["auc_mean"] < CONFIG.min_oos_auc or wf["prec_mean"] < CONFIG.min_oos_precision:
        return {"valid": False,
                "reason": f"gate AUC={wf['auc_mean']:.3f} P={wf['prec_mean']:.3f}"}

    weights = np.exp(-np.arange(len(y))[::-1] / 500.0)
    model = build_model()
    model.fit(X, y, sample_weight=weights)
    _save_cached_model(model)
    log.info("🎓 Model trained | AUC=%.3f P=%.3f n=%d", wf["auc_mean"], wf["prec_mean"], len(X))
    return {"valid": True, "model": model, "cached": False, "auc": wf["auc_mean"]}


# ===========================================================================
# SIGNAL LIFECYCLE (dipanggil SEBELUM cooldown check)
# ===========================================================================
def resolve_active_signals(state: Dict[str, Any], df: pd.DataFrame) -> None:
    if not state.get("active_signals"):
        return
    now, still_active, resolved = now_utc(), [], []

    for sig in state["active_signals"]:
        try:
            sig_time = datetime.fromisoformat(sig["time"])
            expires = datetime.fromisoformat(sig["expires_at"])
            entry, sl, tp = float(sig["entry"]), float(sig["sl"]), float(sig["tp"])
            lot = float(sig["lot"])
            side = sig["type"]
        except (KeyError, ValueError, TypeError):
            log.warning("Corrupt signal, skip")
            continue

        sig_df = df[df["datetime"] > sig_time]
        outcome, exit_price = None, None

        for _, bar in sig_df.iterrows():
            hi, lo = float(bar["high"]), float(bar["low"])
            if side == "BUY":
                if lo <= sl:
                    outcome, exit_price = "loss", sl
                    break
                if hi >= tp:
                    outcome, exit_price = "win", tp
                    break
            else:
                if hi >= sl:
                    outcome, exit_price = "loss", sl
                    break
                if lo <= tp:
                    outcome, exit_price = "win", tp
                    break

        if outcome is None:
            if now >= expires:
                if len(sig_df) > 0:
                    exit_price = float(sig_df["close"].iloc[-1])
                    pnl = ((exit_price - entry) if side == "BUY"
                           else (entry - exit_price)) * lot * CONFIG.contract_size
                else:
                    exit_price = entry
                    pnl = 0.0
                resolved.append((sig, "timeout", exit_price, pnl))
            else:
                still_active.append(sig)
        else:
            pnl = ((exit_price - entry) if side == "BUY"
                   else (entry - exit_price)) * lot * CONFIG.contract_size
            resolved.append((sig, outcome, exit_price, pnl))

    for sig, outcome, exit_price, pnl in resolved:
        ph = state.get("performance_history", {})
        if outcome == "win":
            ph["wins"] = ph.get("wins", 0) + 1
        elif outcome == "loss":
            ph["losses"] = ph.get("losses", 0) + 1
        else:
            ph["timeouts"] = ph.get("timeouts", 0) + 1
        state["performance_history"] = ph
        state["daily_pnl_usd"] = state.get("daily_pnl_usd", 0.0) + pnl
        state["recent_outcomes"].append({
            "outcome": outcome,
            "prob": sig.get("prob", 0.0),
            "pnl": round(pnl, 2),
        })
        log.info("📊 Resolved %s → %s @ %.2f (PnL $%.2f)",
                 sig["type"], outcome, exit_price, pnl)

    state["recent_outcomes"] = state["recent_outcomes"][-50:]
    state["active_signals"] = still_active

    recent = [o for o in state["recent_outcomes"] if o["outcome"] in ("win", "loss")]
    if len(recent) >= 10:
        r20 = recent[-20:]
        win_rate = sum(1 for o in r20 if o["outcome"] == "win") / len(r20)
        new_th = float(np.clip(
            CONFIG.base_min_signal_probability + (0.55 - win_rate) * 0.5,
            CONFIG.min_threshold, CONFIG.max_threshold))
        old_th = state["adaptive_threshold"]
        if abs(new_th - old_th) > 0.005:
            state["adaptive_threshold"] = round(new_th, 4)
            log.info("🎚️ Threshold %.3f → %.3f (win_rate=%.2f n=%d)",
                     old_th, new_th, win_rate, len(r20))


# ===========================================================================
# TRADE COMPUTATION
# ===========================================================================
def compute_trade(sig_type: str, df_feat: pd.DataFrame, prob: float, source: str) -> Dict[str, Any]:
    last = df_feat.iloc[-1]
    price, atr = float(last["close"]), float(last["atr"])
    sl_dist = atr * CONFIG.atr_sl_multiplier
    tp_dist = sl_dist * CONFIG.risk_reward
    half_spread = CONFIG.spread_usd / 2

    if sig_type == "BUY":
        entry = price + half_spread
        sl = price - sl_dist
        tp = price + tp_dist
    else:
        entry = price - half_spread
        sl = price + sl_dist
        tp = price - tp_dist

    risk_usd = CONFIG.account_equity * (CONFIG.risk_percent / 100.0)
    sl_abs = abs(entry - sl)
    lot = (
        CONFIG.min_lot if sl_abs <= 0
        else max(CONFIG.min_lot, min(
            round(risk_usd / (sl_abs * CONFIG.contract_size), 2), CONFIG.max_lot))
    )

    return {
        "type": sig_type,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "lot": lot,
        "risk_usd": round(risk_usd, 2),
        "prob": prob,
        "source": source,
        "time": now_utc().isoformat(),
        "expires_at": (
            now_utc() + timedelta(minutes=CONFIG.timeframe_minutes * CONFIG.forward_bars * 2)
        ).isoformat(),
    }


def format_message(t: Dict, source: str) -> str:
    return (
        f"<b>{'🟢' if t['type'] == 'BUY' else '🔴'} XAUUSD {t['type']}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕒 {t['time'][:16].replace('T', ' ')} UTC\n"
        f"📡 Source: {source}\n"
        f"🎯 Prob: <b>{t['prob'] * 100:.1f}%</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 Entry: <b>{t['entry']:.2f}</b>\n"
        f"🛑 SL: <b>{t['sl']:.2f}</b>\n"
        f"🎯 TP: <b>{t['tp']:.2f}</b>\n"
        f"⚖️ Lot: <b>{t['lot']}</b> (Risk ${t['risk_usd']:.2f})\n"
        f"<i>Expires: {t['expires_at'][:16].replace('T', ' ')} UTC</i>"
    )


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    log.info("=" * 60)
    log.info("XAUUSD v28.1 — FINAL DEFINITIVE EDITION")
    log.info("=" * 60)
    log.info("ACCOUNT_EQUITY: $%.2f", CONFIG.account_equity)

    _diag = CONFIG.diagnostic_mode
    if _diag:
        log.warning("🔬 DIAGNOSTIC MODE — gates bypassed, no Telegram")

    if CONFIG.account_equity <= 0:
        log.critical("🛑 ACCOUNT_EQUITY invalid")
        return

    state = load_state()
    today = now_utc().strftime("%Y-%m-%d")
    if state.get("last_reset_date") != today:
        state["daily_signals_count"] = 0
        state["daily_pnl_usd"] = 0.0
        state["last_reset_date"] = today
        log.info("🔄 Daily state reset")

    # -----------------------------------------------------------------------
    # PHASE 1: Cheap checks (no data needed)
    # -----------------------------------------------------------------------
    if not _diag:
        if _env_bool(CONFIG.kill_switch_env, False) or os.path.exists(CONFIG.kill_switch_path):
            log.critical("🛑 Kill switch active")
            return
        if not is_market_open():
            log.info("📅 Market closed")
            return

    # -----------------------------------------------------------------------
    # PHASE 2: Fetch data (needed for signal resolution)
    # -----------------------------------------------------------------------
    try:
        df, source = fetch_xauusd_data()
        df = sanitize_df(df)
        if df is None:
            return
    except Exception as e:
        log.error("❌ Fetch failed: %s", e)
        return

    state["last_data_source"] = source
    state["last_run_time"] = now_utc().isoformat()

    # -----------------------------------------------------------------------
    # PHASE 3: Resolve active signals (SEBELUM cooldown — fix v28.1)
    # -----------------------------------------------------------------------
    try:
        resolve_active_signals(state, df)
    except Exception as e:
        log.warning("Resolve error: %s", e)

    # -----------------------------------------------------------------------
    # PHASE 4: Cooldown + daily limits (setelah resolution)
    # -----------------------------------------------------------------------
    if not _diag:
        if state["daily_signals_count"] >= CONFIG.max_daily_signals:
            log.info("⛔ Daily signal limit (%d)", CONFIG.max_daily_signals)
            save_state(state)
            return
        max_loss = CONFIG.max_daily_loss_pct * CONFIG.account_equity / 100.0
        if state.get("daily_pnl_usd", 0) <= -max_loss:
            log.critical("🛑 Daily loss limit ($%.2f)", state["daily_pnl_usd"])
            save_state(state)
            return
        if state.get("last_signal_time"):
            try:
                mins = (now_utc() - datetime.fromisoformat(
                    state["last_signal_time"])).total_seconds() / 60
                if mins < CONFIG.cooldown_minutes:
                    log.info("⏳ Cooldown aktif (%.0f/%d min)", mins, CONFIG.cooldown_minutes)
                    save_state(state)
                    return
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # PHASE 5: Feature engineering
    # -----------------------------------------------------------------------
    try:
        df_feat = calculate_features(df)
    except Exception as e:
        log.error("❌ Features failed: %s", e)
        save_state(state)
        return

    if len(df_feat) < 100:
        log.warning("⚠️ Too few features (%d)", len(df_feat))
        save_state(state)
        return

    last = df_feat.iloc[-1]

    # Choppy filter
    entropy = float(last["shannon_entropy"])
    hurst = float(last["hurst"])
    if not _diag and (entropy > CONFIG.max_entropy_threshold or hurst < CONFIG.min_hurst_threshold):
        log.info("🌊 Choppy (Ent=%.2f Hurst=%.2f)", entropy, hurst)
        state["last_choppy_time"] = now_utc().isoformat()
        save_state(state)
        return

    # Wick filter
    rng_ = float(last["high"] - last["low"])
    body = abs(float(last["close"] - last["open"]))
    wick = (rng_ - body) / (rng_ + 1e-12) if rng_ > 0 else 1.0
    if not _diag and wick > CONFIG.max_wick_ratio_threshold:
        log.info("🕯️ Wick %.2f > %.2f", wick, CONFIG.max_wick_ratio_threshold)
        state["last_choppy_time"] = now_utc().isoformat()
        save_state(state)
        return

    # -----------------------------------------------------------------------
    # PHASE 6: Train / load model
    # -----------------------------------------------------------------------
    models = train_model(df_feat)
    if not models.get("valid"):
        log.info("⚠️ Model invalid: %s", models.get("reason"))
        save_state(state)
        return

    if models.get("cached"):
        log.info("ℹ️ Model from cache")
    else:
        log.info("🧠 Model trained | AUC=%.3f", models.get("auc", 0.0))

    # -----------------------------------------------------------------------
    # PHASE 7: Inference
    # -----------------------------------------------------------------------
    X_live = last[FEATURES].values.reshape(1, -1)
    model = models["model"]
    pred = int(model.predict(X_live)[0])
    proba = model.predict_proba(X_live)[0]
    prob_dict = {int(c): float(proba[i]) for i, c in enumerate(model.classes_)}

    log.info("🔎 pred=%d | P(BUY)=%.3f P(SELL)=%.3f P(0)=%.3f",
             pred, prob_dict.get(1, 0.0), prob_dict.get(-1, 0.0), prob_dict.get(0, 0.0))

    if pred == 0:
        log.info("ℹ️ Neutral, skip")
        save_state(state)
        return

    sig_type = "BUY" if pred == 1 else "SELL"
    prob = prob_dict.get(pred, 0.0)
    th = float(state["adaptive_threshold"])

    if not _diag and prob < th:
        log.info("📉 Prob %.3f < threshold %.3f", prob, th)
        save_state(state)
        return

    # -----------------------------------------------------------------------
    # PHASE 8: Rule-based filter
    # -----------------------------------------------------------------------
    close_price = float(last["close"])
    ema_trend = float(last["ema_trend"])
    adx = float(last["adx"])
    rsi = float(last["rsi"])

    if not _diag:
        if sig_type == "BUY" and close_price < ema_trend:
            log.info("🛡️ BUY but price<EMA50, skip")
            save_state(state)
            return
        if sig_type == "SELL" and close_price > ema_trend:
            log.info("🛡️ SELL but price>EMA50, skip")
            save_state(state)
            return
        if adx < 20.0:
            log.info("🛡️ ADX %.1f < 20", adx)
            save_state(state)
            return
        if sig_type == "BUY" and rsi > 75:
            log.info("🛡️ RSI %.1f overbought", rsi)
            save_state(state)
            return
        if sig_type == "SELL" and rsi < 25:
            log.info("🛡️ RSI %.1f oversold", rsi)
            save_state(state)
            return

    log.info("🔥 CONFIRMED | %s prob=%.3f ADX=%.1f RSI=%.1f",
             sig_type, prob, adx, rsi)

    # -----------------------------------------------------------------------
    # PHASE 9: Compute trade + send (or diagnostic output)
    # -----------------------------------------------------------------------
    try:
        trade = compute_trade(sig_type, df_feat, prob, source)
    except Exception as e:
        log.error("❌ compute_trade: %s", e)
        save_state(state)
        return

    if _diag:
        log.info("🔬 DIAGNOSTIC — signal that WOULD be sent:")
        for line in format_message(trade, source).split("\n"):
            clean = line.replace("<b>", "").replace("</b>", "").replace(
                "<i>", "").replace("</i>", "").strip()
            if clean:
                log.info("    %s", clean)
        log.info("🔬 DIAGNOSTIC — state NOT modified")
        return

    if not send_telegram(format_message(trade, source)):
        log.error("❌ Telegram failed")
        save_state(state)
        return

    state["active_signals"].append(trade)
    if len(state["active_signals"]) > CONFIG.max_active_signals:
        state["active_signals"] = state["active_signals"][-CONFIG.max_active_signals:]
    state["daily_signals_count"] += 1
    state["last_signal_time"] = now_utc().isoformat()
    state["last_choppy_time"] = None
    save_state(state)
    log.info("✅ %s sent | Entry=%.2f Lot=%.2f", sig_type, trade["entry"], trade["lot"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("⚠️ Interrupted")
    except Exception as e:
        log.critical("💥 Fatal: %s", e, exc_info=True)
        sys.exit(1)
