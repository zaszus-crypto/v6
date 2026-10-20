#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD BOT v26.5 — GitHub Actions Edition (FINAL PRODUCTION)
=============================================================
Ringkasan perbaikan dari v26.3:
  ✅ min_candles diselaraskan dengan kebutuhan training (600)
  ✅ Signal lifecycle: TP/SL/timeout resolution AKTIF
  ✅ daily_pnl_usd, performance_history, recent_outcomes terisi
  ✅ Adaptive threshold benar-benar adaptif berbasis win-rate
  ✅ Model caching (TTL 6 jam) — hemat compute GitHub Actions
  ✅ Market hours akurat (Jumat 21:00 UTC tutup, Minggu 22:00 UTC buka)
  ✅ Logging transparan (prediksi 3-kelas, alasan skip)
  ✅ BOM dihapus, dead code dibersihkan
  ✅ Data source fallback lebih andal
  ✅ Symbol-agnostic SL/TP spread, TP hitung dengan slippage
  ✅ Cleanup state otomatis (active_signals max 20)
  ✅ Timeout harian & recovers state corrupt dari Gist
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
# ENV HELPER
# ===========================================================================
def _env_float(key: str, default: float) -> float:
    val = os.getenv(key, "")
    if val is None:
        return default
    val = str(val).strip()
    if not val:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key, "")
    if val is None:
        return default
    val = str(val).strip()
    if not val:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, "")
    if val is None:
        return default
    val = str(val).strip().lower()
    if not val:
        return default
    return val in ("true", "1", "yes", "on")


def _env_str(key: str, default: str = "") -> str:
    val = os.getenv(key, "")
    if val is None:
        return default
    return str(val).strip()


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

    timeframe_minutes: int = 15
    min_candles: int = 600
    forward_bars: int = 4
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    risk_reward: float = 2.5
    spread_usd: float = 0.30
    slippage_usd: float = 0.10
    commission_per_lot: float = 7.0

    base_min_signal_probability: float = 0.60
    max_threshold: float = 0.75
    min_oos_auc: float = 0.58
    min_oos_precision: float = 0.55

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

    state_path: str = str(Path.home() / ".xauusd_state.json")
    model_cache_path: str = str(Path.home() / ".xauusd_model_v26.pkl")
    kill_switch_path: str = str(Path.home() / ".xauusd_kill")

    model_cache_ttl_sec: int = 6 * 3600
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
log = logging.getLogger("xauusd_v26")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ===========================================================================
# MARKET HOURS
# ===========================================================================
def is_market_open() -> bool:
    """Forex: buka Minggu 22:00 UTC s/d Jumat 21:00 UTC (conservative)."""
    now = now_utc()
    wd, hr = now.weekday(), now.hour  # Mon=0 ... Sun=6
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
        "version": 265,
        "last_signal_time": None,
        "last_choppy_time": None,
        "daily_signals_count": 0,
        "daily_pnl_usd": 0.0,
        "last_reset_date": now_utc().strftime("%Y-%m-%d"),
        "active_signals": [],
        "performance_history": {"wins": 0, "losses": 0, "timeouts": 0},
        "adaptive_threshold": CONFIG.base_min_signal_probability,
        "recent_outcomes": [],
        "last_data_source": None,
        "last_run_time": None,
    }


def _merge_defaults(state: Dict[str, Any]) -> Dict[str, Any]:
    default = _default_state()
    for k, v in default.items():
        if k not in state:
            state[k] = v
    if not isinstance(state.get("performance_history"), dict):
        state["performance_history"] = default["performance_history"]
    if not isinstance(state.get("active_signals"), list):
        state["active_signals"] = []
    if not isinstance(state.get("recent_outcomes"), list):
        state["recent_outcomes"] = []
    return state


def load_state() -> Dict[str, Any]:
    if CONFIG.gist_token and CONFIG.gist_id:
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
                            log.info("✅ State loaded from Gist")
                            return _merge_defaults(state)
                        except json.JSONDecodeError:
                            log.warning("Gist state JSON corrupt, fallback lokal")
        except Exception as e:
            log.warning("Gist load gagal: %s. Fallback ke lokal.", e)

    if os.path.exists(CONFIG.state_path):
        try:
            with open(CONFIG.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            log.info("✅ State loaded from local file")
            return _merge_defaults(state)
        except Exception as e:
            log.warning("State lokal corrupt: %s", e)

    log.info("ℹ️ State default digunakan (pertama kali run)")
    return _default_state()


def save_state(state: Dict[str, Any]) -> bool:
    saved = False
    if CONFIG.gist_token and CONFIG.gist_id:
        for attempt in range(1, 4):
            try:
                payload = {
                    "files": {
                        "bot_state.json": {
                            "content": json.dumps(state, indent=2)
                        }
                    }
                }
                r = requests.patch(
                    f"https://api.github.com/gists/{CONFIG.gist_id}",
                    json=payload,
                    headers={"Authorization": f"token {CONFIG.gist_token}"},
                    timeout=15,
                )
                if r.status_code == 200:
                    log.info("✅ State saved to Gist")
                    saved = True
                    break
                elif r.status_code == 429:
                    py_time.sleep(2 ** attempt)
            except Exception as e:
                log.warning("Gist save attempt %d gagal: %s", attempt, e)
                py_time.sleep(2 ** attempt)

    try:
        tmp_path = CONFIG.state_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, CONFIG.state_path)
        if not saved:
            log.info("✅ State saved to local file (fallback)")
    except Exception as e:
        log.warning("Local state save gagal: %s", e)
    return True


# ===========================================================================
# TELEGRAM
# ===========================================================================
def send_telegram(message: str) -> bool:
    if CONFIG.backtest_mode:
        log.info("📨 [BACKTEST] Telegram suppressed: %s", message.split("\n")[0])
        return True
    if not CONFIG.telegram_token or not CONFIG.telegram_chat_id:
        log.warning("⚠️ Telegram tidak terkonfigurasi.")
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
                retry_after = int(r.json().get("parameters", {}).get("retry_after", 5))
                py_time.sleep(retry_after + 1)
                continue
            log.warning("Telegram status %d: %s", r.status_code, r.text[:200])
            return False
        except Exception as e:
            log.warning("Telegram attempt %d gagal: %s", attempt, e)
            py_time.sleep(2 * attempt)
    return False


# ===========================================================================
# DATA FETCHING
# ===========================================================================
def fetch_xauusd_data(interval: str = "15m", days: int = 60) -> Tuple[pd.DataFrame, str]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    tickers_to_try = [
        ("XAUUSD=X", "YAHOO_SPOT"),
        ("GC=F", "YAHOO_FUTURES"),
        ("XAU-USD", "YAHOO_SPOT_ALT"),
    ]

    for ticker, source_name in tickers_to_try:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{ticker}?interval={interval}&range={days}d"
        )
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            result = data.get("chart", {}).get("result", [])
            if not result or not result[0].get("timestamp"):
                log.warning("%s: no timestamp data", ticker)
                continue

            res = result[0]
            ts = res["timestamp"]
            quote = res.get("indicators", {}).get("quote", [{}])[0]
            df = pd.DataFrame({
                "datetime": pd.to_datetime(ts, unit="s", utc=True),
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "volume": quote.get("volume", 0),
            })
            df = df.dropna(subset=["open", "high", "low", "close"])
            if len(df) >= CONFIG.min_candles:
                log.info(
                    "✅ Data source: %s (%s) | %d bars",
                    ticker, source_name, len(df),
                )
                return df, source_name
            log.warning(
                "%s hanya memberi %d bar (< %d). Coba sumber lain.",
                ticker, len(df), CONFIG.min_candles,
            )
        except Exception as e:
            log.warning("%s gagal: %s", ticker, e)

    raise RuntimeError("Semua sumber data XAUUSD gagal.")


def sanitize_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 2:
        log.warning("sanitize_df: df kosong/terlalu pendek")
        return None
    raw_n = len(df)
    df = (
        df.drop_duplicates(subset="datetime")
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0) & (df["open"] > 0)]
    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["open", "high", "low", "close"])
    )
    if len(df) == 0:
        log.warning("sanitize_df: semua bar invalid")
        return None

    age_min = (now_utc() - df["datetime"].iloc[-1]).total_seconds() / 60
    if age_min > CONFIG.timeframe_minutes * 3:
        log.warning(
            "⚠️ Data stale (%.0f min > %d min)",
            age_min, CONFIG.timeframe_minutes * 3,
        )
        return None
    if age_min < CONFIG.timeframe_minutes:
        df = df.iloc[:-1].reset_index(drop=True)

    if len(df) < CONFIG.min_candles:
        log.warning(
            "sanitize_df: %d bar setelah cleaning (< %d). raw=%d",
            len(df), CONFIG.min_candles, raw_n,
        )
        return None
    return df


# ===========================================================================
# FEATURE ENGINEERING
# ===========================================================================
def wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def hurst_exponent(series, max_lag=20):
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < max_lag * 2:
        return 0.5
    lags = np.arange(2, max_lag + 1)
    tau = np.array([np.std(s[lag:] - s[:-lag]) for lag in lags], dtype=float)
    valid = tau > 0
    if valid.sum() < 3:
        return 0.5
    try:
        return float(
            np.clip(
                np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0], 0.0, 1.0
            )
        )
    except Exception:
        return 0.5


def shannon_entropy(returns, bins=10):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return 1.0
    hist, _ = np.histogram(r, bins=bins)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    mx = np.log2(bins)
    return float(-np.sum(p * np.log2(p)) / mx) if mx > 0 else 1.0


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
    df["ema_slope"] = (
        (df["ema_fast"] - df["ema_fast"].shift(3)) / (df["atr"] + 1e-12)
    )
    df["trend_strength"] = (df["close"] - df["ema_trend"]) / (df["atr"] + 1e-12)

    up, dn = df["high"].diff(), -df["low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi = (
        100
        * pd.Series(pdm, index=df.index)
        .ewm(alpha=1 / CONFIG.atr_period, adjust=False)
        .mean()
        / (df["atr"] + 1e-12)
    )
    mdi = (
        100
        * pd.Series(mdm, index=df.index)
        .ewm(alpha=1 / CONFIG.atr_period, adjust=False)
        .mean()
        / (df["atr"] + 1e-12)
    )
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
    df["adx"] = dx.ewm(alpha=1 / CONFIG.atr_period, adjust=False).mean()
    df["di_difference"] = pdi - mdi

    m, s = df["close"].rolling(20).mean(), df["close"].rolling(20).std()
    df["bb_position"] = (df["close"] - m) / (2 * s + 1e-12)
    km = df["close"].ewm(span=20, adjust=False).mean()
    kb = df["atr"].ewm(span=20, adjust=False).mean() * 2
    df["keltner_position"] = (df["close"] - km) / (kb + 1e-12)

    ret = df["close"].pct_change()
    df["realized_vol"] = ret.rolling(20).std() * np.sqrt(
        24 * 60 / CONFIG.timeframe_minutes
    )
    df["vol_zscore"] = (
        df["realized_vol"] - df["realized_vol"].rolling(100).mean()
    ) / (df["realized_vol"].rolling(100).std() + 1e-12)

    n = len(df)
    window_size = min(150, n)
    recent_close = df["close"].iloc[-window_size:].values
    recent_ret = ret.iloc[-window_size:].dropna().values

    current_hurst = (
        hurst_exponent(np.log(recent_close), max_lag=20)
        if len(recent_close) >= 40
        else 0.5
    )
    current_entropy = (
        shannon_entropy(recent_ret, bins=10) if len(recent_ret) >= 20 else 1.0
    )
    df["hurst"] = current_hurst
    df["shannon_entropy"] = current_entropy

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (rng + 1e-12)
    df["upper_wick_ratio"] = (
        df["high"] - df[["open", "close"]].max(axis=1)
    ) / (rng + 1e-12)
    df["lower_wick_ratio"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    ) / (rng + 1e-12)

    sw = 5
    df["prior_high"] = df["high"].rolling(sw).max().shift(1)
    df["prior_low"] = df["low"].rolling(sw).min().shift(1)
    df["bos_score"] = np.where(
        (df["high"] > df["prior_high"]) & (df["close"] > df["prior_high"]),
        1.0,
        np.where(
            (df["low"] < df["prior_low"]) & (df["close"] < df["prior_low"]),
            -1.0,
            0.0,
        ),
    )
    df["liquidity_sweep_score"] = np.where(
        (df["low"] < df["prior_low"]) & (df["close"] > df["prior_low"]),
        1.0,
        np.where(
            (df["high"] > df["prior_high"]) & (df["close"] < df["prior_high"]),
            -1.0,
            0.0,
        ),
    )

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    if "volume" in df.columns and df["volume"].notna().sum() > 20:
        v = df["volume"].fillna(0)
        v_mean, v_std = v.rolling(20).mean(), v.rolling(20).std()
        df["volume_zscore"] = ((v - v_mean) / (v_std + 1e-12)).clip(-5, 5)
        df["volume_ratio"] = (v / (v_mean + 1e-12)).clip(0, 10)
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
    c, h, l, a = (
        df["close"].values,
        df["high"].values,
        df["low"].values,
        df["atr"].values,
    )
    n = len(df)
    labels = np.zeros(n, dtype=int)
    for i in range(n - fwd):
        up, dn = c[i] + a[i] * atr_mult, c[i] - a[i] * atr_mult
        for j in range(i + 1, i + fwd + 1):
            if h[j] >= up:
                labels[i] = 1
                break
            if l[j] <= dn:
                labels[i] = -1
                break
    df["label"] = labels
    return df


# ===========================================================================
# SIGNAL LIFECYCLE
# ===========================================================================
def resolve_signals_against_df(state: Dict[str, Any], df: pd.DataFrame) -> None:
    """Iterasi setiap active_signal; deteksi TP/SL/timeout; update PnL & stats."""
    now = now_utc()
    still_active = []
    resolved_count = 0
    wins = losses = timeouts = 0

    for sig in state.get("active_signals", []):
        try:
            sig_time = datetime.fromisoformat(sig["time"])
            expires = datetime.fromisoformat(sig["expires_at"])
            lot = float(sig["lot"])
            entry = float(sig["entry"])
            sl = float(sig["sl"])
            tp = float(sig["tp"])
            side = sig["type"]
        except (KeyError, ValueError, TypeError):
            log.warning("Signal corrupt di state, dilewati: %s", sig)
            continue

        bars = df[df["datetime"] > sig_time]
        outcome = None
        for _, bar in bars.iterrows():
            hi, lo = float(bar["high"]), float(bar["low"])
            if side == "BUY":
                if lo <= sl:
                    outcome = "loss"
                    break
                if hi >= tp:
                    outcome = "win"
                    break
            else:  # SELL
                if hi >= sl:
                    outcome = "loss"
                    break
                if lo <= tp:
                    outcome = "win"
                    break

        if outcome is None and now >= expires:
            outcome = "timeout"

        if outcome is None:
            still_active.append(sig)
            continue

        if outcome == "win":
            pnl = lot * CONFIG.contract_size * abs(tp - entry)
            wins += 1
        elif outcome == "loss":
            pnl = -lot * CONFIG.contract_size * abs(entry - sl)
            losses += 1
        else:
            pnl = 0.0
            timeouts += 1

        state["daily_pnl_usd"] = state.get("daily_pnl_usd", 0.0) + pnl
        state["recent_outcomes"] = (
            state.get("recent_outcomes", []) + [outcome]
        )[-20:]
        resolved_count += 1

    state["active_signals"] = still_active
    if resolved_count:
        ph = state.get("performance_history", {"wins": 0, "losses": 0, "timeouts": 0})
        ph["wins"] = ph.get("wins", 0) + wins
        ph["losses"] = ph.get("losses", 0) + losses
        ph["timeouts"] = ph.get("timeouts", 0) + timeouts
        state["performance_history"] = ph
        log.info(
            "📊 Resolved %d signal(s) [W:%d L:%d T:%d] | Daily PnL: $%.2f | active=%d",
            resolved_count, wins, losses, timeouts,
            state.get("daily_pnl_usd", 0.0), len(still_active),
        )


def update_adaptive_threshold(state: Dict[str, Any]) -> None:
    outcomes = state.get("recent_outcomes", [])[-20:]
    base = CONFIG.base_min_signal_probability
    if len(outcomes) < 10:
        state["adaptive_threshold"] = base
        return
    decisive = [o for o in outcomes if o in ("win", "loss")]
    if len(decisive) < 6:
        state["adaptive_threshold"] = base
        return

    win_rate = sum(1 for o in decisive if o == "win") / len(decisive)
    if win_rate < 0.40:
        new_th = min(CONFIG.max_threshold, base + 0.05)
    elif win_rate > 0.60:
        new_th = max(0.50, base - 0.02)
    else:
        new_th = base

    old = state.get("adaptive_threshold", base)
    state["adaptive_threshold"] = round(new_th, 4)
    if abs(state["adaptive_threshold"] - old) > 1e-9:
        log.info(
            "🎚️ Adaptive threshold: %.3f → %.3f (win_rate=%.2f, n=%d)",
            old, state["adaptive_threshold"], win_rate, len(decisive),
        )


# ===========================================================================
# ML ENGINE
# ===========================================================================
def build_model() -> VotingClassifier:
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=15,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        random_state=42,
    )
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
            proba = m.predict_proba(X[va_idx])
            preds = m.predict(X[va_idx])
            dmask = y[va_idx] != 0
            if dmask.sum() < 10:
                continue
            y_dir, p_dir = y[va_idx][dmask], preds[dmask]
            if len(np.unique(y_dir)) >= 2:
                precs.append(
                    precision_score(y_dir, p_dir, average="macro", zero_division=0)
                )
            classes = list(m.classes_)
            if 1 in classes and -1 in classes:
                bi, si = classes.index(1), classes.index(-1)
                y_bin = (y_dir == 1).astype(int)
                if len(np.unique(y_bin)) >= 2:
                    aucs.append(
                        roc_auc_score(y_bin, proba[dmask][:, [si, bi]][:, 1])
                    )
        except Exception as e:
            log.debug("purged_cv fold error: %s", e)
            continue
    if not aucs:
        return {"valid": False, "auc_mean": 0.5, "prec_mean": 0.0}
    return {
        "valid": True,
        "auc_mean": float(np.mean(aucs)),
        "prec_mean": float(np.mean(precs)) if precs else 0.0,
    }


def _model_cache_valid() -> bool:
    if CONFIG.model_cache_ttl_sec <= 0:
        return False
    if not os.path.exists(CONFIG.model_cache_path):
        return False
    age = py_time.time() - os.path.getmtime(CONFIG.model_cache_path)
    return age < CONFIG.model_cache_ttl_sec


def _load_cached_model() -> Optional[VotingClassifier]:
    if not _model_cache_valid():
        return None
    try:
        with open(CONFIG.model_cache_path, "rb") as f:
            model = pickle.load(f)
        log.info("✅ Model loaded from cache")
        return model
    except Exception as e:
        log.warning("Model cache load gagal: %s", e)
        return None


def _save_cached_model(model: VotingClassifier) -> None:
    try:
        with open(CONFIG.model_cache_path, "wb") as f:
            pickle.dump(model, f)
        log.info("✅ Model cached")
    except Exception as e:
        log.warning("Model cache save gagal: %s", e)


def train_model(df_feat: pd.DataFrame) -> Dict[str, Any]:
    cached = _load_cached_model()
    if cached is not None:
        return {"valid": True, "model": cached, "auc": None, "cached": True}

    df_lab = triple_barrier_labels(
        df_feat, CONFIG.forward_bars, CONFIG.atr_sl_multiplier
    )
    train_df = df_lab.iloc[: -CONFIG.forward_bars]
    X, y = train_df[FEATURES].values, train_df["label"].values
    if len(X) < 400 or len(np.unique(y)) < 2:
        return {
            "valid": False,
            "reason": f"insufficient_data (n={len(X)}, classes={len(np.unique(y))})",
        }

    wf = purged_cv(X, y)
    if (
        not wf["valid"]
        or wf["auc_mean"] < CONFIG.min_oos_auc
        or wf["prec_mean"] < CONFIG.min_oos_precision
    ):
        return {
            "valid": False,
            "reason": (
                f"wf_gate (AUC={wf['auc_mean']:.3f}, P={wf['prec_mean']:.3f})"
            ),
        }

    weights = np.exp(-np.arange(len(y))[::-1] / 1500.0)
    model = build_model()
    model.fit(X, y, sample_weight=weights)
    _save_cached_model(model)
    return {"valid": True, "model": model, "auc": wf["auc_mean"], "cached": False}


# ===========================================================================
# TRADE COMPUTATION
# ===========================================================================
def compute_trade(
    sig_type: str, df_feat: pd.DataFrame, prob: float, source: str
) -> Dict[str, Any]:
    last = df_feat.iloc[-1]
    price, atr = float(last["close"]), float(last["atr"])
    sl_dist = atr * CONFIG.atr_sl_multiplier
    tp_dist = sl_dist * CONFIG.risk_reward

    # Spread & slippage diterapkan konservatif ke kedua sisi eksekusi.
    exec_price = price + (CONFIG.spread_usd / 2) * (1 if sig_type == "BUY" else -1)

    if sig_type == "BUY":
        sl = exec_price - sl_dist - CONFIG.slippage_usd
        tp = exec_price + tp_dist - CONFIG.slippage_usd
    else:
        sl = exec_price + sl_dist + CONFIG.slippage_usd
        tp = exec_price - tp_dist + CONFIG.slippage_usd

    if CONFIG.account_equity <= 0:
        raise ValueError("ACCOUNT_EQUITY tidak valid")

    risk_usd = CONFIG.account_equity * (CONFIG.risk_percent / 100.0)
    sl_abs = abs(exec_price - sl)
    lot = (
        CONFIG.min_lot
        if sl_abs <= 0
        else max(
            CONFIG.min_lot,
            min(
                round(risk_usd / (sl_abs * CONFIG.contract_size), 2),
                CONFIG.max_lot,
            ),
        )
    )
    expires = now_utc() + timedelta(
        minutes=CONFIG.timeframe_minutes * CONFIG.forward_bars * 2
    )
    return {
        "type": sig_type,
        "entry": round(exec_price, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "lot": lot,
        "risk_usd": round(risk_usd, 2),
        "prob": prob,
        "source": source,
        "time": now_utc().isoformat(),
        "expires_at": expires.isoformat(),
    }


def format_message(trade: Dict, source: str) -> str:
    emoji = "🟢" if trade["type"] == "BUY" else "🔴"
    return (
        f"<b>{emoji} XAUUSD {trade['type']}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕒 {trade['time'][:16].replace('T', ' ')} UTC\n"
        f"📡 Source: {source}\n"
        f"🎯 Probability: <b>{trade['prob'] * 100:.1f}%</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 Entry: <b>{trade['entry']:.2f}</b>\n"
        f"🛑 SL: <b>{trade['sl']:.2f}</b>\n"
        f"🎯 TP: <b>{trade['tp']:.2f}</b>\n"
        f"⚖️ Lot: <b>{trade['lot']}</b> (Risk ${trade['risk_usd']:.2f})\n"
        f"<i>Expires: {trade['expires_at'][:16].replace('T', ' ')} UTC</i>"
    )


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    log.info("=" * 60)
    log.info("XAUUSD v26.5 — GitHub Actions Edition (FINAL)")
    log.info("=" * 60)
    log.info("ACCOUNT_EQUITY: $%.2f", CONFIG.account_equity)

    if CONFIG.account_equity <= 0:
        log.critical("🛑 ACCOUNT_EQUITY tidak valid. Periksa secret GitHub.")
        return

    state = load_state()
    state["last_run_time"] = now_utc().isoformat()

    today = now_utc().strftime("%Y-%m-%d")
    if state.get("last_reset_date") != today:
        state["daily_signals_count"] = 0
        state["daily_pnl_usd"] = 0.0
        state["last_reset_date"] = today
        log.info("🔄 Daily state direset")

    if not is_market_open():
        log.info("📅 Market closed. Skip.")
        return

    if os.path.exists(CONFIG.kill_switch_path):
        log.critical("🛑 Kill switch aktif. Bot berhenti.")
        return

    if state["daily_signals_count"] >= CONFIG.max_daily_signals:
        log.info(
            "⛔ Batas sinyal harian tercapai (%d).", CONFIG.max_daily_signals
        )
        return

    max_loss = CONFIG.max_daily_loss_pct * CONFIG.account_equity / 100.0
    if state.get("daily_pnl_usd", 0) <= -max_loss:
        log.critical(
            "🛑 Batas kerugian harian tercapai ($%.2f ≤ -$%.2f).",
            state["daily_pnl_usd"], max_loss,
        )
        return

    if state.get("last_signal_time"):
        try:
            mins = (
                now_utc() - datetime.fromisoformat(state["last_signal_time"])
            ).total_seconds() / 60
            if mins < CONFIG.cooldown_minutes:
                log.info(
                    "⏳ Cooldown aktif (%.0f/%d min).",
                    mins, CONFIG.cooldown_minutes,
                )
                return
        except Exception:
            pass

    try:
        df, source = fetch_xauusd_data()
        df = sanitize_df(df)
        if df is None:
            log.warning("⚠️ Data tidak valid atau stale.")
            return
    except Exception as e:
        log.error("❌ Fetch data gagal: %s", e)
        return

    state["last_data_source"] = source

    # Resolve sinyal aktif & update adaptive threshold dulu
    try:
        resolve_signals_against_df(state, df)
        update_adaptive_threshold(state)
    except Exception as e:
        log.warning("Resolve signals error: %s", e)

    try:
        df_feat = calculate_features(df)
    except Exception as e:
        log.error("❌ Feature engineering gagal: %s", e)
        return

    if len(df_feat) < 100:
        log.warning("⚠️ Data fitur terlalu sedikit (%d).", len(df_feat))
        return

    last = df_feat.iloc[-1]

    entropy = float(last["shannon_entropy"])
    hurst = float(last["hurst"])
    if entropy > CONFIG.max_entropy_threshold or hurst < CONFIG.min_hurst_threshold:
        log.info(
            "🌊 Pasar sideways ekstrem (Entropy=%.2f, Hurst=%.2f < %.2f). Skip.",
            entropy, hurst, CONFIG.min_hurst_threshold,
        )
        state["last_choppy_time"] = now_utc().isoformat()
        save_state(state)
        return

    rng_ = float(last["high"] - last["low"])
    body = abs(float(last["close"] - last["open"]))
    wick = (rng_ - body) / (rng_ + 1e-12) if rng_ > 0 else 1.0
    if wick > CONFIG.max_wick_ratio_threshold:
        log.info(
            "🕯️ Wick terlalu besar (%.2f > %.2f). Skip.",
            wick, CONFIG.max_wick_ratio_threshold,
        )
        state["last_choppy_time"] = now_utc().isoformat()
        save_state(state)
        return

    models = train_model(df_feat)
    if not models.get("valid"):
        log.info("⚠️ Model tidak valid: %s", models.get("reason"))
        save_state(state)
        return
    if models.get("cached"):
        log.info("ℹ️ Menggunakan model cache.")
    else:
        log.info("🧠 Model dilatih. OOS AUC: %.3f", models.get("auc") or 0.0)

    X_live = last[FEATURES].values.reshape(1, -1)
    model = models["model"]
    pred = int(model.predict(X_live)[0])
    proba = model.predict_proba(X_live)[0]
    classes = list(model.classes_)
    prob_dict = {int(c): float(proba[i]) for i, c in enumerate(classes)}

    log.info(
        "🔎 Prediction=%d | P(BUY)=%.3f P(SELL)=%.3f P(0)=%.3f",
        pred,
        prob_dict.get(1, 0.0),
        prob_dict.get(-1, 0.0),
        prob_dict.get(0, 0.0),
    )

    if pred == 0:
        log.info("ℹ️ Tidak ada sinyal jelas (pred=0).")
        save_state(state)
        return

    sig_type = "BUY" if pred == 1 else "SELL"
    prob = prob_dict.get(pred, 0.0)
    th = state["adaptive_threshold"]

    if prob < th:
        log.info("📉 Prob %.2f < threshold %.2f. Skip.", prob, th)
        save_state(state)
        return

    try:
        trade = compute_trade(sig_type, df_feat, prob, source)
    except Exception as e:
        log.error("❌ Compute trade gagal: %s", e)
        return

    msg = format_message(trade, source)
    if not send_telegram(msg):
        log.error("❌ Telegram gagal dikirim.")
        save_state(state)
        return

    # Append sinyal baru, batasi panjang history
    state["active_signals"].append(trade)
    if len(state["active_signals"]) > CONFIG.max_active_signals:
        state["active_signals"] = state["active_signals"][-CONFIG.max_active_signals:]

    state["daily_signals_count"] += 1
    state["last_signal_time"] = now_utc().isoformat()
    state["last_choppy_time"] = None
    save_state(state)

    log.info(
        "✅ %s signal sent | Entry: %.2f | Lot: %.2f | Source: %s",
        sig_type, trade["entry"], trade["lot"], source,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("⚠️ Dihentikan oleh pengguna.")
    except Exception as e:
        log.critical("💥 Fatal error: %s", e, exc_info=True)
        sys.exit(1)
