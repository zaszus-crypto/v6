#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD BOT v26.1 — GitHub Actions Edition (Hardened)
Fix: Robust env parsing, tahan empty string dari GitHub Secrets.
State disimpan di GitHub Gist karena GitHub Actions bersifat ephemeral.
"""

import os
import sys
import json
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
# ENV HELPER — Robust parsing untuk GitHub Actions (tahan empty string)
# ===========================================================================
def _env_float(key: str, default: float) -> float:
    """Parse env var ke float, fallback ke default jika kosong/invalid."""
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
    # API & State
    telegram_token: str = _env_str("TELEGRAM_TOKEN")
    telegram_chat_id: str = _env_str("TELEGRAM_CHAT_ID")
    gist_token: str = _env_str("GH_GIST_TOKEN")
    gist_id: str = _env_str("GH_GIST_ID")
    backtest_mode: bool = _env_bool("BACKTEST_MODE", False)

    # Trading Logic
    timeframe_minutes: int = 15
    min_candles: int = 300
    forward_bars: int = 4
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    risk_reward: float = 2.5
    spread_usd: float = 0.30
    slippage_usd: float = 0.10
    commission_per_lot: float = 7.0

    # ML Thresholds
    base_min_signal_probability: float = 0.60
    max_threshold: float = 0.75
    min_oos_auc: float = 0.58
    min_oos_precision: float = 0.55

    max_entropy_threshold: float = 0.95
    min_hurst_threshold: float = 0.40
    max_wick_ratio_threshold: float = 0.70

    # Risk Management
    cooldown_minutes: int = 120
    max_daily_signals: int = 6
    max_daily_loss_pct: float = 4.0
    account_equity: float = _env_float("ACCOUNT_EQUITY", 1000.0)
    risk_percent: float = _env_float("RISK_PERCENT", 1.0)
    contract_size: float = 100.0
    max_lot: float = 1.0
    min_lot: float = 0.01

    # System Paths
    state_path: str = str(Path.home() / ".xauusd_state.json")
    model_cache_path: str = str(Path.home() / ".xauusd_model_v26.pkl")
    kill_switch_path: str = str(Path.home() / ".xauusd_kill")


CONFIG = Config()

FEATURES = [
    "rsi", "adx", "di_difference", "ema_distance", "ema_slope",
    "trend_strength", "atr_pct", "atr_regime", "realized_vol", "vol_zscore",
    "bb_position", "keltner_position", "hurst_proxy", "shannon_entropy_proxy",
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "bos_score",
    "liquidity_sweep_score", "return_1", "return_3", "return_5",
    "volume_zscore", "volume_ratio", "hour_sin", "hour_cos", "dayofweek",
    "session_london", "session_newyork", "session_overlap",
]

# ===========================================================================
# LOGGING
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stderr,
    force=True,
)
log = logging.getLogger("xauusd_v26")


# ===========================================================================
# UTILITIES
# ===========================================================================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ===========================================================================
# STATE MANAGEMENT — GITHUB GIST (untuk GitHub Actions ephemeral)
# ===========================================================================
def _default_state() -> Dict[str, Any]:
    return {
        "version": 261,
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
    """Load state dari Gist (prioritas) atau file lokal (fallback)."""
    default = _default_state()

    # 1. Coba load dari Gist
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
                        state = json.loads(fc["content"])
                        for k, v in default.items():
                            if k not in state:
                                state[k] = v
                        log.info("✅ State loaded from Gist")
                        return state
            else:
                log.warning("Gist load HTTP %d", r.status_code)
        except Exception as e:
            log.warning("Gist load gagal: %s. Fallback ke lokal.", e)

    # 2. Fallback ke file lokal
    if os.path.exists(CONFIG.state_path):
        try:
            with open(CONFIG.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            for k, v in default.items():
                if k not in state:
                    state[k] = v
            log.info("✅ State loaded from local file")
            return state
        except Exception as e:
            log.warning("State lokal corrupt: %s", e)

    log.info("ℹ️ State default digunakan (pertama kali run)")
    return default


def save_state(state: Dict[str, Any]) -> bool:
    """Simpan state ke Gist (prioritas) DAN file lokal (backup)."""
    saved = False

    # 1. Simpan ke Gist
    if CONFIG.gist_token and CONFIG.gist_id:
        for attempt in range(1, 4):
            try:
                payload = {
                    "files": {
                        "bot_state.json": {"content": json.dumps(state, indent=2)}
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
                    continue
                else:
                    log.warning("Gist save HTTP %d (attempt %d)", r.status_code, attempt)
            except Exception as e:
                log.warning("Gist save attempt %d gagal: %s", attempt, e)
                py_time.sleep(2 ** attempt)

    # 2. Backup ke file lokal (atomic write)
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
        return True
    if not CONFIG.telegram_token or not CONFIG.telegram_chat_id:
        log.warning("⚠️ Telegram tidak terkonfigurasi. Pesan tidak terkirim.")
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
                log.warning("Telegram rate limit, tunggu %ds", retry_after)
                py_time.sleep(retry_after + 1)
                continue
            log.warning("Telegram HTTP %d", r.status_code)
            return False
        except Exception as e:
            log.warning("Telegram attempt %d gagal: %s", attempt, e)
            py_time.sleep(2 * attempt)
    return False


# ===========================================================================
# DATA FETCHING — XAUUSD Spot (Yahoo Finance)
# ===========================================================================
def fetch_xauusd_data(interval: str = "15m", days: int = 60) -> Tuple[pd.DataFrame, str]:
    """Fetch data XAUUSD dari Yahoo Finance. Spot primer, Futures fallback."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # 1. XAUUSD=X (Spot Gold)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/XAUUSD=X?interval={interval}&range={days}d"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        result = data.get("chart", {}).get("result", [])
        if result and result[0].get("timestamp"):
            res = result[0]
            ts = res["timestamp"]
            quote = res.get("indicators", {}).get("quote", [{}])[0]
            df = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(ts, unit="s", utc=True),
                    "open": quote.get("open"),
                    "high": quote.get("high"),
                    "low": quote.get("low"),
                    "close": quote.get("close"),
                    "volume": quote.get("volume", 0),
                }
            )
            df = df.dropna(subset=["open", "high", "low", "close"])
            if len(df) >= CONFIG.min_candles:
                log.info("✅ XAUUSD=X (Spot) | %d bars", len(df))
                return df, "YAHOO_SPOT"
    except Exception as e:
        log.warning("XAUUSD=X gagal: %s", e)

    # 2. Fallback GC=F (Futures)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval={interval}&range={days}d"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        res = r.json().get("chart", {}).get("result", [{}])[0]
        ts = res.get("timestamp", [])
        quote = res.get("indicators", {}).get("quote", [{}])[0]
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(ts, unit="s", utc=True),
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "volume": quote.get("volume", 0),
            }
        )
        df = df.dropna(subset=["open", "high", "low", "close"])
        if len(df) >= CONFIG.min_candles:
            log.warning("⚠️ Fallback ke GC=F (Futures)")
            return df, "YAHOO_FUTURES"
    except Exception as e:
        log.error("GC=F gagal: %s", e)

    raise RuntimeError("Semua sumber data XAUUSD gagal.")


def sanitize_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Bersihkan dan validasi DataFrame."""
    if df is None or len(df) < 2:
        return None

    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0) & (df["open"] > 0)]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["open", "high", "low", "close"]
    )

    if len(df) < 2:
        return None

    age_min = (now_utc() - df["datetime"].iloc[-1]).total_seconds() / 60
    if age_min > CONFIG.timeframe_minutes * 3:
        log.warning("⚠️ Data stale (%.0f min)", age_min)
        return None

    if age_min < CONFIG.timeframe_minutes:
        df = df.iloc[:-1].reset_index(drop=True)

    return df if len(df) >= CONFIG.min_candles else None


# ===========================================================================
# FEATURE ENGINEERING (OPTIMIZED)
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


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung semua fitur ML. Dioptimasi untuk performa."""
    df = df.copy()

    # 1. RSI
    d = df["close"].diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + g / (l + 1e-12))

    # 2. ATR & Volatility
    df["atr"] = wilder_atr(df, CONFIG.atr_period)
    df["atr_pct"] = df["atr"] / (df["close"] + 1e-12)
    df["atr_regime"] = df["atr"] / (df["atr"].rolling(100).mean() + 1e-12)

    # 3. EMA
    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_trend"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_distance"] = (df["ema_fast"] - df["ema_slow"]) / (df["atr"] + 1e-12)
    df["ema_slope"] = (df["ema_fast"] - df["ema_fast"].shift(3)) / (df["atr"] + 1e-12)
    df["trend_strength"] = (df["close"] - df["ema_trend"]) / (df["atr"] + 1e-12)

    # 4. ADX
    up, dn = df["high"].diff(), -df["low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi = (
        100
        * pd.Series(pdm, index=df.index).ewm(alpha=1 / CONFIG.atr_period, adjust=False).mean()
        / (df["atr"] + 1e-12)
    )
    mdi = (
        100
        * pd.Series(mdm, index=df.index).ewm(alpha=1 / CONFIG.atr_period, adjust=False).mean()
        / (df["atr"] + 1e-12)
    )
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
    df["adx"] = dx.ewm(alpha=1 / CONFIG.atr_period, adjust=False).mean()
    df["di_difference"] = pdi - mdi

    # 5. Bollinger & Keltner
    m, s = df["close"].rolling(20).mean(), df["close"].rolling(20).std()
    df["bb_position"] = (df["close"] - m) / (2 * s + 1e-12)
    km = df["close"].ewm(span=20, adjust=False).mean()
    kb = df["atr"].ewm(span=20, adjust=False).mean() * 2
    df["keltner_position"] = (df["close"] - km) / (kb + 1e-12)

    # 6. Realized Volatility
    ret = df["close"].pct_change()
    df["realized_vol"] = ret.rolling(20).std() * np.sqrt(24 * 60 / CONFIG.timeframe_minutes)
    df["vol_zscore"] = (df["realized_vol"] - df["realized_vol"].rolling(100).mean()) / (
        df["realized_vol"].rolling(100).std() + 1e-12
    )

    # 7. Hurst & Entropy (OPTIMIZED: step=5, forward-fill)
    step = 5
    n = len(df)
    h_arr = np.full(n, 0.5)
    e_arr = np.full(n, 1.0)
    logp = np.log(df["close"].values)

    for i in range(100, n, step):
        w_logp = logp[i - 99 : i + 1]
        w_ret = ret.values[i - 99 : i + 1]
        if np.std(w_logp) > 1e-12:
            h_arr[i] = np.clip(
                np.std(np.diff(w_logp)) / (np.std(w_ret) + 1e-12) * 0.5, 0.0, 1.0
            )
        hist, _ = np.histogram(w_ret, bins=10)
        p = hist / (hist.sum() + 1e-12)
        p = p[p > 0]
        e_arr[i] = float(-np.sum(p * np.log2(p)) / np.log2(10)) if len(p) > 0 else 1.0

    df["hurst_proxy"] = pd.Series(h_arr).ffill().bfill()
    df["shannon_entropy_proxy"] = pd.Series(e_arr).ffill().bfill()

    # 8. Candle Structure
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (rng + 1e-12)
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (rng + 1e-12)
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (rng + 1e-12)

    # 9. Market Structure (BOS & Sweep)
    sw = 5
    df["prior_high"] = df["high"].rolling(sw).max().shift(1)
    df["prior_low"] = df["low"].rolling(sw).min().shift(1)
    df["bos_score"] = np.where(
        (df["high"] > df["prior_high"]) & (df["close"] > df["prior_high"]),
        1.0,
        np.where(
            (df["low"] < df["prior_low"]) & (df["close"] < df["prior_low"]), -1.0, 0.0
        ),
    )
    df["liquidity_sweep_score"] = np.where(
        (df["low"] < df["prior_low"]) & (df["close"] > df["prior_low"]),
        1.0,
        np.where(
            (df["high"] > df["prior_high"]) & (df["close"] < df["prior_high"]), -1.0, 0.0
        ),
    )

    # 10. Returns
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    # 11. Volume
    if "volume" in df.columns and df["volume"].notna().sum() > 20:
        v = df["volume"].fillna(0)
        v_mean, v_std = v.rolling(20).mean(), v.rolling(20).std()
        df["volume_zscore"] = ((v - v_mean) / (v_std + 1e-12)).clip(-5, 5)
        df["volume_ratio"] = (v / (v_mean + 1e-12)).clip(0, 10)
    else:
        df["volume_zscore"], df["volume_ratio"] = 0.0, 1.0

    # 12. Time Features
    hr = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["session_london"] = hr.between(7, 15).astype(int)
    df["session_newyork"] = hr.between(12, 20).astype(int)
    df["session_overlap"] = hr.between(12, 15).astype(int)

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def triple_barrier_labels(df: pd.DataFrame, fwd: int, atr_mult: float) -> pd.DataFrame:
    """Labeling dengan triple barrier method."""
    df = df.copy()
    c, h, l, a = df["close"].values, df["high"].values, df["low"].values, df["atr"].values
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
# ML ENGINE
# ===========================================================================
def build_model():
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
    """Purged walk-forward cross-validation untuk mencegah data leakage."""
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
                    aucs.append(roc_auc_score(y_bin, proba[dmask][:, [si, bi]][:, 1]))
        except Exception:
            continue

    if not aucs:
        return {"valid": False, "auc_mean": 0.5, "prec_mean": 0.0}

    return {
        "valid": True,
        "auc_mean": float(np.mean(aucs)),
        "prec_mean": float(np.mean(precs)) if precs else 0.0,
    }


def train_model(df_feat: pd.DataFrame) -> Dict[str, Any]:
    """Latih model dengan validasi ketat."""
    df_lab = triple_barrier_labels(df_feat, CONFIG.forward_bars, CONFIG.atr_sl_multiplier)
    train_df = df_lab.iloc[: -CONFIG.forward_bars]
    X, y = train_df[FEATURES].values, train_df["label"].values

    if len(X) < 400 or len(np.unique(y)) < 2:
        return {"valid": False, "reason": "insufficient_data"}

    wf = purged_cv(X, y)
    if (
        not wf["valid"]
        or wf["auc_mean"] < CONFIG.min_oos_auc
        or wf["prec_mean"] < CONFIG.min_oos_precision
    ):
        return {"valid": False, "reason": f"wf_gate (AUC:{wf['auc_mean']:.2f})"}

    # Final training dengan sample weighting (data baru lebih penting)
    weights = np.exp(-np.arange(len(y))[::-1] / 500.0)
    model = build_model()
    model.fit(X, y, sample_weight=weights)

    return {"valid": True, "model": model, "auc": wf["auc_mean"]}


# ===========================================================================
# TRADE COMPUTATION
# ===========================================================================
def compute_trade(
    sig_type: str, df_feat: pd.DataFrame, prob: float, source: str
) -> Dict[str, Any]:
    """Hitung parameter trade (entry, SL, TP, lot)."""
    last = df_feat.iloc[-1]
    price, atr = float(last["close"]), float(last["atr"])

    sl_dist = atr * CONFIG.atr_sl_multiplier
    tp_dist = sl_dist * CONFIG.risk_reward

    if sig_type == "BUY":
        sl, tp = price - sl_dist - CONFIG.spread_usd, price + tp_dist
    else:
        sl, tp = price + sl_dist + CONFIG.spread_usd, price - tp_dist

    if CONFIG.account_equity <= 0:
        raise ValueError("ACCOUNT_EQUITY tidak valid")

    risk_usd = CONFIG.account_equity * (CONFIG.risk_percent / 100.0)
    sl_abs = abs(price - sl)

    if sl_abs <= 0:
        lot = CONFIG.min_lot
    else:
        lot = max(
            CONFIG.min_lot,
            min(
                round(risk_usd / (sl_abs * CONFIG.contract_size), 2),
                CONFIG.max_lot,
            ),
        )

    expires = now_utc() + timedelta(
        minutes=CONFIG.timeframe_minutes * CONFIG.forward_bars * 2
    )

    return {
        "type": sig_type,
        "entry": round(price, 2),
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
    """Format pesan Telegram."""
    emoji = "🟢" if trade["type"] == "BUY" else "🔴"
    return (
        f"<b>{emoji} XAUUSD {trade['type']}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕒 {trade['time'][:16].replace('T', ' ')} UTC\n"
        f"📡 Source: {source}\n"
        f"🎯 Probability: <b>{trade['prob']*100:.1f}%</b>\n"
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
    log.info("XAUUSD v26.1 — GitHub Actions Edition")
    log.info("=" * 60)
    log.info("ACCOUNT_EQUITY: $%.2f", CONFIG.account_equity)

    # Validasi konfigurasi awal
    if CONFIG.account_equity <= 0:
        log.critical("🛑 ACCOUNT_EQUITY tidak valid. Periksa secret GitHub.")
        return

    state = load_state()
    today = now_utc().strftime("%Y-%m-%d")

    # Reset harian
    if state.get("last_reset_date") != today:
        state["daily_signals_count"] = 0
        state["daily_pnl_usd"] = 0.0
        state["last_reset_date"] = today
        log.info("🔄 Daily state direset")

    # Safety checks
    if now_utc().weekday() >= 5:
        log.info("📅 Weekend. Pasar forex tutup. Skip.")
        return

    if os.path.exists(CONFIG.kill_switch_path):
        log.critical("🛑 Kill switch aktif. Bot berhenti.")
        return

    if state["daily_signals_count"] >= CONFIG.max_daily_signals:
        log.info("⛔ Batas sinyal harian tercapai (%d).", CONFIG.max_daily_signals)
        return

    max_loss = CONFIG.max_daily_loss_pct * CONFIG.account_equity / 100.0
    if state.get("daily_pnl_usd", 0) <= -max_loss:
        log.critical("🛑 Batas kerugian harian tercapai ($%.2f).", state["daily_pnl_usd"])
        return

    # Cooldown check
    if state.get("last_signal_time"):
        try:
            mins = (
                now_utc() - datetime.fromisoformat(state["last_signal_time"])
            ).total_seconds() / 60
            if mins < CONFIG.cooldown_minutes:
                log.info("⏳ Cooldown aktif (%.0f/%d min).", mins, CONFIG.cooldown_minutes)
                return
        except Exception:
            pass

    # Fetch data
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

    # Feature engineering
    try:
        df_feat = calculate_features(df)
    except Exception as e:
        log.error("❌ Feature engineering gagal: %s", e)
        return

    if len(df_feat) < 100:
        log.warning("⚠️ Data fitur terlalu sedikit (%d).", len(df_feat))
        return

    last = df_feat.iloc[-1]

    # Market regime filters
    entropy = float(last["shannon_entropy_proxy"])
    hurst = float(last["hurst_proxy"])
    if entropy > CONFIG.max_entropy_threshold or hurst < CONFIG.min_hurst_threshold:
        log.info(
            "🌊 Pasar choppy (Entropy=%.2f, Hurst=%.2f). Skip.", entropy, hurst
        )
        state["last_choppy_time"] = now_utc().isoformat()
        save_state(state)
        return

    rng_ = float(last["high"] - last["low"])
    body = abs(float(last["close"] - last["open"]))
    wick = (rng_ - body) / (rng_ + 1e-12) if rng_ > 0 else 1.0
    if wick > CONFIG.max_wick_ratio_threshold:
        log.info("🕯️ Wick terlalu besar (%.2f). Skip.", wick)
        state["last_choppy_time"] = now_utc().isoformat()
        save_state(state)
        return

    # ML inference
    models = train_model(df_feat)
    if not models.get("valid"):
        log.info("⚠️ Model tidak valid: %s", models.get("reason"))
        save_state(state)
        return

    X_live = last[FEATURES].values.reshape(1, -1)
    model = models["model"]
    pred = int(model.predict(X_live)[0])
    proba = model.predict_proba(X_live)[0]
    classes = list(model.classes_)
    prob_dict = {int(c): float(proba[i]) for i, c in enumerate(classes)}

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

    # Execute trade
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

    # Update state
    state["active_signals"].append(trade)
    state["daily_signals_count"] += 1
    state["last_signal_time"] = now_utc().isoformat()
    state["last_choppy_time"] = None
    save_state(state)

    log.info(
        "✅ %s signal sent | Entry: %.2f | Lot: %.2f | Source: %s",
        sig_type,
        trade["entry"],
        trade["lot"],
        source,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("⚠️ Dihentikan oleh pengguna.")
    except Exception as e:
        log.critical("💥 Fatal error: %s", e, exc_info=True)
        sys.exit(1)
