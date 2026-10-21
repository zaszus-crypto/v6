#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD BOT v30.1 — INSTITUTIONAL HARDENED + TIMEOUT-SAFE EDITION
=================================================================
Audit remediation (v30.0) + provider timeout fixes (v30.1).

FIX MAP:
    F-001 Model cache validation semantics
    F-002 Gist revision-checked write
    H-001 Purged walk-forward + embargo
    H-002 Intrabar ambiguity (stop-first conservative)
    H-003 Trial count recording
    M-001 Rolling Hurst/Entropy fresh on last bar
    M-002 Timeout exit full spread + slippage
    M-003 Actual PnL from exit_price
    M-004 External kill-switch file
    M-005 Peak-to-trough weekly drawdown enforcement
    M-006 Signal idempotency (pending_send state)
    L-001 Hurst numerical threshold (tau > 1e-12)
    T-001 Deriv ticks_history hard timeout (asyncio.wait_for 15s)
    T-002 Per-provider ThreadPool timeout (45s) + global deadline (90s)
    T-003 YFinance download ThreadPool timeout (25s)
    T-004 Verbose per-provider logging (no more silent failures)
"""

from __future__ import annotations

import os
import sys
import json
import time as py_time
import hashlib
import pickle
import asyncio
import concurrent.futures
import logging
import traceback
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Iterator

try:
    import numpy as np
    import pandas as pd
    import requests
    import yfinance as yf
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.metrics import precision_score, roc_auc_score
except ImportError as e:
    print(f"FATAL: missing dependency: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from deriv_api import DerivAPI
    _HAS_DERIV = True
except ImportError:
    _HAS_DERIV = False


# ===========================================================================
# ENV HELPERS
# ===========================================================================
def _env_float(k: str, d: float) -> float:
    v = (os.getenv(k) or "").strip()
    try: return float(v) if v else d
    except ValueError: return d


def _env_bool(k: str, d: bool = False) -> bool:
    v = (os.getenv(k) or "").strip().lower()
    return v in ("true", "1", "yes", "on") if v else d


def _env_str(k: str, d: str = "") -> str:
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
    stale_data_max_minutes: int = 60

    account_equity: float = _env_float("ACCOUNT_EQUITY", 1000.0)
    risk_percent: float = _env_float("RISK_PERCENT", 1.0)
    contract_size: float = 100.0
    max_lot: float = 1.0
    min_lot: float = 0.01

    hurst_window: int = 150
    hurst_stride: int = 5
    entropy_window: int = 150
    entropy_stride: int = 5

    model_cache_path: str = str(Path.home() / ".xauusd_model_v30.pkl")
    model_max_age_minutes: int = 60 * 12
    state_path: str = str(Path.home() / ".xauusd_state_v30.json")
    state_lock_path: str = str(Path.home() / ".xauusd_state_v30.lock")
    state_lock_timeout_sec: int = 300
    kill_switch_env: str = "KILL_SWITCH"
    kill_switch_file: str = _env_str("KILL_SWITCH_FILE", str(Path.home() / ".xauusd_kill_switch"))

    purge_embargo_bars: int = 10
    min_train_samples: int = 200

    audit_log_path: str = str(Path.home() / ".xauusd_audit.jsonl")

    # T-002: provider timeouts
    provider_timeout_sec: int = 45
    fetch_deadline_sec: int = 90
    deriv_timeout_sec: int = 15
    yfinance_timeout_sec: int = 25


CONFIG = Config()

FEATURES = [
    "rsi", "adx", "di_difference", "ema_distance", "ema_slope",
    "trend_strength", "atr_pct", "atr_regime", "realized_vol", "vol_zscore",
    "bb_position", "keltner_position", "hurst", "shannon_entropy",
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "bos_score",
    "liquidity_sweep_score", "return_1", "return_3", "return_5",
    "volume_zscore", "volume_ratio", "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
    "session_london", "session_newyork", "session_overlap",
]


# ===========================================================================
# LOGGING (structured JSON)
# ===========================================================================
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("xauusd_v30")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(JsonFormatter())
    logger.addHandler(sh)

    try:
        fh = logging.FileHandler(CONFIG.audit_log_path, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        logger.addHandler(fh)
    except Exception:
        pass

    return logger


log = _setup_logging()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ===========================================================================
# FILE LOCK
# ===========================================================================
class FileLock:
    def __init__(self, path: str, timeout: int = 300):
        self.path = path
        self.timeout = timeout

    def __enter__(self):
        deadline = py_time.time() + self.timeout
        while py_time.time() < deadline:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if py_time.time() - os.path.getmtime(self.path) > self.timeout * 2:
                        os.remove(self.path)
                        continue
                except OSError:
                    pass
                py_time.sleep(0.5)
        raise TimeoutError("State lock timeout")

    def __exit__(self, *args):
        try:
            os.remove(self.path)
        except OSError:
            pass


# ===========================================================================
# STATE
# ===========================================================================
class StateConcurrencyError(RuntimeError):
    pass


def _default_state() -> Dict[str, Any]:
    return {
        "version": 301,
        "_revision": 0,
        "last_signal_time": None,
        "daily_signals_count": 0,
        "daily_pnl_usd": 0.0,
        "last_reset_date": now_utc().strftime("%Y-%m-%d"),
        "active_signals": [],
        "pending_send": [],
        "performance_history": {"wins": 0, "losses": 0, "ambiguous": 0},
        "adaptive_threshold": CONFIG.base_min_signal_probability,
        "recent_outcomes": [],
        "consecutive_losses": 0,
        "week_start_equity": CONFIG.account_equity,
        "week_start_date": now_utc().strftime("%Y-%W"),
        "peak_equity": CONFIG.account_equity,
        "last_data_source": None,
        "open_signal_ids": [],
        "last_spot_price": 0.0,
        "last_futures_basis": 0.0,
        "trial_count": 0,
        "last_data_datetime": None,
    }


def _merge_defaults(s: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in _default_state().items():
        if k not in s:
            s[k] = v
    return s


def _fetch_gist_state() -> Optional[Dict[str, Any]]:
    if not (CONFIG.gist_token and CONFIG.gist_id):
        return None
    for _ in range(3):
        try:
            r = requests.get(
                f"https://api.github.com/gists/{CONFIG.gist_id}",
                headers={"Authorization": f"token {CONFIG.gist_token}"},
                timeout=15,
            )
            if r.status_code == 200:
                for fn, fc in r.json().get("files", {}).items():
                    if fn.endswith("bot_state.json"):
                        return _merge_defaults(json.loads(fc["content"]))
            else:
                log.warning(f"Gist load HTTP {r.status_code}")
        except Exception as e:
            log.warning(f"Gist load failed: {e}")
            py_time.sleep(2)
    return None


def load_state() -> Dict[str, Any]:
    if os.path.exists(CONFIG.state_path):
        try:
            with open(CONFIG.state_path, "r", encoding="utf-8") as f:
                return _merge_defaults(json.load(f))
        except Exception as e:
            log.warning(f"Local state load failed: {e}")
    remote = _fetch_gist_state()
    if remote is not None:
        log.info("State loaded from Gist")
        return remote
    return _default_state()


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_state_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _save_gist_revision_checked(state: Dict[str, Any]) -> bool:
    if not (CONFIG.gist_token and CONFIG.gist_id):
        return True
    remote = _fetch_gist_state()
    if remote is not None:
        remote_rev = int(remote.get("_revision", 0))
        local_rev = int(state.get("_revision", 0))
        if remote_rev > local_rev:
            raise StateConcurrencyError(
                f"Gist revision {remote_rev} > local {local_rev}; refusing overwrite."
            )
    for _ in range(3):
        try:
            r = requests.patch(
                f"https://api.github.com/gists/{CONFIG.gist_id}",
                json={"files": {"bot_state.json": {"content": json.dumps(state)}}},
                headers={"Authorization": f"token {CONFIG.gist_token}"},
                timeout=15,
            )
            if r.status_code == 200:
                return True
            if r.status_code == 412:
                raise StateConcurrencyError("Gist 412 Precondition Failed.")
            log.warning(f"Gist save HTTP {r.status_code}")
            py_time.sleep(2)
        except StateConcurrencyError:
            raise
        except Exception as e:
            log.warning(f"Gist save failed: {e}")
            py_time.sleep(2)
    return False


def save_state(state: Dict[str, Any]) -> bool:
    if CONFIG.backtest_mode:
        return True
    state["_revision"] = int(state.get("_revision", 0)) + 1
    try:
        _atomic_write_json(CONFIG.state_path, state)
    except Exception as e:
        log.error(f"Local save failed: {e}")
        return False
    if CONFIG.gist_token and CONFIG.gist_id:
        return _save_gist_revision_checked(state)
    return True


# ===========================================================================
# TELEGRAM
# ===========================================================================
def send_telegram(message: str, signal_id: Optional[str] = None) -> bool:
    if CONFIG.backtest_mode:
        return True
    if not CONFIG.telegram_token or not CONFIG.telegram_chat_id:
        return False
    url = f"https://api.telegram.org/bot{CONFIG.telegram_token}/sendMessage"
    payload = {
        "chat_id": CONFIG.telegram_chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 200:
                return True
            if r.status_code == 429:
                try:
                    retry_after = int(r.json().get("parameters", {}).get("retry_after", 5))
                except Exception:
                    retry_after = 5
                py_time.sleep(retry_after + 1)
                continue
            log.warning(f"Telegram HTTP {r.status_code}: {r.text[:200]}")
            py_time.sleep(2 + attempt * 2)
        except requests.RequestException as e:
            log.warning(f"Telegram send error: {e}")
            py_time.sleep(2 + attempt * 2)
    return False


# ===========================================================================
# DATA PROVIDERS
# ===========================================================================
REQUIRED_COLS = ["datetime", "open", "high", "low", "close", "volume"]


def _validate_ohlc(df: Optional[pd.DataFrame], name: str) -> Optional[pd.DataFrame]:
    if df is None or len(df) < CONFIG.min_candles:
        return None
    if any(c not in df.columns for c in REQUIRED_COLS):
        return None
    if not df["datetime"].is_monotonic_increasing:
        df = df.sort_values("datetime").reset_index(drop=True)
    if df[["open", "high", "low", "close"]].isna().any().any():
        return None
    return df


def _fetch_external_spot() -> Optional[float]:
    if not CONFIG.goldapi_key:
        return None
    try:
        r = requests.get(
            "https://www.goldapi.io/api/XAU/USD",
            headers={"x-access-token": CONFIG.goldapi_key},
            timeout=10,
        )
        if r.status_code == 200:
            price = float(r.json().get("price", 0))
            log.info(f"Gold-API spot: {price}")
            return price or None
        log.warning(f"Gold-API HTTP {r.status_code}")
    except Exception as e:
        log.warning(f"Gold-API failed: {e}")
    return None


def _price_matches(df: pd.DataFrame, ext_spot: Optional[float], name: str) -> bool:
    if ext_spot is None:
        return True
    if name in ("YAHOO_FUTURES", "YFINANCE_OFFSET"):
        return True
    last = float(df["close"].iloc[-1])
    dev = abs(last - ext_spot) / ext_spot * 100
    if dev > CONFIG.price_deviation_max_pct:
        log.warning(f"{name} dev {dev:.2f}% > {CONFIG.price_deviation_max_pct}% — reject")
        return False
    return True


# --- TwelveData (T-004: verbose logging) -----------------------------------
def _fetch_twelvedata_multi(interval: str, bars: int) -> Optional[pd.DataFrame]:
    keys = [k for k in (CONFIG.twelvedata_key, CONFIG.twelvedata_key_2) if k]
    if not keys:
        log.warning("TwelveData: no API keys configured")
        return None
    for i, key in enumerate(keys, 1):
        log.info(f"TwelveData: trying key #{i} (...{key[-4:]})")
        df = _fetch_twelvedata_single(key, interval, bars)
        if df is not None:
            log.info(f"TwelveData OK: {len(df)} bars via key #{i}")
            return df
        log.warning(f"TwelveData key #{i} failed")
    return None


def _fetch_twelvedata_single(api_key: str, interval: str, bars: int) -> Optional[pd.DataFrame]:
    if not api_key:
        return None
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD",
        "interval": interval,
        "outputsize": min(bars, 5000),
        "apikey": api_key,
        "format": "JSON",
        "timezone": "UTC",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            log.warning(f"TwelveData HTTP {r.status_code}: {r.text[:200]}")
            return None
        j = r.json()
        if "values" not in j:
            log.warning(f"TwelveData no values: {json.dumps(j)[:300]}")
            return None
        df = pd.DataFrame(j["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0)
        return df.sort_values("datetime").reset_index(drop=True)[REQUIRED_COLS]
    except requests.Timeout:
        log.warning("TwelveData TIMEOUT 15s")
        return None
    except Exception as e:
        log.warning(f"TwelveData error: {e}")
        return None


# --- Deriv (T-001: hard timeout) -------------------------------------------
async def _fetch_deriv_async(interval: str, bars: int) -> Optional[pd.DataFrame]:
    if not _HAS_DERIV:
        log.info("Deriv: library not installed, skipping")
        return None
    gran_map = {"15min": 900, "1h": 3600, "5min": 300, "1min": 60}
    api = DerivAPI(app_id=1089)
    try:
        response = await asyncio.wait_for(
            api.ticks_history({
                "ticks_history": "frxXAUUSD",
                "style": "candles",
                "granularity": gran_map.get(interval, 900),
                "count": min(bars, 5000),
                "end": "latest",
            }),
            timeout=CONFIG.deriv_timeout_sec,
        )
        if "candles" not in response:
            log.warning("Deriv: response without candles")
            return None
        df = pd.DataFrame(response["candles"])
        df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
        df["volume"] = 0.0
        return df[REQUIRED_COLS].sort_values("datetime").reset_index(drop=True)
    except asyncio.TimeoutError:
        log.warning(f"Deriv ticks_history TIMEOUT {CONFIG.deriv_timeout_sec}s")
        return None
    except Exception as e:
        log.warning(f"Deriv async failed: {e}")
        return None
    finally:
        try:
            await asyncio.wait_for(api.disconnect(), timeout=5.0)
        except Exception:
            pass


def _fetch_deriv(interval: str, bars: int) -> Optional[pd.DataFrame]:
    try:
        return asyncio.run(_fetch_deriv_async(interval, bars))
    except Exception as e:
        log.warning(f"Deriv top-level failed: {e}")
        return None


# --- YFinance offset (T-003: ThreadPool timeout) ---------------------------
def _fetch_yfinance_offset(interval: str, bars: int, state: Dict[str, Any]) -> Optional[pd.DataFrame]:
    try:
        spot_anchor = state.get("last_spot_price", 0.0)
        if spot_anchor == 0.0 and CONFIG.twelvedata_key:
            df_spot = _fetch_twelvedata_single(CONFIG.twelvedata_key, interval, 10)
            if df_spot is not None:
                spot_anchor = float(df_spot["close"].iloc[-1])

        period = "5d" if "min" in interval else "1mo"

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                yf.download, "GC=F", period=period,
                interval=interval, progress=False, threads=False,
            )
            try:
                df_fut = fut.result(timeout=CONFIG.yfinance_timeout_sec)
            except concurrent.futures.TimeoutError:
                log.warning(f"YFinance download TIMEOUT {CONFIG.yfinance_timeout_sec}s")
                return None

        if df_fut is None or df_fut.empty:
            log.warning("YFinance: empty dataframe")
            return None

        last_fut = float(df_fut["Close"].iloc[-1])
        basis = state.get("last_futures_basis", 0.0)
        if spot_anchor > 0:
            basis = last_fut - spot_anchor
            state["last_futures_basis"] = basis
            log.info(f"YFinance basis applied: {basis:.2f} (fut={last_fut:.2f}, spot={spot_anchor:.2f})")
        else:
            log.warning("YFinance: no spot anchor, using historical basis")

        df = pd.DataFrame({
            "datetime": df_fut.index,
            "open": df_fut["Open"] - basis,
            "high": df_fut["High"] - basis,
            "low": df_fut["Low"] - basis,
            "close": df_fut["Close"] - basis,
            "volume": df_fut["Volume"],
        }).reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df.tail(bars).reset_index(drop=True)
    except Exception as e:
        log.warning(f"YFinance offset failed: {e}")
        return None


# --- OANDA -----------------------------------------------------------------
def _fetch_oanda(interval: str, bars: int) -> Optional[pd.DataFrame]:
    if not CONFIG.oanda_api_key:
        log.info("OANDA: no API key configured, skipping")
        return None
    gran = {"15min": "M15", "1h": "H1", "5min": "M5"}.get(interval, "M15")
    host = "https://api-fxpractice.oanda.com" if CONFIG.oanda_env == "practice" \
        else "https://api-fxtrade.oanda.com"
    try:
        r = requests.get(
            f"{host}/v3/instruments/XAU_USD/candles",
            params={"granularity": gran, "count": bars, "price": "M"},
            headers={"Authorization": f"Bearer {CONFIG.oanda_api_key}"},
            timeout=20,
        )
        if r.status_code != 200:
            log.warning(f"OANDA HTTP {r.status_code}: {r.text[:200]}")
            return None
        rows = []
        for c in r.json().get("candles", []):
            if not c.get("complete", False):
                continue
            m = c["mid"]
            rows.append({
                "datetime": pd.to_datetime(c["time"], utc=True),
                "open": float(m["o"]),
                "high": float(m["h"]),
                "low": float(m["l"]),
                "close": float(m["c"]),
                "volume": int(c.get("volume", 0)),
            })
        return pd.DataFrame(rows) if rows else None
    except Exception as e:
        log.warning(f"OANDA failed: {e}")
        return None


# --- Orchestrator (T-002: per-provider timeout + global deadline) ---------
def fetch_xauusd_data(state: Dict[str, Any], interval: str = "15min", bars: int = 5000) -> Tuple[pd.DataFrame, str]:
    started = py_time.time()
    log.info("Fetching external spot (Gold-API)...")
    ext_spot = _fetch_external_spot()

    providers = [
        ("TWELVE_DATA_SPOT", lambda: _fetch_twelvedata_multi(interval, bars)),
        ("DERIV",            lambda: _fetch_deriv(interval, bars)),
        ("YFINANCE_OFFSET",  lambda: _fetch_yfinance_offset(interval, bars, state)),
        ("OANDA_V20",        lambda: _fetch_oanda(interval, bars)),
    ]

    for name, fn in providers:
        elapsed = py_time.time() - started
        if elapsed > CONFIG.fetch_deadline_sec:
            raise RuntimeError(f"Global data deadline exceeded ({elapsed:.0f}s) at {name}")
        log.info(f"Provider attempt: {name} (elapsed={elapsed:.1f}s)")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn)
            try:
                raw = fut.result(timeout=CONFIG.provider_timeout_sec)
            except concurrent.futures.TimeoutError:
                log.warning(f"Provider {name} HARD TIMEOUT {CONFIG.provider_timeout_sec}s — skipping")
                continue
            except Exception as e:
                log.warning(f"Provider {name} exception: {e}")
                continue

        try:
            df = _validate_ohlc(raw, name)
            if df is None:
                log.info(f"Provider {name}: insufficient or invalid bars")
                continue
            if not _price_matches(df, ext_spot, name):
                log.info(f"Provider {name}: price mismatch rejected")
                continue
            state["last_spot_price"] = float(df["close"].iloc[-1])
            log.info(f"Data OK: {name} | {len(df)} bars | last={state['last_spot_price']:.2f}")
            return df, name
        except Exception as e:
            log.warning(f"Provider {name} post-processing error: {e}")

    raise RuntimeError(f"All providers failed (elapsed={py_time.time()-started:.1f}s)")


def sanitize_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 2:
        return None
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    age_min = (now_utc() - df["datetime"].iloc[-1]).total_seconds() / 60
    if age_min < CONFIG.timeframe_minutes:
        df = df.iloc[:-1].reset_index(drop=True)
        if len(df) > 0:
            age_min = (now_utc() - df["datetime"].iloc[-1]).total_seconds() / 60
    if age_min > CONFIG.stale_data_max_minutes:
        log.warning(f"Stale data: last bar {age_min:.1f} min old > {CONFIG.stale_data_max_minutes}")
        return None
    return df if len(df) >= CONFIG.min_candles else None


# ===========================================================================
# FEATURES
# ===========================================================================
def wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
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
    valid = tau > 1e-12
    if valid.sum() < 3:
        return 0.5
    try:
        slope = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0]
        return float(np.clip(slope, 0.0, 1.0))
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
    last_valid = np.nan
    for i in range(window - 1, n):
        if i == n - 1 or (i - (window - 1)) % stride == 0 or not np.isfinite(last_valid):
            last_valid = hurst_exponent(log_s[i - window + 1: i + 1])
        out[i] = last_valid
    return out


def rolling_entropy(returns: np.ndarray, window: int, stride: int) -> np.ndarray:
    n = len(returns)
    out = np.full(n, np.nan)
    last_valid = np.nan
    for i in range(window - 1, n):
        if i == n - 1 or (i - (window - 1)) % stride == 0 or not np.isfinite(last_valid):
            last_valid = shannon_entropy(returns[i - window + 1: i + 1])
        out[i] = last_valid
    return out


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
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / (df["atr"] + 1e-12)
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / (df["atr"] + 1e-12)
    df["adx"] = (100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)).ewm(alpha=1 / 14, adjust=False).mean()
    df["di_difference"] = pdi - mdi

    m, s = df["close"].rolling(20).mean(), df["close"].rolling(20).std()
    df["bb_position"] = (df["close"] - m) / (2 * s + 1e-12)
    km = df["close"].ewm(span=20, adjust=False).mean()
    kb = df["atr"].ewm(span=20, adjust=False).mean() * 2
    df["keltner_position"] = (df["close"] - km) / (kb + 1e-12)

    ret = df["close"].pct_change()
    df["realized_vol"] = ret.rolling(20).std() * np.sqrt(24 * 60 / CONFIG.timeframe_minutes)
    df["vol_zscore"] = (df["realized_vol"] - df["realized_vol"].rolling(100).mean()) / (
        df["realized_vol"].rolling(100).std() + 1e-12
    )
    df["hurst"] = rolling_hurst(df["close"].values, CONFIG.hurst_window, CONFIG.hurst_stride)
    df["shannon_entropy"] = rolling_entropy(ret.fillna(0).values, CONFIG.entropy_window, CONFIG.entropy_stride)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (rng + 1e-12)
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (rng + 1e-12)
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (rng + 1e-12)

    sw = 5
    df["prior_high"] = df["high"].rolling(sw).max().shift(1)
    df["prior_low"] = df["low"].rolling(sw).min().shift(1)
    df["bos_score"] = np.where(
        (df["high"] > df["prior_high"]) & (df["close"] > df["prior_high"]), 1.0,
        np.where((df["low"] < df["prior_low"]) & (df["close"] < df["prior_low"]), -1.0, 0.0)
    )
    df["liquidity_sweep_score"] = np.where(
        (df["low"] < df["prior_low"]) & (df["close"] > df["prior_low"]), 1.0,
        np.where((df["high"] > df["prior_high"]) & (df["close"] < df["prior_high"]), -1.0, 0.0)
    )
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    v = df["volume"].fillna(0)
    vm, vs = v.rolling(20).mean(), v.rolling(20).std()
    if (v > 0).sum() > 20:
        df["volume_zscore"] = ((v - vm) / (vs + 1e-12)).clip(-5, 5)
        df["volume_ratio"] = (v / (vm + 1e-12)).clip(0, 10)
    else:
        df["volume_zscore"] = 0.0
        df["volume_ratio"] = 1.0

    hr = df["datetime"].dt.hour
    dow = df["datetime"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * dow / 7)
    df["session_london"] = hr.between(7, 15).astype(int)
    df["session_newyork"] = hr.between(13, 21).astype(int)
    df["session_overlap"] = hr.between(13, 16).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df if len(df) >= CONFIG.min_candles else df.iloc[0:0]


# ===========================================================================
# TRIPLE BARRIER + PURGED WALK-FORWARD
# ===========================================================================
def triple_barrier_labels_with_end(
    df: pd.DataFrame, fwd: int, atr_mult: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    a = df["atr"].values
    n = len(df)
    labels = np.zeros(n, dtype=int)
    label_end = np.full(n, -1, dtype=int)

    for i in range(n - fwd):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        up = c[i] + a[i] * atr_mult
        dn = c[i] - a[i] * atr_mult
        hh = h[i + 1: i + fwd + 1]
        ll = l[i + 1: i + fwd + 1]
        hit_up = hh >= up
        hit_dn = ll <= dn
        trig = hit_up | hit_dn
        if not trig.any():
            labels[i] = 0
            label_end[i] = i + fwd
            continue
        j = int(np.argmax(trig))
        label_end[i] = i + 1 + j
        if hit_up[j] and hit_dn[j]:
            labels[i] = 0
        elif hit_up[j]:
            labels[i] = 1
        elif hit_dn[j]:
            labels[i] = -1
    df["label"] = labels
    return df, label_end


def _purged_walk_forward_splits(
    n_samples: int,
    label_end: np.ndarray,
    n_splits: int = 4,
    embargo_bars: int = 10,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    valid_idx = np.flatnonzero(label_end >= 0)
    if len(valid_idx) < n_splits * 2:
        return
    fold_boundaries = np.array_split(valid_idx, n_splits + 1)
    for k in range(1, n_splits + 1):
        test_idx = fold_boundaries[k]
        if len(test_idx) == 0:
            continue
        test_start = int(test_idx[0])
        candidate = valid_idx[valid_idx < test_start]
        if len(candidate) < CONFIG.min_train_samples:
            continue
        purged = candidate[label_end[candidate] < test_start]
        train_idx = purged[purged < (test_start - embargo_bars)]
        if len(train_idx) < CONFIG.min_train_samples:
            continue
        yield train_idx, test_idx


def build_model() -> VotingClassifier:
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=20,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=42,
    )
    return VotingClassifier([("rf", rf), ("gb", gb)], voting="soft")


def purged_embargoed_cv(
    X: np.ndarray, y: np.ndarray, label_end: np.ndarray,
) -> Dict[str, Any]:
    aucs: List[float] = []
    precs: List[float] = []
    for tr_idx, va_idx in _purged_walk_forward_splits(
        len(X), label_end, n_splits=4, embargo_bars=CONFIG.purge_embargo_bars,
    ):
        if len(np.unique(y[tr_idx])) < 2:
            continue
        try:
            m = build_model()
            n_tr = len(tr_idx)
            w = np.exp(-(n_tr - 1 - np.arange(n_tr)) / 500.0)
            m.fit(X[tr_idx], y[tr_idx], sample_weight=w)
            proba = m.predict_proba(X[va_idx])
            preds = m.predict(X[va_idx])
            dmask = y[va_idx] != 0
            if dmask.sum() < 10:
                continue
            y_dir = y[va_idx][dmask]
            p_dir = preds[dmask]
            mask_dir = p_dir != 0
            if mask_dir.sum() >= 10 and len(np.unique(y_dir[mask_dir])) >= 2:
                precs.append(precision_score(y_dir[mask_dir], p_dir[mask_dir],
                                             average="weighted", zero_division=0))
            classes = list(m.classes_)
            if 1 in classes and -1 in classes:
                bi = classes.index(1)
                y_bin = (y_dir == 1).astype(int)
                if len(np.unique(y_bin)) >= 2:
                    aucs.append(roc_auc_score(y_bin, proba[dmask][:, bi]))
        except Exception as e:
            log.warning(f"CV fold error: {e}")
            continue
    if not aucs:
        return {"valid": False, "auc_mean": 0.5, "prec_mean": 0.0, "n_folds": 0}
    return {
        "valid": True,
        "auc_mean": float(np.mean(aucs)),
        "prec_mean": float(np.mean(precs)) if precs else 0.0,
        "n_folds": len(aucs),
    }


def _model_fingerprint(df_feat: pd.DataFrame) -> str:
    n = len(df_feat)
    last_ts = df_feat["datetime"].iloc[-1].isoformat() if n else ""
    return hashlib.sha256(
        f"{n}|{last_ts}|{CONFIG.forward_bars}|{CONFIG.atr_sl_multiplier}".encode()
    ).hexdigest()[:16]


def _load_valid_cache(df_feat: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if not os.path.exists(CONFIG.model_cache_path):
        return None
    try:
        age_min = (py_time.time() - os.path.getmtime(CONFIG.model_cache_path)) / 60
        if age_min > CONFIG.model_max_age_minutes:
            return None
        with open(CONFIG.model_cache_path, "rb") as f:
            cached = pickle.load(f)
        if not isinstance(cached, dict) or not cached.get("valid"):
            return None
        if cached.get("data_fingerprint") != _model_fingerprint(df_feat):
            log.info("Cache fingerprint mismatch — retraining")
            return None
        return cached
    except Exception as e:
        log.warning(f"Cache load failed: {e}")
        return None


def train_model(df_feat: pd.DataFrame, force: bool = False) -> Dict[str, Any]:
    if not force:
        cached = _load_valid_cache(df_feat)
        if cached is not None:
            log.info(f"Cache hit: model v{cached.get('model_version')} AUC={cached.get('auc'):.3f}")
            return cached

    df_lab, label_end = triple_barrier_labels_with_end(
        df_feat, CONFIG.forward_bars, CONFIG.atr_sl_multiplier,
    )
    train_idx = np.flatnonzero(label_end >= 0)
    if len(train_idx) < CONFIG.min_train_samples:
        return {"valid": False, "reason": "insufficient_data"}

    X = df_feat.iloc[train_idx][FEATURES].values
    y = df_lab.iloc[train_idx]["label"].values
    le = label_end[train_idx]
    if len(np.unique(y)) < 2:
        return {"valid": False, "reason": "single_class"}

    log.info(f"Training: {len(X)} samples, running purged CV...")
    wf = purged_embargoed_cv(X, y, le)
    log.info(f"CV result: AUC={wf['auc_mean']:.3f} PREC={wf['prec_mean']:.3f} folds={wf['n_folds']}")
    if not wf["valid"] or wf["auc_mean"] < CONFIG.min_oos_auc or wf["prec_mean"] < CONFIG.min_oos_precision:
        return {
            "valid": False,
            "reason": f"wf_gate AUC:{wf['auc_mean']:.2f} PREC:{wf['prec_mean']:.2f} folds:{wf['n_folds']}",
        }

    n = len(y)
    weights = np.exp(-(n - 1 - np.arange(n)) / 500.0)
    model = build_model()
    model.fit(X, y, sample_weight=weights)

    fingerprint = _model_fingerprint(df_feat)
    out = {
        "valid": True,
        "model": model,
        "auc": wf["auc_mean"],
        "prec": wf["prec_mean"],
        "n_folds": wf["n_folds"],
        "n_train_samples": int(n),
        "data_fingerprint": fingerprint,
        "model_version": fingerprint,
        "trained_at": now_utc().isoformat(),
    }
    if not CONFIG.backtest_mode:
        try:
            with open(CONFIG.model_cache_path + ".tmp", "wb") as f:
                pickle.dump(out, f)
            os.replace(CONFIG.model_cache_path + ".tmp", CONFIG.model_cache_path)
        except Exception as e:
            log.warning(f"Cache write failed: {e}")
    return out


# ===========================================================================
# SIGNAL RESOLUTION
# ===========================================================================
def resolve_active_signals(state: Dict[str, Any], df: pd.DataFrame) -> None:
    if not state.get("active_signals"):
        return
    now = now_utc()
    still_active: List[Dict[str, Any]] = []
    resolved: List[Tuple[Dict[str, Any], str, float, float, bool]] = []

    for sig in state["active_signals"]:
        try:
            sig_time = datetime.fromisoformat(sig["time"])
            expires = datetime.fromisoformat(sig["expires_at"])
        except Exception:
            continue
        sig_df = df[df["datetime"] > sig_time]
        outcome: Optional[str] = None
        exit_price: Optional[float] = None
        ambiguous = False

        for _, bar in sig_df.iterrows():
            if sig["type"] == "BUY":
                hit_sl = float(bar["low"]) <= sig["sl"]
                hit_tp = float(bar["high"]) >= sig["tp"]
            else:
                hit_sl = float(bar["high"]) >= sig["sl"]
                hit_tp = float(bar["low"]) <= sig["tp"]

            if hit_sl and hit_tp:
                outcome, exit_price = "loss", sig["sl"]
                ambiguous = True
                break
            if hit_sl:
                outcome, exit_price = "loss", sig["sl"]
                break
            if hit_tp:
                outcome, exit_price = "win", sig["tp"]
                break

        if outcome is None:
            if now >= expires and len(sig_df) > 0:
                last_close = float(sig_df["close"].iloc[-1])
                if sig["type"] == "BUY":
                    exit_price = last_close - CONFIG.spread_usd / 2 - CONFIG.slippage_usd
                    pnl = (exit_price - sig["entry"]) * sig["lot"] * CONFIG.contract_size
                else:
                    exit_price = last_close + CONFIG.spread_usd / 2 + CONFIG.slippage_usd
                    pnl = (sig["entry"] - exit_price) * sig["lot"] * CONFIG.contract_size
                resolved.append((sig, "timeout", exit_price, pnl, False))
            else:
                still_active.append(sig)
        else:
            cost = (CONFIG.spread_usd + CONFIG.slippage_usd) * sig["lot"] * CONFIG.contract_size
            if sig["type"] == "BUY":
                raw_pnl = (exit_price - sig["entry"]) * sig["lot"] * CONFIG.contract_size
            else:
                raw_pnl = (sig["entry"] - exit_price) * sig["lot"] * CONFIG.contract_size
            pnl = raw_pnl - cost
            resolved.append((sig, outcome, exit_price, pnl, ambiguous))

    for sig, outcome, exit_price, pnl, ambiguous in resolved:
        if outcome == "win":
            state["performance_history"]["wins"] += 1
            state["consecutive_losses"] = 0
        elif outcome == "loss":
            state["performance_history"]["losses"] += 1
            state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1
        if ambiguous:
            state["performance_history"]["ambiguous"] += 1
        state["daily_pnl_usd"] += pnl
        new_equity = CONFIG.account_equity + state["daily_pnl_usd"]
        state["peak_equity"] = max(state.get("peak_equity", CONFIG.account_equity), new_equity)
        state["recent_outcomes"].append({
            "outcome": outcome,
            "prob": sig.get("prob", 0.0),
            "signal_id": sig.get("signal_id"),
            "ambiguous": ambiguous,
            "pnl": round(pnl, 4),
        })
        log.info(f"Signal resolved: {sig.get('signal_id')} {outcome} pnl={pnl:.2f} ambiguous={ambiguous}")

    state["recent_outcomes"] = state["recent_outcomes"][-50:]
    state["active_signals"] = still_active
    state["open_signal_ids"] = [s.get("signal_id") for s in still_active if s.get("signal_id")]

    recent = [o for o in state["recent_outcomes"] if o["outcome"] in ("win", "loss")]
    if len(recent) >= 15:
        window = recent[-20:]
        wr = sum(1 for o in window if o["outcome"] == "win") / len(window)
        state["adaptive_threshold"] = float(np.clip(
            CONFIG.base_min_signal_probability + (0.55 - wr) * 0.5,
            CONFIG.min_threshold,
            CONFIG.max_threshold,
        ))


# ===========================================================================
# TRADE CONSTRUCTION
# ===========================================================================
def compute_trade(sig_type: str, df_feat: pd.DataFrame, prob: float, source: str) -> Optional[Dict[str, Any]]:
    last = df_feat.iloc[-1]
    price = float(last["close"])
    atr = float(last["atr"])
    spread_half = CONFIG.spread_usd / 2

    sl_dist = atr * CONFIG.atr_sl_multiplier + CONFIG.spread_usd + CONFIG.slippage_usd
    tp_dist = atr * CONFIG.atr_sl_multiplier * CONFIG.risk_reward - CONFIG.spread_usd - CONFIG.slippage_usd
    if tp_dist <= 0:
        return None

    if sig_type == "BUY":
        entry = price + spread_half + CONFIG.slippage_usd
        sl = price - sl_dist
        tp = price + tp_dist
    else:
        entry = price - spread_half - CONFIG.slippage_usd
        sl = price + sl_dist
        tp = price - tp_dist

    risk_target = CONFIG.account_equity * (CONFIG.risk_percent / 100.0)
    sl_ticks = abs(entry - sl)
    if sl_ticks <= 0:
        return None
    lot = round(risk_target / (sl_ticks * CONFIG.contract_size), 2)
    if lot < CONFIG.min_lot:
        return None
    lot = min(lot, CONFIG.max_lot)
    actual_risk = lot * sl_ticks * CONFIG.contract_size
    if actual_risk > risk_target * 1.05:
        return None

    sig_id = hashlib.sha256(
        f"{sig_type}:{entry:.2f}:{now_utc().isoformat()}".encode()
    ).hexdigest()[:16]
    return {
        "signal_id": sig_id,
        "type": sig_type,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "lot": lot,
        "risk_usd": round(actual_risk, 2),
        "prob": prob,
        "source": source,
        "time": now_utc().isoformat(),
        "expires_at": (now_utc() + timedelta(
            minutes=CONFIG.timeframe_minutes * CONFIG.forward_bars * 2
        )).isoformat(),
    }


def format_message(t: Dict[str, Any], source: str) -> str:
    return (
        f"<b>{'🟢' if t['type'] == 'BUY' else '🔴'} XAUUSD {t['type']}</b>\n"
        f"🕒 {t['time'][:16].replace('T', ' ')} UTC\n"
        f"📡 Source: {source}\n"
        f"🔑 ID: {t['signal_id']}\n"
        f"🎯 Prob: <b>{t['prob'] * 100:.1f}%</b>\n"
        f"📍 Entry: <b>{t['entry']:.2f}</b>\n"
        f"🛑 SL: <b>{t['sl']:.2f}</b>\n"
        f"🎯 TP: <b>{t['tp']:.2f}</b>\n"
        f"⚖️ Lot: <b>{t['lot']}</b> (Risk ${t['risk_usd']:.2f})"
    )


# ===========================================================================
# CIRCUIT BREAKERS
# ===========================================================================
def _kill_switch_active() -> bool:
    if _env_bool(CONFIG.kill_switch_env, False):
        return True
    try:
        if os.path.exists(CONFIG.kill_switch_file):
            with open(CONFIG.kill_switch_file, "r", encoding="utf-8") as f:
                return f.read().strip().lower() in ("true", "1", "yes", "on")
    except Exception:
        pass
    return False


def circuit_breakers_ok(state: Dict[str, Any], df_feat: pd.DataFrame) -> Tuple[bool, str]:
    if _kill_switch_active():
        return False, "external_kill_switch"

    if state["daily_pnl_usd"] <= -(CONFIG.max_daily_loss_pct * CONFIG.account_equity / 100.0):
        return False, "daily_loss_limit"

    if state.get("consecutive_losses", 0) >= CONFIG.max_consecutive_losses:
        return False, "consecutive_losses"

    peak = state.get("peak_equity", CONFIG.account_equity)
    current = CONFIG.account_equity + state["daily_pnl_usd"]
    if peak > 0:
        dd_pct = (peak - current) / peak * 100.0
        if dd_pct >= CONFIG.max_weekly_drawdown_pct:
            return False, f"peak_drawdown_{dd_pct:.1f}%"

    if len(df_feat) > 100:
        q = df_feat["atr"].tail(500).quantile(CONFIG.extreme_atr_percentile)
        if float(df_feat.iloc[-1]["atr"]) >= q:
            return False, "extreme_atr"

    return True, ""


# ===========================================================================
# TELEGRAM IDEMPOTENCY
# ===========================================================================
def _flush_pending_sends(state: Dict[str, Any]) -> None:
    if not state.get("pending_send"):
        return
    log.info(f"Flushing {len(state['pending_send'])} pending sends...")
    remaining = []
    for item in state["pending_send"]:
        try:
            ok = send_telegram(item["message"], signal_id=item["signal_id"])
        except Exception as e:
            log.warning(f"Pending send exception: {e}")
            ok = False
        if ok:
            state["active_signals"].append(item["trade"])
            state["open_signal_ids"].append(item["signal_id"])
            state["daily_signals_count"] += 1
            state["last_signal_time"] = item.get("queued_at", now_utc().isoformat())
            log.info(f"Pending send flushed: {item['signal_id']}")
        else:
            remaining.append(item)
            log.warning(f"Pending send still failing: {item['signal_id']}")
    state["pending_send"] = remaining


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    log.info("XAUUSD v30.1 — Institutional Hardened + Timeout-Safe starting")

    with FileLock(CONFIG.state_lock_path, CONFIG.state_lock_timeout_sec):
        log.info("Loading state...")
        state = load_state()

        today = now_utc().strftime("%Y-%m-%d")
        if state.get("last_reset_date") != today:
            state["daily_signals_count"] = 0
            state["daily_pnl_usd"] = 0.0
            state["last_reset_date"] = today

        week = now_utc().strftime("%Y-%W")
        if state.get("week_start_date") != week:
            state["week_start_equity"] = CONFIG.account_equity + state.get("daily_pnl_usd", 0.0)
            state["week_start_date"] = week

        _flush_pending_sends(state)

        if _kill_switch_active():
            log.info("Kill switch active")
            save_state(state)
            return

        if now_utc().weekday() >= 5:
            log.info("Weekend — market closed")
            save_state(state)
            return

        log.info("Fetching market data...")
        try:
            df, source = fetch_xauusd_data(state)
            df = sanitize_df(df)
            if df is None:
                log.info("No valid data after sanitize")
                save_state(state)
                return
        except Exception as e:
            log.error(f"All providers failed: {e}")
            return

        log.info(f"Data source: {source} | last bar: {df['datetime'].iloc[-1]}")
        state["last_data_source"] = source
        state["last_data_datetime"] = df["datetime"].iloc[-1].isoformat()

        resolve_active_signals(state, df)
        save_state(state)

        if state["daily_signals_count"] >= CONFIG.max_daily_signals:
            log.info("Daily signal limit reached")
            return
        if state["daily_pnl_usd"] <= -(CONFIG.max_daily_loss_pct * CONFIG.account_equity / 100.0):
            log.info("Daily loss limit reached")
            return
        if state.get("last_signal_time"):
            try:
                elapsed = (now_utc() - datetime.fromisoformat(state["last_signal_time"])).total_seconds() / 60
                if elapsed < CONFIG.cooldown_minutes:
                    log.info(f"Cooldown: {elapsed:.0f}/{CONFIG.cooldown_minutes} min")
                    return
            except Exception:
                pass

        log.info("Computing features...")
        df_feat = calculate_features(df)
        if len(df_feat) < CONFIG.min_candles:
            log.info(f"Insufficient feature rows: {len(df_feat)}")
            return
        last = df_feat.iloc[-1]

        h_val = float(last["hurst"])
        e_val = float(last["shannon_entropy"])
        log.info(f"Regime: hurst={h_val:.3f} (min={CONFIG.min_hurst_threshold}) "
                 f"entropy={e_val:.3f} (max={CONFIG.max_entropy_threshold})")
        if e_val > CONFIG.max_entropy_threshold or h_val < CONFIG.min_hurst_threshold:
            log.info("Filtered by regime")
            return

        cb_ok, cb_reason = circuit_breakers_ok(state, df_feat)
        if not cb_ok:
            log.warning(f"Circuit breaker: {cb_reason}")
            return

        state["trial_count"] = state.get("trial_count", 0) + 1

        log.info("Training / validating model...")
        models = train_model(df_feat)
        if not models.get("valid"):
            log.info(f"Invalid model: {models.get('reason')}")
            save_state(state)
            return

        X_live = last[FEATURES].values.reshape(1, -1)
        model = models["model"]
        pred = int(model.predict(X_live)[0])
        proba = model.predict_proba(X_live)[0]
        prob_dict = {int(c): float(proba[i]) for i, c in enumerate(model.classes_)}
        log.info(f"Prediction: {pred} probs={prob_dict}")

        if pred == 0:
            log.info("Neutral prediction")
            return

        sig_type = "BUY" if pred == 1 else "SELL"
        prob = prob_dict.get(pred, 0.0)
        if prob < float(state["adaptive_threshold"]):
            log.info(f"Prob {prob:.2f} < threshold {state['adaptive_threshold']:.2f}")
            return

        close_price = float(last["close"])
        ema_trend = float(last["ema_trend"])
        adx = float(last["adx"])
        rsi = float(last["rsi"])
        if sig_type == "BUY" and (close_price < ema_trend or adx < 20.0 or rsi > 75):
            log.info(f"Rule filter rejected BUY (close={close_price:.2f} ema={ema_trend:.2f} adx={adx:.1f} rsi={rsi:.1f})")
            return
        if sig_type == "SELL" and (close_price > ema_trend or adx < 20.0 or rsi < 25):
            log.info(f"Rule filter rejected SELL (close={close_price:.2f} ema={ema_trend:.2f} adx={adx:.1f} rsi={rsi:.1f})")
            return

        trade = compute_trade(sig_type, df_feat, prob, source)
        if trade is None:
            log.info("Trade sizing rejected")
            return
        if trade["signal_id"] in state.get("open_signal_ids", []):
            log.info("Duplicate signal_id")
            return

        message = format_message(trade, source)
        state["pending_send"].append({
            "signal_id": trade["signal_id"],
            "message": message,
            "trade": trade,
            "queued_at": now_utc().isoformat(),
        })
        save_state(state)

        log.info(f"Sending Telegram for {trade['signal_id']}...")
        ok = send_telegram(message, signal_id=trade["signal_id"])
        if not ok:
            log.error("Telegram send failed; kept in pending_send for retry")
            return

        state["pending_send"] = [p for p in state["pending_send"] if p["signal_id"] != trade["signal_id"]]
        state["active_signals"].append(trade)
        state["open_signal_ids"].append(trade["signal_id"])
        state["daily_signals_count"] += 1
        state["last_signal_time"] = now_utc().isoformat()
        save_state(state)

        log.info(f"{sig_type} SENT | Entry={trade['entry']:.2f} Lot={trade['lot']} "
                 f"Risk=${trade['risk_usd']:.2f} Model={models.get('model_version', 'n/a')}")


if __name__ == "__main__":
    try:
        main()
    except StateConcurrencyError as e:
        log.critical(f"State concurrency failure — manual reconciliation required: {e}")
        sys.exit(2)
    except Exception as e:
        log.critical(f"Fatal: {e}\n{traceback.format_exc()}")
        sys.exit(1)
