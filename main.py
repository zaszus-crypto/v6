#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD BOT v27.4 — The Ultimate Perfect Edition
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
# CONFIG (FORENSICALLY OPTIMIZED FOR 3-CLASS)
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
    forward_bars: int = 8             
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5    
    risk_reward: float = 2.0          
    spread_usd: float = 0.30

    # ✅ PERBAIKAN FORENSIK: Threshold 0.38 untuk 3 kelas (Batas acak = 0.33)
    base_min_signal_probability: float = 0.38
    max_threshold: float = 0.50
    min_threshold: float = 0.35
    
    min_oos_auc: float = 0.45         
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

# ===========================================================================
# STATE MANAGEMENT (WITH FORCED THRESHOLD OVERRIDE)
# ===========================================================================
def _default_state() -> Dict[str, Any]:
    return {
        "version": 274, 
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
                            
                            # ✅ FORENSIC FIX: Paksa reset threshold jika Gist menyimpan angka usang (>0.50)
                            if state.get("adaptive_threshold", 0.60) > 0.50:
                                log.info("🔄 Gist cache is outdated. Forcing threshold to 0.38")
                                state["adaptive_threshold"] = CONFIG.base_min_signal_probability
                                state["version"] = 274
                                
                            log.info("✅ State loaded from Gist")
                            return state
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

    return True

# ===========================================================================
# TELEGRAM
# ===========================================================================
def send_telegram(message: str) -> bool:
    if CONFIG.backtest_mode: return True
    if not CONFIG.telegram_token or not CONFIG.telegram_chat_id:
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
                py_time.sleep(int(r.json().get("parameters", {}).get("retry_after", 5)) + 1)
                continue
        except Exception:
            py_time.sleep(2 * attempt)
    return False

# ===========================================================================
# DATA FETCHING (TWELVE DATA)
# ===========================================================================
def fetch_xauusd_data(interval: str = "15min", bars: int = 5000) -> Tuple[pd.DataFrame, str]:
    api_key = _env_str("TWELVE_DATA_KEY")
    if not api_key:
        return _fetch_yahoo_futures(interval, bars)

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD", "interval": interval, "outputsize": min(bars, 5000),
        "apikey": api_key, "format": "JSON", "timezone": "UTC",
    }
    for attempt in range(1, 4):
        try:
            r = requests.get(url, params=params, timeout=20)
            data = r.json()
            values = data.get("values", [])
            if not values:
                raise RuntimeError("Empty values")
            
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
            
            log.info("✅ Data: XAU/USD spot (TWELVE_DATA) | %d bars", len(df))
            return df, "TWELVE_DATA_SPOT"
        except Exception as e:
            log.warning("TwelveData attempt %d: %s", attempt, e)
            py_time.sleep(2 ** attempt)
            
    return _fetch_yahoo_futures(interval, bars)

def _fetch_yahoo_futures(interval: str = "15m", bars: int = 5000) -> Tuple[pd.DataFrame, str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    yf_interval = "15m" if interval == "15min" else interval
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval={yf_interval}&range=60d"
    r = requests.get(url, headers=headers, timeout=15)
    res = r.json().get("chart", {}).get("result", [])[0]
    q = res.get("indicators", {}).get("quote", [{}])[0]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(res["timestamp"], unit="s", utc=True),
        "open": q.get("open"), "high": q.get("high"),
        "low": q.get("low"), "close": q.get("close"), "volume": q.get("volume", 0),
    }).dropna()
    return df, "YAHOO_FUTURES"

def sanitize_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 2: return None
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    age = (now_utc() - df["datetime"].iloc[-1]).total_seconds() / 60
    if age < CONFIG.timeframe_minutes:
        df = df.iloc[:-1].reset_index(drop=True)
    return df if len(df) >= CONFIG.min_candles else None

# ===========================================================================
# FEATURE ENGINEERING
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
    return float(np.clip(np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0], 0.0, 1.0))

def shannon_entropy(returns: np.ndarray, bins: int = 10) -> float:
    r = returns[np.isfinite(returns)]
    if len(r) < 20: return 1.0
    hist, _ = np.histogram(r, bins=bins)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(bins))

def rolling_hurst(close: np.ndarray, window: int, stride: int) -> np.ndarray:
    n = len(close); log_s = np.log(np.maximum(close, 1e-10)); out = np.full(n, np.nan)
    for i in range(window - 1, n, stride):
        out[i] = hurst_exponent(log_s[i - window + 1 : i + 1])
    return pd.Series(out).ffill().bfill().fillna(0.5).values

def rolling_entropy(returns: np.ndarray, window: int, stride: int) -> np.ndarray:
    n = len(returns); out = np.full(n, np.nan)
    for i in range(window - 1, n, stride):
        out[i] = shannon_entropy(returns[i - window + 1 : i + 1])
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
    df["bos_score"] = np.where((df["high"] > df["prior_high"]) & (df["close"] > df["prior_high"]), 1.0, np.where((df["low"] < df["prior_low"]) & (df["close"] < df["prior_low"]), -1.0, 0.0))
    df["liquidity_sweep_score"] = np.where((df["low"] < df["prior_low"]) & (df["close"] > df["prior_low"]), 1.0, np.where((df["high"] > df["prior_high"]) & (df["close"] < df["prior_high"]), -1.0, 0.0))
    df["return_1"], df["return_3"], df["return_5"] = df["close"].pct_change(1), df["close"].pct_change(3), df["close"].pct_change(5)

    if "volume" in df.columns and (df["volume"] > 0).sum() > 20:
        v = df["volume"].fillna(0); vm, vs = v.rolling(20).mean(), v.rolling(20).std()
        df["volume_zscore"] = ((v - vm) / (vs + 1e-12)).clip(-5, 5)
        df["volume_ratio"]  = (v / (vm + 1e-12)).clip(0, 10)
    else:
        df["volume_zscore"], df["volume_ratio"] = 0.0, 1.0

    hr = df["datetime"].dt.hour
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["session_london"], df["session_newyork"] = hr.between(7, 15).astype(int), hr.between(12, 20).astype(int)
    df["session_overlap"] = hr.between(12, 15).astype(int)

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

def triple_barrier_labels(df: pd.DataFrame, fwd: int, atr_mult: float) -> pd.DataFrame:
    df = df.copy()
    c, h, l, a = df["close"].values, df["high"].values, df["low"].values, df["atr"].values
    labels = np.zeros(len(df), dtype=int)
    for i in range(len(df) - fwd):
        if not np.isfinite(a[i]) or a[i] <= 0: continue
        up, dn = c[i] + a[i] * atr_mult, c[i] - a[i] * atr_mult
        for j in range(i + 1, i + fwd + 1):
            if h[j] >= up and l[j] <= dn: labels[i] = 0; break
            if h[j] >= up: labels[i] = 1; break
            if l[j] <= dn: labels[i] = -1; break
    df["label"] = labels
    return df

# ===========================================================================
# ML ENGINE
# ===========================================================================
def build_model():
    rf = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=20, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=42)
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
            m = build_model(); m.fit(X[tr_slice], y[tr_slice])
            proba, preds = m.predict_proba(X[va_idx]), m.predict(X[va_idx])
            dmask = y[va_idx] != 0
            if dmask.sum() < 10: continue
            y_dir, p_dir = y[va_idx][dmask], preds[dmask]
            if len(np.unique(y_dir)) >= 2: precs.append(precision_score(y_dir, p_dir, average="weighted", zero_division=0))
            classes = list(m.classes_)
            if 1 in classes and -1 in classes:
                bi = classes.index(1); y_bin = (y_dir == 1).astype(int)
                if len(np.unique(y_bin)) >= 2: aucs.append(roc_auc_score(y_bin, proba[dmask][:, bi]))
        except Exception: continue
    if not aucs: return {"valid": False, "auc_mean": 0.5, "prec_mean": 0.0}
    return {"valid": True, "auc_mean": float(np.mean(aucs)), "prec_mean": float(np.mean(precs)) if precs else 0.0}

def train_model(df_feat: pd.DataFrame) -> Dict[str, Any]:
    df_lab = triple_barrier_labels(df_feat, CONFIG.forward_bars, CONFIG.atr_sl_multiplier)
    train_df = df_lab.iloc[: -CONFIG.forward_bars]
    X, y = train_df[FEATURES].values, train_df["label"].values
    if len(X) < 400 or len(np.unique(y)) < 2: return {"valid": False, "reason": f"insufficient_data"}
    wf = purged_cv(X, y)
    if not wf["valid"] or wf["auc_mean"] < CONFIG.min_oos_auc or wf["prec_mean"] < CONFIG.min_oos_precision:
        return {"valid": False, "reason": f"wf_gate AUC:{wf['auc_mean']:.2f} PREC:{wf['prec_mean']:.2f}"}
    
    weights = np.exp(-np.arange(len(y))[::-1] / 500.0)
    model = build_model(); model.fit(X, y, sample_weight=weights)
    log.info("🎓 Model trained | AUC=%.3f PREC=%.3f n=%d", wf["auc_mean"], wf["prec_mean"], len(X))
    return {"valid": True, "model": model}

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
            if now >= expires:
                exit_price = float(sig_df["close"].iloc[-1]) if len(sig_df) else sig["entry"]
                pnl = ((exit_price - sig["entry"]) if sig["type"] == "BUY" else (sig["entry"] - exit_price)) * sig["lot"] * CONFIG.contract_size
                resolved.append((sig, "timeout", exit_price, pnl))
            else: still_active.append(sig)
        else:
            pnl = (sig["risk_usd"] * CONFIG.risk_reward) if outcome == "win" else (-sig["risk_usd"])
            resolved.append((sig, outcome, exit_price, pnl))

    for sig, outcome, exit_price, pnl in resolved:
        if outcome == "win": state["performance_history"]["wins"] += 1
        elif outcome == "loss": state["performance_history"]["losses"] += 1
        state["daily_pnl_usd"] += pnl
        state["recent_outcomes"].append({"outcome": outcome, "prob": sig.get("prob", 0.0)})
        log.info("📊 Resolved %s → %s (PnL $%.2f)", sig["type"], outcome, pnl)
    state["recent_outcomes"] = state["recent_outcomes"][-50:]
    state["active_signals"] = still_active

    recent = [o for o in state["recent_outcomes"] if o["outcome"] in ("win","loss")]
    if len(recent) >= 10:
        win_rate = sum(1 for o in recent[-20:] if o["outcome"] == "win") / len(recent[-20:])
        new_th = float(np.clip(CONFIG.base_min_signal_probability + (0.55 - win_rate) * 0.5, CONFIG.min_threshold, CONFIG.max_threshold))
        if abs(new_th - state["adaptive_threshold"]) > 0.005:
            state["adaptive_threshold"] = new_th

def compute_trade(sig_type: str, df_feat: pd.DataFrame, prob: float, source: str) -> Dict[str, Any]:
    last = df_feat.iloc[-1]; price, atr = float(last["close"]), float(last["atr"])
    sl_dist, tp_dist = atr * CONFIG.atr_sl_multiplier, atr * CONFIG.atr_sl_multiplier * CONFIG.risk_reward
    entry, sl, tp = (price + 0.15, price - sl_dist, price + tp_dist) if sig_type == "BUY" else (price - 0.15, price + sl_dist, price - tp_dist)
    risk_usd = CONFIG.account_equity * (CONFIG.risk_percent / 100.0)
    lot = max(CONFIG.min_lot, min(round(risk_usd / (abs(entry - sl) * CONFIG.contract_size), 2), CONFIG.max_lot))
    return {
        "type": sig_type, "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
        "lot": lot, "risk_usd": round(risk_usd, 2), "prob": prob, "source": source,
        "time": now_utc().isoformat(), "expires_at": (now_utc() + timedelta(minutes=CONFIG.timeframe_minutes * CONFIG.forward_bars * 2)).isoformat(),
    }

def format_message(t: Dict, source: str) -> str:
    return (
        f"<b>{'🟢' if t['type'] == 'BUY' else '🔴'} XAUUSD {t['type']}</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕒 {t['time'][:16].replace('T',' ')} UTC\n📡 Source: {source}\n🎯 Prob: <b>{t['prob']*100:.1f}%</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 Entry: <b>{t['entry']:.2f}</b>\n🛑 SL: <b>{t['sl']:.2f}</b>\n🎯 TP: <b>{t['tp']:.2f}</b>\n"
        f"⚖️ Lot: <b>{t['lot']}</b> (Risk ${t['risk_usd']:.2f})\n<i>Expires: {t['expires_at'][:16].replace('T',' ')} UTC</i>"
    )

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    log.info("=" * 60)
    log.info("XAUUSD v27.4 — The Ultimate Perfect Edition")
    log.info("=" * 60)
    
    state = load_state()
    if state.get("last_reset_date") != now_utc().strftime("%Y-%m-%d"):
        state["daily_signals_count"], state["daily_pnl_usd"], state["last_reset_date"] = 0, 0.0, now_utc().strftime("%Y-%m-%d")

    if _env_bool(CONFIG.kill_switch_env, False) or now_utc().weekday() >= 5 or (now_utc().weekday() == 4 and now_utc().hour >= 21):
        log.info("📅 Market closed or Kill Switch active."); return

    try:
        df, source = fetch_xauusd_data()
        df = sanitize_df(df)
        if df is None: return
    except Exception as e:
        log.error("❌ Fetch failed: %s", e); return

    resolve_active_signals(state, df)
    save_state(state)

    if state["daily_signals_count"] >= CONFIG.max_daily_signals or state["daily_pnl_usd"] <= -(CONFIG.max_daily_loss_pct * CONFIG.account_equity / 100.0):
        log.info("⛔ Daily limits reached."); return

    if state.get("last_signal_time"):
        if (now_utc() - datetime.fromisoformat(state["last_signal_time"])).total_seconds() / 60 < CONFIG.cooldown_minutes:
            return

    df_feat = calculate_features(df)
    last = df_feat.iloc[-1]

    if float(last["shannon_entropy"]) > CONFIG.max_entropy_threshold or float(last["hurst"]) < CONFIG.min_hurst_threshold:
        log.info("🌊 Choppy market. Skip."); save_state(state); return

    models = train_model(df_feat)
    if not models.get("valid"):
        log.info("⚠️ Invalid model: %s", models.get("reason")); save_state(state); return

    X_live = last[FEATURES].values.reshape(1, -1)
    model = models["model"]
    pred = int(model.predict(X_live)[0])
    proba = model.predict_proba(X_live)[0]
    prob_dict = {int(c): float(proba[i]) for i, c in enumerate(model.classes_)}

    if pred == 0: log.info("ℹ️ Neutral prediction."); return

    sig_type = "BUY" if pred == 1 else "SELL"
    prob = prob_dict.get(pred, 0.0)
    
    # ✅ Threshold sekarang 0.38 (Cocok untuk 3 kelas)
    if prob < float(state["adaptive_threshold"]):
        log.info("📉 Prob %.2f < threshold %.2f. Skip.", prob, state["adaptive_threshold"]); return

    # =======================================================================
    # SMART RULE-BASED FILTER (PENGAMAN TREN & MOMENTUM)
    # =======================================================================
    close_price, ema_trend = float(last["close"]), float(last["ema_trend"])
    adx, rsi = float(last["adx"]), float(last["rsi"])

    if sig_type == "BUY" and close_price < ema_trend:
        log.info("🛡️ Rule Filter: ML=BUY, but Price < EMA50. Skip."); save_state(state); return
    if sig_type == "SELL" and close_price > ema_trend:
        log.info("🛡️ Rule Filter: ML=SELL, but Price > EMA50. Skip."); save_state(state); return
    
    # ADX Filter diturunkan ke 20.0 (Gold sering trending dengan ADX 20-22)
    if adx < 20.0:
        log.info("🛡️ Rule Filter: ADX %.1f < 20 (Sideways). Skip.", adx); save_state(state); return
    
    if sig_type == "BUY" and rsi > 75:
        log.info("🛡️ Rule Filter: RSI %.1f (Overbought). Skip.", rsi); save_state(state); return
    if sig_type == "SELL" and rsi < 25:
        log.info("🛡️ Rule Filter: RSI %.1f (Oversold). Skip.", rsi); save_state(state); return

    log.info("🔥 Signal Confirmed by ML + Rules | %s (Prob %.2f)", sig_type, prob)

    trade = compute_trade(sig_type, df_feat, prob, source)
    if not send_telegram(format_message(trade, source)):
        log.error("❌ Telegram failed."); return

    state["active_signals"].append(trade)
    state["daily_signals_count"] += 1
    state["last_signal_time"] = now_utc().isoformat()
    save_state(state)
    log.info("✅ %s sent | Entry=%.2f Lot=%.2f", sig_type, trade["entry"], trade["lot"])

if __name__ == "__main__":
    try: main()
    except Exception as e: log.critical("💥 Fatal: %s", e, exc_info=True)
