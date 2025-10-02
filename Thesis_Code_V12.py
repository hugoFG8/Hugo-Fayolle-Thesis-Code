#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thesis Qualitative Rebalancer — V12
----------------------------------
Builds monthly adjusted weights from Reddit-derived opinion sentences.

Adds on top of V5:
  • Target Active-Weight Scaler (band 6–10% by default), with safeguards:
      - Only activates if enough "strong" signals this month
      - Respects sector-neutral option and per-name caps; reconciles up to N passes
  • Logs pre/post active weight so you can see impact

Keeps:
  • Historical mode (reads per-sentence from Opinion_Sentences) vs Live mode (V3 + Yahoo fallback)
  • Weighted FinBERT sentence sentiment (+_w = score + 3*comments)
  • BART-MNLI risk score (optional)
  • EMA memory for thin months, N-based shrinkage, z-winsor, per-name cap
  • Output format compatible with Portfolio_Return_Calculator (Qual_Rebalance columns)

Requires: pandas, numpy, openpyxl, transformers, torch, yfinance, praw (only if live mode uses reddit)
"""

import os, re, sys, time, json, warnings, traceback, importlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore", message=r".*asynchronous environment.*", category=UserWarning)

# ========================= USER CONFIG =========================
EXCEL_PATH = "/Users/hugofayolle/Downloads/SP500_Stocks_and_Prices_per_month/SP500_prices_matrix.xlsx"
PORTFOLIO_SHEET   = "Momentum_Portfolio"
OPINION_SHEET     = "Opinion_Sentences"
OUT_SHEET         = "Qual_Rebalance"
CSV_BACKUP        = "/Users/hugofayolle/Downloads/Qual_Rebalance_backup.csv"

# Month window (either order ok; output order can be newest-first)
HISTORICAL_MODE    = True
START_MONTH        = "2025-08"
END_MONTH          = "2010-07"
ORDER_NEWEST_FIRST = True

# Baseline weighting
BASELINE_EQUAL_WEIGHT = True  # equal-weight within monthly momentum picks

# ---------- Impact knobs ----------
# (1) Signal-strength scaling
N0_SHRINK     = 3            # alpha_i = N_i / (N_i + N0)
# (2) Quality-weighted sentiment (historical)
HIST_WEIGHTED_SENTIMENT = True
# (3) Short memory for thin months
EMA_MIN_SENT  = 5
EMA_ALPHA     = 0.80
# (4) Larger bet with cap
TILT_PCT      = 0.25         # max +/- tilt per name
K_FACTOR      = 1.70         # scales z before clipping
Z_WINSOR      = 3.0
MAX_WEIGHT_MULT = 2.0        # adj_weight_i <= 2x base

# --- Polarity/strength gate ---
USE_POLARITY_GATE = True    # concentrate risk on clear views only
POLARITY_ABS_TH   = 0.05    # zero out |adj_eff| below 5%

# ===== t-stat boosters =====
# (B) Engagement-weighted dispersion-aware shrink for sentence aggregation
USE_DISPERSION_SHRINK  = True   # shrinks noisy mean polarity
DISP_TAU               = 0.12   # shrink scale (lower => stronger shrink when noisy)

# --- Tracking-Error (TE) normalization of overlay ---
USE_TE_NORM       = True
TE_ANNUAL_TARGET  = 0.04
COV_LOOKBACK_M    = 24
COV_MIN_OBS       = 8
USE_LEDOWIT_WOLF  = True

# (C) Dynamic coverage guard around polarity gate
USE_DYNAMIC_COVERAGE_GUARD = True
MIN_TILTED_NAMES           = 6
POLARITY_FLOOR             = 0.045
POLARITY_STEP              = 0.005

# ---------- Target Active Weight (band) ----------
ENABLE_TARGET_ACTIVE  = True
TARGET_ACTIVE_MIN     = 0.08   # 8%
TARGET_ACTIVE_MAX     = 0.10   # 10%
# Scaler activation safeguards
STRONG_ABS_TH         = 0.05   # require |adj_eff| >= this to count as "strong"
SCALER_MIN_STRONG     = 4      # min # of strong names OR ...
SCALER_MIN_FRAC       = 0.20   # ... at least this fraction of names strong
SCALER_MAX_PASSES     = 4      # reconcile after caps/sector-neutral (at most this many passes)

# Risk node
USE_RISK_NODE = True
RISK_PENALTY  = 0.30          # adjusted = sentiment - λ*risk

# Opinion labels (info only)
SENT_POS_TH = 0.10
SENT_NEG_TH = -0.10

# --- Optimizer (post-processor) ---
USE_OPTIMIZER          = True
TE_ANNUAL_TARGET_OPT = 0.04     # annual active risk budget for OPT
OPT_CAP_MULT         = 3.0       # per-name cap as multiple of index weight
OPT_TILT_BAND        = 0.5      # if not None, |x - b| <= band (absolute weights)
TURNOVER_PENALTY     = 0.0003   # L1 penalty on |x - x_prev|
OUT_SHEET_OPT     = "Qual_Rebalance_Opt"  # new sheet for optimized weights
OPT_TILT_BAND     = globals().get('TILT_BAND_OPT', 0.5)        # optional band around base; set None to disable

# --- Polisher aggressiveness ---
POLISH_RISK_AVERSION = 0.05   # lambda: weight on variance term
POLISH_STAY_CLOSE    = 1.0    # eta:    weight on ||x-wq||^2
POLISH_TE_FRACTION   = 0.90   # optional: bind TE to 90% of current wq TE if that is tighter

# ---------- Live mode (HISTORICAL_MODE=False) ----------
LIVE_SOURCE = "reddit_v3"  # "reddit_v3" or "yahoo"

# Where your V3 lives (folder with Reddit_Text_Retriever_V3.py) and module name (no .py)
V3_DIR          = "/Users/hugofayolle/Downloads"
V3_MODULE_NAME  = "Reddit_Text_Retriever_V3"
V3_FORCE_RELOAD = True

# Live fetch breadth/strictness
LIVE_SUBREDDITS    = [
    "stocks","investing","StockMarket","wallstreetbets","options","stocksDD",
    "ValueInvesting","GrowthStocks","Dividends","techstocks","AskStocks","FinancialCareers"
]
LIVE_STRICT_OPINION = True
LIVE_MAX_SENTENCES  = 20

# Yahoo fallback (title + summaries)
MAX_NEWS_HEADLINES = 6

# OPTIONAL: sector-neutral tilting
SECTOR_NEUTRAL = True

# Models
FINBERT_MODEL   = "ProsusAI/finbert"
BART_MNLI_MODEL = "facebook/bart-large-mnli"
# ===============================================================

# Make V3 importable in live mode
if V3_DIR and V3_DIR not in sys.path:
    sys.path.append(V3_DIR)

# Transformers (lazy load)
try:
    from transformers import pipeline, AutoTokenizer
except Exception:
    pipeline = None
    AutoTokenizer = None
    
try:
    from sklearn.covariance import LedoitWolf
    _HAVE_LW = True
except Exception:
    _HAVE_LW = False
    
# Optional exact QP solver (fallback provided if missing)
try:
    import cvxpy as cp
    _HAVE_CVXPY = True
except Exception:
    _HAVE_CVXPY = False

# Device selection
try:
    import torch
except Exception:
    torch = None

def _hf_device():
    if torch is None: return -1
    if torch.cuda.is_available(): return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return -1

# ========================= Utilities =========================
def parse_ym(s: str) -> Tuple[int,int]:
    dt = datetime.strptime(s, "%Y-%m"); return dt.year, dt.month

def ym_range(start_ym: str, end_ym: str) -> List[str]:
    y0, m0 = parse_ym(start_ym)
    y1, m1 = parse_ym(end_ym)
    cur = datetime(y0, m0, 1); end = datetime(y1, m1, 1)
    step = 1 if cur <= end else -1
    out = []
    while (step == 1 and cur <= end) or (step == -1 and cur >= end):
        out.append(cur.strftime("%Y-%m"))
        cur += relativedelta(months=step)
    return out

def fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60)
    if h: return f"{h}h {m}m"
    if m: return f"{m}m {s}s"
    return f"{s}s"

def read_momentum_long(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=PORTFOLIO_SHEET, engine="openpyxl")
    if df.columns[0].lower().startswith("unnamed"):
        df = df.drop(columns=[df.columns[0]])
    long = df.melt(var_name="month", value_name="id_like").dropna(subset=["id_like"]).reset_index(drop=True)
    long["month"]  = long["month"].astype(str)
    long["ticker"] = (long["id_like"].astype(str)
                                 .str.strip()
                                 .str.replace(r"\s+[A-Z]{1,3}\s+Equity$", "", regex=True)
                                 .str.upper())
    return long[["month","ticker"]]

def read_opinion_sentences(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=OPINION_SHEET, engine="openpyxl")
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame()
    out["month"]    = df[cols.get("month")].astype(str)
    out["ticker"]   = df[cols.get("ticker")].astype(str).str.upper()
    out["sentence"] = df[cols.get("sentence")].astype(str)
    if "post_score" in cols and "num_comments" in cols:
        out["_w"] = df[cols["post_score"]].fillna(0).astype(float) + 3.0*df[cols["num_comments"]].fillna(0).astype(float)
    else:
        out["_w"] = 0.0
    return out

def build_text_and_sentences(df_sents: pd.DataFrame, month: str, ticker: str, max_sents: int = 60) -> Tuple[str, List[str], List[float]]:
    cur = df_sents[(df_sents["month"] == month) & (df_sents["ticker"] == ticker)]
    if cur.empty: return "", [], []
    cur = cur.sort_values("_w", ascending=False)
    sents = (cur["sentence"].astype(str).tolist())[:max_sents]
    weights = (cur["_w"].astype(float).tolist())[:max_sents]
    if HIST_WEIGHTED_SENTIMENT:
        wsum = float(np.sum(weights))
        if wsum <= 0: weights = [1.0] * len(sents)
        else:         weights = [w/wsum for w in weights]
    else:
        weights = [1.0/len(sents)] * len(sents) if sents else []
    text  = " ".join(sents)
    text  = re.sub(r"\s+", " ", text).strip()
    return text[:6000], sents, weights

# ========================= Live fetchers =========================
def fetch_live_sentences_v3(month: str, ticker: str) -> Tuple[List[str], List[float]]:
    try:
        r3 = importlib.import_module(V3_MODULE_NAME)
        if V3_FORCE_RELOAD: r3 = importlib.reload(r3)
    except Exception as e:
        sys.stderr.write(f"[WARN] Could not import {V3_MODULE_NAME} from {V3_DIR}: {e}\n")
        return [], []
    y, m = parse_ym(month)
    sentences = []
    try:
        if hasattr(r3, "fetch_for_ticker_month"):
            sentences_raw = r3.fetch_for_ticker_month(
                reddit=None, ticker=ticker, year=y, month=m, id_like=ticker,
                subreddits=LIVE_SUBREDDITS, strict_opinion=LIVE_STRICT_OPINION,
                max_sentences=LIVE_MAX_SENTENCES
            )
        elif hasattr(r3, "fetch_reddit_sentences"):
            sentences_raw = r3.fetch_reddit_sentences(
                None, ticker, y, m,
                subreddits=LIVE_SUBREDDITS, strict_opinion=LIVE_STRICT_OPINION,
                max_sentences=LIVE_MAX_SENTENCES
            )
        else:
            raise RuntimeError("V3 missing fetch_for_ticker_month / fetch_reddit_sentences")
        for r in sentences_raw or []:
            if isinstance(r, dict) and "sentence" in r: sentences.append(str(r["sentence"]))
            else: sentences.append(str(r))
    except Exception as e:
        sys.stderr.write(f"[WARN] V3 fetch error for {ticker} {month}: {e}\n")
        return [], []
    if LIVE_STRICT_OPINION:
        cue_rx = re.compile(
            r"\b(i think|i believe|imo|imho|in my opinion|bullish|bearish|overvalued|undervalued|"
            r"should|would|might|looks (cheap|expensive)|seems (cheap|expensive)|"
            r"(strong|weak)\s+(buy|sell)|(buy|sell|hold)\b)", re.IGNORECASE
        )
        newsy_rx = re.compile(
            r"(reports Q[1-4]|press release|downgrade|upgrade|price target|guidance|"
            r"announces|launches|appoints|results|week recap|market update|closing bell)", re.IGNORECASE
        )
        sentences = [s for s in (sentences or []) if isinstance(s, str) and len(s.strip()) >= 8
                     and cue_rx.search(s) and not newsy_rx.search(s)]
    if not sentences: return [], []
    weights = [1.0/len(sentences)] * len(sentences)
    return sentences[:LIVE_MAX_SENTENCES], weights[:LIVE_MAX_SENTENCES]

def fetch_live_text_yahoo(ticker: str) -> str:
    parts = []
    try:
        news = yf.Ticker(ticker).news
        if isinstance(news, list):
            for item in news[:MAX_NEWS_HEADLINES]:
                title = item.get("title") or ""
                summ  = item.get("summary") or ""
                if title: parts.append(title)
                if summ:  parts.append(summ)
    except Exception:
        pass
    blob = " ".join(parts)
    blob = re.sub(r"http\S+", " ", blob)
    blob = re.sub(r"\s+", " ", blob).strip()
    return blob[:6000]

# ========================= Sentiment & Risk =========================
_finbert = None; _bart = None; _tokenizer = None

def get_finbert():
    global _finbert, _tokenizer
    if _finbert is None:
        if pipeline is None: raise RuntimeError("transformers not installed. pip install transformers torch")
        dev = _hf_device()
        _finbert = pipeline("sentiment-analysis", model=FINBERT_MODEL, device=dev)
        try:
            if AutoTokenizer is not None:
                _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL, use_fast=True)
        except Exception:
            _tokenizer = None
        print(f"[FinBERT] device: {dev}")
    return _finbert

def get_bart():
    global _bart
    if _bart is None:
        if pipeline is None: raise RuntimeError("transformers not installed. pip install transformers torch")
        dev = _hf_device()
        _bart = pipeline("zero-shot-classification", model=BART_MNLI_MODEL, device=dev)
        print(f"[BART] device: {dev}")
    return _bart

def finbert_score_sentences(sents: List[str]) -> List[float]:
    if not sents: return []
    fb = get_finbert()
    scores = []
    B = 16
    for i in range(0, len(sents), B):
        batch = [re.sub(r"\s+", " ", s).strip()[:1000] for s in sents[i:i+B]]
        try:
            try:
                out = fb(batch, top_k=None, truncation=True, batch_size=len(batch))
            except TypeError:
                out = fb(batch, return_all_scores=True, truncation=True, batch_size=len(batch))
            for r in out:
                pos = next((d["score"] for d in r if "pos" in d["label"].lower()), 0.0)
                neg = next((d["score"] for d in r if "neg" in d["label"].lower()), 0.0)
                scores.append(float(pos) - float(neg))
        except Exception:
            for s in batch:
                try:
                    one = fb(s, truncation=True)
                    if isinstance(one, list): one = one[0]
                    lab = one["label"].lower(); sc = float(one["score"])
                    if "pos" in lab: scores.append(sc)
                    elif "neg" in lab: scores.append(-sc)
                    else: scores.append(0.0)
                except Exception:
                    scores.append(0.0)
    return scores

def bart_risk(text: str) -> float:
    if not USE_RISK_NODE or not text: return 0.0
    bz = get_bart()
    LABELS  = ["High risk","Medium risk","Low risk","No risk"]
    WEIGHTS = {"High risk":1.0,"Medium":0.5,"Low":0.1,"No risk":0.0,
               "Medium risk":0.5}  # guard for label key
    try:
        out = bz(text[:3000], candidate_labels=LABELS, multi_label=False)
        labels = out.get("labels", []); scores = out.get("scores", [])
        if not labels or not scores: return 0.0
        risk = 0.0
        for lab, sc in zip(labels, scores):
            risk += WEIGHTS.get(lab, 0.0) * float(sc)
        return max(0.0, min(1.0, risk))
    except Exception:
        return 0.0

# ========================= Core helpers =========================
def build_base_weights(tickers: List[str]) -> Dict[str, float]:
    n = len(tickers)
    if n == 0: return {}
    w = 1.0 / n if BASELINE_EQUAL_WEIGHT else 1.0 / n
    return {t: w for t in tickers}

def cap_and_normalize_adj(base_w: Dict[str,float], adj_w: Dict[str,float]) -> Dict[str,float]:
    if MAX_WEIGHT_MULT and MAX_WEIGHT_MULT > 0:
        for t, bw in base_w.items():
            cap = float(MAX_WEIGHT_MULT) * float(bw)
            if t in adj_w: adj_w[t] = min(adj_w[t], cap)
    s = sum(adj_w.values())
    if s > 0:
        for t in adj_w:
            adj_w[t] = adj_w[t] / s
    return adj_w

def sector_map_for(tickers: List[str], cache: Dict[str,str]) -> Dict[str,str]:
    out = {}
    for t in tickers:
        if t in cache: out[t] = cache[t]; continue
        sec = None
        try:
            info = yf.Ticker(t).info or {}
            sec = info.get("sector")
        except Exception:
            sec = None
        cache[t] = sec or "UNKNOWN"; out[t] = cache[t]
    return out

def sector_neutralize(base_w: Dict[str,float], prop_adj: Dict[str,float]) -> Dict[str,float]:
    cache = {}
    sec = sector_map_for(list(base_w.keys()), cache)
    base_sec = {}
    for t, w in base_w.items(): base_sec[sec[t]] = base_sec.get(sec[t], 0.0) + w
    prop_sec = {}
    for t, w in prop_adj.items(): prop_sec[sec[t]] = prop_sec.get(sec[t], 0.0) + w
    adj = {}
    for t, w in prop_adj.items():
        s = sec[t]
        if prop_sec.get(s, 0.0) > 0:
            scale = base_sec.get(s, 0.0) / prop_sec[s]
            adj[t] = w * scale
        else:
            adj[t] = w
    s = sum(adj.values())
    if s > 0:
        for t in adj: adj[t] = adj[t] / s
    return adj

def active_weight(base_w: Dict[str,float], adj_w: Dict[str,float]) -> float:
    keys = base_w.keys()
    return 0.5 * sum(abs(adj_w.get(t,0.0) - base_w.get(t,0.0)) for t in keys)

# ===== Helper: TE-scale or just clip/renorm (guarded) =====
def _te_scale_or_clip(month: str,
                      tickers: List[str],
                      base_w: Dict[str,float],
                      adj_w: Dict[str,float],
                      do_te_scale: bool) -> Dict[str,float]:
    """
    Returns a new adj_w dict:
      - if do_te_scale=True: TE-normalize active weights to target, then cap+renorm
      - else: just cap+renorm (no TE scaling here; leave it for the optimizer)
    """
    import numpy as np

    # Align keys to current tickers for deterministic order
    keys = [t for t in tickers if t in base_w and t in adj_w]
    if not keys:
        return adj_w.copy()

    b = np.array([base_w[t] for t in keys], dtype=float)
    x = np.array([adj_w[t]   for t in keys], dtype=float)

    # Caps
    MAX_WEIGHT_CAP  = float(globals().get("MAX_WEIGHT_CAP", 0.10))
    MAX_WEIGHT_MULT = float(globals().get("MAX_WEIGHT_MULT", 2.0))
    cap_vec = np.minimum(
        np.full_like(b, fill_value=MAX_WEIGHT_CAP),
        MAX_WEIGHT_MULT * np.clip(b, 0.0, None)
    )

    # Clip+renorm BEFORE TE scaling
    x = np.clip(x, 0.0, cap_vec)
    s = float(x.sum())
    if s > 0.0:
        x = x / s

    if do_te_scale:
        # Try to build covariance aligned to 'keys'
        Sigma = None
        try:
            Sigma_src = _build_cov_from_pxret(tickers, month)
            if Sigma_src is not None:
                try:
                    # If DataFrame:
                    Sigma = Sigma_src.loc[keys, keys].to_numpy(dtype=float)
                except Exception:
                    Sigma = np.asarray(Sigma_src, dtype=float)
        except Exception:
            Sigma = None

        if Sigma is not None:
            # Regularize
            eps = 1e-6 * np.trace(Sigma) / Sigma.shape[0]
            Sigma = Sigma + eps * np.eye(Sigma.shape[0])

            # Scale active weights to TE target
            TE_ANNUAL_TARGET = float(globals().get("TE_ANNUAL_TARGET", 0.04))
            w_act = x - b
            te_now = float(np.sqrt(max(0.0, w_act @ Sigma @ w_act)))
            if te_now > 0.0 and TE_ANNUAL_TARGET > 0.0:
                scale = TE_ANNUAL_TARGET / te_now
                x = b + scale * w_act

            # Re-impose caps & renorm
            x = np.clip(x, 0.0, cap_vec)
            s = float(x.sum())
            if s > 0.0:
                x = x / s
        # else: leave x as pre-TE (already clipped/renormed)

    # Write back to dict
    out = adj_w.copy()
    for i, t in enumerate(keys):
        out[t] = float(x[i])
    return out
# ===== end helper =====

# ========================= Month reweighting =========================
def compute_month_weights(month: str,
                          tickers: List[str],
                          text_for_risk: Dict[str,str],
                          sentences: Dict[str,List[str]],
                          weights: Dict[str,List[float]],
                          ema_prev: Dict[str,float],
                          do_te_scale: bool = True) -> Tuple[pd.DataFrame, Dict[str,float]]:
    """
    Build weights for a month:
      - weighted FinBERT per sentence, BART risk
      - adjusted score = sentiment - λ*risk
      - EMA fallback when N < EMA_MIN_SENT
      - z-score tilt, winsor, N-shrink alpha, cap, sector-neutral (optional)
      - NEW: Target Active-Weight Scaler (6–10%) with safeguards
    """
    base_w = build_base_weights(tickers)
    if not tickers:
        return pd.DataFrame(columns=[
            "month","ticker","text_len","n_sent","sentiment","risk_score","adjusted","adj_eff",
            "z","delta","alpha","base_weight","adj_weight","label"
        ]), {}

    sentiments: Dict[str,float] = {}
    risks: Dict[str,float] = {}
    n_sent: Dict[str,int] = {}
    text_len: Dict[str,int] = {}
    adjusted: Dict[str,float] = {}
    adj_eff: Dict[str,float] = {}

    fb = get_finbert()

    for t in tickers:
        sents = sentences.get(t, []) or []
        wts   = weights.get(t, []) or []
        text  = text_for_risk.get(t, "") or ""
        text_len[t] = len(text); n_sent[t] = len(sents)

        # sentiment (weighted per-sentence)
        if sents:
            scores = finbert_score_sentences(sents)
            # normalize / repair weights
            if (len(scores) != len(wts)) or (sum(wts) <= 0):
                w = np.ones(len(scores), dtype=float) / float(len(scores)) if scores else np.array([], dtype=float)
            else:
                w = np.array(wts, dtype=float)
                w = w / (w.sum() if w.sum() > 0 else 1.0)
            s = np.array(scores, dtype=float)

            if s.size:
                mean_pol = float(np.dot(s, w))
                if USE_DISPERSION_SHRINK:
                    # weighted variance & effective N under w
                    var = float(np.dot(w, (s - mean_pol) ** 2))
                    sd  = float(np.sqrt(max(0.0, var)))
                    n_eff = float((w.sum() ** 2) / (np.sum(w ** 2) + 1e-12))
                    u = sd / np.sqrt(max(1.0, n_eff))
                    shrink = 1.0 / (1.0 + u / float(DISP_TAU))
                    sent = mean_pol * float(shrink)
                else:
                    sent = mean_pol
            else:
                sent = 0.0
        else:
            sent = 0.0

        # risk
        risk = bart_risk(text)

        sentiments[t] = sent; risks[t] = risk
        adj_raw = float(sent) - (RISK_PENALTY * float(risk) if USE_RISK_NODE else 0.0)

        # EMA fallback
        if n_sent[t] < EMA_MIN_SENT:
            prev = float(ema_prev.get(t, 0.0))
            adj_used = float(EMA_ALPHA) * adj_raw + float(1.0-EMA_ALPHA) * prev
        else:
            adj_used = adj_raw

        adjusted[t] = adj_raw; adj_eff[t] = adj_used

    # --- Polarity/strength gate (with dynamic coverage option) ---
    if 'USE_POLARITY_GATE' in globals() and globals().get('USE_POLARITY_GATE', False):
        th_init = float(globals().get('POLARITY_ABS_TH', 0.06))
        th = th_init
        if globals().get('USE_DYNAMIC_COVERAGE_GUARD', True):
            mask = {t: abs(adj_eff.get(t, 0.0)) >= th for t in tickers}
            while sum(mask.values()) < globals().get('MIN_TILTED_NAMES', 5) and th > globals().get('POLARITY_FLOOR', 0.045):
                th -= float(globals().get('POLARITY_STEP', 0.005))
                mask = {t: abs(adj_eff.get(t, 0.0)) >= th for t in tickers}
            # zero out below the (possibly relaxed) threshold
            for t in tickers:
                if abs(adj_eff.get(t, 0.0)) < th:
                    adj_eff[t] = 0.0
        else:
            for t in tickers:
                if abs(adj_eff.get(t, 0.0)) < th:
                    adj_eff[t] = 0.0

    # z-score within month
    xs = np.array([adj_eff[t] for t in tickers], dtype=float)
    mu = float(np.mean(xs)) if len(xs) else 0.0
    sd = float(np.std(xs)) if len(xs) else 0.0
    z = np.zeros_like(xs) if sd < 1e-8 else (xs - mu) / sd
    if Z_WINSOR and Z_WINSOR > 0: z = np.clip(z, -Z_WINSOR, +Z_WINSOR)

    # delta before scaling: shrink by N, clip by TILT_PCT after K
    K = float(K_FACTOR) * float(TILT_PCT)
    delta = np.clip(K * z, -TILT_PCT, +TILT_PCT)
    alpha = np.array([ (n_sent[t] / (n_sent[t] + N0_SHRINK)) if n_sent[t] > 0 else 0.0 for t in tickers ], dtype=float)
    delta = alpha * delta

    def build_adj_from_delta(delta_vec: np.ndarray) -> Dict[str,float]:
        # multiplicative on base
        prop = {t: max(0.0, base_w[t] * (1.0 + float(d))) for t, d in zip(tickers, delta_vec)}
        # optional sector neutral
        prop2 = sector_neutralize(base_w, prop) if SECTOR_NEUTRAL else prop
        # cap & normalize
        return cap_and_normalize_adj(base_w, prop2)

    # First pass adj
    adj_w = build_adj_from_delta(delta)
    aw0 = active_weight(base_w, adj_w)

    # ----- NEW: target active-weight scaler -----
    strong_mask = np.array([abs(adj_eff[t]) >= STRONG_ABS_TH for t in tickers], dtype=bool)
    strong_n = int(strong_mask.sum())
    enough_strong = (strong_n >= SCALER_MIN_STRONG) or (strong_n / max(1, len(tickers)) >= SCALER_MIN_FRAC)

    if ENABLE_TARGET_ACTIVE and enough_strong:
        aw = aw0
        passes = 0
        # If above max (rare), scale down once
        if aw > TARGET_ACTIVE_MAX:
            factor = TARGET_ACTIVE_MAX / max(aw, 1e-6)
            delta = np.clip(delta * factor, -TILT_PCT, +TILT_PCT)
            adj_w = build_adj_from_delta(delta); aw = active_weight(base_w, adj_w)
        # If below min, scale up iteratively (respect caps/sector-neutral each pass)
        while (aw < TARGET_ACTIVE_MIN) and (passes < SCALER_MAX_PASSES):
            factor = TARGET_ACTIVE_MIN / max(aw, 1e-6)
            # gentle step so caps don't whipsaw; but allow large if aw is tiny
            factor = min(factor, 3.0)
            delta = np.clip(delta * factor, -TILT_PCT, +TILT_PCT)
            adj_w = build_adj_from_delta(delta)
            aw_new = active_weight(base_w, adj_w)
            passes += 1
            if abs(aw_new - aw) < 1e-4:  # converged / capped out
                aw = aw_new; break
            aw = aw_new
        print(f"  [{month}] target-AW scaler: pre {aw0:.3%} → post {aw:.3%} | strong={strong_n}/{len(tickers)} | passes={passes}")
    else:
        if ENABLE_TARGET_ACTIVE:
            print(f"  [{month}] scaler skipped (not enough strong signals: {strong_n}/{len(tickers)})")

    # ---- TE normalization of overlay (monthly target), BEFORE sector-neutral ----
    if do_te_scale and globals().get('USE_TE_NORM', True):        # Access monthly returns matrix (DataFrame: index=tickers, columns='YYYY-MM')
        px_ret_glob = globals().get('PX_RET', None)

        if isinstance(px_ret_glob, pd.DataFrame) and (month in list(px_ret_glob.columns)):
            # arrays aligned with tickers
            w0 = np.array([base_w[t] for t in tickers], dtype=float)
            w1 = np.array([adj_w.get(t, 0.0) for t in tickers], dtype=float)

            # pick trailing months strictly before `month`
            cols = list(px_ret_glob.columns)
            j = cols.index(month)
            hist_cols = cols[max(0, j - int(globals().get('COV_LOOKBACK_M', 36))): j]

            R = px_ret_glob.reindex(index=tickers)[hist_cols].dropna(how="all", axis=1)
            if R.shape[1] >= int(globals().get('COV_MIN_OBS', 18)):
                X = R.T.to_numpy(dtype=float)
                X = X - np.nanmean(X, axis=0, keepdims=True)
                X = np.nan_to_num(X, nan=0.0)

                use_lw = bool(globals().get('USE_LEDOWIT_WOLF', True)) and globals().get('_HAVE_LW', False)
                if use_lw and X.shape[0] >= X.shape[1] + 2:
                    try:
                        Sigma = LedoitWolf().fit(X).covariance_
                    except Exception:
                        Sigma = np.cov(X, rowvar=False)
                else:
                    Sigma = np.cov(X, rowvar=False)

                dw   = (w1 - w0).reshape(-1, 1)
                num  = float(dw.T @ Sigma @ dw)           # current overlay variance
                te_m = float(globals().get('TE_ANNUAL_TARGET', 0.05)) / np.sqrt(12.0)

                if num > 1e-12:
                    s  = float(np.sqrt((te_m ** 2) / num))
                    w1 = w0 + (dw.flatten() * s)

                    # re-apply caps and renormalize
                    caps = np.maximum(w0 * float(globals().get('MAX_WEIGHT_MULT', 2.0)), 1e-12)
                    w1   = np.minimum(np.maximum(w1, 0.0), caps)
                    w1  /= max(w1.sum(), 1e-12)

                    # write back into adj_w for downstream output
                    for i, t in enumerate(tickers):
                        adj_w[t] = float(w1[i])

    # Build output rows; recompute per-ticker final delta from resulting adj_w
    rows = []
    for idx, t in enumerate(tickers):
        bw = float(base_w.get(t,0.0)); awt = float(adj_w.get(t,0.0))
        final_delta = (awt / bw - 1.0) if bw > 0 else 0.0
        lab = "Neutral"
        if adj_eff[t] >= SENT_POS_TH: lab = "Favorable"
        elif adj_eff[t] <= SENT_NEG_TH: lab = "Unfavorable"
        rows.append({
            "month": month, "ticker": t,
            "text_len": int(text_len[t]), "n_sent": int(n_sent[t]),
            "sentiment": round(float(sentiments[t]), 4),
            "risk_score": round(float(risks[t]), 4),
            "adjusted": round(float(adjusted[t]), 4),
            "adj_eff": round(float(adj_eff[t]), 4),
            "z": round(float(z[idx]), 4),
            "delta": round(float(final_delta), 4),
            "alpha": round(float(alpha[idx]), 4),
            "base_weight": bw, "adj_weight": awt,
            "label": lab
        })

    # EMA state for next month: use adj_eff (signal), not delta
    ema_next = {t: adj_eff[t] for t in tickers}

    df_out = pd.DataFrame(rows)
    # final safety renorm
    ssum = df_out["adj_weight"].sum()
    if ssum > 0: df_out["adj_weight"] = df_out["adj_weight"] / ssum

    return df_out, ema_next

# ========================= I/O =========================
def write_output(excel_path: str, df_add: pd.DataFrame, sheet: str = OUT_SHEET):
    df_add = df_add.sort_values(
        ["month","ticker"],
        ascending=[not ORDER_NEWEST_FIRST, True]
    ).drop_duplicates(["month","ticker"], keep="last")

    try:
        existing = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")
        combo = pd.concat([existing, df_add], ignore_index=True)
        combo = combo.sort_values(
            ["month","ticker"],
            ascending=[not ORDER_NEWEST_FIRST, True]
        ).drop_duplicates(["month","ticker"], keep="last")
    except Exception:
        combo = df_add.copy()

    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        combo.to_excel(w, index=False, sheet_name=sheet)

    # CSV backup
    try:
        parent = os.path.dirname(CSV_BACKUP)
        if parent: os.makedirs(parent, exist_ok=True)
        combo.to_csv(CSV_BACKUP, index=False)
    except Exception:
        pass
    
# =========== Optimizer Helpers ===========

def _build_cov_from_pxret(tickers, month, lookback_m=36, min_obs=18, use_lw=True):
    """Trailing covariance for these tickers before `month`, from PX_RET (monthly returns)."""
    px = globals().get('PX_RET', None)
    if px is None or not isinstance(px, pd.DataFrame) or month not in list(px.columns):
        return None

    cols = list(px.columns)
    if month not in cols:
        return None
    j = cols.index(month)
    hist_cols = cols[max(0, j - int(lookback_m)): j]   # strictly before `month`
    
    # just in case: ensure PX_RET index has no duplicates
    if px.index.duplicated().any():
        px = px[~px.index.duplicated(keep="first")]

    # --- align to requested tickers (normalize like we did for PX_RET creation) ---
    tickers_norm = [str(t).strip().upper() for t in tickers]
    R = px.reindex(index=tickers_norm)[hist_cols]

    # quick diag
    nonnan_by_row = R.notna().sum(axis=1)
    nonnan_by_col = R.notna().sum(axis=0)

    # require at least some data in the window
    if R.shape[1] < int(min_obs) or (nonnan_by_row >= 1).sum() < 2:
        print(f"[{month}] COV skip: window cols={R.shape[1]} (need>={int(min_obs)}), "
              f"names_with_data={(nonnan_by_row>=1).sum()} (need>=2).")
        return None

    # ---- de-mean by column on available data ----
    X = R.T.to_numpy(dtype=float)                 # shape: months x names
    # de-mean each column using available values only
    col_means = np.nanmean(X, axis=0, keepdims=True)  # shape 1 x names
    X = X - col_means
    # replace remaining NaNs (missing months for a name) by 0 so they don't contribute
    X = np.nan_to_num(X, nan=0.0)

    # if LedoitWolf is available and months >= names+2, use it, else sample cov
    use_lw = bool(use_lw) and globals().get('_HAVE_LW', False)
    try:
        if use_lw and X.shape[0] >= X.shape[1] + 2:
            Sigma = LedoitWolf().fit(X).covariance_
        else:
            Sigma = np.cov(X, rowvar=False)       # names x names
    except Exception:
        Sigma = np.cov(X, rowvar=False)

    # validate Sigma shape and dimensionality
    if Sigma is None:
        return None
    Sigma = np.asarray(Sigma)
    if Sigma.ndim != 2 or min(Sigma.shape) < 2:
        print(f"[{month}] COV skip: Sigma shape invalid {Sigma.shape}.")
        return None

    # tiny ridge for stability
    k = Sigma.shape[0]
    Sigma += 1e-8 * (np.trace(Sigma) / max(k, 1)) * np.eye(k)

    # optional quick condition-number diag (safe now)
    try:
        w = np.linalg.eigvalsh(Sigma)
        cond = float((w.max() + 1e-12) / (w.min() + 1e-12))
        print(f"[{month}] COV ok: names={k}, months={len(hist_cols)}, cond≈{cond:.2e}, "
              f"rows≥1obs={(nonnan_by_row>=1).sum()}, cols≥1obs={(nonnan_by_col>=1).sum()}")
    except Exception:
        pass

    return Sigma

def _project_to_capped_simplex(u, caps, z=1.0):
    """
    Project vector u to { x : 0 <= x <= caps, sum x = z } via water-filling with upper bounds.
    """
    caps = np.asarray(caps, float)
    lo = np.zeros_like(caps)
    # bisection on tau s.t. sum clip(u - tau, 0, caps) = z
    def g(tau):
        return np.clip(u - tau, 0.0, caps).sum() - z
    # bounds for tau
    tau_lo, tau_hi = u.max() - caps.max(), u.max()  # coarse
    # expand if needed
    for _ in range(40):
        if g(tau_lo) >= 0: break
        tau_lo -= max(1.0, abs(tau_lo))
    for _ in range(40):
        if g(tau_hi) <= 0: break
        tau_hi += max(1.0, abs(tau_hi))
    # bisection
    for _ in range(80):
        tau_mid = 0.5*(tau_lo + tau_hi)
        val = g(tau_mid)
        if abs(val) < 1e-10: break
        if val > 0: tau_lo = tau_mid
        else:       tau_hi = tau_mid
    x = np.clip(u - tau_mid, 0.0, caps)
    s = x.sum()
    if s <= 0:  # fallback
        x = caps.copy()
        tot = x.sum()
        x = x / tot if tot>0 else x
        return x
    return x / s * z  # renorm exact

def _optimize_month_cvx(mu, Sigma, b, x_prev, te_month, cap_mult=None, tilt_band=None, gamma=0.0003):
    """Exact convex QP with cvxpy. All vectors are numpy arrays aligned to tickers."""
    n = len(b)
    x = cp.Variable(n)
    cons = [cp.sum(x) == 1, x >= 0]
    # TE constraint vs baseline
    dw = x - b
    cons += [cp.quad_form(dw, Sigma) <= float(te_month**2)]
    # caps
    if cap_mult is None: cap_mult = 2.0
    caps = np.minimum(cap_mult*b, 1.0)
    cons += [x <= caps]
    # optional band
    if tilt_band is not None and tilt_band > 0:
        band = float(tilt_band) * b
        cons += [dw <= band, -dw <= band]
    # objective: maximize mu^T x - gamma * ||x - x_prev||_1
    obj = cp.Maximize(mu @ x - float(gamma) * cp.norm1(x - x_prev))
    try:
        cp.Problem(obj, cons).solve(solver=cp.OSQP, verbose=False)
        if x.value is None:
            raise RuntimeError("cvxpy solver failed")
        val = np.array(x.value, dtype=float)
        val = np.clip(val, 0.0, caps)
        s = val.sum()
        return val/s if s>0 else b
    except Exception:
        return None

def _optimize_month_fallback(mu, Sigma, b, x_prev, te_month, cap_mult=None, tilt_band=None, gamma=0.0003):
    """
    Closed-form style overlay: v ~ Sigma^{-1} mu, TE-scale, cap, project to simplex with caps,
    then blend a little toward x_prev to control turnover.
    """
    n = len(b)
    if cap_mult is None: cap_mult = 2.0
    caps = np.minimum(cap_mult*b, 1.0)
    # direction
    try:
        v = np.linalg.solve(Sigma, mu)
    except Exception:
        v = np.linalg.pinv(Sigma) @ mu
    # center around baseline via overlay
    x_raw = b + v
    x_raw = np.clip(x_raw, 0.0, caps)
    x     = _project_to_capped_simplex(x_raw, caps, z=1.0)
    # TE scale
    dw = (x - b).reshape(-1,1)
    num = float((dw.T @ Sigma @ dw).item())
    if num > 1e-12:
        s = float(np.sqrt((te_month**2)/num))
        x = b + s * (x - b)
        x = np.clip(x, 0.0, caps)
        x = _project_to_capped_simplex(x, caps, z=1.0)
    # turnover blend
    if gamma and gamma > 0 and x_prev is not None and len(x_prev)==n:
        rho = min(0.5, float(gamma)*1000.0)  # simple mapping; small blend
        x = (1.0 - rho)*x + rho*np.asarray(x_prev, float)
        x = np.clip(x, 0.0, caps)
        x = _project_to_capped_simplex(x, caps, z=1.0)
    return x

def _polish_month_cvx(Sigma, b, wq, x_prev, te_month, cap_mult=2.0, tilt_band=None, gamma=0.0003):
    import numpy as np, cvxpy as cp
    n = len(b)
    caps = np.minimum(cap_mult * b, 1.0)
    x = cp.Variable(n)

    cons = [cp.sum(x) == 1, x >= 0, x <= caps]
    dw = x - b
    if te_month >= 0:
        cons += [cp.quad_form(dw, Sigma) <= float(te_month**2)]
    if tilt_band is not None and tilt_band > 0:
        band = float(tilt_band) * b
        cons += [dw <= band, -dw <= band]

    lam = float(globals().get('POLISH_RISK_AVERSION', 0.05))
    eta = float(globals().get('POLISH_STAY_CLOSE', 1.0))
    obj = cp.Minimize(lam * cp.quad_form(dw, Sigma) + eta * cp.sum_squares(x - wq) + float(gamma) * cp.norm1(x - x_prev))

    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        val = np.array(x.value, dtype=float) if x.value is not None else None
        if val is None: return None
        val = np.clip(val, 0.0, caps)
        s = val.sum()
        return val/s if s > 0 else b
    except Exception:
        return None
    
def _polish_month_fallback(Sigma, b, wq, x_prev, te_month, cap_mult=2.0, tilt_band=None, gamma=0.0003):
    import numpy as np
    n = len(b)
    caps = np.minimum(cap_mult * b, 1.0)
    x = np.minimum(np.maximum(wq.copy(), 0.0), caps)
    s = x.sum(); x = x/s if s>0 else b.copy()

    if tilt_band is not None and tilt_band > 0:
        band = float(tilt_band) * b
        lo = np.maximum(0.0, b - band); hi = np.minimum(caps, b + band)
        x = np.minimum(np.maximum(x, lo), hi); x /= x.sum()

    # If TE not binding, apply a tiny risk-shrink toward baseline using Sigma
    dw = (x - b).reshape(-1,1)
    num = float((dw.T @ Sigma @ dw).item())
    if num > 1e-12 and te_month >= 0:
        s = float(np.sqrt((te_month**2) / num))
        s = min(1.0, s)
        x = b + s * (x - b)

    # re-cap + project
    x = np.minimum(np.maximum(x, 0.0), caps)
    x = _project_to_capped_simplex(x, caps, z=1.0)

    if gamma and gamma > 0 and x_prev is not None and len(x_prev)==n:
        rho = min(0.15, float(gamma)*1000.0)
        x = (1.0 - rho)*x + rho*np.asarray(x_prev, float)
        x = np.minimum(np.maximum(x, 0.0), caps)
        x = _project_to_capped_simplex(x, caps, z=1.0)
    return x

def optimize_month_weights_from_df(month, df_m, x_prev_dict=None):
    """
    Polisher: keep the optimizer close to the pre-optimized adj weights (wq),
    only fixing them to respect TE, caps, bands, and turnover.
    """
    import numpy as np

    tickers = df_m['ticker'].tolist()
    b  = df_m['base_weight'].to_numpy(float)
    wq = df_m['adj_weight'].to_numpy(float)

    # Covariance from PX_RET (as you already built it)
    Sigma = _build_cov_from_pxret(
        tickers, month,
        lookback_m=int(globals().get('COV_LOOKBACK_M', 36)),
        min_obs=int(globals().get('COV_MIN_OBS', 18)),
        use_lw=bool(globals().get('USE_LEDOWIT_WOLF', True))
    )
    if Sigma is None or np.ndim(Sigma) != 2 or min(Sigma.shape) < 2:
        shp = None if Sigma is None else tuple(np.shape(Sigma))
        print(f"[{month}] Sigma invalid for OPT; shape={shp}. Skipping optimizer for this month.")
        return df_m.copy(), {t: w for t, w in zip(tickers, wq)}

    # Small ridge
    eps = 1e-6 * np.trace(Sigma) / Sigma.shape[0]
    Sigma = Sigma + eps * np.eye(Sigma.shape[0])

    # Align x_prev to this ticker set
    if x_prev_dict:
        x_prev = np.array([x_prev_dict.get(t, wq[i]) for i, t in enumerate(tickers)], float)
    else:
        x_prev = wq.copy()

    # --- Polisher settings ---
    # Use the *monthly* TE bound derived from your monthly scaling target.
    # For polishing, we bind TE vs the baseline to your TE_ANNUAL_TARGET (not the larger OPT one).
    # base monthly TE from the stricter of the two knobs
    te_ann_cfg = float(min(
        globals().get('TE_ANNUAL_TARGET_OPT', 0.05),
        globals().get('TE_ANNUAL_TARGET',     0.05)
    ))
    te_m_cfg   = te_ann_cfg / np.sqrt(12.0)

    # current TE of your pre-optimized weights vs baseline
    te_pre = float(((wq - b) @ Sigma @ (wq - b)) ** 0.5)
    
    # optionally tighten: e.g., 0.90 means “cap at 90% of current TE”
    frac = float(globals().get('POLISH_TE_FRACTION', 1.0))
    te_m = min(te_m_cfg, frac * te_pre)

    cap_m  = float(globals().get('OPT_CAP_MULT', 3.0))
    band   = globals().get('OPT_TILT_BAND', 0.5)
    gamma  = float(globals().get('TURNOVER_PENALTY', 0.0003))  # a bit stronger than overlay

    # --- Solve polish problem: "closest x to wq" under constraints ---
    if globals().get('_HAVE_CVXPY', False):
        x = _polish_month_cvx(Sigma, b, wq, x_prev, te_m, cap_mult=cap_m, tilt_band=band, gamma=gamma)
        if x is None:
            x = _polish_month_fallback(Sigma, b, wq, x_prev, te_m, cap_mult=cap_m, tilt_band=band, gamma=gamma)
    else:
        x = _polish_month_fallback(Sigma, b, wq, x_prev, te_m, cap_mult=cap_m, tilt_band=band, gamma=gamma)

    # Diagnostics: how far did we move?
    try:
        te_base = float(np.sqrt((wq - b) @ Sigma @ (wq - b)))
        te_opt  = float(np.sqrt((x  - b) @ Sigma @ (x  - b)))
        l1      = float(np.abs(x - wq).sum())
        print(f"[{month}] POLISH: TE_pre={te_base:.4%}, TE_post={te_opt:.4%}, L1|x-wq|={l1:.6f}")
    except Exception:
        pass

    df_opt = df_m.copy()
    df_opt['adj_weight'] = x
    x_prev_out = {t: float(xi) for t, xi in zip(tickers, x)}
    return df_opt, x_prev_out

# ========================= Main =========================
def main():
    try:
        df_momo = read_momentum_long(EXCEL_PATH)
    except Exception as e:
        sys.stderr.write(f"[FATAL] Cannot read '{PORTFOLIO_SHEET}': {e}\n"); sys.exit(1)

    df_ops = None
    if HISTORICAL_MODE:
        try:
            df_ops = read_opinion_sentences(EXCEL_PATH)
        except Exception as e:
            sys.stderr.write(f"[FATAL] Cannot read '{OPINION_SHEET}': {e}\n"); sys.exit(1)

    # --- Build monthly returns matrix for TE normalization (PX_RET) ---
    try:
        # Read wide price matrix (rows=tickers, cols='YYYY-MM')
        px_wide = pd.read_excel(EXCEL_PATH, sheet_name="S&P500", engine="openpyxl")
        
        # drop any left-over index column
        if px_wide.columns[0].lower().startswith("unnamed"):
            px_wide = px_wide.drop(columns=[px_wide.columns[0]])

        # ensure we have a Ticker index
        ticker_col = "Ticker" if "Ticker" in px_wide.columns else px_wide.columns[0]
        px_wide = px_wide.rename(columns={ticker_col: "Ticker"}).set_index("Ticker")

        # normalize tickers to *bare symbols* to match Momentum_Portfolio cleaning
        px_wide.index = (px_wide.index.astype(str)
                             .str.strip()
                             .str.replace(r"\s+[A-Z]{1,3}\s+Equity$", "", regex=True)
                             .str.upper())
        
        # collapse duplicate tickers created by the normalization (keep first occurrence)
        if px_wide.index.duplicated().any():
            dups = px_wide.index[px_wide.index.duplicated()].unique().tolist()
            print(f"[PX_RET] dropping {len(dups)} duplicate tickers after normalization; e.g. {dups[:5]}")
            px_wide = px_wide[~px_wide.index.duplicated(keep="first")]
        
        # sanitize: coerce common placeholders to NaN BEFORE numeric conversion
        px_wide = px_wide.replace(["--", "-", "N/A", "NA", "", " "], np.nan)
        
        # keep only YYYY-MM columns (no regex to avoid extra imports)
        def is_ym(col: str) -> bool:
            return isinstance(col, str) and len(col) == 7 and col[4] == "-" \
                and col[:4].isdigit() and col[5:7].isdigit()
                
        month_cols = [c for c in px_wide.columns if is_ym(c)]
        month_cols = sorted(month_cols)  # ascending chronology
        
        # numeric coercion column-wise; any leftover non-numeric -> NaN
        px = px_wide[month_cols].apply(pd.to_numeric, errors="coerce")
        
        # drop months that are entirely NaN (should be rare)
        px = px.dropna(how="all", axis=1)

        # compute monthly returns across time (columns)
        px = px.sort_index(axis=1)            # ensure chronological order
        ret = px.T.pct_change().T             # monthly returns
        ret = ret.dropna(how="all", axis=0)   # drop tickers with no returns at all
        ret = ret.dropna(how="all", axis=1)   # drop months with no returns at all

        globals()['PX_RET'] = ret
        print(f"[PX_RET] built: tickers={ret.shape[0]}, months={ret.shape[1]}, "
              f"missing_cells={int(np.isnan(ret.to_numpy()).sum())}")
    except Exception as e:
        globals()['PX_RET'] = None
        print(f"[WARN] Could not build PX_RET for TE normalization: {e}")
        
    months_out = ym_range(START_MONTH, END_MONTH)
    m0, m1 = (START_MONTH, END_MONTH)
    months_chrono = ym_range(min(m0, m1), max(m0, m1))

    total_tickers = 0
    for ms in months_out:
        total_tickers += len(df_momo[df_momo["month"] == ms]["ticker"].dropna().astype(str).str.upper().unique())
    progress = {"done": 0, "total": total_tickers, "t0": time.time()}

    print(f"Qual Rebalance V6 | HIST={HISTORICAL_MODE} | months_out={months_out[0]}…{months_out[-1]} | total tickers={total_tickers}")

    month_outputs: Dict[str, pd.DataFrame] = {}
    ema_state: Dict[str, float] = {}
    # --- optimizer state (carry last month’s optimized weights) ---
    opt_prev: Dict[str, float] = {}
    for ms in months_chrono:
        tickers = (df_momo[df_momo["month"] == ms]["ticker"]
                   .dropna().astype(str).str.upper().unique().tolist())

        if not tickers:
            month_outputs[ms] = pd.DataFrame(columns=[
                "month","ticker","text_len","n_sent","sentiment","risk_score","adjusted","adj_eff",
                "z","delta","alpha","base_weight","adj_weight","label"
            ])
            continue

        text_for_risk: Dict[str,str] = {}
        sentences: Dict[str,List[str]] = {}
        weights: Dict[str,List[float]] = {}

        if HISTORICAL_MODE:
            for t in tickers:
                txt, sents, wts = build_text_and_sentences(df_ops, ms, t)
                text_for_risk[t] = txt; sentences[t] = sents; weights[t] = wts
        else:
            for t in tickers:
                sents, wts = fetch_live_sentences_v3(ms, t) if LIVE_SOURCE == "reddit_v3" else ([], [])
                if not sents:
                    text_for_risk[t] = fetch_live_text_yahoo(t)
                else:
                    text_for_risk[t] = " ".join(sents)[:6000]
                sentences[t] = sents
                weights[t]   = wts if (wts and abs(sum(wts)-1.0) < 1e-6) else ([1.0/len(sents)]*len(sents) if sents else [])

        df_out_m, ema_next = compute_month_weights(
            ms, tickers, text_for_risk, sentences, weights, ema_state,
            do_te_scale = (not USE_OPTIMIZER)
            )
        month_outputs[ms] = df_out_m  # --- optimizer post-processor (risk-aware TE-sized polish) ---
        if USE_OPTIMIZER:
            try:
                df_opt_m, opt_prev = optimize_month_weights_from_df(ms, df_out_m, x_prev_dict=opt_prev)
                # movement diagnostics: compare OPT vs BASE-ADJ
                try:
                    x_base = df_out_m["adj_weight"].to_numpy(float)
                    x_opt  = df_opt_m["adj_weight"].to_numpy(float)
                    diff   = np.abs(x_opt - x_base)
                    moved  = int((diff > 1e-8).sum())
                    print(f"[{ms}] OPT movement: max|Δw|={diff.max():.5f}, names_changed={moved}")
                except Exception:
                    pass
                month_outputs[f"{ms}__OPT"] = df_opt_m
            except Exception as e:
                print(f"[{ms}] OPT skip: {e}")
        ema_state.update(ema_next)
        progress["done"] += len(tickers)
        avg = (time.time() - progress["t0"]) / max(1, progress["done"])
        eta = avg * max(0, progress["total"] - progress["done"])
        print(f"=== {ms} | tickers: {len(tickers)} | avg {avg:.1f}s | ETA {fmt_eta(eta)}")

    for ms in months_out:
        df_out_m = month_outputs.get(ms, pd.DataFrame())
        if not df_out_m.empty:
            write_output(EXCEL_PATH, df_out_m, sheet=OUT_SHEET)
            print(f"[{ms}] wrote {len(df_out_m)} rows to '{OUT_SHEET}'")
            # also write optimized sheet (same schema; only adj_weight differs)
            if USE_OPTIMIZER:
                df_opt_m = month_outputs.get(f"{ms}__OPT", pd.DataFrame())
                if not df_opt_m.empty:
                    write_output(EXCEL_PATH, df_opt_m, sheet=OUT_SHEET_OPT)
                    print(f"[{ms}] wrote {len(df_opt_m)} rows to '{OUT_SHEET_OPT}'")
        else:
            print(f"[{ms}] no rows produced.")

    elapsed = time.time() - progress["t0"]
    print(f"\nDone. Elapsed: {fmt_eta(elapsed)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write("FATAL ERROR:\n" + "".join(traceback.format_exception(e)) + "\n")
        sys.exit(1)