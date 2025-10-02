#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from pathlib import Path

# ===================== CONFIG =====================
EXCEL_PATH = "/Users/hugofayolle/Downloads/SP500_Stocks_and_Prices_per_month/SP500_prices_matrix.xlsx"
PRICES_SHEET = "S&P500"
MOMO_SHEET   = "Momentum_Portfolio"     # fallback if a month missing in Qual_Rebalance
QUAL_SHEET   = "Qual_Rebalance_Opt"

OUT_RETURNS  = "Backtest_Returns_Opt"
OUT_SUMMARY  = "Backtest_Summary_Opt"
OUT_DIAG     = "Backtest_Diagnostics_Opt"
OUT_DIAG_SUM = "Backtest_Diagnostics_Summary_Op"

SHIFT_WEIGHTS = 1          # fixed: avoid look-ahead
USE_COSTS     = False      # set True to subtract simple turnover costs
COST_BPS      = 10         # per-month cost if USE_COSTS=True
EPS_COV       = 0.0005     # 5 bps — counts as “tilted” name

# >>> NEW: limit the backtest window (inclusive). Set to None to use all.
START_MONTH = "2015-01"    # e.g. "2015-01", or None
END_MONTH   = "2025-08"    # e.g. "2025-08", or None
# ==================================================

# ---------- helpers ----------
def first_token(bbg: str) -> str:
    return str(bbg).split()[0].upper().strip()

def ann_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty: return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan}
    idx_dt = pd.to_datetime(r.index + "-01")
    yrs = (idx_dt[-1] - idx_dt[0]).days / 365.25
    cagr = (1 + r).prod()**(1/yrs) - 1 if yrs > 0 else np.nan
    vol  = r.std() * np.sqrt(12)
    sharpe = (r.mean()*12) / vol if vol > 0 else np.nan
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe}

def cum_nav(r: pd.Series) -> pd.Series:
    return (1 + r.fillna(0)).cumprod()

def simple_turnover(w: pd.DataFrame) -> pd.Series:
    return 0.5 * w.fillna(0).diff().abs().sum(axis=1).fillna(0)

def add_one_month(ym: str) -> str:
    y, m = map(int, ym.split("-"))
    m += 1
    if m == 13: y, m = y+1, 1
    return f"{y:04d}-{m:02d}"

def window_mask(index_ym: pd.Index, start: str | None, end: str | None) -> pd.Series:
    """Return boolean mask for inclusive [min(start,end), max(start,end)] on 'YYYY-MM' string index."""
    if start is None and end is None:
        return pd.Series(True, index=index_ym)
    if start is None:
        lo, hi = index_ym.min(), str(end)
    elif end is None:
        lo, hi = str(start), index_ym.max()
    else:
        lo, hi = sorted([str(start), str(end)])
    idx = index_ym.astype(str).str.slice(0, 7)
    return (idx >= lo) & (idx <= hi)

def info_ratio_and_tstat(x: pd.Series):
    x = x.dropna()
    if x.empty: return np.nan, np.nan
    ir = x.mean() / x.std() if x.std() > 0 else np.nan
    tstat = ir * np.sqrt(12)
    return ir, tstat

# ---------- 1) Prices → monthly returns ----------
px_wide = pd.read_excel(EXCEL_PATH, sheet_name=PRICES_SHEET)

if "Ticker" not in px_wide.columns:
    raise ValueError(f"'{PRICES_SHEET}' must have a 'Ticker' column.")

month_cols = sorted([c for c in px_wide.columns if re.fullmatch(r"\d{4}-\d{2}", str(c))])
if not month_cols:
    raise ValueError("No YYYY-MM columns found in price sheet.")

px = px_wide.set_index("Ticker")[month_cols].apply(pd.to_numeric, errors="coerce")

# returns across time (columns); silence pct_change fill warning by fill_method=None
ret = px.pct_change(axis=1, fill_method=None).T.dropna(how="all")
ret.index.name = "month"
ret.columns.name = "Ticker"

# ---------- 2) Read Qual_Rebalance (month,ticker,base_weight,adj_weight) ----------
q = pd.read_excel(EXCEL_PATH, sheet_name=QUAL_SHEET)
cols = {c.lower(): c for c in q.columns}

req = {"month", "ticker", "base_weight", "adj_weight"}
if not req.issubset(set(cols.keys())):
    raise ValueError(
        f"'{QUAL_SHEET}' must have columns: {sorted(req)} (case-insensitive). "
        f"Found: {list(q.columns)}"
    )

month_norm = (
    q[cols["month"]]
    .astype(str)
    .str.strip()
    .str.slice(0, 7)
)

q_norm = pd.DataFrame({
    "month":       month_norm,
    "symbol":      q[cols["ticker"]].astype(str).str.upper().str.strip(),
    "base_weight": pd.to_numeric(q[cols["base_weight"]], errors="coerce"),
    "adj_weight":  pd.to_numeric(q[cols["adj_weight"]],  errors="coerce"),
}).dropna(subset=["month", "symbol"])

# Keep only months that exist in the returns panel
q_norm = q_norm[q_norm["month"].isin(ret.index)]

# ---------- 3) Map symbols → BBG tickers in price panel ----------
symbols_to_bbg = {}
for bbg in ret.columns:
    sym = first_token(bbg)
    symbols_to_bbg.setdefault(sym, []).append(bbg)

def expand_weights(df_month: pd.DataFrame, weight_col: str) -> pd.Series:
    if df_month.empty:
        return pd.Series(dtype=float, name=None)
    s = df_month[weight_col].sum()
    if s and abs(s - 1.0) > 1e-6:
        df_month = df_month.assign(**{weight_col: df_month[weight_col] / s})
    w_map = {}
    for sym, w in zip(df_month["symbol"], df_month[weight_col]):
        bbg_list = symbols_to_bbg.get(sym, [])
        if not bbg_list:
            continue
        split = float(w) / len(bbg_list)
        for b in bbg_list:
            w_map[b] = w_map.get(b, 0.0) + split
    return pd.Series(w_map, name=df_month["month"].iloc[0])

# Build base/adj from Qual_Rebalance
base_rows, adj_rows = [], []
for m, g in q_norm.groupby("month"):
    base_rows.append(expand_weights(g, "base_weight").rename(m))
    adj_rows.append(expand_weights(g, "adj_weight").rename(m))

base_w = pd.DataFrame(base_rows).fillna(0.0)
adj_w  = pd.DataFrame(adj_rows).fillna(0.0)
base_w.index.name = adj_w.index.name = "month"

# ---------- 4) Fallback for missing months using Momentum_Portfolio ----------
momo = pd.read_excel(EXCEL_PATH, sheet_name=MOMO_SHEET)
if momo.columns[0].lower().startswith("unnamed"):
    momo = momo.drop(columns=[momo.columns[0]])
momo_months = [c for c in momo.columns if re.fullmatch(r"\d{4}-\d{2}", str(c))]

base_momo_rows = []
for m in momo_months:
    picks = momo[m].dropna().astype(str).tolist()
    if picks:
        w = pd.Series(1.0/len(picks), index=picks, name=m)
        base_momo_rows.append(w)

base_momo = pd.DataFrame(base_momo_rows).fillna(0.0)
base_momo.index.name = "month"

# Align fallback and Qual frames to ret
base_momo = base_momo.reindex(index=ret.index).fillna(0.0)
base_momo = base_momo.reindex(columns=ret.columns, fill_value=0.0)

base_w = base_w.reindex(index=ret.index).fillna(0.0).reindex(columns=ret.columns, fill_value=0.0)
adj_w  = adj_w.reindex(index=ret.index).fillna(0.0).reindex(columns=ret.columns, fill_value=0.0)

# Fill missing base with momentum equal-weight; missing adj with base
row_sum_base_q = base_w.sum(axis=1)
base_w.loc[row_sum_base_q == 0] = base_momo.loc[row_sum_base_q == 0]
row_sum_adj_q = adj_w.sum(axis=1)
adj_w.loc[row_sum_adj_q == 0] = base_w.loc[row_sum_adj_q == 0]

# Normalize rows (safety) before shifting
base_w = base_w.div(base_w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
adj_w  = adj_w.div(adj_w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

# ---------- 5) Shift weights; trim to first tradeable month ----------
base_w = base_w.shift(SHIFT_WEIGHTS)
adj_w  = adj_w.shift(SHIFT_WEIGHTS)

# First tradeable month = earliest weight month + 1
earliest_w_months = []
if not q_norm.empty:
    earliest_w_months.append(q_norm["month"].min())
if momo_months:
    earliest_w_months.append(min(momo_months))
if earliest_w_months:
    first_trade_month = add_one_month(min(earliest_w_months))
    mask_trim = ret.index >= first_trade_month
    ret    = ret.loc[mask_trim]
    base_w = base_w.loc[mask_trim]
    adj_w  = adj_w.loc[mask_trim]

# ---------- 6) Apply user date window (inclusive; reversed allowed) ----------
mask_win = window_mask(ret.index, START_MONTH, END_MONTH)
ret    = ret.loc[mask_win]
base_w = base_w.loc[mask_win]
adj_w  = adj_w.loc[mask_win]

if ret.empty:
    raise ValueError("Selected date window contains no months after shift/trim. "
                     f"Available: {first_trade_month if earliest_w_months else ret.index.min()} → "
                     f"{px.columns[-1]}")

print(f"Backtest window: {ret.index.min()} → {ret.index.max()}  (months: {len(ret)})")

# ---------- 7) Portfolio returns ----------
def port_ret(w: pd.DataFrame, r: pd.DataFrame) -> pd.Series:
    row_sums = w.sum(axis=1)
    pr = (w * r).sum(axis=1)
    pr[row_sums == 0] = np.nan
    return pr

base_ret = port_ret(base_w, ret)
qual_ret = port_ret(adj_w,  ret)

if USE_COSTS:
    base_ret = base_ret - simple_turnover(base_w) * (COST_BPS / 10_000)
    qual_ret = qual_ret - simple_turnover(adj_w)  * (COST_BPS / 10_000)

nav_base = cum_nav(base_ret)
nav_qual = cum_nav(qual_ret)

summary = pd.DataFrame([ann_stats(base_ret), ann_stats(qual_ret)], index=["Base","Qual"]).round(4)

# ---------- 8) Diagnostics ----------
diff_bbg = (adj_w - base_w).abs()
coverage_bbg = (diff_bbg > EPS_COV).sum(axis=1) / diff_bbg.shape[1]
active_bbg   = 0.5 * diff_bbg.sum(axis=1)
turnover_bbg = 0.5 * adj_w.diff().abs().sum(axis=1)

excess_ret = (qual_ret - base_ret)
hit = (excess_ret > 0).astype(float)

# Optional symbol-level diagnostics (pre-shift), built from Qual_Rebalance (unwindowed source ok)
if not q_norm.empty:
    q_in_win = q_norm[q_norm["month"].isin(ret.index)]
    def cov_active_sym(df):
        diffs = (df["adj_weight"] - df["base_weight"]).abs()
        return pd.Series({
            "coverage_sym": (diffs > EPS_COV).mean(),
            "active_sym":   0.5 * diffs.sum()
        })
    diag_sym = q_in_win.groupby("month").apply(cov_active_sym).reset_index().set_index("month")

    adj_wide_sym = q_in_win.pivot_table(index="month", columns="symbol", values="adj_weight",
                                        aggfunc="last").sort_index().fillna(0.0)
    turnover_sym = 0.5 * adj_wide_sym.diff().abs().sum(axis=1)
else:
    diag_sym = pd.DataFrame(index=ret.index)
    turnover_sym = pd.Series(index=ret.index, dtype=float)

diag = pd.DataFrame({
    "base_ret": base_ret,
    "qual_ret": qual_ret,
    "excess_ret": excess_ret,
    "hit": hit,
    "coverage_bbg": coverage_bbg,
    "active_bbg": active_bbg,
    "turnover_bbg": turnover_bbg,
})
if not diag_sym.empty:
    diag = diag.join(diag_sym.reindex(diag.index), how="left")
    diag["turnover_sym"] = turnover_sym.reindex(diag.index)

ir_excess, t_excess = info_ratio_and_tstat(excess_ret)

diag_summary = pd.DataFrame({
    "IR_excess":        [ir_excess],
    "tstat_excess":     [t_excess],
    "hit_rate":         [hit.mean()],
    "avg_coverage_bbg": [coverage_bbg.mean()],
    "avg_active_bbg":   [active_bbg.mean()],
    "avg_turnover_bbg": [turnover_bbg.mean()],
})

# ---------- 9) Write outputs ----------
with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
    out = pd.DataFrame({
        "base_ret": base_ret,
        "qual_ret": qual_ret,
        "nav_base": nav_base,
        "nav_qual": nav_qual
    })
    out.index.name = "month"
    out.to_excel(w, sheet_name=OUT_RETURNS)

    summary.to_excel(w, sheet_name=OUT_SUMMARY)

    diag.to_excel(w, sheet_name=OUT_DIAG)
    diag_summary.to_excel(w, sheet_name=OUT_DIAG_SUM, index=False)

print("Wrote sheets:", OUT_RETURNS, OUT_SUMMARY, OUT_DIAG, OUT_DIAG_SUM)
print(summary)
print(summary)