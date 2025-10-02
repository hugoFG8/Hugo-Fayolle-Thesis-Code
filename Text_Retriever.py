import os, re, sys, time, json, logging, warnings, traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd
from dateutil.relativedelta import relativedelta

# Silence common warnings/logs (Spyder/IPython + PRAW async)
warnings.filterwarnings("ignore", message=r".*asynchronous environment.*", category=UserWarning)
logging.getLogger("praw").setLevel(logging.ERROR)
logging.getLogger("prawcore").setLevel(logging.ERROR)

# 3rd-party deps
import requests
import yfinance as yf
try:
    import praw
except Exception:
    print("ERROR: Please install praw (pip install praw).")
    raise

# ===================== USER CONFIG =====================

EXCEL_PATH = "/Users/hugofayolle/Downloads/SP500_Stocks_and_Prices_per_month/SP500_prices_matrix.xlsx"
CSV_BACKUP   = "/Users/hugofayolle/Downloads/Opinion_Sentences_backup.csv"
UNMAPPED_CSV = "/Users/hugofayolle/Downloads/unmapped_ids.csv"

EXCEL_PORTFOLIO_SHEET = "Momentum_Portfolio"
EXCEL_OUT_SHEET       = "Opinion_Sentences"

# Window
START_MONTH = "2025-08"
MIN_MONTH   = "2010-07"

# Subreddits
SUBREDDITS = [
    "stocks","investing","StockMarket","wallstreetbets",
    "investing_discussion","stocksDD","options",
    "ValueInvesting","GrowthStocks","Dividends","techstocks"
]

# PRAW search settings (structure unchanged)
TIME_FILTER = "all"       
SEARCH_SORT = "relevance" 
MAX_POSTS_PER_QUERY = 120

# Mining caps / targets
MAX_SENTENCES_PER_TICKER_MON   = 15
TARGET_MIN_SENTENCES_PER_TICKER = 8

# Reddit quality gates (global defaults)
REDDIT_MIN_SCORE    = 10
REDDIT_MIN_COMMENTS = 5

# Gentler gates for WSB only
WSB_MIN_SCORE    = 2
WSB_MIN_COMMENTS = 1

# Comments: deep sweep (top-scored across all depths)
COMMENT_HARVEST_LIMIT = 80  # cap total comments inspected per post

# Opinion filters
FILTER_PATTERNS = re.compile(
    r"(earnings|reports Q[1-4]|quarter|dividend|guidance|results|sec filing|press release|"
    r"price target|downgrade|upgrade|initiates coverage|announces|launches|appoints|board of directors|"
    r"week recap|weekly wrap|market update|daily update|morning update|closing bell|"
    r"after hours|pre-market|beats estimates|misses estimates|\bup \d+%|\bdown \d+%|"
    r"gain porn|loss porn|my portfolio|judge my portfolio|what should I buy|help me pick)",
    re.IGNORECASE
)
OPINION_CUES = re.compile(
    r"\b(i think|i believe|imo|imho|in my opinion|looks (cheap|expensive|overvalued|undervalued)|"
    r"seems (cheap|expensive|overvalued|undervalued)|"
    r"(bullish|bearish)|overvalued|undervalued|"
    r"i'm (buying|selling|holding)|i am (buying|selling|holding)|"
    r"(great|good|bad|poor) (moat|management)|"
    r"(overpriced|underpriced)|"
    r"risk/?reward|"
    r"accumulate|trim|buy the dip|bagholder|diamond hands|paper hands)"
    , re.IGNORECASE
)

# Reddit API
REDDIT_CLIENT_ID = "FxzwOWWq4ufX35fLI_9qRA"
REDDIT_CLIENT_SECRET = "j_joq-MRUZNHZIM98YpNLeksG2Nxhg"
REDDIT_USER_AGENT = "thesis_qualitative_filter/0.1 by u/ThesisRetrievalUnit"

# OpenFIGI (optional) for Bloomberg-like IDs
OPENFIGI_API_KEY     = "28f47c68-ed65-4c82-81e5-ad29be43f616"

VERBOSE = True

# ===================== Helpers & Filters =====================

def month_bounds_ts(year: int, month: int) -> Tuple[int, int]:
    start = datetime(year, month, 1)
    end   = (start + relativedelta(months=1)) - timedelta(seconds=1)
    return int(start.timestamp()), int(end.timestamp())

def parse_ym(s: str) -> Tuple[int, int]:
    dt = datetime.strptime(s, "%Y-%m"); return dt.year, dt.month

def fmt_eta(sec: float) -> str:
    sec = max(0, float(sec))
    if sec < 60: return f"{sec:.0f}s"
    m, s = divmod(int(sec), 60)
    return f"{m}m {s}s" if sec < 3600 else f"{m//60}h {m%60}m"

def looks_opinionated_regex(txt: str) -> bool:
    if not isinstance(txt, str): return False
    s = txt.strip()
    if len(s) < 8 or re.fullmatch(r"[\W_]+", s): return False
    return not FILTER_PATTERNS.search(s)

def opinionated_enough(sentence: str) -> bool:
    # keep if passes hard filter OR has subjective cues
    return looks_opinionated_regex(sentence) or bool(OPINION_CUES.search(sentence))

# Sentence splitter
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip(): return []
    parts = _SENT_SPLIT.split(text.replace("\n", " ").strip())
    uniq, seen = [], set()
    for p in parts:
        t = p.strip()
        if len(t) >= 8 and t not in seen:
            uniq.append(t); seen.add(t)
    return uniq

# ===================== Mapping IDs to tickers =====================

_BBG_PAT = re.compile(r"^\s*([A-Z0-9]+)\s+[A-Z]{1,3}\s+Equity\s*$", re.IGNORECASE)
_figi_cache: Dict[str, str] = {}

def heuristic_bbg(id_like: str) -> Optional[str]:
    if not isinstance(id_like, str): return None
    s = id_like.strip()
    m = _BBG_PAT.match(s)
    if m:
        core = m.group(1).upper()
        if re.match(r"^[A-Z][A-Z0-9\.-]{0,9}$", core): return core
    if re.match(r"^[A-Z][A-Z0-9\.-]{0,9}$", s.upper()):
        return s.upper()
    return None

def figi_lookup(id_like: str) -> Optional[str]:
    if id_like in _figi_cache: return _figi_cache[id_like]
    if not OPENFIGI_API_KEY or OPENFIGI_API_KEY == "YOUR_OPENFIGI_API_KEY":
        return None
    headers = {"Content-Type":"application/json","X-OPENFIGI-APIKEY":OPENFIGI_API_KEY}
    unique_id = id_like.split()[0].strip()
    payload = [{"idType":"ID_BB_UNIQUE","idValue":unique_id}]
    try:
        r = requests.post("https://api.openfigi.com/v3/mapping", headers=headers, data=json.dumps(payload), timeout=15)
        if r.status_code==200:
            data=r.json()
            if data and data[0].get("data"):
                tk = data[0]["data"][0].get("ticker")
                if tk:
                    _figi_cache[id_like]=tk.upper(); return tk.upper()
    except Exception as e:
        sys.stderr.write(f"[WARN] FIGI fail {id_like}: {e}\n")
    return None

def to_ticker(id_like: str) -> Optional[str]:
    tk = heuristic_bbg(id_like)
    if tk: return tk
    if isinstance(id_like, str) and re.match(r"^\d{6,}[A-Z]\s+[A-Z]{1,3}\s+Equity$", id_like.strip(), re.IGNORECASE):
        tk = figi_lookup(id_like)
        if tk: return tk
        try:
            with open(UNMAPPED_CSV,"a",encoding="utf-8") as f: f.write(id_like+"\n")
        except Exception: pass
    return None

# ===================== Aliases & mention matching =====================

CORP_SUFFIXES = {
    "inc","inc.","corp","corp.","corporation","company","co","co.",
    "ltd","ltd.","plc","sa","nv","ag","llc","holdings","group","lp","plc."
}

def _clean_company_phrase(name: str) -> Optional[str]:
    if not isinstance(name, str): return None
    s = re.sub(r"[^\w&\-\s]", " ", name)
    parts = [p for p in s.split() if p]
    if not parts: return None
    while parts and parts[-1].lower().strip(".,") in CORP_SUFFIXES:
        parts.pop()
    s2 = " ".join(parts).strip()
    return s2 if len(s2) >= 3 else None

def _and_variants(phrase: str) -> List[str]:
    out = {phrase}
    if "&" in phrase: out.add(phrase.replace("&", "and"))
    if " and " in phrase.lower(): out.add(re.sub(r"\band\b", "&", phrase, flags=re.IGNORECASE))
    return list(out)

def _hyphen_variants(phrase: str) -> List[str]:
    if "-" not in phrase: return [phrase]
    return [phrase, phrase.replace("-", " ")]

_alias_cache: Dict[str, List[str]] = {}
def build_aliases_for_ticker(ticker: str) -> List[str]:
    if ticker in _alias_cache: return _alias_cache[ticker]
    aliases = []
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    for k in ("shortName", "longName"):
        nm = info.get(k)
        if isinstance(nm, str) and len(nm) > 2:
            base = _clean_company_phrase(nm)
            if base:
                for a in _and_variants(base):
                    aliases.extend(_hyphen_variants(a))
    # Manual helpful extras
    MANUAL = {
        "META": ["Facebook"],
        "GOOG": ["Google", "Alphabet"],
        "GOOGL": ["Google", "Alphabet"],
        "BRK.B": ["Berkshire Hathaway", "Berkshire"],
        "BRK.A": ["Berkshire Hathaway", "Berkshire"],
        "SQ": ["Square"],
        "PG": ["Procter & Gamble", "Procter and Gamble", "P&G"],
        "V": ["Visa"],
        "T": ["AT&T"],
        "AVGO": ["Avago"],
        "AXON": ["Taser","TASER"]
    }
    aliases += MANUAL.get(ticker.upper(), [])
    # de-dup case-insensitive
    seen, out = set(), []
    for a in aliases:
        k = a.lower().strip()
        if k and k not in seen:
            out.append(a.strip()); seen.add(k)
    _alias_cache[ticker] = out
    return out

def build_queries_for(ticker: str) -> List[str]:
    qs = [f'${ticker}', f'"{ticker}"', f'title:{ticker}', ticker]
    for alias in build_aliases_for_ticker(ticker)[:5]:  # modest cap for speed
        qs += [f'title:"{alias}"', f'"{alias}"']
    # unique, order-preserving
    seen, out = set(), []
    for q in qs:
        k = q.lower().strip()
        if k and k not in seen:
            out.append(q); seen.add(k)
    return out

def build_mention_patterns(ticker: str, aliases: List[str]) -> List[re.Pattern]:
    pats = []
    tkr = ticker.upper()
    pats.append(re.compile(rf"(?<!\w)\${re.escape(tkr)}(?!\w)", re.IGNORECASE))
    if len(tkr) >= 3 and re.fullmatch(r"[A-Z0-9\.]+", tkr):
        pats.append(re.compile(rf"\b{re.escape(tkr)}\b", re.IGNORECASE))
    for a in aliases:
        esc = re.escape(a).replace(r"\ ", r"\s+")
        pats.append(re.compile(rf"\b{esc}\b", re.IGNORECASE))
    return pats

def mention_matches(sentence: str, patterns: List[re.Pattern]) -> bool:
    if not isinstance(sentence, str): return False
    s = sentence.strip()
    if len(s) < 3: return False
    return any(p.search(s) for p in patterns)

# ===================== PRAW client & mining =====================

def get_reddit() -> "praw.Reddit":
    kw = dict(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
    try:
        return praw.Reddit(**kw, check_for_async=False)
    except TypeError:
        return praw.Reddit(**kw)

def extract_sentences_from_submission(subm, patterns: List[re.Pattern]) -> List[Tuple[str,str,int,str,str]]:
    """
    Returns tuples: (sentence, kind, weight, author_name, comment_id)
    kind: 'title' | 'selftext' | 'comment'
    """
    keep = []
    sub_author = str(getattr(getattr(subm, "author", None), "name", "") or "[deleted]")

    # Title as sentence if opinion cues present
    title = getattr(subm, "title", "") or ""
    if title and OPINION_CUES.search(title):
        keep.append((title.strip(), "title", int(getattr(subm, "score", 0) or 0), sub_author, ""))

    # Selftext
    if getattr(subm, "is_self", False):
        text = getattr(subm, "selftext", "") or ""
        if text:
            for s in split_sentences(text):
                if mention_matches(s, patterns):
                    keep.append((s, "selftext", int(getattr(subm,"score",0) or 0), sub_author, ""))

    # COMMENTS: take highest-scored across ALL depths, capped
    try:
        subm.comment_sort = "top"
        subm.comments.replace_more(limit=0)
        all_comments = subm.comments.list()
        all_comments.sort(key=lambda c: int(getattr(c, "score", 0) or 0), reverse=True)
        for c in all_comments[:COMMENT_HARVEST_LIMIT]:
            body = getattr(c, "body", "") or ""
            if not body: continue
            author = str(getattr(getattr(c, "author", None), "name", "") or "[deleted]")
            for s in split_sentences(body):
                if mention_matches(s, patterns):
                    keep.append((s, "comment", int(getattr(c,"score",0) or 0), author, getattr(c, "id", "")))
    except Exception:
        pass

    return keep

# ===================== Excel I/O =====================

def read_momentum_portfolio(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=EXCEL_PORTFOLIO_SHEET, engine="openpyxl")
    if df.columns[0].lower().startswith("unnamed"):
        df = df.drop(df.columns[0], axis=1)
    long = df.melt(var_name="month", value_name="id_like").dropna(subset=["id_like"]).reset_index(drop=True)
    long["month"] = long["month"].astype(str)
    return long

def append_to_excel(excel_path: str, df_add: pd.DataFrame, sheet: str = EXCEL_OUT_SHEET):
    try:
        existing = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")
        combo = pd.concat([existing, df_add], ignore_index=True)
    except Exception:
        combo = df_add.copy()
    combo.drop_duplicates(subset=["month","ticker","sentence"], inplace=True)
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        combo.to_excel(w, index=False, sheet_name=sheet)

def append_to_csv(csv_path: str, df_add: pd.DataFrame):
    try:
        ex = pd.read_csv(csv_path)
        combo = pd.concat([ex, df_add], ignore_index=True)
    except Exception:
        combo = df_add.copy()
    combo.drop_duplicates(subset=["month","ticker","sentence"], inplace=True)
    combo.to_csv(csv_path, index=False, encoding="utf-8")

# ===================== Per-ticker-month fetch =====================

def fetch_for_ticker_month(reddit, ticker: str, year: int, month: int, id_like: str) -> List[dict]:
    month_start_ts, month_end_ts = month_bounds_ts(year, month)
    aliases  = build_aliases_for_ticker(ticker)
    patterns = build_mention_patterns(ticker, aliases)

    rows: List[dict] = []
    seen_sentence_keys: Set[Tuple[str,str,str]] = set()

    # Iterate subs and alias-enriched queries
    for sub_name in SUBREDDITS:
        try:
            sr = reddit.subreddit(sub_name)
        except Exception:
            continue

        for q in build_queries_for(ticker):
            try:
                it = sr.search(q, sort=SEARCH_SORT, time_filter=TIME_FILTER, limit=MAX_POSTS_PER_QUERY)
            except TypeError:
                it = sr.search(q, sort=SEARCH_SORT, limit=MAX_POSTS_PER_QUERY)

            for post in it:
                try:
                    created = int(getattr(post, "created_utc", 0) or 0)
                    if created < month_start_ts or created > month_end_ts:
                        continue

                    score = int(getattr(post, "score", 0) or 0)
                    ncom  = int(getattr(post, "num_comments", 0) or 0)

                    # WSB gentler gates
                    sub_lower = sub_name.lower()
                    if sub_lower == "wallstreetbets":
                        min_score, min_comments = WSB_MIN_SCORE, WSB_MIN_COMMENTS
                    else:
                        min_score, min_comments = REDDIT_MIN_SCORE, REDDIT_MIN_COMMENTS

                    if score < min_score or ncom < min_comments:
                        continue

                    sents = extract_sentences_from_submission(post, patterns)
                    if not sents:
                        continue

                    title = getattr(post, "title", "") or ""
                    flair = (getattr(post, "link_flair_text", "") or "").strip()
                    permalink = getattr(post, "permalink", None)

                    for sent, kind, w, author, comment_id in sents:
                        s_clean = sent.strip()
                        if not opinionated_enough(s_clean):
                            continue
                        key = (f"{year:04d}-{month:02d}", ticker, s_clean.lower())
                        if key in seen_sentence_keys:
                            continue
                        rows.append({
                            "month": f"{year:04d}-{month:02d}",
                            "ticker": ticker,
                            "source": "reddit",
                            "subreddit": sub_name,
                            "kind": kind,                 # 'title' | 'selftext' | 'comment'
                            "title": title,               # context only
                            "sentence": s_clean,          # payload
                            "url": f"https://www.reddit.com{permalink}" if permalink else "",
                            "post_score": score,
                            "num_comments": ncom,
                            "sent_weight": int(w or 0),   # comment/post score proxy
                            "flair": flair,
                            "id_like": id_like,
                            "author": author,
                            "comment_id": comment_id,
                            "submission_id": getattr(post, "id", "")
                        })
                        seen_sentence_keys.add(key)

                except Exception:
                    continue

            time.sleep(0.2)  # polite pacing per query

        # Early stop per ticker if target met (keeps runtime reasonable)
        if len(rows) >= TARGET_MIN_SENTENCES_PER_TICKER:
            break

    # Rank by discussion weight and cap per month
    rows.sort(key=lambda r: (r.get("post_score",0) + 3*r.get("num_comments",0) + r.get("sent_weight",0)), reverse=True)
    out, seen = [], set()
    for r in rows:
        k = (r["month"], r["ticker"], r["sentence"].strip().lower())
        if k in seen: continue
        out.append(r); seen.add(k)
        if len(out) >= MAX_SENTENCES_PER_TICKER_MON:
            break
    return out

# ===================== MAIN =====================

def main():
    df_sel = read_momentum_portfolio(EXCEL_PATH)
    sy, sm = parse_ym(START_MONTH)
    min_m = MIN_MONTH if MIN_MONTH else df_sel["month"].min()

    def m2i(ms): y,m=parse_ym(ms); return y*100 + m
    months = [m for m in sorted(df_sel["month"].unique()) if m2i(min_m) <= m2i(m) <= sy*100+sm]
    months.sort(reverse=True)
    if not months:
        print("No months found."); return

    reddit = get_reddit()
    total_rows = 0
    total_tickers = sum(len(df_sel[df_sel["month"]==ms]) for ms in months)
    done = 0
    t0 = time.time()

    for mi, ms in enumerate(months, start=1):
        y, m = parse_ym(ms)
        df_m = df_sel[df_sel["month"]==ms].copy()
        if VERBOSE:
            print(f"\n=== {ms} ({mi}/{len(months)}) tickers: {len(df_m)} ===")
        batch = []
        for i, row in enumerate(df_m.itertuples(index=False), start=1):
            id_like = row.id_like
            tk = to_ticker(id_like)
            done += 1
            avg = (time.time()-t0)/max(1,done)
            eta = avg*max(0,(total_tickers-done))
            print(f"  [{ms}] {i}/{len(df_m)}  {id_like} -> {tk or 'unmapped'}  | avg {avg:.1f}s | ETA {fmt_eta(eta)}")
            if not tk:
                continue
            recs = fetch_for_ticker_month(reddit, tk, y, m, id_like)
            if recs:
                batch.extend(recs)
                print(f"    kept {len(recs)} sentences")
            else:
                print(f"    no opinion sentences found")
            time.sleep(0.1)

        if batch:
            df_add = pd.DataFrame.from_records(batch)
            try: append_to_excel(EXCEL_PATH, df_add, sheet=EXCEL_OUT_SHEET)
            except Exception as e: sys.stderr.write(f"[ERROR] Excel write {ms}: {e}\n")
            try: append_to_csv(CSV_BACKUP, df_add)
            except Exception as e: sys.stderr.write(f"[ERROR] CSV write {ms}: {e}\n")
            total_rows += len(df_add)
            print(f"[{ms}] saved {len(df_add)} sentences; total ~{total_rows}")

    print(f"\nDone. Saved ~{total_rows} opinion sentences.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write("FATAL ERROR:\n" + "".join(traceback.format_exception(e)) + "\n")
        sys.exit(1)