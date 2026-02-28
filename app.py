import io
import os
from datetime import datetime, timedelta
from math import exp, factorial

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="HYBRID VALUE Engine (CSV Strict)", layout="wide")

# =========================
# CONSTANTS (STRICT)
# =========================
EXEC_BOOK = "Stoiximan"     # ΜΟΝΟ Stoiximan execution
TOTAL_BOOKS_REQUIRED = 7    # 7 συνολικά (μαζί με Stoiximan)
BASELINE_REQUIRED = 6       # 6 baseline (οι υπόλοιπες)
TZ_NAME = "Europe/Athens"   # Ελλάδα


# =========================
# STRICT HELPERS (ENGINE LOGIC UNCHANGED)
# =========================
def poisson_pmf(k: int, lam: float) -> float:
    return exp(-lam) * (lam**k) / factorial(k)

def outcome_probs(lam_home: float, lam_away: float, max_goals: int = 10):
    """
    Returns (p1, pX, p2, tail_home, tail_away).
    Tail mass folded into max_goals bucket (deterministic).
    """
    ph = np.array([poisson_pmf(k, lam_home) for k in range(max_goals + 1)], dtype=float)
    pa = np.array([poisson_pmf(k, lam_away) for k in range(max_goals + 1)], dtype=float)

    tail_h = float(max(0.0, 1.0 - ph.sum()))
    tail_a = float(max(0.0, 1.0 - pa.sum()))
    ph[-1] += tail_h
    pa[-1] += tail_a

    p1 = pX = p2 = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = ph[i] * pa[j]
            if i > j:
                p1 += p
            elif i == j:
                pX += p
            else:
                p2 += p

    return float(p1), float(pX), float(p2), tail_h, tail_a

def normalize_probs_from_odds(o1, ox, o2):
    imp1, impx, imp2 = 1.0/o1, 1.0/ox, 1.0/o2
    overround = imp1 + impx + imp2
    return (imp1/overround, impx/overround, imp2/overround, overround)

def kelly_fractional(p: float, odds: float) -> float:
    """Full Kelly (Κέλλυ πλήρες) for decimal odds -> fraction of bankroll."""
    b = odds - 1.0
    q = 1.0 - p
    if b <= 0:
        return 0.0
    f = (b*p - q) / b
    return max(0.0, f)

def classify_edge(ev_mid: float, ev_worst: float):
    if ev_mid > 0 and ev_worst < 0:
        return "FRAGILE EDGE (εύθραυστο)"
    if ev_worst >= 0:
        return "STRONG EDGE (ισχυρό)"
    return "FAKE EDGE (ψεύτικο)"


# =========================
# CSV LOADER (AUTO SEP)
# =========================
def _read_csv_autodetect(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

def _normalize_cols_teams(df: pd.DataFrame) -> pd.DataFrame:
    """
    STRICT required columns (aliases allowed):
      Team, xG, xGA, M
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()

        if lc in ["team", "squad", "club", "team_name", "name"]:
            col_map[c] = "Team"
        if lc in ["m", "mp", "matches", "played", "games", "n", "apps"]:
            col_map[c] = "M"
        if lc in ["xg", "xg_for", "xgfor", "xg (for)", "xgfor."]:
            col_map[c] = "xG"
        if lc in ["xga", "xg_against", "xgagainst", "xg (against)", "xg_against."]:
            col_map[c] = "xGA"

    df = df.rename(columns=col_map)

    need = ["Team", "xG", "xGA", "M"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Need Team, xG, xGA, M (or aliases).")

    df["Team"] = df["Team"].astype(str).str.strip()
    df["xG"] = pd.to_numeric(df["xG"], errors="coerce")
    df["xGA"] = pd.to_numeric(df["xGA"], errors="coerce")
    df["M"] = pd.to_numeric(df["M"], errors="coerce")

    df = df.dropna(subset=["Team", "xG", "xGA", "M"])
    df = df[df["M"] > 0]
    df = df[df["Team"] != ""]
    return df.reset_index(drop=True)

def league_avg_xg(df: pd.DataFrame) -> float:
    return float(df["xG"].sum() / df["M"].sum())

def _normalize_cols_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    STRICT odds CSV required columns (aliases allowed):
      book, odds_1, odds_x, odds_2
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ["book", "books", "company", "site", "bk"]:
            col_map[c] = "book"
        if lc in ["odds_1", "o1", "1", "home", "odds1"]:
            col_map[c] = "odds_1"
        if lc in ["odds_x", "ox", "x", "draw", "oddsx"]:
            col_map[c] = "odds_x"
        if lc in ["odds_2", "o2", "2", "away", "odds2"]:
            col_map[c] = "odds_2"

    df = df.rename(columns=col_map)

    need = ["book", "odds_1", "odds_x", "odds_2"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Need book, odds_1, odds_x, odds_2.")

    df["book"] = df["book"].astype(str).str.strip()
    df["odds_1"] = pd.to_numeric(df["odds_1"], errors="coerce")
    df["odds_x"] = pd.to_numeric(df["odds_x"], errors="coerce")
    df["odds_2"] = pd.to_numeric(df["odds_2"], errors="coerce")
    df = df.dropna(subset=["book", "odds_1", "odds_x", "odds_2"])

    # normalize book names for strict matching
    df["book_norm"] = df["book"].str.lower().str.replace(r"\s+", "", regex=True)

    # normalize exec book
    exec_norm = EXEC_BOOK.lower().replace(" ", "")

    # enforce positive odds
    if (df[["odds_1", "odds_x", "odds_2"]] <= 1.0).any().any():
        raise ValueError("All odds must be > 1.00")

    # unique by book_norm
    df = df.sort_values("book").drop_duplicates(subset=["book_norm"], keep="first").reset_index(drop=True)

    # strict count
    if df["book_norm"].nunique() != TOTAL_BOOKS_REQUIRED:
        raise ValueError(
            f"Need exactly {TOTAL_BOOKS_REQUIRED} unique books in odds CSV "
            f"(you provided {df['book_norm'].nunique()})."
        )

    if exec_norm not in set(df["book_norm"]):
        raise ValueError(f"Missing {EXEC_BOOK} row in odds CSV.")

    return df

def split_baseline_exec_odds(odds_df: pd.DataFrame):
    exec_norm = EXEC_BOOK.lower().replace(" ", "")
    exec_row = odds_df[odds_df["book_norm"] == exec_norm].iloc[0]
    base_df = odds_df[odds_df["book_norm"] != exec_norm].copy()

    if base_df["book_norm"].nunique() != BASELINE_REQUIRED:
        raise ValueError(f"Need exactly {BASELINE_REQUIRED} baseline books (non-{EXEC_BOOK}).")

    # baseline average odds from baseline books
    avg_o1 = float(base_df["odds_1"].mean())
    avg_ox = float(base_df["odds_x"].mean())
    avg_o2 = float(base_df["odds_2"].mean())

    # execution odds = Stoiximan only
    stoix_1 = float(exec_row["odds_1"])
    stoix_x = float(exec_row["odds_x"])
    stoix_2 = float(exec_row["odds_2"])

    return avg_o1, avg_ox, avg_o2, stoix_1, stoix_x, stoix_2


# =========================
# PERSISTENCE (HISTORY PER RUN = B)
# =========================
def run_history_path(run_id: int) -> str:
    return f"history_run_{run_id}.csv"

def list_run_files():
    files = [f for f in os.listdir(".") if f.startswith("history_run_") and f.endswith(".csv")]
    def _rid(f):
        try:
            return int(f.replace("history_run_", "").replace(".csv", ""))
        except Exception:
            return 0
    return sorted(files, key=_rid)

def load_history(run_id: int) -> pd.DataFrame:
    path = run_history_path(run_id)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_history(run_id: int, df: pd.DataFrame) -> None:
    path = run_history_path(run_id)
    df.to_csv(path, index=False)

def parse_run_id_from_filename(fname: str) -> int:
    try:
        return int(fname.replace("history_run_", "").replace(".csv", ""))
    except Exception:
        return 0


# =========================
# ICS (REMINDER)
# =========================
def _dt_to_ics(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S")

def build_single_match_ics(match_label: str, kickoff_local: datetime, minutes_before: int = 10) -> str:
    remind_time = kickoff_local - timedelta(minutes=minutes_before)
    ics = (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "PRODID:-//HYBRID VALUE//Closing Reminder//EN\r\n"
        "CALSCALE:GREGORIAN\r\n"
        "BEGIN:VEVENT\r\n"
        f"UID:{abs(hash((match_label, kickoff_local.isoformat())))}@hybridvalue\r\n"
        f"DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}\r\n"
        f"DTSTART;TZID={TZ_NAME}:{_dt_to_ics(remind_time)}\r\n"
        f"SUMMARY:Closing odds (Stoiximan) – {match_label}\r\n"
        f"DESCRIPTION:Take closing odds now (10 minutes before kickoff).\r\n"
        "BEGIN:VALARM\r\n"
        "TRIGGER:PT0M\r\n"
        "ACTION:DISPLAY\r\n"
        "DESCRIPTION:Closing odds reminder\r\n"
        "END:VALARM\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )
    return ics


# =========================
# RESEARCH METRICS (CLV SIGN FIX)
# =========================
def clv_pct(open_odds: float, close_odds: float) -> float:
    """
    SIGN FIX:
      Positive CLV = καλύτερη τιμή από το closing (open > close)
      Negative CLV = χειρότερη (open < close)
    """
    if open_odds <= 0 or close_odds <= 0:
        return np.nan
    return (open_odds - close_odds) / open_odds

def settle_pnl(result: str, odds: float, stake: float) -> float:
    # result in {"W","L","V"}
    if result == "V":
        return 0.0
    if result == "W":
        return stake * (odds - 1.0)
    return -stake

def ev_bin(ev: float) -> str:
    if pd.isna(ev):
        return "NA"
    if ev < 0:
        return "<0"
    if ev < 0.01:
        return "0–1%"
    if ev < 0.02:
        return "1–2%"
    if ev < 0.05:
        return "2–5%"
    return ">=5%"


# =========================
# SAFE GUARDS (NO KEYERROR EVER)
# =========================
DEFAULT_COLS = {
    # Run/meta
    "run_id": np.nan,
    "timestamp_local": "",
    "kickoff_local": "",
    "match": "",
    "home_team": "",
    "away_team": "",
    # Pick/bet
    "pick": "NO BET",
    "book": "",
    "odds": np.nan,
    "stake": 0.0,
    "ev_mid_pick": np.nan,
    "ev_worst_pick": np.nan,
    "label": "",
    # Research/editable fields (must persist)
    "played": False,
    "include_in_eval": True,
    "roi_included": True,
    "closing_odds": np.nan,
    "clv_pct": np.nan,
    "settled": False,
    "result": "",  # W/L/V
    "pnl": 0.0,
    "roi": np.nan,
    "flags": "",
    # Optional bins
    "ev_bin": "NA",
}

BOOL_COLS = ["played", "include_in_eval", "roi_included", "settled"]
NUM_COLS = ["odds", "stake", "closing_odds", "clv_pct", "pnl", "ev_mid_pick", "ev_worst_pick", "roi"]

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, default in DEFAULT_COLS.items():
        if col not in df.columns:
            df[col] = default

    # normalize types gently (never crash)
    for c in BOOL_COLS:
        try:
            df[c] = df[c].astype(bool)
        except Exception:
            df[c] = df[c].fillna(DEFAULT_COLS[c]).astype(bool)

    for c in NUM_COLS:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            df[c] = pd.to_numeric(pd.Series([DEFAULT_COLS[c]] * len(df)), errors="coerce")

    # normalize result
    try:
        df["result"] = df["result"].fillna("").astype(str).str.upper().str.strip()
        df.loc[~df["result"].isin(["W", "L", "V", ""]), "result"] = ""
    except Exception:
        df["result"] = ""

    # normalize strings
    for c in ["match", "kickoff_local", "pick", "book", "label", "flags"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    return df

def apply_recalc(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute ONLY derived research fields:
      clv_pct (with sign fix), pnl, roi, ev_bin.
    Does NOT change engine math.
    """
    df = ensure_columns(df_in)

    # CLV (sign-correct)
    df["clv_pct"] = np.nan
    mask_clv = df["odds"].notna() & df["closing_odds"].notna() & (df["odds"] > 0) & (df["closing_odds"] > 0)
    if mask_clv.any():
        df.loc[mask_clv, "clv_pct"] = df.loc[mask_clv].apply(
            lambda r: float(clv_pct(float(r["odds"]), float(r["closing_odds"]))),
            axis=1
        )

    # PnL (only for settled + valid result + valid odds/stake)
    df["pnl"] = pd.to_numeric(df.get("pnl", 0.0), errors="coerce").fillna(0.0)
    mask_settle = df["settled"] & df["odds"].notna() & df["stake"].notna() & df["result"].isin(["W", "L", "V"])
    if mask_settle.any():
        df.loc[mask_settle, "pnl"] = df.loc[mask_settle].apply(
            lambda r: float(settle_pnl(r["result"], float(r["odds"]), float(r["stake"]))),
            axis=1
        )

    # ROI (derived; dashboard filter still uses include/played/settled flags)
    df["roi"] = np.nan
    mask_roi = df["stake"].notna() & (df["stake"] > 0) & df["pnl"].notna()
    if mask_roi.any():
        df.loc[mask_roi, "roi"] = df.loc[mask_roi, "pnl"] / df.loc[mask_roi, "stake"]

    # EV bins (based on ev_worst_pick if present)
    if "ev_worst_pick" in df.columns:
        df["ev_worst_pick"] = pd.to_numeric(df["ev_worst_pick"], errors="coerce")
        df["ev_bin"] = df["ev_worst_pick"].apply(ev_bin)
    else:
        df["ev_bin"] = "NA"

    return df


# =========================
# SESSION STATE
# =========================
def ensure_state():
    if "page" not in st.session_state:
        st.session_state.page = "HOME"

    if "mode" not in st.session_state:
        st.session_state.mode = "STRICT"  # STRICT or RESEARCH

    if "run_active" not in st.session_state:
        st.session_state.run_active = False
    if "run_id" not in st.session_state:
        st.session_state.run_id = 1

    if "home_df" not in st.session_state:
        st.session_state.home_df = None
    if "away_df" not in st.session_state:
        st.session_state.away_df = None
    if "league_home_avg" not in st.session_state:
        st.session_state.league_home_avg = None
    if "league_away_avg" not in st.session_state:
        st.session_state.league_away_avg = None

    if "prev_league_home_avg" not in st.session_state:
        st.session_state.prev_league_home_avg = None
    if "prev_league_away_avg" not in st.session_state:
        st.session_state.prev_league_away_avg = None

    if "run_df" not in st.session_state:
        st.session_state.run_df = pd.DataFrame()

    if "settings_unlocked" not in st.session_state:
        st.session_state.settings_unlocked = False

    # Auto-save buffers
    if "hist_last_saved_fingerprint" not in st.session_state:
        st.session_state.hist_last_saved_fingerprint = ""
    if "hist_current_file" not in st.session_state:
        st.session_state.hist_current_file = None

ensure_state()


# =========================
# NAVIGATION (TYPE B)
# =========================
def nav_to(page: str):
    st.session_state.page = page

def nav_block():
    st.sidebar.markdown("## 🧭 Navigation")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("🏠 HOME", use_container_width=True):
            nav_to("HOME")
    with c2:
        if st.button("🧠 ENGINE", use_container_width=True):
            nav_to("ENGINE")
    c3, c4 = st.sidebar.columns(2)
    with c3:
        if st.button("🎯 MATCH CENTER", use_container_width=True):
            nav_to("MATCH_CENTER")
    with c4:
        if st.button("📚 HISTORY", use_container_width=True):
            nav_to("HISTORY")
    if st.sidebar.button("📊 DASHBOARD", use_container_width=True):
        nav_to("DASHBOARD")
    st.sidebar.divider()


# =========================
# GLOBAL HEADER
# =========================
st.title("HYBRID VALUE 500€ ENGINE (CSV STRICT)")
st.caption(
    "ΜΗΧΑΝΗ (machine) • Home/Away CSV • League avg = ΣxG/ΣM • Cross-check >0.001 STOP • "
    "Odds via CSV upload (7 books incl. Stoiximan) • Baseline=6 (χωρίς Stoiximan) • "
    "Execution ΜΟΝΟ Stoiximan • Full audit + Kelly 1/4 + cap • Research (CLV/equity/drawdown)"
)


# =========================
# SIDEBAR (MODE + SETTINGS + CSV)
# =========================
with st.sidebar:
    nav_block()

    st.header("🧪 Mode")
    # RESEARCH MODE FIX: selector BEFORE any xG guards
    st.session_state.mode = st.radio(
        "Select mode",
        options=["STRICT", "RESEARCH"],
        index=0 if st.session_state.mode == "STRICT" else 1
    )

    st.divider()
    st.header("⚙ Ρυθμίσεις")

    st.session_state.settings_unlocked = st.toggle(
        "🔒 Unlock settings",
        value=st.session_state.settings_unlocked
    )
    LOCKED = not st.session_state.settings_unlocked

    bankroll = st.number_input(
        "Bankroll (€)",
        min_value=1.0,
        value=500.0,
        step=10.0,
        disabled=LOCKED
    )

    # NOTE: default delta = 0.01
    delta = st.slider(
        "Delta (μείωση worst-case)",
        0.00, 0.03, 0.01, 0.005,
        disabled=LOCKED
    )

    ev_worst_min = st.slider(
        "EV_worst_min (κατώφλι worst-case EV)",
        -0.02, 0.02, 0.00, 0.001,
        disabled=LOCKED
    )

    market_w = st.slider(
        "Market weight (βάρος αγοράς)",
        0.0, 1.0, 0.7, 0.05,
        disabled=LOCKED
    )
    xg_w = 1.0 - market_w

    kelly_frac = st.slider(
        "Kelly fraction (ποσοστό Kelly)",
        0.0, 1.0, 0.25, 0.05,
        disabled=LOCKED
    )

    cap_pct = st.slider(
        "Cap % bankroll (ταβάνι %)",
        0.01, 0.10, 0.05, 0.01,
        disabled=LOCKED
    )

    max_goals = st.slider(
        "Poisson max goals",
        7, 12, 10, 1,
        disabled=LOCKED
    )

    st.divider()
    st.header("📄 CSV Inputs (Home/Away)")

    home_file = st.file_uploader("Upload HOME CSV", type=["csv"], key="home_csv")
    away_file = st.file_uploader("Upload AWAY CSV", type=["csv"], key="away_csv")

    if home_file and away_file:
        try:
            home_df = _normalize_cols_teams(_read_csv_autodetect(home_file))
            away_df = _normalize_cols_teams(_read_csv_autodetect(away_file))

            st.session_state.home_df = home_df
            st.session_state.away_df = away_df
            st.session_state.league_home_avg = league_avg_xg(home_df)
            st.session_state.league_away_avg = league_avg_xg(away_df)

            st.success(
                "Loaded ✅\n"
                f"Home league avg xG: {st.session_state.league_home_avg:.6f}\n"
                f"Away league avg xG: {st.session_state.league_away_avg:.6f}"
            )
        except Exception as e:
            st.error(f"CSV load failed: {e}")
    else:
        if st.session_state.mode == "STRICT":
            st.info("STRICT: Ανέβασε ΚΑΙ τα δύο CSV (Home + Away).")
        else:
            st.info("RESEARCH: Δεν χρειάζονται Home/Away CSV uploads.")


# =========================
# RUN CONTROL HELPERS (PERSISTENCE)
# =========================
def sync_from_disk(run_id: int):
    df = load_history(run_id)
    df = ensure_columns(df)
    df = apply_recalc(df)
    st.session_state.run_df = df.copy()
    if not df.empty:
        last = df.iloc[-1]
        st.session_state.prev_league_home_avg = float(last.get("league_home_avg", np.nan)) if pd.notna(last.get("league_home_avg", np.nan)) else None
        st.session_state.prev_league_away_avg = float(last.get("league_away_avg", np.nan)) if pd.notna(last.get("league_away_avg", np.nan)) else None
    else:
        st.session_state.prev_league_home_avg = None
        st.session_state.prev_league_away_avg = None

def get_latest_run_id() -> int:
    files = list_run_files()
    if not files:
        return 1
    return max(parse_run_id_from_filename(f) for f in files) or 1

def load_run_file(fname: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(fname)
    except Exception:
        df = pd.DataFrame()
    df = ensure_columns(df)
    df = apply_recalc(df)
    return df


# =========================
# PAGES
# =========================
def page_home():
    st.subheader("Home")
    st.write("Διάλεξε σελίδα:")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("🧠 ENGINE – Run Match", use_container_width=True):
            nav_to("ENGINE")
    with b2:
        if st.button("🎯 MATCH CENTER – Single Match View", use_container_width=True):
            nav_to("MATCH_CENTER")

    b3, b4 = st.columns(2)
    with b3:
        if st.button("📚 HISTORY – Full Table + Filters", use_container_width=True):
            nav_to("HISTORY")
    with b4:
        if st.button("📊 DASHBOARD – Summary ROI/CLV", use_container_width=True):
            nav_to("DASHBOARD")

    st.divider()
    st.caption(
        "Κανόνας: navigation από παντού προς παντού, χωρίς να χάνεται run/state. "
        "Όλα τα edits (closing_odds/result/played/settled/include_in_eval/roi_included/flags) "
        "γράφονται άμεσα στο run file."
    )

def page_engine():
    st.subheader("🧠 ENGINE")
    st.caption("STRICT runs (Poisson/EV/Kelly/Selection/Weights/Thresholds/Staking) — UNCHANGED.")

    # STRICT xG guards (only in STRICT mode)
    if st.session_state.mode == "STRICT":
        if st.session_state.home_df is None or st.session_state.away_df is None:
            st.warning("STOP: STRICT χρειάζεται Home & Away CSV για να τρέξει το engine.")
            st.stop()

    # RUN CONTROL
    st.markdown("### RUN CONTROL — Persistent history ανά run (Β)")

    rc1, rc2, rc3, rc4, rc5, rc6 = st.columns([1, 1, 1, 1.2, 1.3, 2])
    with rc1:
        start_run = st.button("Start Run")
    with rc2:
        end_run = st.button("End Run")
    with rc3:
        new_run = st.button("New Run (+1)")
    with rc4:
        load_run = st.button("Load Run File")
    with rc5:
        jump_latest = st.button("Jump to Latest Run")
    with rc6:
        st.write(f"Run active: **{st.session_state.run_active}** | Run ID: **{st.session_state.run_id}**")

    if jump_latest:
        st.session_state.run_id = get_latest_run_id()
        sync_from_disk(st.session_state.run_id)
        st.success(f"Jumped to latest: {run_history_path(st.session_state.run_id)}")

    if load_run:
        sync_from_disk(st.session_state.run_id)
        st.success(f"Loaded history from {run_history_path(st.session_state.run_id)}")

    if new_run:
        st.session_state.run_active = False
        st.session_state.run_id += 1
        st.session_state.run_df = pd.DataFrame()
        st.session_state.prev_league_home_avg = None
        st.session_state.prev_league_away_avg = None
        save_history(st.session_state.run_id, ensure_columns(st.session_state.run_df))
        st.success("New Run created (empty).")

    if start_run:
        st.session_state.run_active = True
        sync_from_disk(st.session_state.run_id)
        st.success("Run started.")

    if end_run:
        st.session_state.run_active = False
        st.success("Run ended.")

    if st.session_state.mode != "STRICT":
        st.info("RESEARCH mode: Engine είναι απενεργοποιημένο (δεν απαιτούνται xG uploads).")
        st.stop()

    # STRICT: need home/away loaded
    home_df = st.session_state.home_df
    away_df = st.session_state.away_df
    teams = sorted(set(home_df["Team"].tolist()) | set(away_df["Team"].tolist()))

    # LEAGUE AVG + CROSS-CHECK (unchanged)
    st.divider()
    st.subheader("2A) League averages (από CSV) + cross-check")

    league_home_avg = float(st.session_state.league_home_avg)
    league_away_avg = float(st.session_state.league_away_avg)

    prev_h = st.session_state.prev_league_home_avg
    prev_a = st.session_state.prev_league_away_avg

    diff_h = None if prev_h is None else abs(league_home_avg - prev_h)
    diff_a = None if prev_a is None else abs(league_away_avg - prev_a)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("League home avg xG", f"{league_home_avg:.9f}")
    m2.metric("League away avg xG", f"{league_away_avg:.9f}")
    m3.metric("Δ home vs prev", "—" if diff_h is None else f"{diff_h:.9f}")
    m4.metric("Δ away vs prev", "—" if diff_a is None else f"{diff_a:.9f}")

    if diff_h is not None and (diff_h > 0.001 or diff_a > 0.001):
        st.error("RUN STATUS: NO RUN – LEAGUE AVG MISMATCH (>|0.001|). STOP.")
        st.session_state.run_active = False
        st.stop()

    st.caption("PASS (ή πρώτο ματς του run).")

    # MATCH INPUT (unchanged)
    st.divider()
    st.subheader("New match (ανά ματς)")

    mc1, mc2, mc3, mc4 = st.columns([1, 1, 1.2, 1.2])
    home_team = mc1.selectbox("Home team", teams, index=0)
    away_team = mc2.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)
    match_label = mc3.text_input("Match label", value=f"{home_team} – {away_team}")

    kickoff_date = mc4.date_input("Kickoff date (Ελλάδα)", value=datetime.now().date())
    kickoff_time = st.time_input("Kickoff time (Ελλάδα)", value=datetime.now().replace(second=0, microsecond=0).time())
    kickoff_local = datetime.combine(kickoff_date, kickoff_time)

    if home_team not in set(home_df["Team"]) or away_team not in set(away_df["Team"]):
        st.error("STOP: Home team must exist in HOME CSV and Away team must exist in AWAY CSV.")
        st.stop()

    # .ics
    c_ics1, c_ics2 = st.columns([1, 3])
    with c_ics1:
        make_ics = st.button("Create .ics reminder (10’ πριν)")
    with c_ics2:
        st.caption("Δημιουργεί calendar reminder για closing odds 10 λεπτά πριν το kickoff.")

    if make_ics:
        ics_text = build_single_match_ics(match_label=match_label, kickoff_local=kickoff_local, minutes_before=10)
        filename = f"reminder_{match_label.replace(' ', '_').replace('–','-')}.ics"
        st.download_button(
            "Download .ics",
            data=ics_text.encode("utf-8"),
            file_name=filename,
            mime="text/calendar"
        )

    # Odds CSV upload
    st.markdown("### Odds CSV upload (7 books incl. Stoiximan) — required only for STRICT RUN")
    odds_file = st.file_uploader(
        "Upload ODDS CSV (book, odds_1, odds_x, odds_2)",
        type=["csv"],
        key=f"odds_{st.session_state.run_id}_{len(st.session_state.run_df)}"
    )

    # xG PREVIEW (unchanged)
    st.markdown("### xG Preview (χωρίς odds)")

    hr = home_df[home_df["Team"] == home_team].iloc[0]
    ar = away_df[away_df["Team"] == away_team].iloc[0]

    h_att_avg = float(hr["xG"] / hr["M"])
    h_def_avg = float(hr["xGA"] / hr["M"])
    a_att_avg = float(ar["xG"] / ar["M"])
    a_def_avg = float(ar["xGA"] / ar["M"])

    home_attack = h_att_avg / league_home_avg
    home_def    = h_def_avg / league_away_avg
    away_attack = a_att_avg / league_away_avg
    away_def    = a_def_avg / league_home_avg

    lam_home = league_home_avg * home_attack * away_def
    lam_away = league_away_avg * away_attack * home_def

    p_x1, p_xx, p_x2, tail_h, tail_a = outcome_probs(lam_home, lam_away, max_goals=max_goals)

    px_df = pd.DataFrame([
        {"Outcome": "1", "p_xG": p_x1, "Fair_xG": 1.0/max(p_x1, 1e-12)},
        {"Outcome": "X", "p_xG": p_xx, "Fair_xG": 1.0/max(p_xx, 1e-12)},
        {"Outcome": "2", "p_xG": p_x2, "Fair_xG": 1.0/max(p_x2, 1e-12)},
    ])
    px_disp = px_df.copy()
    px_disp["p_xG"] = px_disp["p_xG"].map(lambda x: f"{x:.6f}")
    px_disp["Fair_xG"] = px_disp["Fair_xG"].map(lambda x: f"{x:.2f}")
    st.dataframe(px_disp, use_container_width=True)

    st.caption(f"λ_home={lam_home:.4f} | λ_away={lam_away:.4f} | tail_h={tail_h:.6f} | tail_a={tail_a:.6f}")

    # RUN MATCH (STRICT) — engine logic unchanged
    run_match = st.button("RUN MATCH (STRICT)")

    if run_match:
        if not st.session_state.run_active and st.session_state.run_df.empty:
            st.error("STOP: Πάτα πρώτα Start Run (ή φόρτωσε υπάρχον run).")
            st.stop()

        if odds_file is None:
            st.error("STOP: Για STRICT run χρειάζεται Odds CSV.")
            st.stop()

        try:
            odds_df = _normalize_cols_odds(_read_csv_autodetect(odds_file))
            avg_o1, avg_ox, avg_o2, stoix_1, stoix_x, stoix_2 = split_baseline_exec_odds(odds_df)
        except Exception as e:
            st.error(f"STOP: Odds CSV invalid: {e}")
            st.stop()

        # STEP 1: Market baseline (6 baseline books)
        p_m1, p_mx, p_m2, overround = normalize_probs_from_odds(avg_o1, avg_ox, avg_o2)

        # STEP 3: Hybrid p_true (engine logic unchanged)
        p_t1 = market_w*p_m1 + xg_w*p_x1
        p_tx = market_w*p_mx + xg_w*p_xx
        p_t2 = market_w*p_m2 + xg_w*p_x2
        mass = p_t1 + p_tx + p_t2

        fair_1, fair_x, fair_2 = 1.0/p_t1, 1.0/p_tx, 1.0/p_t2
        dp1, dpx, dp2 = p_t1 - p_m1, p_tx - p_mx, p_t2 - p_m2

        # STEP 4: EV (Stoiximan ONLY)
        def evs(p_true, o, d):
            ev_mid = o*p_true - 1.0
            ev_worst = o*max(0.0, p_true - d) - 1.0
            return float(ev_mid), float(ev_worst)

        ev1_mid, ev1_worst = evs(p_t1, stoix_1, delta)
        evx_mid, evx_worst = evs(p_tx, stoix_x, delta)
        ev2_mid, ev2_worst = evs(p_t2, stoix_2, delta)

        # STEP 5: Selection rules (1 επιλογή ανά ματς)
        candidates = [
            ("1", EXEC_BOOK, stoix_1, p_t1, p_m1, p_x1, fair_1, dp1, ev1_mid, ev1_worst),
            ("X", EXEC_BOOK, stoix_x, p_tx, p_mx, p_xx, fair_x, dpx, evx_mid, evx_worst),
            ("2", EXEC_BOOK, stoix_2, p_t2, p_m2, p_x2, fair_2, dp2, ev2_mid, ev2_worst),
        ]
        passed = [c for c in candidates if c[9] >= ev_worst_min]
        pick = sorted(passed, key=lambda t: t[9], reverse=True)[0] if passed else None

        # STEP 6: AUDIT checks
        mass_ok = abs(mass - 1.0) <= 0.002

        market_best = max([("1", p_m1), ("X", p_mx), ("2", p_m2)], key=lambda t: t[1])[0]
        xg_best = max([("1", p_x1), ("X", p_xx), ("2", p_x2)], key=lambda t: t[1])[0]
        direction_ok = (market_best == xg_best) or (max(p_m1, p_mx, p_m2) - max(p_x1, p_xx, p_x2) <= 0.05)

        # STEP 7: Stake (Kelly 1/4 + cap)
        stake = 0.0
        kelly_full = 0.0
        final_label = "NO BET"
        bet_out = ""
        bet_book = ""
        bet_odds = np.nan
        bet_ev_mid = np.nan
        bet_ev_worst = np.nan

        if pick is not None and mass_ok:
            out, book, odds, p_true, *_rest, ev_mid, ev_worst = pick
            label = classify_edge(ev_mid, ev_worst)
            if label != "FAKE EDGE (ψεύτικο)":
                kelly_full = kelly_fractional(p_true, odds)
                stake = bankroll * (kelly_full * kelly_frac)
                stake = min(stake, bankroll * cap_pct)

                final_label = label
                bet_out = out
                bet_book = book
                bet_odds = odds
                bet_ev_mid = ev_mid
                bet_ev_worst = ev_worst

        # OUTPUT
        st.success(f"RUN COMPLETE: {match_label}")

        out_df = pd.DataFrame([
            {"Outcome":"1", "p_market":p_m1, "p_xG":p_x1, "p_true":p_t1, "Fair":fair_1, "Δp":dp1,
             "StoixOdds":stoix_1, "EV_mid":ev1_mid, "EV_worst":ev1_worst},
            {"Outcome":"X", "p_market":p_mx, "p_xG":p_xx, "p_true":p_tx, "Fair":fair_x, "Δp":dpx,
             "StoixOdds":stoix_x, "EV_mid":evx_mid, "EV_worst":evx_worst},
            {"Outcome":"2", "p_market":p_m2, "p_xG":p_x2, "p_true":p_t2, "Fair":fair_2, "Δp":dp2,
             "StoixOdds":stoix_2, "EV_mid":ev2_mid, "EV_worst":ev2_worst},
        ])
        disp = out_df.copy()
        for c in ["p_market","p_xG","p_true","Fair","Δp","EV_mid","EV_worst"]:
            disp[c] = disp[c].map(lambda x: f"{x:.6f}")
        disp["StoixOdds"] = disp["StoixOdds"].map(lambda x: f"{x:.2f}")
        st.dataframe(disp, use_container_width=True)

        st.subheader("AUDIT (έλεγχος)")
        st.write(f"Probability mass: {mass:.9f} → {'PASS' if mass_ok else 'FAIL'} (±0.002)")
        st.write(f"Directional logic: market_best={market_best}, xG_best={xg_best} → {'PASS' if direction_ok else 'FLAG'}")
        st.write("EV Stability: EV_mid>0 αλλά EV_worst<0 ⇒ FRAGILE EDGE (εύθραυστο)")

        st.subheader("FINAL (τελικό)")
        if final_label == "NO BET" or np.isnan(bet_odds):
            st.warning("NO BET")
        else:
            st.success(f"{final_label} → BET {bet_book} {bet_out} @ {bet_odds:.2f} | Stake €{stake:.2f}")
            st.caption(f"Kelly full={kelly_full:.6f} | Kelly applied={kelly_frac:.2f} | Cap={cap_pct:.2%}")

        # Append to run history (PERSISTENT)
        run_row = {
            "run_id": st.session_state.run_id,
            "timestamp_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "kickoff_local": kickoff_local.strftime("%Y-%m-%d %H:%M:%S"),
            "match": match_label,
            "home_team": home_team,
            "away_team": away_team,

            "league_home_avg": league_home_avg,
            "league_away_avg": league_away_avg,

            "avg_baseline_odds_1": avg_o1,
            "avg_baseline_odds_x": avg_ox,
            "avg_baseline_odds_2": avg_o2,

            "stoix_odds_1": stoix_1,
            "stoix_odds_x": stoix_x,
            "stoix_odds_2": stoix_2,

            "p_market_1": p_m1, "p_market_x": p_mx, "p_market_2": p_m2,
            "p_xg_1": p_x1, "p_xg_x": p_xx, "p_xg_2": p_x2,
            "p_true_1": p_t1, "p_true_x": p_tx, "p_true_2": p_t2,

            "fair_1": fair_1, "fair_x": fair_x, "fair_2": fair_2,
            "dp_1": dp1, "dp_x": dpx, "dp_2": dp2,

            "delta": delta,
            "ev_worst_min": ev_worst_min,

            "ev_mid_1": ev1_mid, "ev_mid_x": evx_mid, "ev_mid_2": ev2_mid,
            "ev_worst_1": ev1_worst, "ev_worst_x": evx_worst, "ev_worst_2": ev2_worst,

            "pick": ("NO BET" if np.isnan(bet_odds) else bet_out),
            "book": ("" if np.isnan(bet_odds) else bet_book),
            "odds": (np.nan if np.isnan(bet_odds) else bet_odds),
            "stake": (0.0 if np.isnan(bet_odds) else stake),
            "ev_mid_pick": (np.nan if np.isnan(bet_odds) else bet_ev_mid),
            "ev_worst_pick": (np.nan if np.isnan(bet_odds) else bet_ev_worst),
            "label": final_label,

            "lambda_home": lam_home,
            "lambda_away": lam_away,

            # Research fields (editable later) — must persist immediately
            "played": False,
            "include_in_eval": True,
            "roi_included": True,
            "closing_odds": np.nan,
            "clv_pct": np.nan,
            "settled": False,
            "result": "",
            "pnl": 0.0,
            "roi": np.nan,
            "flags": "",
            "ev_bin": ev_bin(bet_ev_worst) if not np.isnan(bet_odds) else "NA",
        }

        df_old = st.session_state.run_df.copy()
        df_new = pd.concat([df_old, pd.DataFrame([run_row])], ignore_index=True)
        df_new = ensure_columns(df_new)
        df_new = apply_recalc(df_new)

        st.session_state.run_df = df_new
        save_history(st.session_state.run_id, df_new)

        # Update prev league avgs for strict cross-check (within run)
        st.session_state.prev_league_home_avg = league_home_avg
        st.session_state.prev_league_away_avg = league_away_avg

        st.info(f"Saved: {run_history_path(st.session_state.run_id)}")


def autosave_editor(df_current: pd.DataFrame, file_path: str, fingerprint_key: str):
    """
    Auto-save immediately to disk if editor content changed.
    """
    try:
        fp = str(pd.util.hash_pandas_object(df_current, index=True).sum())
    except Exception:
        fp = str(len(df_current)) + "|" + str(list(df_current.columns))

    last = st.session_state.get(fingerprint_key, "")
    if fp != last:
        df_to_save = apply_recalc(df_current)
        df_to_save.to_csv(file_path, index=False)
        st.session_state[fingerprint_key] = fp
        return True
    return False


def page_history():
    st.subheader("📚 HISTORY — Full Table + Filters")
    st.caption("Όλα τα edits γράφονται άμεσα στο run file (auto-save). Soft Exclude & Hard Delete υποστηρίζονται.")

    run_files = list_run_files()
    if not run_files:
        st.info("Δεν υπάρχουν saved runs ακόμα.")
        return

    c1, c2, c3 = st.columns([1.4, 1.0, 1.2])
    with c1:
        selected_file = st.selectbox("Run file", run_files, index=len(run_files)-1)
    with c2:
        if st.button("↻ Refresh list"):
            st.rerun()
    with c3:
        st.download_button(
            "Download selected run CSV",
            data=open(selected_file, "rb").read(),
            file_name=selected_file,
            mime="text/csv",
        )

    df_hist = load_run_file(selected_file)
    df_hist = ensure_columns(df_hist)

    st.markdown("### Filters")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        only_bets = st.toggle("Only bets (≠ NO BET)", value=True)
    with f2:
        only_played = st.toggle("Only played", value=False)
    with f3:
        only_settled = st.toggle("Only settled", value=False)
    with f4:
        only_included = st.toggle("Only include_in_eval", value=False)

    mask = pd.Series([True] * len(df_hist))
    if only_bets:
        mask &= df_hist["pick"].astype(str).fillna("").ne("NO BET") & df_hist["odds"].notna()
    if only_played:
        mask &= df_hist["played"]
    if only_settled:
        mask &= df_hist["settled"]
    if only_included:
        mask &= df_hist["include_in_eval"]

    view_df = df_hist[mask].copy()

    st.markdown("### History table (view)")
    show_cols = [
        "kickoff_local", "match", "pick", "odds", "stake",
        "ev_mid_pick", "ev_worst_pick", "label",
        "played", "include_in_eval", "roi_included",
        "closing_odds", "clv_pct",
        "settled", "result", "pnl", "roi", "flags",
    ]
    for c in show_cols:
        if c not in view_df.columns:
            view_df[c] = DEFAULT_COLS.get(c, np.nan)

    st.dataframe(view_df[show_cols], use_container_width=True)

    st.divider()
    st.markdown("### ✏ Editor (auto-save)")

    edit_cols = [
        "kickoff_local", "match", "pick", "odds", "stake",
        "played", "include_in_eval", "roi_included",
        "closing_odds", "settled", "result", "flags",
    ]
    edit_df = df_hist[edit_cols].copy()

    edited = st.data_editor(
        edit_df,
        use_container_width=True,
        num_rows="fixed",
        key=f"hist_editor_{selected_file}",
    )

    # merge back
    df_merged = df_hist.copy()
    for col in edited.columns:
        df_merged[col] = edited[col]

    # normalize + recalc
    df_merged = ensure_columns(df_merged)
    df_merged = apply_recalc(df_merged)

    saved_now = autosave_editor(df_merged, selected_file, "hist_last_saved_fingerprint")
    if saved_now:
        st.success("Auto-saved ✅")

    st.divider()
    st.markdown("### 🔹 Soft Exclude / Include (instant)")
    s1, s2, s3 = st.columns([1.2, 1.2, 2])
    with s1:
        row_idx = st.number_input("Row index (0-based)", min_value=0, max_value=max(0, len(df_hist)-1), value=0, step=1)
    with s2:
        new_inc = st.selectbox("include_in_eval", [True, False], index=0)
    with s3:
        st.caption("Soft exclude = include_in_eval=False. Επηρεάζει άμεσα ROI/Dashboard.")

    if st.button("Apply include_in_eval now"):
        df_tmp = load_run_file(selected_file)
        df_tmp = ensure_columns(df_tmp)
        i = int(row_idx)
        if 0 <= i < len(df_tmp):
            df_tmp.at[i, "include_in_eval"] = bool(new_inc)
            df_tmp = apply_recalc(df_tmp)
            df_tmp.to_csv(selected_file, index=False)
            st.success("Saved include_in_eval.")
            st.rerun()

    st.divider()
    st.markdown("### 🗑 Hard Delete (με επιβεβαίωση)")
    d1, d2, d3 = st.columns([1.2, 1.8, 2.0])
    with d1:
        del_idx = st.number_input("Row index to delete", min_value=0, max_value=max(0, len(df_hist)-1), value=0, step=1, key="del_idx")
    with d2:
        confirm = st.checkbox("CONFIRM DELETE (irreversible)", value=False)
    with d3:
        st.caption("Διαγράφει τη γραμμή ΜΟΝΟ με confirm.")

    if st.button("DELETE ROW NOW"):
        if not confirm:
            st.error("STOP: Πρέπει να τσεκάρεις το CONFIRM DELETE.")
        else:
            df_tmp = load_run_file(selected_file)
            df_tmp = ensure_columns(df_tmp)
            i = int(del_idx)
            if 0 <= i < len(df_tmp):
                df_tmp = df_tmp.drop(index=i).reset_index(drop=True)
                df_tmp = apply_recalc(df_tmp)
                df_tmp.to_csv(selected_file, index=False)
                st.success(f"Deleted row {i}.")
                st.rerun()


def page_match_center():
    st.subheader("🎯 MATCH CENTER — Single Match View")
    st.caption("Επιλέγεις match, βάζεις closing/result/flags και βλέπεις CLV/ROI/PnL/EV/Stake μόνο για αυτό το match.")

    run_files = list_run_files()
    if not run_files:
        st.info("Δεν υπάρχουν saved runs ακόμα.")
        return

    c1, c2 = st.columns([1.4, 1.0])
    with c1:
        selected_file = st.selectbox("Run file", run_files, index=len(run_files)-1, key="mc_run_file")
    with c2:
        st.download_button(
            "Download run CSV",
            data=open(selected_file, "rb").read(),
            file_name=selected_file,
            mime="text/csv",
        )

    df = load_run_file(selected_file)
    df = ensure_columns(df)

    if df.empty:
        st.info("Το run είναι άδειο.")
        return

    # pick a match
    labels = df["match"].astype(str).fillna("").tolist()
    idx_default = 0
    pick_idx = st.selectbox(
        "Select match (row)",
        options=list(range(len(df))),
        format_func=lambda i: f"[{i}] {labels[i]}",
        index=idx_default,
        key="mc_row_idx"
    )
    i = int(pick_idx)
    row = df.iloc[i].copy()

    # Editor (single match) — immediate save
    st.markdown("### Edit (auto-save)")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        played = st.toggle("played", value=bool(row["played"]), key=f"mc_played_{selected_file}_{i}")
        settled = st.toggle("settled", value=bool(row["settled"]), key=f"mc_settled_{selected_file}_{i}")
    with e2:
        include_in_eval = st.toggle("include_in_eval", value=bool(row["include_in_eval"]), key=f"mc_inc_{selected_file}_{i}")
        roi_included = st.toggle("roi_included", value=bool(row["roi_included"]), key=f"mc_roiinc_{selected_file}_{i}")
    with e3:
        closing_odds = st.number_input(
            "closing_odds",
            min_value=1.0,
            value=float(row["closing_odds"]) if pd.notna(row["closing_odds"]) and float(row["closing_odds"]) >= 1.0 else 1.0,
            step=0.01,
            key=f"mc_close_{selected_file}_{i}"
        )
    with e4:
        result = st.selectbox(
            "result",
            options=["", "W", "L", "V"],
            index=["", "W", "L", "V"].index(str(row["result"]).strip()) if str(row["result"]).strip() in ["", "W", "L", "V"] else 0,
            key=f"mc_res_{selected_file}_{i}"
        )

    flags = st.text_input("flags", value=str(row.get("flags", "")), key=f"mc_flags_{selected_file}_{i}")

    # Apply and save immediately
    df.at[i, "played"] = bool(played)
    df.at[i, "settled"] = bool(settled)
    df.at[i, "include_in_eval"] = bool(include_in_eval)
    df.at[i, "roi_included"] = bool(roi_included)
    df.at[i, "closing_odds"] = float(closing_odds) if closing_odds is not None else np.nan
    df.at[i, "result"] = str(result).strip().upper()
    df.at[i, "flags"] = str(flags)

    df = apply_recalc(df)
    df.to_csv(selected_file, index=False)

    # Re-pull the row after recalc
    row2 = df.iloc[i].copy()

    st.success("Saved ✅")

    # Match-level stats (only this match)
    st.markdown("### Match stats (this row only)")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Stake (€)", f"{float(row2['stake']):.2f}" if pd.notna(row2["stake"]) else "—")
    s2.metric("EV mid", f"{float(row2['ev_mid_pick']):.6f}" if pd.notna(row2["ev_mid_pick"]) else "—")
    s3.metric("EV worst", f"{float(row2['ev_worst_pick']):.6f}" if pd.notna(row2["ev_worst_pick"]) else "—")
    s4.metric("CLV %", f"{float(row2['clv_pct'])*100:.2f}%" if pd.notna(row2["clv_pct"]) else "—")
    s5.metric("PnL (€)", f"{float(row2['pnl']):.2f}" if pd.notna(row2["pnl"]) else "—")
    s6.metric("ROI %", f"{float(row2['roi'])*100:.2f}%" if pd.notna(row2["roi"]) else "—")

    st.markdown("### Details")
    detail_cols = [
        "kickoff_local", "match", "pick", "book", "odds", "stake",
        "played", "include_in_eval", "roi_included",
        "closing_odds", "clv_pct",
        "settled", "result", "pnl", "roi", "flags",
        "label", "ev_mid_pick", "ev_worst_pick",
    ]
    st.dataframe(pd.DataFrame([row2[detail_cols]]), use_container_width=True)


def page_dashboard():
    st.subheader("📊 DASHBOARD")
    st.caption("Στατιστικά χωρίς αλλαγή μαθηματικής λογικής. Δεν κρασάρει ποτέ αν λείπουν στήλες (safe guards).")

    run_files = list_run_files()
    if not run_files:
        st.info("Δεν υπάρχουν saved runs ακόμα.")
        return

    scope = st.radio("Scope", options=["ALL RUNS", "SINGLE RUN"], index=0, horizontal=True)
    if scope == "SINGLE RUN":
        selected_file = st.selectbox("Run file", run_files, index=len(run_files)-1, key="dash_single_file")
        dfs = [load_run_file(selected_file)]
    else:
        dfs = [load_run_file(f) for f in run_files]

    df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    df_all = ensure_columns(df_all)
    df_all = apply_recalc(df_all)

    if df_all.empty:
        st.info("Δεν υπάρχουν γραμμές.")
        return

    # Sort timeline
    if "kickoff_local" in df_all.columns:
        try:
            df_all = df_all.sort_values("kickoff_local").reset_index(drop=True)
        except Exception:
            df_all = df_all.reset_index(drop=True)
    else:
        df_all = df_all.reset_index(drop=True)

    # Bets mask
    bets_mask = df_all["pick"].astype(str).fillna("").ne("NO BET") & df_all["odds"].notna()

    # Evaluation filter (required)
    eval_mask = df_all["settled"] & df_all["played"] & df_all["include_in_eval"] & df_all["roi_included"] & bets_mask

    df_eval = df_all[eval_mask].copy()
    df_bets = df_all[bets_mask].copy()

    # Total ROI: sum(pnl)/sum(stake) for eval
    total_pnl = float(df_eval["pnl"].fillna(0.0).sum()) if not df_eval.empty else 0.0
    total_stake = float(df_eval["stake"].fillna(0.0).sum()) if not df_eval.empty else 0.0
    total_roi = (total_pnl / total_stake) if total_stake > 0 else np.nan

    # CLV stats
    clv_mean = float(df_eval["clv_pct"].dropna().mean()) if df_eval["clv_pct"].notna().any() else np.nan
    clv_median = float(df_eval["clv_pct"].dropna().median()) if df_eval["clv_pct"].notna().any() else np.nan

    # Equity & drawdown (eval-only pnl)
    df_all["pnl_eval"] = np.where(eval_mask, df_all["pnl"].fillna(0.0), 0.0)
    df_all["cum_pnl"] = df_all["pnl_eval"].cumsum()
    df_all["equity"] = float(bankroll) + df_all["cum_pnl"]
    df_all["equity_peak"] = df_all["equity"].cummax()
    df_all["drawdown"] = df_all["equity"] - df_all["equity_peak"]
    df_all["drawdown_pct"] = np.where(df_all["equity_peak"] > 0, df_all["drawdown"] / df_all["equity_peak"], 0.0)

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Bets (all)", f"{len(df_bets)}")
    m2.metric("Eval bets", f"{len(df_eval)}")
    m3.metric("Total PnL eval (€)", f"{total_pnl:.2f}")
    m4.metric("Total ROI eval", "—" if pd.isna(total_roi) else f"{total_roi*100:.2f}%")
    m5.metric("Max Drawdown (€)", f"{float(df_all['drawdown'].min()):.2f}")

    st.divider()
    st.markdown("### ROI ανά run")
    # ROI per run (eval-only)
    if "run_id" not in df_all.columns:
        df_all["run_id"] = np.nan

    per_run = df_all.copy()
    per_run["is_eval"] = eval_mask
    per_run = per_run[per_run["is_eval"]].copy()

    if per_run.empty:
        st.info("Δεν υπάρχουν eval bets ακόμα (played+settled+include_in_eval+roi_included).")
    else:
        grp = (
            per_run.groupby("run_id", dropna=False)
            .agg(
                bets=("run_id", "count"),
                pnl=("pnl", "sum"),
                stake=("stake", "sum"),
                clv_mean=("clv_pct", "mean"),
                clv_median=("clv_pct", "median"),
            )
            .reset_index()
        )
        grp["roi"] = np.where(grp["stake"] > 0, grp["pnl"] / grp["stake"], np.nan)
        st.dataframe(grp, use_container_width=True)

    st.divider()
    st.markdown("### CLV (eval only)")
    if pd.isna(clv_mean):
        st.info("CLV: Δεν υπάρχουν ακόμα closing odds σε evaluated settled bets.")
    else:
        st.write(f"Mean CLV%: **{clv_mean*100:.2f}%** | Median CLV%: **{clv_median*100:.2f}%**")

    st.divider()
    st.markdown("### EV bins (all bets, based on ev_worst_pick)")
    if "ev_bin" not in df_all.columns:
        df_all["ev_bin"] = df_all["ev_worst_pick"].apply(ev_bin)

    bins_src = df_all[bets_mask].copy()
    bin_table = (
        bins_src.groupby("ev_bin", dropna=False)
        .agg(
            bets=("ev_bin", "count"),
            settled=("settled", "sum"),
            played=("played", "sum"),
            pnl_eval=("pnl_eval", "sum"),
            avg_ev_worst=("ev_worst_pick", "mean"),
            avg_clv=("clv_pct", "mean"),
        )
        .reset_index()
        .sort_values("ev_bin")
    )
    st.dataframe(bin_table, use_container_width=True)

    st.divider()
    st.markdown("### Equity curve / Drawdown (eval only)")
    chart_df = df_all.copy()
    chart_df["t"] = chart_df.get("kickoff_local", pd.Series(range(len(chart_df)))).astype(str)

    c_eq, c_dd = st.columns(2)
    with c_eq:
        st.line_chart(chart_df.set_index("t")[["equity"]])
    with c_dd:
        st.line_chart(chart_df.set_index("t")[["drawdown"]])

    st.caption("Σημείωση: equity/drawdown υπολογίζονται ΜΟΝΟ για settled+played+include_in_eval+roi_included.")


# =========================
# ROUTER
# =========================
if st.session_state.page == "HOME":
    page_home()
elif st.session_state.page == "ENGINE":
    page_engine()
elif st.session_state.page == "MATCH_CENTER":
    page_match_center()
elif st.session_state.page == "HISTORY":
    page_history()
elif st.session_state.page == "DASHBOARD":
    page_dashboard()
else:
    st.session_state.page = "HOME"
    st.rerun()