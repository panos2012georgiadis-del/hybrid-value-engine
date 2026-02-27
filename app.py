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
    Optional column:
      closing_time (ignored by engine; used for reminder only if present)
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
        if lc in ["match_time", "kickoff", "start_time", "datetime", "date_time", "time"]:
            col_map[c] = "match_time"

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

def load_history(run_id: int) -> pd.DataFrame:
    path = run_history_path(run_id)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_history(run_id: int, df: pd.DataFrame) -> None:
    path = run_history_path(run_id)
    df.to_csv(path, index=False)

# =========================
# ICS (REMINDER)
# =========================
def _dt_to_ics(dt: datetime) -> str:
    # Local-naive converted to "floating" with TZID in VEVENT
    return dt.strftime("%Y%m%dT%H%M%S")

def build_single_match_ics(match_label: str, kickoff_local: datetime, minutes_before: int = 10) -> str:
    remind_time = kickoff_local - timedelta(minutes=minutes_before)
    # VEVENT uses TZID for DTSTART
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
# RESEARCH METRICS
# =========================
def clv_pct(open_odds: float, close_odds: float) -> float:
    # Simple odds-based CLV % (closing vs open)
    # Positive means closing odds higher than open (you beat the close if your open was lower?) – depends on convention.
    # We keep deterministic and transparent:
    return (close_odds - open_odds) / open_odds

def settle_pnl(outcome_pick: str, result: str, odds: float, stake: float) -> float:
    # result in {"W","L","V"}
    if result == "V":
        return 0.0
    if result == "W":
        return stake * (odds - 1.0)
    # Loss
    return -stake

def ev_bin(ev: float) -> str:
    # bins for EV_worst (can adjust later without touching engine)
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
# SESSION STATE
# =========================
def ensure_state():
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

ensure_state()

# =========================
# UI HEADER
# =========================
st.title("HYBRID VALUE 500€ ENGINE (CSV STRICT)")
st.caption(
    "ΜΗΧΑΝΗ (machine) • Home/Away CSV • League avg = ΣxG/ΣM • Cross-check >0.001 STOP • "
    "Odds via CSV upload (7 books incl. Stoiximan) • Baseline=6 (χωρίς Stoiximan) • "
    "Execution ΜΟΝΟ Stoiximan • Full audit + Kelly 1/4 + cap • Research Mode (history/CLV/equity)"
)

# =========================
# SIDEBAR CONTROLS
# =========================
# =========================
# SIDEBAR CONTROLS
# =========================
with st.sidebar:

    st.header("⚙ Ρυθμίσεις")

    # ===== LOCK SYSTEM =====
    if "settings_unlocked" not in st.session_state:
        st.session_state.settings_unlocked = False

    st.session_state.settings_unlocked = st.toggle(
        "🔒 Unlock settings",
        value=st.session_state.settings_unlocked
    )

    LOCKED = not st.session_state.settings_unlocked

    # ===== CORE SETTINGS =====
    bankroll = st.number_input(
        "Bankroll (€)",
        min_value=1.0,
        value=500.0,
        step=10.0,
        disabled=LOCKED
    )

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

    # ===== CSV INPUTS =====
    st.header("📄 CSV Inputs (Home/Away)")

    home_file = st.file_uploader("Upload HOME CSV", type=["csv"])
    away_file = st.file_uploader("Upload AWAY CSV", type=["csv"])

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
        st.info("Ανέβασε ΚΑΙ τα δύο CSV (Home + Away).")

# Must have CSVs to proceed
if st.session_state.home_df is None or st.session_state.away_df is None:
    st.warning("STOP: Χρειάζονται Home & Away CSV για να τρέξει το engine.")
    st.stop()

home_df = st.session_state.home_df
away_df = st.session_state.away_df
teams = sorted(set(home_df["Team"].tolist()) | set(away_df["Team"].tolist()))

# =========================
# RUN CONTROL + PERSISTENCE
# =========================
st.subheader("RUN CONTROL (πολλά ματς στο ίδιο run) — Persistent history ανά run (Β)")

rc1, rc2, rc3, rc4, rc5 = st.columns([1, 1, 1, 1, 2])
with rc1:
    start_run = st.button("Start Run")
with rc2:
    end_run = st.button("End Run")
with rc3:
    reset_all = st.button("Reset All")
with rc4:
    load_run = st.button("Load Run File")
with rc5:
    st.write(f"Run active: **{st.session_state.run_active}** | Run ID: **{st.session_state.run_id}**")

# Always keep run_df in sync with disk if exists
def _sync_from_disk():
    df = load_history(st.session_state.run_id)
    st.session_state.run_df = df.copy()
    # Set prev league averages to last entry (so cross-check continues if you resume run)
    if not df.empty:
        last = df.iloc[-1]
        st.session_state.prev_league_home_avg = float(last.get("league_home_avg", np.nan)) if pd.notna(last.get("league_home_avg", np.nan)) else None
        st.session_state.prev_league_away_avg = float(last.get("league_away_avg", np.nan)) if pd.notna(last.get("league_away_avg", np.nan)) else None
    else:
        st.session_state.prev_league_home_avg = None
        st.session_state.prev_league_away_avg = None

if load_run:
    _sync_from_disk()
    st.success(f"Loaded history from {run_history_path(st.session_state.run_id)}")

if reset_all:
    st.session_state.run_active = False
    st.session_state.prev_league_home_avg = None
    st.session_state.prev_league_away_avg = None
    st.session_state.run_id += 1
    st.session_state.run_df = pd.DataFrame()
    st.success("Reset complete. (New Run ID created)")

if start_run:
    st.session_state.run_active = True
    # If a file already exists for this run_id, load it; else start empty
    _sync_from_disk()
    st.success("Run started. (Persistent history enabled)")

if end_run:
    st.session_state.run_active = False
    st.success("Run ended.")

if not st.session_state.run_active and st.session_state.run_df.empty:
    st.info("Πάτα **Start Run** για να ξεκινήσεις νέο run.")
    st.stop()

# =========================
# LEAGUE AVGS + CROSS-CHECK (RECALC EACH MATCH)
# =========================
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

# =========================
# MATCH INPUT
# =========================
st.subheader("New match (ανά ματς)")

mc1, mc2, mc3, mc4 = st.columns([1, 1, 1.2, 1.2])
home_team = mc1.selectbox("Home team", teams, index=0)
away_team = mc2.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)
match_label = mc3.text_input("Match label", value=f"{home_team} – {away_team}")

# Kickoff date + time (for reminder only)
kickoff_date = mc4.date_input("Kickoff date (Ελλάδα)", value=datetime.now().date())
kickoff_time = st.time_input("Kickoff time (Ελλάδα)", value=datetime.now().replace(second=0, microsecond=0).time())
kickoff_local = datetime.combine(kickoff_date, kickoff_time)

if home_team not in set(home_df["Team"]) or away_team not in set(away_df["Team"]):
    st.error("STOP: Home team must exist in HOME CSV and Away team must exist in AWAY CSV.")
    st.stop()

st.markdown("### Odds CSV upload (7 books incl. Stoiximan)")
odds_file = st.file_uploader(
    "Upload ODDS CSV (book, odds_1, odds_x, odds_2)",
    type=["csv"],
    key=f"odds_{st.session_state.run_id}_{len(st.session_state.run_df)}"
)

c_ics1, c_ics2 = st.columns([1, 3])
with c_ics1:
    make_ics = st.button("Create .ics reminder (10’ πριν)")
with c_ics2:
    st.caption("Δημιουργεί ένα αρχείο calendar reminder για closing odds 10 λεπτά πριν το kickoff.")

if make_ics:
    ics_text = build_single_match_ics(match_label=match_label, kickoff_local=kickoff_local, minutes_before=10)
    filename = f"reminder_{match_label.replace(' ', '_').replace('–','-')}.ics"
    st.download_button(
        "Download .ics",
        data=ics_text.encode("utf-8"),
        file_name=filename,
        mime="text/calendar"
    )

run_match = st.button("RUN MATCH (STRICT)")

# =========================
# RUN MATCH (STRICT) — ENGINE LOGIC UNCHANGED
# =========================
if run_match:
    if odds_file is None:
        st.error("STOP: Πρέπει να ανεβάσεις Odds CSV για αυτό το ματς.")
        st.stop()

    try:
        odds_df = _normalize_cols_odds(_read_csv_autodetect(odds_file))
        avg_o1, avg_ox, avg_o2, stoix_1, stoix_x, stoix_2 = split_baseline_exec_odds(odds_df)
    except Exception as e:
        st.error(f"STOP: Odds CSV invalid: {e}")
        st.stop()

    # -------------------------
    # STEP 1: Market baseline (6 baseline books)
    # -------------------------
    p_m1, p_mx, p_m2, overround = normalize_probs_from_odds(avg_o1, avg_ox, avg_o2)

    # -------------------------
    # STEP 2B: Poisson lambdas from CSV split
    # -------------------------
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

    # -------------------------
    # STEP 3: Hybrid p_true
    # -------------------------
    p_t1 = market_w*p_m1 + xg_w*p_x1
    p_tx = market_w*p_mx + xg_w*p_xx
    p_t2 = market_w*p_m2 + xg_w*p_x2
    mass = p_t1 + p_tx + p_t2

    fair_1, fair_x, fair_2 = 1.0/p_t1, 1.0/p_tx, 1.0/p_t2
    dp1, dpx, dp2 = p_t1 - p_m1, p_tx - p_mx, p_t2 - p_m2

    # -------------------------
    # STEP 4: EV (Stoiximan ONLY)
    # -------------------------
    def evs(p_true, o, d):
        ev_mid = o*p_true - 1.0
        ev_worst = o*max(0.0, p_true - d) - 1.0
        return float(ev_mid), float(ev_worst)

    ev1_mid, ev1_worst = evs(p_t1, stoix_1, delta)
    evx_mid, evx_worst = evs(p_tx, stoix_x, delta)
    ev2_mid, ev2_worst = evs(p_t2, stoix_2, delta)

    # -------------------------
    # STEP 5: Selection rules (1 επιλογή ανά ματς)
    # -------------------------
    candidates = [
        ("1", EXEC_BOOK, stoix_1, p_t1, p_m1, p_x1, fair_1, dp1, ev1_mid, ev1_worst),
        ("X", EXEC_BOOK, stoix_x, p_tx, p_mx, p_xx, fair_x, dpx, evx_mid, evx_worst),
        ("2", EXEC_BOOK, stoix_2, p_t2, p_m2, p_x2, fair_2, dp2, ev2_mid, ev2_worst),
    ]
    passed = [c for c in candidates if c[9] >= ev_worst_min]
    pick = sorted(passed, key=lambda t: t[9], reverse=True)[0] if passed else None

    # -------------------------
    # STEP 6: AUDIT checks
    # -------------------------
    mass_ok = abs(mass - 1.0) <= 0.002

    market_best = max([("1", p_m1), ("X", p_mx), ("2", p_m2)], key=lambda t: t[1])[0]
    xg_best = max([("1", p_x1), ("X", p_xx), ("2", p_x2)], key=lambda t: t[1])[0]
    direction_ok = (market_best == xg_best) or (max(p_m1, p_mx, p_m2) - max(p_x1, p_xx, p_x2) <= 0.05)

    # -------------------------
    # STEP 7: Stake (Kelly 1/4 + cap)
    # -------------------------
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

    # -------------------------
    # OUTPUT (FULL)
    # -------------------------
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
    st.write(f"Directional logic: market_best={market_best}, xg_best={xg_best} → {'PASS' if direction_ok else 'FLAG'}")
    st.write("EV Stability: EV_mid>0 αλλά EV_worst<0 ⇒ FRAGILE EDGE (εύθραυστο)")

    st.subheader("FINAL (τελικό)")
    if final_label == "NO BET" or np.isnan(bet_odds):
        st.warning("NO BET")
    else:
        st.success(f"{final_label} → BET {bet_book} {bet_out} @ {bet_odds:.2f} | Stake €{stake:.2f}")
        st.caption(f"Kelly full={kelly_full:.6f} | Kelly applied={kelly_frac:.2f} | Cap={cap_pct:.2%}")

    # -------------------------
    # Append to run history (PERSISTENT)
    # -------------------------
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

        # Research additions (no engine change)
        "closing_odds": np.nan,
        "clv_pct": np.nan,
        "settled": False,
        "result": "",
        "pnl": 0.0,
        "ev_bin": ev_bin(bet_ev_worst) if not np.isnan(bet_odds) else "NA",
    }

    df_old = st.session_state.run_df.copy()
    df_new = pd.concat([df_old, pd.DataFrame([run_row])], ignore_index=True)
    st.session_state.run_df = df_new
    save_history(st.session_state.run_id, df_new)

    # Update prev league avgs for strict cross-check (within run)
    st.session_state.prev_league_home_avg = league_home_avg
    st.session_state.prev_league_away_avg = league_away_avg

    st.info(f"Saved: {run_history_path(st.session_state.run_id)}")

# =========================
# FULL RESEARCH MODE
# =========================
st.divider()
st.subheader("Full Research Mode (ιστορικό / CLV / equity / drawdown)")

df = st.session_state.run_df.copy()
if df.empty:
    st.info("Δεν υπάρχει ακόμα ιστορικό σε αυτό το run.")
    st.stop()

# Download history CSV
c_dl1, c_dl2 = st.columns([1, 3])
with c_dl1:
    st.download_button(
        "Download history CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=run_history_path(st.session_state.run_id),
        mime="text/csv",
    )
with c_dl2:
    st.caption("Persistent history ανά run (Β): αποθηκεύεται σε αρχείο CSV δίπλα στο app.py")

# Show core history table
show_cols = [
    "kickoff_local", "match", "pick", "odds", "stake",
    "ev_mid_pick", "ev_worst_pick", "label",
    "closing_odds", "clv_pct", "settled", "result", "pnl"
]
for col in show_cols:
    if col not in df.columns:
        df[col] = np.nan

disp_hist = df[show_cols].copy()
for c in ["odds", "stake", "ev_mid_pick", "ev_worst_pick", "closing_odds", "clv_pct", "pnl"]:
    if c in disp_hist.columns:
        disp_hist[c] = pd.to_numeric(disp_hist[c], errors="coerce")

st.dataframe(disp_hist, use_container_width=True)

# =========================
# SETTLE + CLV UI (ROI Controlled)
# =========================

st.markdown("### Settle Bet + CLV (ROI controlled)")
st.caption("Βάλε closing odds και επίλεξε αν το bet θα μετρήσει στο ROI.")

# Ensure column exists (backward compatible)
if "roi_included" not in df.columns:
    df["roi_included"] = True
else:
    df["roi_included"] = df["roi_included"].fillna(True).astype(bool)

bet_rows = df.index[df["pick"].astype(str) != "NO BET"].tolist()

if not bet_rows:
    st.info("Δεν υπάρχουν bets σε αυτό το run.")
else:
    options = [
        (i, f"{df.loc[i,'kickoff_local']} | {df.loc[i,'match']} | "
            f"{df.loc[i,'pick']} @ {df.loc[i,'odds']} | €{df.loc[i,'stake']}")
        for i in bet_rows
    ]

    idx = st.selectbox(
        "Select bet to update",
        options=options,
        format_func=lambda x: x[1]
    )[0]

    row = df.loc[idx].copy()

    st.write(f"**Selected:** {row['match']} | Pick: **{row['pick']}** @ {row['odds']}")

    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])

    with c1:
        closing = st.number_input(
            "Closing odds (Stoiximan)",
            min_value=1.01,
            value=float(row["closing_odds"]) if pd.notna(row.get("closing_odds", None)) else float(row["odds"]),
            step=0.01,
            key=f"closing_{st.session_state.run_id}_{idx}"
        )

        roi_included = st.checkbox(
            "Count in ROI / Equity",
            value=bool(row.get("roi_included", True)),
            key=f"roi_{st.session_state.run_id}_{idx}"
        )

    with c2:
        btn_w = st.button("Set WIN", key=f"win_{st.session_state.run_id}_{idx}")
    with c3:
        btn_l = st.button("Set LOSS", key=f"loss_{st.session_state.run_id}_{idx}")
    with c4:
        btn_v = st.button("Set VOID", key=f"void_{st.session_state.run_id}_{idx}")

    def _apply_settle(result_code: str):
        open_odds = float(row["odds"])
        stake = float(row["stake"])

        pnl = settle_pnl(
            outcome_pick=str(row["pick"]),
            result=result_code,
            odds=open_odds,
            stake=stake
        )

        df.at[idx, "closing_odds"] = float(closing)
        df.at[idx, "clv_pct"] = float(clv_pct(open_odds=open_odds, close_odds=float(closing)))
        df.at[idx, "settled"] = True
        df.at[idx, "result"] = result_code
        df.at[idx, "pnl"] = float(pnl)

        # NEW: ROI flag
        df.at[idx, "roi_included"] = bool(roi_included)

        st.session_state.run_df = df
        save_history(st.session_state.run_id, df)

        st.success(f"Settled: {result_code} | PnL={pnl:.2f}")
        st.rerun()

    if btn_w:
        _apply_settle("W")
    if btn_l:
        _apply_settle("L")
    if btn_v:
        _apply_settle("V")
# =========================
# RESEARCH DASHBOARD (CLV / EV BINS / EQUITY / DRAWDOWN)
# =========================
st.divider()
st.subheader("Research Dashboard")

df = st.session_state.run_df.copy()
df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
df["stake"] = pd.to_numeric(df["stake"], errors="coerce")
df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
df["ev_worst_pick"] = pd.to_numeric(df["ev_worst_pick"], errors="coerce")
df["clv_pct"] = pd.to_numeric(df["clv_pct"], errors="coerce")
df["settled"] = df["settled"].astype(bool) if "settled" in df.columns else False

# Equity curve (based on settled pnl)
df_sorted = df.sort_values("kickoff_local").reset_index(drop=True)
df_sorted["cum_pnl"] = df_sorted["pnl"].fillna(0.0).cumsum()
df_sorted["equity"] = float(bankroll) + df_sorted["cum_pnl"]
df_sorted["equity_peak"] = df_sorted["equity"].cummax()
df_sorted["drawdown"] = df_sorted["equity"] - df_sorted["equity_peak"]
df_sorted["drawdown_pct"] = np.where(df_sorted["equity_peak"] > 0, df_sorted["drawdown"] / df_sorted["equity_peak"], 0.0)

# Summary metrics
settled_df = df_sorted[df_sorted["settled"] == True].copy()
bets_df = df_sorted[df_sorted["pick"].astype(str) != "NO BET"].copy()

mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
mcol1.metric("Bets", f"{len(bets_df)}")
mcol2.metric("Settled", f"{len(settled_df)}")
mcol3.metric("Total PnL (€)", f"{settled_df['pnl'].sum():.2f}")
mcol4.metric("Equity (€)", f"{df_sorted['equity'].iloc[-1]:.2f}")
mcol5.metric("Max Drawdown (€)", f"{df_sorted['drawdown'].min():.2f}")

# CLV summary
if settled_df["clv_pct"].notna().any():
    st.markdown("#### CLV (Closing Line Value)")
    clv_mean = float(settled_df["clv_pct"].dropna().mean())
    clv_med = float(settled_df["clv_pct"].dropna().median())
    st.write(f"Mean CLV%: **{clv_mean*100:.2f}%** | Median CLV%: **{clv_med*100:.2f}%**")
else:
    st.info("CLV: Δεν υπάρχουν ακόμα closing odds σε settled bets.")

# EV bins table
st.markdown("#### EV bins (βάσει EV_worst_pick)")
bets_df["ev_bin"] = bets_df["ev_bin"].astype(str)
bin_table = (
    bets_df.groupby("ev_bin", dropna=False)
    .agg(
        bets=("ev_bin", "count"),
        settled=("settled", "sum"),
        pnl=("pnl", "sum"),
        avg_ev_worst=("ev_worst_pick", "mean"),
        avg_clv=("clv_pct", "mean"),
    )
    .reset_index()
    .sort_values("ev_bin")
)
st.dataframe(bin_table, use_container_width=True)

# Equity curve + drawdown (simple charts)
st.markdown("#### Equity curve / Drawdown")
chart_df = df_sorted[["kickoff_local", "equity", "drawdown", "drawdown_pct"]].copy()
chart_df["kickoff_local"] = chart_df["kickoff_local"].astype(str)

c_eq, c_dd = st.columns(2)
with c_eq:
    st.line_chart(chart_df.set_index("kickoff_local")[["equity"]])
with c_dd:
    st.line_chart(chart_df.set_index("kickoff_local")[["drawdown"]])

st.caption("Σημείωση: η equity ενημερώνεται μόνο όταν κάνεις Settle (WIN/LOSS/VOID).")
