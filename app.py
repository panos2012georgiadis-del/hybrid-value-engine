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

def list_run_files() -> list[str]:
    files = [f for f in os.listdir(".") if f.startswith("history_run_") and f.endswith(".csv")]
    # sort by run_id numeric if possible
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
# RESEARCH METRICS
# =========================
def clv_pct(open_odds: float, close_odds: float) -> float:
    return (close_odds - open_odds) / open_odds

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

    if "settings_unlocked" not in st.session_state:
        st.session_state.settings_unlocked = False

ensure_state()


# =========================
# UI HEADER
# =========================
st.title("HYBRID VALUE 500€ ENGINE (CSV STRICT)")
st.caption(
    "ΜΗΧΑΝΗ (machine) • Home/Away CSV • League avg = ΣxG/ΣM • Cross-check >0.001 STOP • "
    "Odds via CSV upload (7 books incl. Stoiximan) • Baseline=6 (χωρίς Stoiximan) • "
    "Execution ΜΟΝΟ Stoiximan • Full audit + Kelly 1/4 + cap • Research (CLV/equity/drawdown)"
)


# =========================
# SIDEBAR (SETTINGS + CSV)
# =========================
with st.sidebar:
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

    # NOTE: default delta = 0.01 όπως είχες ζητήσει (και μπορείς να το αλλάζεις όταν ξεκλειδώσεις)
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
        st.info("Ανέβασε ΚΑΙ τα δύο CSV (Home + Away).")


# Must have Home/Away CSVs
if st.session_state.home_df is None or st.session_state.away_df is None:
    st.warning("STOP: Χρειάζονται Home & Away CSV για να τρέξει το engine.")
    st.stop()

home_df = st.session_state.home_df
away_df = st.session_state.away_df
teams = sorted(set(home_df["Team"].tolist()) | set(away_df["Team"].tolist()))


# =========================
# RUN CONTROL + PERSISTENCE
# =========================
st.subheader("RUN CONTROL — Persistent history ανά run (Β)")

def _sync_from_disk(run_id: int):
    df = load_history(run_id)
    st.session_state.run_df = df.copy()
    if not df.empty:
        last = df.iloc[-1]
        st.session_state.prev_league_home_avg = float(last.get("league_home_avg", np.nan)) if pd.notna(last.get("league_home_avg", np.nan)) else None
        st.session_state.prev_league_away_avg = float(last.get("league_away_avg", np.nan)) if pd.notna(last.get("league_away_avg", np.nan)) else None
    else:
        st.session_state.prev_league_home_avg = None
        st.session_state.prev_league_away_avg = None

rc1, rc2, rc3, rc4, rc5 = st.columns([1, 1, 1, 1.3, 2])
with rc1:
    start_run = st.button("Start Run")
with rc2:
    end_run = st.button("End Run")
with rc3:
    new_run = st.button("New Run (+1)")
with rc4:
    load_run = st.button("Load Run File")
with rc5:
    st.write(f"Run active: **{st.session_state.run_active}** | Run ID: **{st.session_state.run_id}**")

if load_run:
    _sync_from_disk(st.session_state.run_id)
    st.success(f"Loaded history from {run_history_path(st.session_state.run_id)}")

if new_run:
    st.session_state.run_active = False
    st.session_state.run_id += 1
    st.session_state.run_df = pd.DataFrame()
    st.session_state.prev_league_home_avg = None
    st.session_state.prev_league_away_avg = None
    st.success("New Run created (empty).")

if start_run:
    st.session_state.run_active = True
    _sync_from_disk(st.session_state.run_id)
    st.success("Run started.")

if end_run:
    st.session_state.run_active = False
    st.success("Run ended.")


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

# Odds CSV (optional until you actually RUN STRICT)
st.markdown("### Odds CSV upload (7 books incl. Stoiximan) — required only for STRICT RUN")
odds_file = st.file_uploader(
    "Upload ODDS CSV (book, odds_1, odds_x, odds_2)",
    type=["csv"],
    key=f"odds_{st.session_state.run_id}_{len(st.session_state.run_df)}"
)

# =========================
# xG PREVIEW (no odds needed)
# =========================
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


# =========================
# RUN MATCH (STRICT) — engine logic unchanged
# =========================
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

        # Research fields (editable later)
        "played": False,                 # αν το έπαιξες
        "include_in_eval": True,         # αν θα μετρήσει στην αξιολόγηση
        "roi_included": True,            # τελικό φίλτρο ROI (default True)
        "closing_odds": np.nan,
        "clv_pct": np.nan,
        "settled": False,
        "result": "",                    # W/L/V
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
# RESEARCH / HISTORY (ALWAYS VISIBLE UNDER ENGINE)  <-- mode 2
# =========================
st.divider()
st.subheader("📚 Research / History (πάντα κάτω)")

# Choose which run to inspect/edit
run_files = list_run_files()
c_r1, c_r2, c_r3 = st.columns([1.3, 1.2, 1.5])

with c_r1:
    if run_files:
        selected_file = st.selectbox("Run file", run_files, index=len(run_files)-1)
    else:
        selected_file = None
with c_r2:
    refresh_hist = st.button("Refresh list")
with c_r3:
    if selected_file:
        st.download_button(
            "Download selected run CSV",
            data=open(selected_file, "rb").read(),
            file_name=selected_file,
            mime="text/csv",
        )

if refresh_hist:
    st.rerun()

if selected_file is None:
    st.info("Δεν υπάρχουν saved runs ακόμα.")
    st.stop()

try:
    df_hist = pd.read_csv(selected_file)
except Exception as e:
    st.error(f"Δεν μπορώ να ανοίξω το αρχείο: {selected_file} ({e})")
    st.stop()

# Ensure columns exist
need_cols = {
    "played": False,
    "include_in_eval": True,
    "roi_included": True,
    "closing_odds": np.nan,
    "clv_pct": np.nan,
    "settled": False,
    "result": "",
    "pnl": 0.0,
}
for col, default in need_cols.items():
    if col not in df_hist.columns:
        df_hist[col] = default

# Show compact table
st.markdown("### History table")
show_cols = [
    "kickoff_local", "match", "pick", "odds", "stake",
    "ev_mid_pick", "ev_worst_pick", "label",
    "played", "include_in_eval", "roi_included",
    "closing_odds", "clv_pct",
    "settled", "result", "pnl",
]
for c in show_cols:
    if c not in df_hist.columns:
        df_hist[c] = np.nan

st.dataframe(df_hist[show_cols], use_container_width=True)

# Settlement editor (edit later)
st.markdown("### ✏ Settlement Editor (βάζεις closing odds / αποτέλεσμα / flags αργότερα)")

edit_cols = [
    "kickoff_local", "match", "pick", "odds", "stake",
    "played", "include_in_eval", "roi_included",
    "closing_odds", "settled", "result",
]
edit_df = df_hist[edit_cols].copy()

edited = st.data_editor(
    edit_df,
    use_container_width=True,
    num_rows="fixed"
)

c_s1, c_s2, c_s3 = st.columns([1, 1, 2])
with c_s1:
    save_edits = st.button("💾 Save edits")
with c_s2:
    recalc_now = st.button("↻ Recalc CLV/PNL")

def _apply_recalc(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    df["odds"] = pd.to_numeric(df.get("odds"), errors="coerce")
    df["stake"] = pd.to_numeric(df.get("stake"), errors="coerce")
    df["closing_odds"] = pd.to_numeric(df.get("closing_odds"), errors="coerce")

    # result normalize
    df["result"] = df["result"].fillna("").astype(str).str.upper().str.strip()
    df.loc[~df["result"].isin(["W", "L", "V", ""]), "result"] = ""

    # settled normalize
    df["settled"] = df["settled"].astype(bool)

    # CLV
    df["clv_pct"] = np.nan
    mask_clv = df["odds"].notna() & df["closing_odds"].notna() & (df["odds"] > 0) & (df["closing_odds"] > 0)
    df.loc[mask_clv, "clv_pct"] = (df.loc[mask_clv, "closing_odds"] - df.loc[mask_clv, "odds"]) / df.loc[mask_clv, "odds"]

    # PnL
    df["pnl"] = pd.to_numeric(df.get("pnl", 0.0), errors="coerce").fillna(0.0)
    mask_settle = df["settled"] & df["odds"].notna() & df["stake"].notna() & df["result"].isin(["W", "L", "V"])
    df.loc[mask_settle, "pnl"] = df.loc[mask_settle].apply(
        lambda r: float(settle_pnl(r["result"], float(r["odds"]), float(r["stake"]))),
        axis=1
    )

    # EV bins (based on ev_worst_pick if present)
    if "ev_worst_pick" in df.columns:
        df["ev_worst_pick"] = pd.to_numeric(df["ev_worst_pick"], errors="coerce")
        df["ev_bin"] = df["ev_worst_pick"].apply(ev_bin)
    else:
        df["ev_bin"] = "NA"

    return df

if recalc_now:
    # merge edited back into df_hist
    df_tmp = df_hist.copy()
    for col in edited.columns:
        df_tmp[col] = edited[col]
    df_tmp = _apply_recalc(df_tmp)

    df_tmp.to_csv(selected_file, index=False)
    st.success("Recalc OK & saved.")
    st.rerun()

if save_edits:
    df_tmp = df_hist.copy()
    for col in edited.columns:
        df_tmp[col] = edited[col]
    df_tmp = _apply_recalc(df_tmp)

    df_tmp.to_csv(selected_file, index=False)
    st.success("Saved.")
    st.rerun()

# Quick settle buttons by row index
st.markdown("### ✅ Quick Settle (κουμπιά)")
idx = st.number_input("Row index to settle", min_value=0, max_value=max(0, len(df_hist)-1), value=0, step=1)

q1, q2, q3, q4 = st.columns([1, 1, 1, 2])
with q1:
    btn_w = st.button("WIN (W)")
with q2:
    btn_l = st.button("LOSS (L)")
with q3:
    btn_v = st.button("VOID (V)")
with q4:
    st.caption("Πριν πατήσεις, βάλε closing_odds/played/include_in_eval/roi_included στο editor αν χρειάζεται.")

def _quick_settle(code: str):
    df_tmp = pd.read_csv(selected_file)
    # ensure cols
    for col, default in need_cols.items():
        if col not in df_tmp.columns:
            df_tmp[col] = default

    df_tmp.at[int(idx), "settled"] = True
    df_tmp.at[int(idx), "result"] = code

    df_tmp = _apply_recalc(df_tmp)
    df_tmp.to_csv(selected_file, index=False)
    st.success(f"Settled row {idx}: {code}")
    st.rerun()

if btn_w:
    _quick_settle("W")
if btn_l:
    _quick_settle("L")
if btn_v:
    _quick_settle("V")


# =========================
# DASHBOARD (CLV / EV BINS / EQUITY / DRAWDOWN)
# =========================
st.divider()
st.subheader("📊 Research Dashboard (selected run)")

df_dash = pd.read_csv(selected_file)
for col, default in need_cols.items():
    if col not in df_dash.columns:
        df_dash[col] = default

df_dash["odds"] = pd.to_numeric(df_dash.get("odds"), errors="coerce")
df_dash["stake"] = pd.to_numeric(df_dash.get("stake"), errors="coerce")
df_dash["pnl"] = pd.to_numeric(df_dash.get("pnl"), errors="coerce")
df_dash["ev_worst_pick"] = pd.to_numeric(df_dash.get("ev_worst_pick"), errors="coerce")
df_dash["clv_pct"] = pd.to_numeric(df_dash.get("clv_pct"), errors="coerce")

df_dash["settled"] = df_dash["settled"].astype(bool)
df_dash["played"] = df_dash["played"].astype(bool)
df_dash["include_in_eval"] = df_dash["include_in_eval"].astype(bool)
df_dash["roi_included"] = df_dash["roi_included"].astype(bool)

# sort timeline
if "kickoff_local" in df_dash.columns:
    df_sorted = df_dash.sort_values("kickoff_local").reset_index(drop=True)
else:
    df_sorted = df_dash.reset_index(drop=True)

# filters for evaluation
eval_mask = df_sorted["settled"] & df_sorted["played"] & df_sorted["include_in_eval"] & df_sorted["roi_included"]
bets_mask = df_sorted["pick"].astype(str).fillna("").ne("NO BET") & df_sorted["odds"].notna()

settled_eval = df_sorted[eval_mask].copy()
bets_all = df_sorted[bets_mask].copy()

# equity curve based on eval PnL
df_sorted["pnl_eval"] = np.where(eval_mask, df_sorted["pnl"].fillna(0.0), 0.0)
df_sorted["cum_pnl"] = df_sorted["pnl_eval"].cumsum()
df_sorted["equity"] = float(bankroll) + df_sorted["cum_pnl"]
df_sorted["equity_peak"] = df_sorted["equity"].cummax()
df_sorted["drawdown"] = df_sorted["equity"] - df_sorted["equity_peak"]
df_sorted["drawdown_pct"] = np.where(df_sorted["equity_peak"] > 0, df_sorted["drawdown"] / df_sorted["equity_peak"], 0.0)

mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
mcol1.metric("Bets (all)", f"{len(bets_all)}")
mcol2.metric("Settled+Played+Included (eval)", f"{len(settled_eval)}")
mcol3.metric("Total PnL eval (€)", f"{settled_eval['pnl_eval'].sum():.2f}")
mcol4.metric("Equity (€)", f"{df_sorted['equity'].iloc[-1]:.2f}")
mcol5.metric("Max Drawdown (€)", f"{df_sorted['drawdown'].min():.2f}")

# CLV
if settled_eval["clv_pct"].notna().any():
    clv_mean = float(settled_eval["clv_pct"].dropna().mean())
    clv_med = float(settled_eval["clv_pct"].dropna().median())
    st.markdown("#### CLV (Closing Line Value)")
    st.write(f"Mean CLV%: **{clv_mean*100:.2f}%** | Median CLV%: **{clv_med*100:.2f}%**")
else:
    st.info("CLV: Δεν υπάρχουν ακόμα closing odds σε evaluated settled bets.")

# EV bins
st.markdown("#### EV bins (βάσει EV_worst_pick)")
if "ev_bin" not in df_sorted.columns:
    df_sorted["ev_bin"] = df_sorted["ev_worst_pick"].apply(ev_bin)

bins_src = df_sorted[bets_mask].copy()
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

# Charts
st.markdown("#### Equity curve / Drawdown")
chart_df = df_sorted.copy()
chart_df["kickoff_local"] = chart_df.get("kickoff_local", pd.Series(range(len(chart_df)))).astype(str)

c_eq, c_dd = st.columns(2)
with c_eq:
    st.line_chart(chart_df.set_index("kickoff_local")[["equity"]])
with c_dd:
    st.line_chart(chart_df.set_index("kickoff_local")[["drawdown"]])

st.caption("Σημείωση: equity/drawdown υπολογίζονται ΜΟΝΟ για settled+played+include_in_eval+roi_included.")