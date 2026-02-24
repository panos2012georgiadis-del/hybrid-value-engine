import io
import numpy as np
import pandas as pd
import streamlit as st
from math import exp, factorial

st.set_page_config(page_title="HYBRID VALUE Engine (CSV Strict)", layout="wide")

# =========================
# STRICT HELPERS
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

def contains_execution_book(name: str) -> bool:
    s = (name or "").strip().lower()
    return ("stoix" in s) or ("bet365" in s) or (" 365" in s) or ("365 " in s) or (s == "365")

# =========================
# CSV LOADER (AUTO SEP)
# =========================
def _read_csv_autodetect(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    # sep=None + engine=python autodetects comma/semicolon/tab
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    STRICT required columns (aliases allowed):
      Team, xG, xGA, M
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()

        # TEAM
        if lc in ["team", "squad", "club", "team_name", "name"]:
            col_map[c] = "Team"

        # MATCHES
        if lc in ["m", "mp", "matches", "played", "games", "n", "apps"]:
            col_map[c] = "M"

        # xG FOR
        if lc in ["xg", "xg_for", "xgfor", "xg (for)", "xgfor."]:
            col_map[c] = "xG"

        # xG AGAINST
        if lc in ["xga", "xg_against", "xgagainst", "xg (against)", "xg_against."]:
            col_map[c] = "xGA"

        # common understat exports sometimes: "xG.1" etc — leave as is

    df = df.rename(columns=col_map)

    need = ["Team", "xG", "xGA", "M"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Need Team, xG, xGA, M (or aliases)."
        )

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

# =========================
# SESSION STATE
# =========================
def ensure_state():
    if "run_active" not in st.session_state:
        st.session_state.run_active = False
    if "run_id" not in st.session_state:
        st.session_state.run_id = 0

    if "baseline_books" not in st.session_state:
        st.session_state.baseline_books = []
    if "baseline_locked" not in st.session_state:
        st.session_state.baseline_locked = False

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

    if "run_log" not in st.session_state:
        st.session_state.run_log = []
    if "paper_log" not in st.session_state:
        st.session_state.paper_log = []

ensure_state()

# =========================
# UI HEADER
# =========================
st.title("HYBRID VALUE 500€ ENGINE (CSV STRICT)")
st.caption(
    "ΜΗΧΑΝΗ (machine) • Δεδομένα από CSV Home/Away • League avg = ΣxG/ΣM • "
    "Cross-check >0.001 STOP • Baseline 6–10 books (ίσος μέσος όρος) χωρίς Stoix/Bet365 • "
    "Execution μόνο Stoiximan/Bet365 • Full audit + Kelly 1/4 + cap • Paper ROI (W/L/V)"
)

# =========================
# SIDEBAR CONTROLS (MODULAR)
# =========================
with st.sidebar:
    st.header("⚙ Ρυθμίσεις")
    bankroll = st.number_input("Bankroll (€)", min_value=1.0, value=500.0, step=10.0)

    delta = st.slider("Delta (μείωση worst-case)", 0.00, 0.03, 0.02, 0.005)
    ev_worst_min = st.slider("EV_worst_min (κατώφλι worst-case EV)", -0.02, 0.02, 0.00, 0.001)

    market_w = st.slider("Market weight (βάρος αγοράς)", 0.0, 1.0, 0.7, 0.05)
    xg_w = 1.0 - market_w

    kelly_frac = st.slider("Kelly fraction (ποσοστό Kelly)", 0.0, 1.0, 0.25, 0.05)
    cap_pct = st.slider("Cap % bankroll (ταβάνι %)", 0.01, 0.10, 0.05, 0.01)

    max_goals = st.slider("Poisson max goals", 7, 12, 10, 1)

    st.divider()
    st.header("📄 CSV Inputs (Home/Away)")

    home_file = st.file_uploader("Upload HOME CSV", type=["csv"], key="home_csv")
    away_file = st.file_uploader("Upload AWAY CSV", type=["csv"], key="away_csv")

    if home_file and away_file:
        try:
            home_df = _normalize_cols(_read_csv_autodetect(home_file))
            away_df = _normalize_cols(_read_csv_autodetect(away_file))

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

# Teams list from union
teams = sorted(set(home_df["Team"].tolist()) | set(away_df["Team"].tolist()))

# =========================
# RUN CONTROL
# =========================
st.subheader("RUN CONTROL (πολλά ματς στο ίδιο run)")

rc1, rc2, rc3, rc4 = st.columns([1, 1, 1, 2])
with rc1:
    start_run = st.button("Start Run")
with rc2:
    end_run = st.button("End Run")
with rc3:
    reset_all = st.button("Reset All")
with rc4:
    st.write(f"Run active: **{st.session_state.run_active}** | Run ID: **{st.session_state.run_id}**")

if reset_all:
    st.session_state.run_active = False
    st.session_state.baseline_books = []
    st.session_state.baseline_locked = False
    st.session_state.prev_league_home_avg = None
    st.session_state.prev_league_away_avg = None
    st.session_state.run_log = []
    st.session_state.paper_log = []
    st.session_state.run_id += 1
    st.success("Reset complete.")

if start_run:
    st.session_state.run_active = True
    st.session_state.prev_league_home_avg = None
    st.session_state.prev_league_away_avg = None
    st.session_state.run_log = []
    st.session_state.paper_log = []
    st.session_state.baseline_books = ["Book1","Book2","Book3","Book4","Book5","Book6"]
    st.session_state.baseline_locked = False
    st.session_state.run_id += 1
    st.success("Run started. Κλείδωσε baseline book names (6–10).")

if end_run:
    st.session_state.run_active = False
    st.success("Run ended. (Μπορείς να συνεχίσεις να περνάς W/L/V στο paper.)")

if not st.session_state.run_active and not st.session_state.run_log:
    st.info("Πάτα **Start Run** για να ξεκινήσεις νέο run.")
    st.stop()

# =========================
# BASELINE BOOKS (NAMES ONCE PER RUN)
# =========================
st.subheader("Baseline books (ονόματα) — μία φορά ανά run")
st.caption("6–10 books, ίσος μέσος όρος. Stoiximan/Bet365 ΑΠΑΓΟΡΕΥΟΝΤΑΙ στο baseline.")

b1, b2, b3 = st.columns([1, 1, 2])
with b1:
    n_books = st.number_input("N baseline books", min_value=6, max_value=10, value=len(st.session_state.baseline_books), step=1)
with b2:
    if st.button("Resize list"):
        cur = st.session_state.baseline_books
        if len(cur) < n_books:
            cur = cur + [f"Book{len(cur)+i+1}" for i in range(n_books - len(cur))]
        else:
            cur = cur[:n_books]
        st.session_state.baseline_books = cur
with b3:
    st.write(f"Locked: **{st.session_state.baseline_locked}**")

books_inputs = []
for i in range(len(st.session_state.baseline_books)):
    books_inputs.append(
        st.text_input(
            f"Baseline book #{i+1}",
            value=st.session_state.baseline_books[i],
            key=f"bn_{st.session_state.run_id}_{i}"
        )
    )

c_lock, c_unlock = st.columns(2)
with c_lock:
    lock_btn = st.button("Lock baseline book names")
with c_unlock:
    unlock_btn = st.button("Unlock (edit names)")

if unlock_btn:
    st.session_state.baseline_locked = False
    st.warning("Baseline unlocked. (Θα χρειαστεί να ξανακλειδώσεις.)")

if lock_btn:
    if len(books_inputs) < 6 or len(books_inputs) > 10:
        st.error("STOP: baseline books must be 6–10.")
        st.stop()
    if any(contains_execution_book(x) for x in books_inputs):
        st.error("STOP: Stoiximan/Bet365 cannot be used in baseline book names.")
        st.stop()
    st.session_state.baseline_books = books_inputs
    st.session_state.baseline_locked = True
    st.success("Baseline book names LOCKED for this run.")

if not st.session_state.baseline_locked:
    st.warning("Πρέπει να πατήσεις **Lock baseline book names** πριν τρέξεις ματς.")
    st.stop()

baseline_books = st.session_state.baseline_books

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

# Cross-check rule: STOP if mismatch
if diff_h is not None and (diff_h > 0.001 or diff_a > 0.001):
    st.error("RUN STATUS: NO RUN – LEAGUE AVG MISMATCH (>|0.001|). STOP.")
    st.session_state.run_active = False
    st.stop()

st.caption("PASS (ή πρώτο ματς του run).")

# =========================
# MATCH INPUT
# =========================
st.subheader("New match (ανά ματς)")

mc1, mc2, mc3 = st.columns([1, 1, 1.2])
home_team = mc1.selectbox("Home team", teams, index=0)
away_team = mc2.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)
match_label = mc3.text_input("Match label", value=f"{home_team} – {away_team}")

# Ensure teams exist in correct tables
if home_team not in set(home_df["Team"]) or away_team not in set(away_df["Team"]):
    st.error("STOP: Home team must exist in HOME CSV and Away team must exist in AWAY CSV.")
    st.stop()

st.markdown("### 1) Market baseline odds (6–10 books) — αποδόσεις ανά ματς")
baseline_odds = []
for i, book in enumerate(baseline_books):
    r = st.columns([1.4, 1, 1, 1])
    r[0].markdown(f"**{book}**")
    o1 = r[1].number_input("1", min_value=1.01, value=2.00, step=0.01, key=f"bo1_{st.session_state.run_id}_{len(st.session_state.run_log)}_{i}")
    ox = r[2].number_input("X", min_value=1.01, value=3.20, step=0.01, key=f"box_{st.session_state.run_id}_{len(st.session_state.run_log)}_{i}")
    o2 = r[3].number_input("2", min_value=1.01, value=3.80, step=0.01, key=f"bo2_{st.session_state.run_id}_{len(st.session_state.run_log)}_{i}")
    baseline_odds.append((float(o1), float(ox), float(o2)))

st.markdown("### 4) Stoiximan / Bet365 odds (execution only)")
scol = st.columns(6)
stoix_1 = float(scol[0].number_input("Stoiximan 1", min_value=1.01, value=2.00, step=0.01))
stoix_x = float(scol[1].number_input("Stoiximan X", min_value=1.01, value=3.20, step=0.01))
stoix_2 = float(scol[2].number_input("Stoiximan 2", min_value=1.01, value=3.80, step=0.01))
b365_1  = float(scol[3].number_input("Bet365 1", min_value=1.01, value=2.00, step=0.01))
b365_x  = float(scol[4].number_input("Bet365 X", min_value=1.01, value=3.20, step=0.01))
b365_2  = float(scol[5].number_input("Bet365 2", min_value=1.01, value=3.80, step=0.01))

run_match = st.button("RUN MATCH (STRICT)")

if run_match:
    # -------------------------
    # STEP 1: Market baseline
    # -------------------------
    avg_o1 = float(np.mean([x[0] for x in baseline_odds]))
    avg_ox = float(np.mean([x[1] for x in baseline_odds]))
    avg_o2 = float(np.mean([x[2] for x in baseline_odds]))
    p_m1, p_mx, p_m2, overround = normalize_probs_from_odds(avg_o1, avg_ox, avg_o2)

    # -------------------------
    # STEP 2B: Poisson lambdas from CSV split
    # We require per-team HOME: xG & xGA per match, AWAY: xG & xGA per match.
    # -------------------------
    hr = home_df[home_df["Team"] == home_team].iloc[0]
    ar = away_df[away_df["Team"] == away_team].iloc[0]

    h_att_avg = float(hr["xG"] / hr["M"])
    h_def_avg = float(hr["xGA"] / hr["M"])
    a_att_avg = float(ar["xG"] / ar["M"])
    a_def_avg = float(ar["xGA"] / ar["M"])

    # Strength ratios (deterministic)
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
    # STEP 4: EV (Stoix/Bet365 only)
    # -------------------------
    def evs(p_true, o, d):
        ev_mid = o*p_true - 1.0
        ev_worst = o*max(0.0, p_true - d) - 1.0
        return float(ev_mid), float(ev_worst)

    best_1 = max(stoix_1, b365_1); best_1_book = "Stoiximan" if stoix_1 >= b365_1 else "Bet365"
    best_x = max(stoix_x, b365_x); best_x_book = "Stoiximan" if stoix_x >= b365_x else "Bet365"
    best_2 = max(stoix_2, b365_2); best_2_book = "Stoiximan" if stoix_2 >= b365_2 else "Bet365"

    ev1_mid, ev1_worst = evs(p_t1, best_1, delta)
    evx_mid, evx_worst = evs(p_tx, best_x, delta)
    ev2_mid, ev2_worst = evs(p_t2, best_2, delta)

    # -------------------------
    # STEP 5: Selection rules
    # -------------------------
    candidates = [
        ("1", best_1_book, best_1, p_t1, p_m1, p_x1, fair_1, dp1, ev1_mid, ev1_worst),
        ("X", best_x_book, best_x, p_tx, p_mx, p_xx, fair_x, dpx, evx_mid, evx_worst),
        ("2", best_2_book, best_2, p_t2, p_m2, p_x2, fair_2, dp2, ev2_mid, ev2_worst),
    ]
    passed = [c for c in candidates if c[9] >= ev_worst_min]
    pick = sorted(passed, key=lambda t: t[9], reverse=True)[0] if passed else None

    # -------------------------
    # STEP 6: AUDIT checks (deterministic flags)
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
         "BestOdds":best_1, "Book":best_1_book, "EV_mid":ev1_mid, "EV_worst":ev1_worst},
        {"Outcome":"X", "p_market":p_mx, "p_xG":p_xx, "p_true":p_tx, "Fair":fair_x, "Δp":dpx,
         "BestOdds":best_x, "Book":best_x_book, "EV_mid":evx_mid, "EV_worst":evx_worst},
        {"Outcome":"2", "p_market":p_m2, "p_xG":p_x2, "p_true":p_t2, "Fair":fair_2, "Δp":dp2,
         "BestOdds":best_2, "Book":best_2_book, "EV_mid":ev2_mid, "EV_worst":ev2_worst},
    ])
    disp = out_df.copy()
    for c in ["p_market","p_xG","p_true","Fair","Δp","EV_mid","EV_worst"]:
        disp[c] = disp[c].map(lambda x: f"{x:.6f}")
    disp["BestOdds"] = disp["BestOdds"].map(lambda x: f"{x:.2f}")
    st.dataframe(disp, use_container_width=True)

    st.subheader("AUDIT (έλεγχος)")
    st.write(f"Probability mass (μάζα πιθανοτήτων): {mass:.9f} → {'PASS' if mass_ok else 'FAIL'} (±0.002)")
    st.write(f"Directional logic (κατεύθυνση): market_best={market_best}, xG_best={xg_best} → {'PASS' if direction_ok else 'FLAG'}")
    st.write("EV Stability: EV_mid>0 αλλά EV_worst<0 ⇒ FRAGILE EDGE (εύθραυστο)")

    st.subheader("FINAL (τελικό)")
    if final_label == "NO BET" or np.isnan(bet_odds):
        st.warning("NO BET")
    else:
        st.success(f"{final_label} → BET {bet_book} {bet_out} @ {bet_odds:.2f} | Stake €{stake:.2f}")
        st.caption(f"Kelly full={kelly_full:.6f} | Kelly applied={kelly_frac:.2f} | Cap={cap_pct:.2%}")

    # -------------------------
    # Append to run log
    # -------------------------
    run_row = {
        "match": match_label,
        "home_team": home_team,
        "away_team": away_team,
        "league_home_avg": league_home_avg,
        "league_away_avg": league_away_avg,
        "baseline_n": len(baseline_books),
        "avg_baseline_odds_1": avg_o1,
        "avg_baseline_odds_x": avg_ox,
        "avg_baseline_odds_2": avg_o2,
        "p_market_1": p_m1, "p_market_x": p_mx, "p_market_2": p_m2,
        "p_xg_1": p_x1, "p_xg_x": p_xx, "p_xg_2": p_x2,
        "p_true_1": p_t1, "p_true_x": p_tx, "p_true_2": p_t2,
        "delta": delta,
        "ev_worst_min": ev_worst_min,
        "pick": ("NO BET" if np.isnan(bet_odds) else bet_out),
        "book": ("" if np.isnan(bet_odds) else bet_book),
        "odds": (np.nan if np.isnan(bet_odds) else bet_odds),
        "stake": (0.0 if np.isnan(bet_odds) else stake),
        "ev_mid_pick": bet_ev_mid,
        "ev_worst_pick": bet_ev_worst,
        "label": final_label,
        "lambda_home": lam_home,
        "lambda_away": lam_away,
        "overround": overround,
        "tails": {"home": tail_h, "away": tail_a},
    }
    st.session_state.run_log.append(run_row)

    # UPDATE CROSS-CHECK STATE AFTER SUCCESSFUL MATCH RUN
    st.session_state.prev_league_home_avg = league_home_avg
    st.session_state.prev_league_away_avg = league_away_avg

# =========================
# RUN LOG + PAPER ROI
# =========================
st.divider()
st.subheader("RUN LOG (όλα τα ματς του run)")

if st.session_state.run_log:
    log_df = pd.DataFrame(st.session_state.run_log)
    show_cols = ["match","pick","book","odds","stake","ev_worst_pick","label"]
    show = log_df[show_cols].copy()

    show["odds"] = show["odds"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    show["stake"] = show["stake"].map(lambda x: f"{x:.2f}")
    show["ev_worst_pick"] = show["ev_worst_pick"].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    st.dataframe(show, use_container_width=True)
else:
    st.info("Δεν έχουν τρέξει ματς ακόμα σε αυτό το run.")

st.subheader("PAPER MODE ROI (manual W/L/V)")
st.caption("W=Win (κέρδος), L=Loss (χάσιμο), V=Void (επιστροφή).")

if st.session_state.run_log:
    bets_only = [
        r for r in st.session_state.run_log
        if (isinstance(r.get("odds"), (float,int)) and not pd.isna(r["odds"]) and r.get("stake", 0) > 0)
    ]

    if not bets_only:
        st.info("Δεν υπάρχουν bets για να καταχωρήσεις αποτελέσματα.")
    else:
        rows = []
        for r in bets_only:
            # existing?
            existing = None
            for pr in st.session_state.paper_log:
                if pr.get("match") == r["match"] and pr.get("pick") == r["pick"] and pr.get("book") == r["book"]:
                    existing = pr
                    break
            rows.append({
                "match": r["match"],
                "pick": r["pick"],
                "book": r["book"],
                "odds": float(r["odds"]),
                "stake": float(r["stake"]),
                "result": (existing["result"] if existing else "—"),
            })

        paper_df = pd.DataFrame(rows)

        # UI per row
        for i in range(len(paper_df)):
            cols = st.columns([2.4, 0.6, 0.9, 0.8, 0.8, 0.8])
            cols[0].write(paper_df.loc[i, "match"])
            cols[1].write(paper_df.loc[i, "pick"])
            cols[2].write(paper_df.loc[i, "book"])
            cols[3].write(f'{paper_df.loc[i, "odds"]:.2f}')
            cols[4].write(f'{paper_df.loc[i, "stake"]:.2f}')
            sel = cols[5].selectbox(
                "Result",
                ["—","W","L","V"],
                index=["—","W","L","V"].index(paper_df.loc[i, "result"]),
                key=f"res_{st.session_state.run_id}_{i}"
            )
            paper_df.loc[i, "result"] = sel

        if st.button("Save paper results"):
            new_log = []
            for i in range(len(paper_df)):
                r = paper_df.loc[i]
                stake = float(r["stake"])
                odds = float(r["odds"])
                res = r["result"]

                if res == "W":
                    pnl = stake * (odds - 1.0)
                elif res == "L":
                    pnl = -stake
                elif res == "V":
                    pnl = 0.0
                else:
                    pnl = np.nan

                new_log.append({
                    "match": r["match"],
                    "pick": r["pick"],
                    "book": r["book"],
                    "odds": odds,
                    "stake": stake,
                    "result": res,
                    "pnl": pnl
                })

            st.session_state.paper_log = new_log
            st.success("Saved.")

        if st.session_state.paper_log:
            prdf = pd.DataFrame(st.session_state.paper_log)
            scored = prdf[prdf["result"].isin(["W","L","V"])].copy()

            if scored.empty:
                st.info("Δεν υπάρχουν ολοκληρωμένα αποτελέσματα ακόμη.")
            else:
                total_staked = float(scored["stake"].sum())
                total_pnl = float(scored["pnl"].sum())
                roi = (total_pnl / total_staked) if total_staked > 0 else 0.0

                wins = int((scored["result"] == "W").sum())
                losses = int((scored["result"] == "L").sum())
                voids = int((scored["result"] == "V").sum())
                n = len(scored)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Bets counted", f"{n}")
                c2.metric("PnL (€)", f"{total_pnl:.2f}")
                c3.metric("ROI (Απόδοση)", f"{roi*100:.2f}%")
                c4.metric("W/L/V", f"{wins}/{losses}/{voids}")

                st.dataframe(scored, use_container_width=True)

with st.expander("Diagnostics"):
    st.write({
        "run_active": st.session_state.run_active,
        "baseline_books": st.session_state.baseline_books,
        "baseline_locked": st.session_state.baseline_locked,
        "league_home_avg": st.session_state.league_home_avg,
        "league_away_avg": st.session_state.league_away_avg,
        "prev_league_avgs": [st.session_state.prev_league_home_avg, st.session_state.prev_league_away_avg],
        "run_log_rows": len(st.session_state.run_log),
        "paper_rows": len(st.session_state.paper_log),
        "home_csv_rows": len(home_df),
        "away_csv_rows": len(away_df),
    })
