import streamlit as st
import requests, re, json
import pandas as pd
import numpy as np
from math import exp, factorial

st.set_page_config(page_title="HYBRID VALUE Engine", layout="wide")

# =========================
# Helpers
# =========================
def poisson_pmf(k: int, lam: float) -> float:
    return exp(-lam) * (lam**k) / factorial(k)

def outcome_probs(lam_home: float, lam_away: float, max_goals: int = 10):
    """
    Returns (p_home, p_draw, p_away, tail_home, tail_away)
    Tail mass folded into max_goals bucket to keep full probability mass.
    """
    ph = np.array([poisson_pmf(k, lam_home) for k in range(max_goals + 1)], dtype=float)
    pa = np.array([poisson_pmf(k, lam_away) for k in range(max_goals + 1)], dtype=float)

    tail_h = float(max(0.0, 1.0 - ph.sum()))
    tail_a = float(max(0.0, 1.0 - pa.sum()))
    ph[-1] += tail_h
    pa[-1] += tail_a

    p_home = p_draw = p_away = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = ph[i] * pa[j]
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p

    return float(p_home), float(p_draw), float(p_away), tail_h, tail_a

def normalize_probs_from_odds(o1, ox, o2):
    imp1, impx, imp2 = 1.0/o1, 1.0/ox, 1.0/o2
    overround = imp1 + impx + imp2
    return (imp1/overround, impx/overround, imp2/overround, overround)

def kelly_fractional(p: float, odds: float) -> float:
    """Full Kelly for decimal odds. Returns fraction of bankroll."""
    b = odds - 1.0
    q = 1.0 - p
    if b <= 0:
        return 0.0
    f = (b*p - q) / b
    return max(0.0, f)

def classify_edge(ev_mid: float, ev_worst: float):
    # Audit label based on your rule
    if ev_mid > 0 and ev_worst < 0:
        return "FRAGILE EDGE (εύθραυστο)"
    if ev_worst >= 0:
        return "STRONG EDGE (ισχυρό)"
    return "FAKE EDGE (ψεύτικο)"

def contains_execution_book(name: str) -> bool:
    s = (name or "").strip().lower()
    return ("stoix" in s) or ("bet365" in s) or (s == "365") or (" 365" in s) or ("365 " in s)

# =========================
# Understat pull (24h cache)
# =========================
@st.cache_data(ttl=86400)
def get_understat_teams_df(league: str, season: str) -> pd.DataFrame:
    url = f"https://understat.com/league/{league}/{season}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    html = r.text

    m = re.search(r"teamsData\s*=\s*JSON.parse\('([^']+)'\)", html)
    if not m:
        raise RuntimeError("Understat JSON block not found (teamsData).")

    data = json.loads(m.group(1).encode("utf-8").decode("unicode_escape"))

    rows = []
    for _, team in data.items():
        history = team["history"]
        home_matches = [h for h in history if h.get("h_a") == "h"]
        away_matches = [h for h in history if h.get("h_a") == "a"]

        rows.append({
            "team": team["title"],
            "home_matches": len(home_matches),
            "away_matches": len(away_matches),
            "home_xG": sum(float(h["xG"]) for h in home_matches),
            "home_xGA": sum(float(h["xGA"]) for h in home_matches),
            "away_xG": sum(float(h["xG"]) for h in away_matches),
            "away_xGA": sum(float(h["xGA"]) for h in away_matches),
        })

    df = pd.DataFrame(rows).sort_values("team").reset_index(drop=True)
    return df

def compute_league_avgs(df: pd.DataFrame):
    league_home_avg = df["home_xG"].sum() / df["home_matches"].sum()
    league_away_avg = df["away_xG"].sum() / df["away_matches"].sum()
    return float(league_home_avg), float(league_away_avg)

# =========================
# Session State (Run machine)
# =========================
def ensure_state():
    if "teams_df" not in st.session_state:
        st.session_state.teams_df = None
    if "run_active" not in st.session_state:
        st.session_state.run_active = False
    if "baseline_books" not in st.session_state:
        st.session_state.baseline_books = []
    if "prev_league_home_avg" not in st.session_state:
        st.session_state.prev_league_home_avg = None
    if "prev_league_away_avg" not in st.session_state:
        st.session_state.prev_league_away_avg = None
    if "run_log" not in st.session_state:
        st.session_state.run_log = []  # list of dict rows
    if "paper_log" not in st.session_state:
        st.session_state.paper_log = []  # results W/L/V with pnl
    if "run_id" not in st.session_state:
        st.session_state.run_id = 0

ensure_state()

# =========================
# UI
# =========================
st.title("HYBRID VALUE 500€ ENGINE (STRICT / Machine)")
st.caption(
    "Understat live (cache 24h) • Home/Away split • League avg cross-check κάθε ματς • "
    "Baseline 6–10 books (χωρίς Stoix/Bet365) • Execution μόνο Stoiximan/Bet365 • "
    "Paper ROI (χαρτί/δοκιμή) με W/L/V."
)

with st.sidebar:
    st.header("⚙ Model Controls (ρυθμίσεις)")
    bankroll = st.number_input("Bankroll (€)", min_value=1.0, value=500.0, step=10.0)

    # Modular sliders (όπως είπες Β)
    delta = st.slider("Delta (μείωση worst-case)", 0.00, 0.03, 0.02, 0.005)
    ev_worst_min = st.slider("EV_worst_min (κατώφλι worst-case EV)", -0.02, 0.02, 0.00, 0.001)

    market_w = st.slider("Market weight (βάρος αγοράς)", 0.0, 1.0, 0.7, 0.05)
    xg_w = 1.0 - market_w

    kelly_frac = st.slider("Kelly fraction (ποσοστό Kelly)", 0.0, 1.0, 0.25, 0.05)
    cap_pct = st.slider("Cap % bankroll (ταβάνι %)", 0.01, 0.10, 0.05, 0.01)

    max_goals = st.slider("Poisson max goals", 7, 12, 10, 1)

    st.divider()
    st.header("League (Understat)")
    league = st.text_input("League code", value="La_liga")
    season = st.text_input("Season", value="2024")

    if st.button("Load / Refresh Understat (cache 24h)"):
        try:
            st.session_state.teams_df = get_understat_teams_df(league, season)
            st.success("Loaded Understat data.")
        except Exception as e:
            st.error(f"Load failed: {e}")

# Ensure we have league data
if st.session_state.teams_df is None:
    st.info("Πάτα στο sidebar: **Load / Refresh Understat** για να φορτώσει δεδομένα.")
    st.stop()

df = st.session_state.teams_df
teams = df["team"].tolist()

# =========================
# RUN CONTROL
# =========================
st.subheader("RUN CONTROL (State machine)")

rc1, rc2, rc3, rc4 = st.columns([1,1,1,2])
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
    st.session_state.baseline_books = []
    st.session_state.run_id += 1
    st.success("Run started. Set baseline books below.")

if end_run:
    st.session_state.run_active = False
    st.success("Run ended. (You can still edit paper results below.)")

if not st.session_state.run_active and not st.session_state.run_log:
    st.warning("Δεν υπάρχει ενεργό run. Πάτα **Start Run**.")
    st.stop()

# =========================
# BASELINE BOOK NAMES (set once per run)
# =========================
st.subheader("Baseline books (ονόματα) — ΜΟΝΟ μία φορά ανά run")
st.caption("Επιτρέπονται **6–10** books. **Stoiximan/Bet365 απαγορεύονται** στο baseline.")

if not st.session_state.baseline_books:
    st.session_state.baseline_books = ["Book1","Book2","Book3","Book4","Book5","Book6"]

bcol1, bcol2 = st.columns([2,1])
with bcol1:
    n_books = st.number_input("Πόσα baseline books;", min_value=6, max_value=10, value=len(st.session_state.baseline_books), step=1)
with bcol2:
    if st.button("Resize baseline list"):
        cur = st.session_state.baseline_books
        if len(cur) < n_books:
            cur = cur + [f"Book{len(cur)+i+1}" for i in range(n_books - len(cur))]
        else:
            cur = cur[:n_books]
        st.session_state.baseline_books = cur

books_inputs = []
for i in range(len(st.session_state.baseline_books)):
    books_inputs.append(st.text_input(f"Baseline book #{i+1}", value=st.session_state.baseline_books[i], key=f"bn_{i}"))

if st.button("Lock baseline book names"):
    # Validate
    if len(books_inputs) < 6 or len(books_inputs) > 10:
        st.error("STOP: baseline books must be 6–10.")
        st.stop()
    if any(contains_execution_book(x) for x in books_inputs):
        st.error("STOP: Stoiximan/Bet365 cannot be used in baseline book names.")
        st.stop()
    st.session_state.baseline_books = books_inputs
    st.success("Baseline book names locked for this run.")

# Quick validation display
if any(contains_execution_book(x) for x in st.session_state.baseline_books):
    st.error("ERROR: baseline contains Stoiximan/Bet365. Fix names and lock again.")
    st.stop()

# =========================
# MATCH INPUT (per match)
# =========================
st.divider()
st.subheader("New Match (κάθε ματς)")
league_home_avg, league_away_avg = compute_league_avgs(df)

# STEP 2A cross-check (MANDATORY per match)
prev_h = st.session_state.prev_league_home_avg
prev_a = st.session_state.prev_league_away_avg
diff_h = None if prev_h is None else abs(league_home_avg - prev_h)
diff_a = None if prev_a is None else abs(league_away_avg - prev_a)

c1, c2, c3, c4 = st.columns(4)
c1.metric("League home avg xG", f"{league_home_avg:.9f}")
c2.metric("League away avg xG", f"{league_away_avg:.9f}")
c3.metric("Δ home vs prev", "—" if diff_h is None else f"{diff_h:.9f}")
c4.metric("Δ away vs prev", "—" if diff_a is None else f"{diff_a:.9f}")

if diff_h is not None and (diff_h > 0.001 or diff_a > 0.001):
    st.error("RUN STATUS: NO RUN – LEAGUE AVG MISMATCH (>|0.001|). STOP RUN.")
    st.session_state.run_active = False
    st.stop()

st.caption("PASS cross-check (ή πρώτο ματς του run).")

m1, m2, m3 = st.columns([1,1,1])
home_team = m1.selectbox("Home team", teams, index=0)
away_team = m2.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)
match_label = m3.text_input("Match label", value=f"{home_team} – {away_team}")

st.markdown("### Baseline odds (ανά ματς) — 6–10 books, equal weight")
st.caption("Εδώ βάζεις ΜΟΝΟ τις αποδόσεις. Τα ονόματα baseline books έχουν κλειδώσει.")

baseline_odds = []
for i, book in enumerate(st.session_state.baseline_books):
    r = st.columns([1.2, 1, 1, 1])
    r[0].markdown(f"**{book}**")
    o1 = r[1].number_input("1", min_value=1.01, value=2.00, step=0.01, key=f"bo1_{st.session_state.run_id}_{len(st.session_state.run_log)}_{i}")
    ox = r[2].number_input("X", min_value=1.01, value=3.20, step=0.01, key=f"box_{st.session_state.run_id}_{len(st.session_state.run_log)}_{i}")
    o2 = r[3].number_input("2", min_value=1.01, value=3.80, step=0.01, key=f"bo2_{st.session_state.run_id}_{len(st.session_state.run_log)}_{i}")
    baseline_odds.append((float(o1), float(ox), float(o2)))

st.markdown("### Stoiximan / Bet365 odds (execution only)")
scol = st.columns(6)
stoix_1 = float(scol[0].number_input("Stoiximan 1", min_value=1.01, value=2.00, step=0.01))
stoix_x = float(scol[1].number_input("Stoiximan X", min_value=1.01, value=3.20, step=0.01))
stoix_2 = float(scol[2].number_input("Stoiximan 2", min_value=1.01, value=3.80, step=0.01))
b365_1  = float(scol[3].number_input("Bet365 1", min_value=1.01, value=2.00, step=0.01))
b365_x  = float(scol[4].number_input("Bet365 X", min_value=1.01, value=3.20, step=0.01))
b365_2  = float(scol[5].number_input("Bet365 2", min_value=1.01, value=3.80, step=0.01))

run_match = st.button("RUN MATCH (STRICT)")

if run_match:
    # =========================
    # STEP 1: Market baseline
    # =========================
    avg_o1 = float(np.mean([x[0] for x in baseline_odds]))
    avg_ox = float(np.mean([x[1] for x in baseline_odds]))
    avg_o2 = float(np.mean([x[2] for x in baseline_odds]))
    p_m1, p_mx, p_m2, overround = normalize_probs_from_odds(avg_o1, avg_ox, avg_o2)

    # =========================
    # STEP 2B: Lambdas + Poisson (Home/Away split)
    # =========================
    home_row = df[df["team"] == home_team].iloc[0]
    away_row = df[df["team"] == away_team].iloc[0]

    # per match avgs
    h_att_avg = float(home_row["home_xG"] / home_row["home_matches"])
    h_def_avg = float(home_row["home_xGA"] / home_row["home_matches"])
    a_att_avg = float(away_row["away_xG"] / away_row["away_matches"])
    a_def_avg = float(away_row["away_xGA"] / away_row["away_matches"])

    # strengths
    home_attack = h_att_avg / league_home_avg
    home_def    = h_def_avg / league_away_avg
    away_attack = a_att_avg / league_away_avg
    away_def    = a_def_avg / league_home_avg

    lam_home = league_home_avg * home_attack * away_def
    lam_away = league_away_avg * away_attack * home_def

    p_x1, p_xx, p_x2, tail_h, tail_a = outcome_probs(lam_home, lam_away, max_goals=max_goals)

    # =========================
    # STEP 3: Hybrid
    # =========================
    p_t1 = market_w*p_m1 + xg_w*p_x1
    p_tx = market_w*p_mx + xg_w*p_xx
    p_t2 = market_w*p_m2 + xg_w*p_x2
    mass = p_t1 + p_tx + p_t2

    fair_1, fair_x, fair_2 = 1.0/p_t1, 1.0/p_tx, 1.0/p_t2
    dp1, dpx, dp2 = p_t1 - p_m1, p_tx - p_mx, p_t2 - p_m2

    # =========================
    # STEP 4: EV (Αναμενόμενη Αξία) Stoix/365
    # =========================
    def evs(p_true, o, delta):
        ev_mid = o*p_true - 1.0
        ev_worst = o*max(0.0, p_true - delta) - 1.0
        return float(ev_mid), float(ev_worst)

    best_1 = max(stoix_1, b365_1); best_1_book = "Stoiximan" if stoix_1 >= b365_1 else "Bet365"
    best_x = max(stoix_x, b365_x); best_x_book = "Stoiximan" if stoix_x >= b365_x else "Bet365"
    best_2 = max(stoix_2, b365_2); best_2_book = "Stoiximan" if stoix_2 >= b365_2 else "Bet365"

    ev1_mid, ev1_worst = evs(p_t1, best_1, delta)
    evx_mid, evx_worst = evs(p_tx, best_x, delta)
    ev2_mid, ev2_worst = evs(p_t2, best_2, delta)

    # =========================
    # STEP 5: Selection rules
    # =========================
    candidates = [
        ("1", best_1_book, best_1, p_t1, p_m1, p_x1, fair_1, dp1, ev1_mid, ev1_worst),
        ("X", best_x_book, best_x, p_tx, p_mx, p_xx, fair_x, dpx, evx_mid, evx_worst),
        ("2", best_2_book, best_2, p_t2, p_m2, p_x2, fair_2, dp2, ev2_mid, ev2_worst),
    ]
    passed = [c for c in candidates if c[9] >= ev_worst_min]
    pick = sorted(passed, key=lambda t: t[9], reverse=True)[0] if passed else None

    # =========================
    # STEP 6: Audit checks
    # =========================
    mass_ok = abs(mass - 1.0) <= 0.002

    market_best = max([("1", p_m1), ("X", p_mx), ("2", p_m2)], key=lambda t: t[1])[0]
    xg_best = max([("1", p_x1), ("X", p_xx), ("2", p_x2)], key=lambda t: t[1])[0]

    # Directional logic (κατεύθυνση): απλό flag, όχι ερμηνεία
    direction_ok = (market_best == xg_best) or (max(p_m1, p_mx, p_m2) - max(p_x1, p_xx, p_x2) <= 0.05)

    # =========================
    # STEP 7: Stake (Ποντάρισμα) Kelly 1/4 + cap
    # =========================
    stake = 0.0
    kelly_full = 0.0
    label = "NO BET"
    bet_out = ""
    bet_book = ""
    bet_odds = None
    bet_ev_mid = None
    bet_ev_worst = None

    if pick and mass_ok:
        out, book, odds, p_true, p_market, p_xg, fair, dp, ev_mid, ev_worst = pick
        label = classify_edge(ev_mid, ev_worst)

        if label != "FAKE EDGE (ψεύτικο)":
            kelly_full = kelly_fractional(p_true, odds)
            stake = bankroll * (kelly_full * kelly_frac)
            stake = min(stake, bankroll * cap_pct)

            bet_out = out
            bet_book = book
            bet_odds = odds
            bet_ev_mid = ev_mid
            bet_ev_worst = ev_worst
        else:
            label = "NO BET"
    else:
        label = "NO BET"

    # =========================
    # OUTPUT (full per match)
    # =========================
    st.success(f"RUN MATCH COMPLETE: {match_label}")

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
    if label == "NO BET" or bet_odds is None:
        st.warning("NO BET")
    else:
        st.success(f"{label} → BET {bet_book} {bet_out} @ {bet_odds:.2f} | Stake €{stake:.2f}")
        st.caption(f"Kelly full={kelly_full:.6f} | Kelly applied={kelly_frac:.2f} | Cap={cap_pct:.2%}")

    # =========================
    # Append to run log
    # =========================
    run_row = {
        "match": match_label,
        "home_team": home_team,
        "away_team": away_team,
        "league_home_avg": league_home_avg,
        "league_away_avg": league_away_avg,
        "baseline_n": len(st.session_state.baseline_books),
        "avg_baseline_odds_1": avg_o1,
        "avg_baseline_odds_x": avg_ox,
        "avg_baseline_odds_2": avg_o2,
        "p_market_1": p_m1, "p_market_x": p_mx, "p_market_2": p_m2,
        "p_xg_1": p_x1, "p_xg_x": p_xx, "p_xg_2": p_x2,
        "p_true_1": p_t1, "p_true_x": p_tx, "p_true_2": p_t2,
        "delta": delta,
        "ev_worst_min": ev_worst_min,
        "pick": ("NO BET" if bet_odds is None else bet_out),
        "book": ("" if bet_odds is None else bet_book),
        "odds": (np.nan if bet_odds is None else bet_odds),
        "stake": (0.0 if bet_odds is None else stake),
        "ev_mid_pick": (np.nan if bet_ev_mid is None else bet_ev_mid),
        "ev_worst_pick": (np.nan if bet_ev_worst is None else bet_ev_worst),
        "label": label,
        "lambda_home": lam_home,
        "lambda_away": lam_away,
        "overround": overround
    }
    st.session_state.run_log.append(run_row)

    # Update cross-check state AFTER successful match run
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
    show["ev_worst_pick"] = show["ev_worst_pick"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
