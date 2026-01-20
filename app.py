import time
import json
import difflib
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import requests
from io import StringIO
from nba_api.stats.endpoints import leaguedashplayerstats, playergamelog

# ==========================
# CONFIG
# ==========================
SEASON = "2025-26"
GIST_FINAL = "final.csv"
GIST_OUT = "out.json"

LEAGUE_TIMEOUT = 20
GAMELOG_TIMEOUT = 12
GAMELOG_RETRIES = 2

# one-sided z scores
Z = {
    0.70: 0.5244005127080409,
    0.80: 0.8416212335729143,
    0.90: 1.2815515655446004,
}

MIN_SCALE = 0.70
MAX_SCALE = 1.30

# ==========================
# PAGE
# ==========================
st.set_page_config(layout="wide")
st.title("Confidence Floors (70/80/90%) — from Optimizer Projections")

st.markdown("""
This builds one-sided **confidence floors**:
> “With X% confidence, player gets at least ___”

Pulled from your optimizer `final.csv` + minutes-adjusted volatility from this season’s game logs.
""")

# ==========================
# HELPERS
# ==========================
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

def deaccent(s: str) -> str:
    return (
        unicodedata.normalize("NFKD", str(s))
        .encode("ascii", "ignore")
        .decode("ascii")
    )

def clean_name(s: str) -> str:
    s = deaccent(s).lower()
    s = s.replace(".", "").replace(",", "").replace("’", "'").replace("`", "'")
    return " ".join(s.split())

def strip_suffix(name: str) -> str:
    parts = clean_name(name).split()
    if parts and parts[-1] in SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts)

def parse_minutes_min(x):
    s = str(x)
    if ":" not in s:
        try:
            return float(s)
        except Exception:
            return np.nan
    m, sec = s.split(":")
    try:
        return float(m) + float(sec) / 60
    except Exception:
        return np.nan

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ==========================
# GIST
# ==========================
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GIST_ID = st.secrets["GIST_ID"]

def gh_headers():
    return {"Authorization": f"token {GITHUB_TOKEN}"}

def gist_get():
    r = requests.get(f"https://api.github.com/gists/{GIST_ID}", headers=gh_headers(), timeout=25)
    r.raise_for_status()
    return r.json()

def gist_read(filename: str):
    g = gist_get()
    if filename not in g.get("files", {}):
        return None
    f = g["files"][filename]
    if not f.get("truncated"):
        return f.get("content")
    rr = requests.get(f["raw_url"], timeout=25)
    rr.raise_for_status()
    return rr.text

# ==========================
# NBA DATA
# ==========================
@st.cache_data(ttl=900)
def league_player_df():
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        per_mode_detailed="PerGame",
        timeout=LEAGUE_TIMEOUT
    ).get_data_frames()[0]

    df = df[["PLAYER_ID", "PLAYER_NAME"]].copy()
    df.columns = ["PLAYER_ID", "NBA_Name"]
    df["NBA_Name_clean"] = df["NBA_Name"].apply(clean_name)
    df["NBA_Name_stripped"] = df["NBA_Name"].apply(strip_suffix)
    df["NBA_Last"] = df["NBA_Name_clean"].apply(lambda x: x.split()[-1] if isinstance(x, str) and x.split() else "")
    return df

def match_player_id(name: str, nba_df: pd.DataFrame):
    cn = clean_name(name)
    sn = strip_suffix(name)

    exact = nba_df[nba_df["NBA_Name_clean"] == cn]
    if not exact.empty:
        return int(exact.iloc[0]["PLAYER_ID"])

    exact2 = nba_df[nba_df["NBA_Name_stripped"] == sn]
    if not exact2.empty:
        return int(exact2.iloc[0]["PLAYER_ID"])

    parts = sn.split()
    if parts:
        last = parts[-1]
        cand = nba_df[nba_df["NBA_Last"] == last]
        if not cand.empty:
            best_id, best_score = None, 0.0
            for _, row in cand.iterrows():
                score = difflib.SequenceMatcher(None, sn, row["NBA_Name_stripped"]).ratio()
                if score > best_score:
                    best_score = score
                    best_id = int(row["PLAYER_ID"])
            if best_id is not None and best_score >= 0.75:
                return best_id

    hit = difflib.get_close_matches(cn, nba_df["NBA_Name_clean"].tolist(), n=1, cutoff=0.90)
    if hit:
        return int(nba_df[nba_df["NBA_Name_clean"] == hit[0]].iloc[0]["PLAYER_ID"])
    return None

@st.cache_data(ttl=3600)
def pull_pm_volatility(player_id: int, last_n: int):
    """
    Returns per-minute σ for PTS/REB/AST/FG3M and avg minutes in the sample.
    """
    last_err = None
    for attempt in range(1, GAMELOG_RETRIES + 1):
        try:
            gl = playergamelog.PlayerGameLog(
                player_id=int(player_id),
                season=SEASON,
                timeout=GAMELOG_TIMEOUT
            ).get_data_frames()[0]

            gl = gl.head(int(last_n)).copy()
            if gl.empty:
                return None

            gl["MIN_f"] = gl["MIN"].apply(parse_minutes_min)
            gl = gl[gl["MIN_f"].fillna(0) > 0].copy()
            if gl.empty:
                return None

            for c in ["PTS","REB","AST","FG3M"]:
                if c not in gl.columns:
                    gl[c] = 0.0
                gl[c] = pd.to_numeric(gl[c], errors="coerce").fillna(0.0)
                gl[f"PM_{c}"] = gl[c] / gl["MIN_f"]

            avg_min = float(gl["MIN_f"].mean())
            games = int(len(gl))
            out = {"avg_min": avg_min, "games": games}
            for c in ["PTS","REB","AST","FG3M"]:
                out[f"pm_sigma_{c}"] = float(gl[f"PM_{c}"].std(ddof=1)) if games >= 2 else 0.0
            return out
        except Exception as e:
            last_err = str(e)
            time.sleep(0.35 * attempt)
    return None

# ==========================
# LOAD PROJECTIONS FROM OPTIMIZER
# ==========================
final_text = gist_read(GIST_FINAL)
if not final_text:
    st.error("Could not load final.csv from the optimizer. Run Step B in optimizer first.")
    st.stop()

proj = pd.read_csv(StringIO(final_text))

needed = ["Name", "Salary", "Minutes", "PTS", "REB", "AST", "FG3M"]
missing = [c for c in needed if c not in proj.columns]
if missing:
    st.error(f"final.csv is missing required columns: {missing}")
    st.stop()

proj["Name_clean"] = proj["Name"].apply(clean_name)
proj["Salary"] = pd.to_numeric(proj["Salary"], errors="coerce")
proj["Minutes"] = pd.to_numeric(proj["Minutes"], errors="coerce")

# remove OUT players (if present)
out_set = set()
out_text = gist_read(GIST_OUT)
if out_text:
    try:
        out_flags = json.loads(out_text)
        out_set = {k for k, v in out_flags.items() if v}
    except Exception:
        out_set = set()

if out_set:
    proj = proj[~proj["Name_clean"].isin(out_set)].copy()

# ==========================
# SETTINGS
# ==========================
st.sidebar.subheader("Volatility")
last_n = st.sidebar.slider("Use last N games", 5, 30, 15, 1)
min_sigma_pm = st.sidebar.slider("Minimum per-minute σ floor", 0.0, 0.50, 0.02, 0.01)

st.sidebar.subheader("Selection Rules")
top50_salary = 50
top25_reb = 25
top25_ast = 25

# Build sets (unique players)
top_salary_df = proj.dropna(subset=["Salary"]).sort_values("Salary", ascending=False).head(top50_salary).copy()
top_reb_df = proj.dropna(subset=["REB"]).sort_values("REB", ascending=False).head(top25_reb).copy()
top_ast_df = proj.dropna(subset=["AST"]).sort_values("AST", ascending=False).head(top25_ast).copy()

need_names = set(top_salary_df["Name_clean"]) | set(top_reb_df["Name_clean"]) | set(top_ast_df["Name_clean"])

st.write(f"Will compute volatility for **{len(need_names)} unique players** (Top 50 salary + Top 25 REB + Top 25 AST).")

if not st.button("Build 70/80/90% floors"):
    st.dataframe(proj[["Name","Salary","Minutes","PTS","REB","AST","FG3M"]].head(30), use_container_width=True)
    st.stop()

nba_df = league_player_df()

# Precompute volatility for needed players only
vol_map = {}  # name_clean -> vol dict + player_id
progbar = st.progress(0, text="Pulling game logs for selected players...")

need_list = sorted(list(need_names))
for i, nm in enumerate(need_list):
    name = proj.loc[proj["Name_clean"] == nm, "Name"].iloc[0]
    progbar.progress((i + 1) / len(need_list), text=f"{name} ({i+1}/{len(need_list)})")

    pid = match_player_id(name, nba_df)
    if pid is None:
        vol_map[nm] = {"status": "ERR_NAME"}
        continue

    v = pull_pm_volatility(pid, last_n=last_n)
    time.sleep(0.12)  # tiny throttle helps cloud/rate-limits
    if v is None:
        vol_map[nm] = {"status": "ERR_LOGS"}
    else:
        v["status"] = "OK"
        v["player_id"] = pid
        vol_map[nm] = v

def make_floor_table(df_in: pd.DataFrame, stats: list, title: str):
    df = df_in.copy()
    rows = []
    for _, r in df.iterrows():
        nm = r["Name_clean"]
        v = vol_map.get(nm, {"status": "ERR"})
        if v.get("status") != "OK":
            rows.append({
                "Name": r["Name"],
                "Team": r.get("Team",""),
                "Opp": r.get("Opp",""),
                "Salary": r.get("Salary", np.nan),
                "ProjMin": r.get("Minutes", np.nan),
                "Status": v.get("status","ERR"),
            })
            continue

        mu_min = float(r["Minutes"]) if np.isfinite(r["Minutes"]) else 0.0
        if mu_min <= 0:
            rows.append({
                "Name": r["Name"],
                "Team": r.get("Team",""),
                "Opp": r.get("Opp",""),
                "Salary": r.get("Salary", np.nan),
                "ProjMin": r.get("Minutes", np.nan),
                "Status": "ERR_MIN",
            })
            continue

        avg_min = float(v.get("avg_min", mu_min))
        scale = clamp(mu_min / max(avg_min, 1e-6), MIN_SCALE, MAX_SCALE)

        out = {
            "Name": r["Name"],
            "Team": r.get("Team",""),
            "Opp": r.get("Opp",""),
            "Salary": r.get("Salary", np.nan),
            "ProjMin": round(mu_min, 1),
            "VolGames": int(v.get("games", 0)),
            "Status": "OK",
        }

        for stat in stats:
            mu = float(r[stat])
            pm_sigma = float(v.get(f"pm_sigma_{stat}", 0.0))
            pm_sigma = max(pm_sigma, float(min_sigma_pm))
            sigma_adj = pm_sigma * np.sqrt(mu_min) * scale

            out[f"{stat}_Proj"] = round(mu, 2)
            out[f"{stat}_SigmaAdj"] = round(sigma_adj, 2)

            for p, z in Z.items():
                out[f"{stat}_Floor{int(p*100)}"] = round(mu - z * sigma_adj, 2)

        rows.append(out)

    out_df = pd.DataFrame(rows)
    st.subheader(title)
    st.dataframe(out_df, use_container_width=True)

    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"Download {title} CSV",
        data=csv_bytes,
        file_name=re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower() + ".csv",
        mime="text/csv"
    )

# TABLE 1: Top 50 salary → PTS + FG3M floors
make_floor_table(
    top_salary_df,
    stats=["PTS","FG3M"],
    title="Top 50 Salary — Points + 3PM Floors (70/80/90%)"
)

# TABLE 2: Top 25 projected REB → REB floors
make_floor_table(
    top_reb_df,
    stats=["REB"],
    title="Top 25 Projected Rebounds — REB Floors (70/80/90%)"
)

# TABLE 3: Top 25 projected AST → AST floors
make_floor_table(
    top_ast_df,
    stats=["AST"],
    title="Top 25 Projected Assists — AST Floors (70/80/90%)"
)

st.caption("Floor70/80/90 are one-sided floors: projected to exceed that value about 70/80/90% of the time under this model.")
