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
LEAGUE_TIMEOUT = 20
GAMELOG_TIMEOUT = 12
GAMELOG_RETRIES = 2

Z_ONE_SIDED_90 = 1.2815515655446004

MIN_GAMES_FOR_VOL = 5
MIN_SCALE = 0.70
MAX_SCALE = 1.30

# Must match optimizer app gist filenames
GIST_FINAL = "final.csv"
GIST_OUT = "out.json"

# ==========================
# PAGE
# ==========================
st.set_page_config(layout="wide")
st.title("Props App — 90% Floors (Auto-load from Optimizer)")

st.markdown(
    """
This app auto-loads your projections from the **Optimizer app** so you only mark OUT players once.

Workflow:
1) In Optimizer app: Upload slate → mark OUT → Run Projections  
2) Here: Load `final.csv` + `out.json` from the same Gist → Build 90% floors
"""
)

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
# GIST READ
# ==========================
def gh_headers(token: str):
    return {"Authorization": f"token {token}"}

def gist_read(gist_id: str, token: str, filename: str):
    r = requests.get(f"https://api.github.com/gists/{gist_id}", headers=gh_headers(token), timeout=25)
    r.raise_for_status()
    g = r.json()
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

    keep = ["PLAYER_ID", "PLAYER_NAME"]
    df = df[keep].copy()
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
def player_rates_volatility(player_id: int, last_n: int):
    """
    Per-minute σ for PTS/REB/AST/FG3M from last_n games.
    Returns avg_min for scaling.
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
            time.sleep(0.4 * attempt)
    return None

# ==========================
# SIDEBAR
# ==========================
st.sidebar.subheader("Source")
source = st.sidebar.radio("Load from:", ["Optimizer Gist (recommended)", "Upload projections CSV"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Volatility Settings")
last_n = st.sidebar.slider("Use last N games", 5, 30, 15, 1)
min_games = st.sidebar.slider("Minimum games required", 3, 15, 5, 1)
min_sigma_pm = st.sidebar.slider("Minimum per-minute σ floor", 0.0, 0.50, 0.02, 0.01)

# ==========================
# LOAD PROJECTIONS
# ==========================
proj = None
out_set = set()

if source == "Optimizer Gist (recommended)":
    if "GITHUB_TOKEN" not in st.secrets or "GIST_ID" not in st.secrets:
        st.error("This mode requires GITHUB_TOKEN and GIST_ID in Streamlit secrets.")
        st.stop()

    final_text = gist_read(st.secrets["GIST_ID"], st.secrets["GITHUB_TOKEN"], GIST_FINAL)
    if not final_text:
        st.error("Could not load final.csv from Gist. Run projections in optimizer first.")
        st.stop()

    proj = pd.read_csv(StringIO(final_text))

    out_text = gist_read(st.secrets["GIST_ID"], st.secrets["GITHUB_TOKEN"], GIST_OUT)
    if out_text:
        try:
            out_flags = json.loads(out_text)
            out_set = {k for k, v in out_flags.items() if v}
        except Exception:
            out_set = set()

else:
    up = st.sidebar.file_uploader("Upload projections CSV", type="csv")
    if not up:
        st.info("Upload a projections CSV to begin.")
        st.stop()
    proj_text = up.getvalue().decode("utf-8", errors="ignore")
    proj = pd.read_csv(StringIO(proj_text))

# Validate needed columns
needed = ["Name", "Minutes", "PTS", "REB", "AST", "FG3M"]
missing = [c for c in needed if c not in proj.columns]
if missing:
    st.error(f"Projections file missing required columns: {missing}")
    st.stop()

# Remove OUT players if they exist in projection file
proj["Name_clean"] = proj["Name"].apply(clean_name)
if out_set:
    proj = proj[~proj["Name_clean"].isin(out_set)].copy()

proj["Minutes"] = pd.to_numeric(proj["Minutes"], errors="coerce").fillna(0.0)

st.subheader("Projections Loaded")
st.dataframe(proj[["Name","Team","Opp","PrimaryPos","Salary","Minutes","PTS","REB","AST","FG3M"]].head(50), use_container_width=True)

# ==========================
# BUILD FLOORS
# ==========================
if not st.button("Build 90% floors for slate"):
    st.stop()

nba_df = league_player_df()

rows = []
progbar = st.progress(0, text="Computing minutes-adjusted volatility...")

for i, r in proj.iterrows():
    name = str(r["Name"])
    progbar.progress((i + 1) / len(proj), text=f"{name} ({i+1}/{len(proj)})")

    pid = match_player_id(name, nba_df)
    if pid is None:
        rows.append({"Name": name, "Status": "ERR_NAME", "Notes": "Could not match name"})
        continue

    mu_min = float(r["Minutes"]) if np.isfinite(r["Minutes"]) else 0.0
    if mu_min <= 0:
        rows.append({"Name": name, "Status": "ERR_MIN", "Notes": "No projected minutes"})
        continue

    # mean projections from optimizer app
    mu = {
        "PTS": float(r["PTS"]),
        "REB": float(r["REB"]),
        "AST": float(r["AST"]),
        "FG3M": float(r["FG3M"]),
    }

    vol = player_rates_volatility(pid, last_n=last_n)
    if vol is None or vol.get("games", 0) < int(min_games):
        rows.append({"Name": name, "Status": "ERR_LOGS", "Notes": f"Need >= {min_games} games for σ"})
        continue

    avg_min_sample = float(vol["avg_min"]) if vol.get("avg_min") else mu_min
    scale = clamp(mu_min / max(avg_min_sample, 1e-6), MIN_SCALE, MAX_SCALE)

    out = {
        "Name": name,
        "Team": r.get("Team",""),
        "Opp": r.get("Opp",""),
        "PrimaryPos": r.get("PrimaryPos",""),
        "Salary": r.get("Salary", np.nan),
        "ProjMin": round(mu_min, 2),
        "VolGames": int(vol["games"]),
        "Status": "OK",
        "Notes": f"VOL(last{last_n}) scale={scale:.2f}"
    }

    for stat in ["PTS","REB","AST","FG3M"]:
        pm_sigma = float(vol.get(f"pm_sigma_{stat}", 0.0))
        pm_sigma = max(pm_sigma, float(min_sigma_pm))

        # minutes-adjusted σ: per-minute σ scaled by sqrt(minutes) and role-change scale
        sigma = pm_sigma * np.sqrt(mu_min) * scale

        floor90 = float(mu[stat]) - Z_ONE_SIDED_90 * float(sigma)

        out[f"{stat}_Proj"] = round(float(mu[stat]), 2)
        out[f"{stat}_SigmaAdj"] = round(float(sigma), 2)
        out[f"{stat}_Floor90"] = round(float(floor90), 2)

    rows.append(out)

props = pd.DataFrame(rows)

st.subheader("90% Floors (Auto from Optimizer Projections)")
st.dataframe(props, use_container_width=True)

csv_bytes = props.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download floors CSV",
    data=csv_bytes,
    file_name="props_floor90_from_optimizer.csv",
    mime="text/csv"
)

st.caption(
    "These floors use your optimizer projections as the mean (μ) and minutes-adjusted volatility (σ) from recent games."
)

