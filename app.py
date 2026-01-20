import time
import json
import difflib
import unicodedata
import re
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import requests

from nba_api.stats.endpoints import leaguedashplayerstats, playergamelog

# ==========================
# CONFIG
# ==========================
SEASON = "2025-26"
LEAGUE_TIMEOUT = 20
GAMELOG_TIMEOUT = 12
GAMELOG_RETRIES = 2

# One-sided 90% z-score: P(X >= floor)=0.90
Z_ONE_SIDED_90 = 1.2815515655446004

# Minimum game sample to compute σ reliably
MIN_GAMES_FOR_VOL = 5

# If projected minutes are bumped way up/down, cap scaling so we don't go crazy
MIN_SCALE = 0.70
MAX_SCALE = 1.30

# ==========================
# PAGE
# ==========================
st.set_page_config(layout="wide")
st.title("DK Slate Props — 90% Confidence Floors (Minutes-Adjusted Volatility)")

st.markdown(
    """
Upload your **DraftKings salary slate CSV** and this app will:

1) Pull **this season** player game logs (NBA API)  
2) Build **minutes-adjusted volatility** per stat (PTS/REB/AST/3PM)  
3) Output a **90% floor** for each stat for each player on the slate  

**Floor90** is a *one-sided* 90% floor:
> a number the player is projected to exceed about **90%** of the time under this model.

Minutes-adjusted = volatility scales with projected minutes (so injury-minute bumps matter).
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

# DK FP not needed in this app, but we keep this for possible extensions
def dk_fp_from_stats_row(r):
    pts = float(r["PTS"])
    reb = float(r["REB"])
    ast = float(r["AST"])
    stl = float(r["STL"])
    blk = float(r["BLK"])
    tov = float(r["TOV"])
    fg3m = float(r["FG3M"])

    fp = pts + 1.25*reb + 1.5*ast + 2.0*stl + 2.0*blk - 0.5*tov + 0.5*fg3m
    cats = sum([pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10])
    if cats >= 2:
        fp += 1.5
    if cats >= 3:
        fp += 3.0
    return fp

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

    keep = ["PLAYER_ID", "PLAYER_NAME", "MIN", "PTS", "REB", "AST", "FG3M"]
    df = df[keep].copy()
    df.columns = ["PLAYER_ID", "NBA_Name", "MIN", "PTS", "REB", "AST", "FG3M"]

    df["NBA_Name_clean"] = df["NBA_Name"].apply(clean_name)
    df["NBA_Name_stripped"] = df["NBA_Name"].apply(strip_suffix)
    df["NBA_Last"] = df["NBA_Name_clean"].apply(lambda x: x.split()[-1] if isinstance(x, str) and x.split() else "")
    return df

def match_player_row(name: str, nba_df: pd.DataFrame):
    cn = clean_name(name)
    sn = strip_suffix(name)

    exact = nba_df[nba_df["NBA_Name_clean"] == cn]
    if not exact.empty:
        return exact.iloc[0]

    exact2 = nba_df[nba_df["NBA_Name_stripped"] == sn]
    if not exact2.empty:
        return exact2.iloc[0]

    parts = sn.split()
    if parts:
        last = parts[-1]
        cand = nba_df[nba_df["NBA_Last"] == last]
        if not cand.empty:
            best_row, best_score = None, 0.0
            for _, row in cand.iterrows():
                score = difflib.SequenceMatcher(None, sn, row["NBA_Name_stripped"]).ratio()
                if score > best_score:
                    best_score = score
                    best_row = row
            if best_row is not None and best_score >= 0.75:
                return best_row

    hit = difflib.get_close_matches(cn, nba_df["NBA_Name_clean"].tolist(), n=1, cutoff=0.90)
    if hit:
        return nba_df[nba_df["NBA_Name_clean"] == hit[0]].iloc[0]

    return None

@st.cache_data(ttl=3600)
def player_rates_volatility(player_id: int, last_n: int):
    """
    Pull last_n game logs and compute:
      - avg_minutes (from sample)
      - per-minute standard deviation for PTS/REB/AST/FG3M
    We'll scale per-minute σ by projected minutes later.

    Returns: dict with keys:
      avg_min, pm_sigma_{stat}, games
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

            # Ensure stats exist
            for c in ["PTS","REB","AST","FG3M"]:
                if c not in gl.columns:
                    gl[c] = 0.0
                gl[c] = pd.to_numeric(gl[c], errors="coerce").fillna(0.0)

            # per-minute rates each game
            for c in ["PTS","REB","AST","FG3M"]:
                gl[f"PM_{c}"] = gl[c] / gl["MIN_f"]

            avg_min = float(gl["MIN_f"].mean())
            games = int(len(gl))

            out = {"avg_min": avg_min, "games": games}
            for c in ["PTS","REB","AST","FG3M"]:
                # ddof=1 if enough games; else 0
                if games >= 2:
                    out[f"pm_sigma_{c}"] = float(gl[f"PM_{c}"].std(ddof=1))
                else:
                    out[f"pm_sigma_{c}"] = 0.0

            return out
        except Exception as e:
            last_err = str(e)
            time.sleep(0.4 * attempt)
    return None

# ==========================
# INPUT: DK SLATE
# ==========================
st.sidebar.subheader("Upload DK Slate")
slate_file = st.sidebar.file_uploader("DraftKings salary slate CSV", type="csv")

if not slate_file:
    st.info("Upload your DraftKings slate CSV to begin.")
    st.stop()

slate_text = slate_file.getvalue().decode("utf-8", errors="ignore")
df = pd.read_csv(StringIO(slate_text))

# Minimal required columns
required = ["Name", "Salary", "TeamAbbrev", "Position"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"DK CSV missing columns: {missing}")
    st.stop()

# Controls
st.sidebar.markdown("---")
st.sidebar.subheader("Volatility Settings")
last_n = st.sidebar.slider("Use last N games for volatility", 5, 30, 15, 1)
min_games = st.sidebar.slider("Minimum games required", 3, 15, 5, 1)
min_sigma_pm = st.sidebar.slider("Minimum per-minute σ floor", 0.0, 0.50, 0.02, 0.01)

st.sidebar.subheader("Minutes Source")
minutes_mode = st.sidebar.radio(
    "Projected minutes (μ minutes) source",
    ["Use season avg minutes (NBA)", "Enter a constant minutes for all players"],
    index=0
)
const_minutes = None
if minutes_mode == "Enter a constant minutes for all players":
    const_minutes = st.sidebar.number_input("Constant minutes", min_value=10.0, max_value=48.0, value=32.0, step=0.5)

run = st.button("Build 90% floors for slate")

if not run:
    st.dataframe(df.head(25), use_container_width=True)
    st.stop()

nba_df = league_player_df()

# Build slate list
slate = pd.DataFrame({
    "Name": df["Name"].astype(str),
    "Salary": pd.to_numeric(df["Salary"], errors="coerce"),
    "Team": df["TeamAbbrev"].astype(str).str.upper(),
    "Position": df["Position"].astype(str),
})
slate["Name_clean"] = slate["Name"].apply(clean_name)

# Output rows
rows = []
prog = st.progress(0, text="Pulling game logs & computing minutes-adjusted volatility...")

for i, r in slate.iterrows():
    name = r["Name"]
    prog.progress((i + 1) / len(slate), text=f"{name} ({i+1}/{len(slate)})")

    hit = match_player_row(name, nba_df)
    if hit is None:
        rows.append({
            "Name": name,
            "Team": r["Team"],
            "Pos": r["Position"],
            "Salary": r["Salary"],
            "Status": "ERR_NAME",
            "Notes": "Could not match name to NBA player"
        })
        continue

    pid = int(hit["PLAYER_ID"])

    # Projected minutes (μ minutes)
    if minutes_mode == "Enter a constant minutes for all players":
        mu_min = float(const_minutes)
        min_note = "MIN:CONST"
    else:
        mu_min = float(hit["MIN"]) if pd.notna(hit["MIN"]) else np.nan
        min_note = "MIN:NBA"

    # Season per-minute means (μ stat rates)
    # We'll use season per-game and season minutes to get per-minute rates, then multiply by mu_min
    season_min = float(hit["MIN"]) if pd.notna(hit["MIN"]) else np.nan
    if not np.isfinite(season_min) or season_min <= 0:
        rows.append({
            "Name": name,
            "Team": r["Team"],
            "Pos": r["Position"],
            "Salary": r["Salary"],
            "Status": "ERR_MIN",
            "Notes": "No season minutes from NBA"
        })
        continue

    mu = {}
    for stat in ["PTS","REB","AST","FG3M"]:
        mu_rate = float(hit[stat]) / season_min
        mu[stat] = mu_rate * mu_min

    # Pull per-minute volatility from game logs
    vol = player_rates_volatility(pid, last_n=last_n)
    if vol is None or vol.get("games", 0) < int(min_games):
        rows.append({
            "Name": name,
            "Team": r["Team"],
            "Pos": r["Position"],
            "Salary": r["Salary"],
            "Status": "ERR_LOGS",
            "Notes": f"Not enough game logs for volatility (need {min_games})"
        })
        continue

    avg_min_sample = float(vol["avg_min"]) if vol.get("avg_min") else season_min
    scale = clamp(mu_min / max(avg_min_sample, 1e-6), MIN_SCALE, MAX_SCALE)

    out = {
        "Name": name,
        "Team": r["Team"],
        "Pos": r["Position"],
        "Salary": r["Salary"],
        "ProjMin": round(mu_min, 2),
        "VolGames": int(vol["games"]),
        "Status": "OK",
        "Notes": f"{min_note} VOL(last{last_n}) scale={scale:.2f}"
    }

    for stat in ["PTS","REB","AST","FG3M"]:
        pm_sigma = float(vol.get(f"pm_sigma_{stat}", 0.0))
        pm_sigma = max(pm_sigma, float(min_sigma_pm))

        # Minutes-adjusted sigma: scale per-minute σ by sqrt(minutes) and a minutes scaling factor
        # - sqrt(mu_min) treats the stat as "accumulating" over minutes
        # - scale adjusts for changes in role/minutes vs sample average
        sigma = pm_sigma * np.sqrt(mu_min) * scale

        floor90 = float(mu[stat]) - Z_ONE_SIDED_90 * float(sigma)

        out[f"{stat}_Proj"] = round(float(mu[stat]), 2)
        out[f"{stat}_SigmaAdj"] = round(float(sigma), 2)
        out[f"{stat}_Floor90"] = round(float(floor90), 2)

    rows.append(out)

props = pd.DataFrame(rows)

st.subheader("90% Floors for Slate (PTS / REB / AST / 3PM)")
st.dataframe(
    props.sort_values(["Status", "Salary"], ascending=[True, False]),
    use_container_width=True
)

csv_bytes = props.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download 90% floors CSV",
    data=csv_bytes,
    file_name="dk_slate_props_floor90.csv",
    mime="text/csv"
)

st.caption(
    "Floor90 is a one-sided 90% floor: projected to exceed that stat about 90% of the time under this model. "
    "Volatility is minutes-adjusted using per-minute σ from recent games."
)
