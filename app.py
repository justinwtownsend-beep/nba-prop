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

# Vol cache persisted in gist (keyed by player_id|last_n)
GIST_VOL_CACHE = f"vol_cache_{SEASON}.json"

LEAGUE_TIMEOUT = 20
GAMELOG_TIMEOUT = 12
GAMELOG_RETRIES = 2

# one-sided z scores
Z = {
    0.70: 0.5244005127080409,
    0.80: 0.8416212335729143,
    0.90: 1.2815515655446004,
}

# minutes scaling caps
MIN_SCALE = 0.70
MAX_SCALE = 1.30

# ==========================
# PAGE
# ==========================
st.set_page_config(layout="wide")
st.title("Confidence Floors (70/80/90%) — Top 25 by Stat (with Volatility)")

st.markdown("""
This app uses your optimizer projections (`final.csv`) and computes **player-specific volatility**
for ONLY the union of:

- Top 25 projected **PTS**
- Top 25 projected **REB**
- Top 25 projected **AST**
- Top 25 projected **3PM**

Then it produces **Floor70 / Floor80 / Floor90** for each stat group.

To keep it fast, volatility is **cached to your GitHub Gist**.
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

def cache_key(pid: int, last_n: int) -> str:
    return f"{int(pid)}|{int(last_n)}"

# ==========================
# GIST (read/write)
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

def gist_write(files: dict):
    payload = {"files": {k: {"content": v} for k, v in files.items()}}
    r = requests.patch(f"https://api.github.com/gists/{GIST_ID}", headers=gh_headers(), json=payload, timeout=25)
    r.raise_for_status()

# ==========================
# NBA ID Matching
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
    Per-minute σ for PTS/REB/AST/FG3M from last_n games.
    Returns avg_min, games, pm_sigma_{stat}
    """
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

            for c in ["PTS", "REB", "AST", "FG3M"]:
                if c not in gl.columns:
                    gl[c] = 0.0
                gl[c] = pd.to_numeric(gl[c], errors="coerce").fillna(0.0)
                gl[f"PM_{c}"] = gl[c] / gl["MIN_f"]

            avg_min = float(gl["MIN_f"].mean())
            games = int(len(gl))

            out = {"avg_min": avg_min, "games": games}
            for c in ["PTS", "REB", "AST", "FG3M"]:
                out[f"pm_sigma_{c}"] = float(gl[f"PM_{c}"].std(ddof=1)) if games >= 2 else 0.0
            return out
        except Exception:
            time.sleep(0.35 * attempt)
    return None

# ==========================
# LOAD projections from optimizer
# ==========================
final_text = gist_read(GIST_FINAL)
if not final_text:
    st.error("Could not load final.csv from optimizer. Run projections in optimizer app first.")
    st.stop()

proj = pd.read_csv(StringIO(final_text))

needed = ["Name", "Minutes", "PTS", "REB", "AST", "FG3M"]
missing = [c for c in needed if c not in proj.columns]
if missing:
    st.error(f"final.csv missing required columns: {missing}")
    st.stop()

# normalize / numeric
proj["Name_clean"] = proj["Name"].apply(clean_name)
for c in ["Minutes", "PTS", "REB", "AST", "FG3M"]:
    proj[c] = pd.to_numeric(proj[c], errors="coerce")

# remove OUT players
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

# dedupe players (important!)
proj = proj.dropna(subset=["Name_clean"]).drop_duplicates(subset=["Name_clean"]).copy()

st.sidebar.subheader("Settings")
top_n = st.sidebar.number_input("Top N per stat", min_value=5, max_value=100, value=25, step=5)
last_n = st.sidebar.slider("Volatility from last N games", 5, 30, 15, 1)
min_sigma_pm = st.sidebar.slider("Min per-minute σ floor", 0.0, 0.50, 0.02, 0.01)
refresh_cache = st.sidebar.checkbox("Refresh volatility cache (slow)", value=False)

# ==========================
# Build Top-N sets (your rule)
# ==========================
top_pts = proj.dropna(subset=["PTS"]).sort_values("PTS", ascending=False).head(int(top_n)).copy()
top_reb = proj.dropna(subset=["REB"]).sort_values("REB", ascending=False).head(int(top_n)).copy()
top_ast = proj.dropna(subset=["AST"]).sort_values("AST", ascending=False).head(int(top_n)).copy()
top_3pm = proj.dropna(subset=["FG3M"]).sort_values("FG3M", ascending=False).head(int(top_n)).copy()

need_names = set(top_pts["Name_clean"]) | set(top_reb["Name_clean"]) | set(top_ast["Name_clean"]) | set(top_3pm["Name_clean"])
need_df = proj[proj["Name_clean"].isin(need_names)].copy()

st.write(f"✅ Will compute volatility for **{len(need_df)} unique players** total (Top {int(top_n)} by each stat, union).")
st.subheader("Players included (union list)")
st.dataframe(
    need_df[["Name", "Minutes", "PTS", "REB", "AST", "FG3M"]].sort_values("PTS", ascending=False),
    use_container_width=True
)

# ==========================
# Load volatility cache from gist
# ==========================
cache_text = gist_read(GIST_VOL_CACHE)
if cache_text:
    try:
        VOL_CACHE = json.loads(cache_text)
    except Exception:
        VOL_CACHE = {}
else:
    VOL_CACHE = {}

if not st.button("Build floors (70/80/90) with volatility"):
    st.stop()

nba_df = league_player_df()

# Compute volatility only for needed players
vol_map = {}  # name_clean -> volatility dict/status
updated_cache = False

progbar = st.progress(0, text="Computing volatility for selected players...")
need_list = need_df[["Name", "Name_clean"]].to_dict("records")

for i, rr in enumerate(need_list):
    progbar.progress((i + 1) / len(need_list), text=f"{rr['Name']} ({i+1}/{len(need_list)})")

    pid = match_player_id(rr["Name"], nba_df)
    if pid is None:
        vol_map[rr["Name_clean"]] = {"status": "ERR_NAME"}
        continue

    key = cache_key(pid, last_n)
    v = None if refresh_cache else VOL_CACHE.get(key)

    if v is None:
        v = pull_pm_volatility(pid, last_n=last_n)
        time.sleep(0.10)  # tiny throttle helps cloud
        if v is not None:
            # apply sigma floors
            for stat in ["PTS", "REB", "AST", "FG3M"]:
                v[f"pm_sigma_{stat}"] = max(float(v.get(f"pm_sigma_{stat}", 0.0)), float(min_sigma_pm))
            VOL_CACHE[key] = v
            updated_cache = True

    if v is None:
        vol_map[rr["Name_clean"]] = {"status": "ERR_LOGS"}
    else:
        v2 = dict(v)
        v2["status"] = "OK"
        v2["player_id"] = pid
        vol_map[rr["Name_clean"]] = v2

# Write updated cache back to gist
if updated_cache:
    gist_write({GIST_VOL_CACHE: json.dumps(VOL_CACHE)})

def floors_for_group(df_group: pd.DataFrame, stat: str, title: str):
    rows = []
    for _, r in df_group.iterrows():
        nm = r["Name_clean"]
        v = vol_map.get(nm, {"status": "ERR"})
        mu_min = float(r["Minutes"]) if np.isfinite(r["Minutes"]) else 0.0

        base = {
            "Name": r["Name"],
            "Team": r.get("Team", ""),
            "Opp": r.get("Opp", ""),
            "Minutes": round(mu_min, 1),
            "Proj": round(float(r[stat]), 2) if np.isfinite(r[stat]) else np.nan,
            "Status": v.get("status", "ERR"),
        }

        if base["Status"] != "OK" or mu_min <= 0 or not np.isfinite(r[stat]):
            rows.append(base)
            continue

        avg_min = float(v.get("avg_min", mu_min))
        scale = clamp(mu_min / max(avg_min, 1e-6), MIN_SCALE, MAX_SCALE)

        pm_sigma = float(v.get(f"pm_sigma_{stat}", 0.0))
        pm_sigma = max(pm_sigma, float(min_sigma_pm))

        sigma_adj = pm_sigma * np.sqrt(mu_min) * scale

        base["SigmaAdj"] = round(sigma_adj, 2)
        base["Floor70"] = round(float(r[stat]) - Z[0.70] * sigma_adj, 2)
        base["Floor80"] = round(float(r[stat]) - Z[0.80] * sigma_adj, 2)
        base["Floor90"] = round(float(r[stat]) - Z[0.90] * sigma_adj, 2)
        base["Scale"] = round(scale, 2)
        base["VolGames"] = int(v.get("games", 0))
        rows.append(base)

    out = pd.DataFrame(rows).sort_values("Proj", ascending=False)
    st.subheader(title)
    st.dataframe(out, use_container_width=True)

    st.download_button(
        f"Download {title} CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"{title.lower().replace(' ', '_').replace('%', '')}.csv",
        mime="text/csv"
    )

# Output: top 25 (or your chosen Top N) per stat group
floors_for_group(top_pts, "PTS", f"Top {int(top_n)} Projected Points — Floors 70/80/90")
floors_for_group(top_reb, "REB", f"Top {int(top_n)} Projected Rebounds — Floors 70/80/90")
floors_for_group(top_ast, "AST", f"Top {int(top_n)} Projected Assists — Floors 70/80/90")
floors_for_group(top_3pm, "FG3M", f"Top {int(top_n)} Projected 3PM — Floors 70/80/90")

st.caption("Volatility is per-player from game logs (last N games), minutes-adjusted, and cached to your Gist for speed.")
