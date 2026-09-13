from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUT = ROOT / "analysis"
ASSETS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

DATA_URL = (
    "https://raw.githubusercontent.com/twodotone/finalLOL/main/data/csv/"
    "2022_LoL_esports_match_data_from_OraclesElixir.csv"
)
DATA_PATH = Path("/tmp/lol2022.csv")
TIMEPOINTS = [10, 15, 20, 25]
MAJOR_LEAGUES = ["LCK", "LPL", "LEC", "LCS"]
RANDOM_STATE = 42


def download_data() -> None:
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 50_000_000:
        return
    print("Downloading 2022 Oracle's Elixir mirror...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def needed_columns() -> list[str]:
    cols = [
        "gameid", "datacompleteness", "league", "playoffs", "date", "patch",
        "participantid", "side", "teamname", "teamid", "gamelength", "result",
        "turretplates", "opp_turretplates",
    ]
    for t in TIMEPOINTS:
        cols += [
            f"goldat{t}", f"xpat{t}", f"csat{t}",
            f"opp_goldat{t}", f"opp_xpat{t}", f"opp_csat{t}",
            f"golddiffat{t}", f"xpdiffat{t}", f"csdiffat{t}",
            f"killsat{t}", f"opp_killsat{t}",
        ]
    return cols


def load_team_rows() -> pd.DataFrame:
    wanted = set(needed_columns())
    df = pd.read_csv(DATA_PATH, usecols=lambda c: c in wanted, low_memory=False)
    df = df[df["participantid"].isin([100, 200])].copy()
    if "datacompleteness" in df.columns:
        df = df[df["datacompleteness"].eq("complete")].copy()
    df = df[df["result"].isin([0, 1])].copy()
    df["result"] = df["result"].astype(int)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["side_blue"] = df["side"].eq("Blue").astype(int)

    # Attach the opposing team name to each team-level row.
    opp = df[["gameid", "side", "teamname"]].copy()
    opp["side"] = opp["side"].map({"Blue": "Red", "Red": "Blue"})
    opp = opp.rename(columns={"teamname": "opponent"})
    df = df.merge(opp, on=["gameid", "side"], how="left")
    return df


def state_at(df: pd.DataFrame, t: int) -> tuple[pd.DataFrame, list[str], str]:
    g = f"golddiffat{t}"
    xp = f"xpdiffat{t}"
    cs = f"csdiffat{t}"
    kills = f"killsat{t}"
    opp_kills = f"opp_killsat{t}"

    cols = [g, xp, cs, kills, opp_kills, "side_blue"]
    d = df[df["gamelength"] >= t * 60].copy()
    d["killdiff"] = pd.to_numeric(d[kills], errors="coerce") - pd.to_numeric(
        d[opp_kills], errors="coerce"
    )
    full = [g, xp, cs, "killdiff", "side_blue"]

    # Turret plates disappear at 14:00, so their final plate count is known by
    # every prediction checkpoint from 15 minutes onward.
    if t >= 15:
        d["plate_diff"] = pd.to_numeric(d["turretplates"], errors="coerce") - pd.to_numeric(
            d["opp_turretplates"], errors="coerce"
        )
        full.append("plate_diff")

    numeric = list(dict.fromkeys(cols + full))
    for c in numeric:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=[g, "result", "gameid"]).copy()
    return d, full, g


def gold_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
    ])


def state_model() -> Pipeline:
    # Degree-2 interactions let the model distinguish game states where the
    # same gold lead is supported (or contradicted) by XP, CS, kills and plates.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.10, max_iter=4000, random_state=RANDOM_STATE)),
    ])


def oof_predictions(d: pd.DataFrame, full_features: list[str], gold_col: str) -> pd.DataFrame:
    y = d["result"].to_numpy()
    groups = d["gameid"].to_numpy()
    Xg = d[[gold_col, "side_blue"]]
    Xf = d[full_features]
    pg = np.full(len(d), np.nan)
    pf = np.full(len(d), np.nan)

    cv = GroupKFold(n_splits=5)
    for train_idx, test_idx in cv.split(Xg, y, groups=groups):
        gm = gold_model()
        fm = state_model()
        gm.fit(Xg.iloc[train_idx], y[train_idx])
        fm.fit(Xf.iloc[train_idx], y[train_idx])
        pg[test_idx] = gm.predict_proba(Xg.iloc[test_idx])[:, 1]
        pf[test_idx] = fm.predict_proba(Xf.iloc[test_idx])[:, 1]

    out = d.copy()
    out["p_gold"] = pg
    out["p_state"] = pf
    out["state_edge"] = out["p_state"] - out["p_gold"]
    out["residual"] = out["result"] - out["p_state"]
    return out


def metrics(y: pd.Series, p: pd.Series) -> dict[str, float]:
    yv = y.to_numpy()
    pv = np.clip(p.to_numpy(), 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(yv, pv)),
        "log_loss": float(log_loss(yv, pv)),
        "auc": float(roc_auc_score(yv, pv)),
    }


def fit_gold_curve(d: pd.DataFrame, gold_col: str, label: str) -> tuple[pd.DataFrame, dict[str, float | None]]:
    m = gold_model()
    m.fit(d[[gold_col, "side_blue"]], d["result"])

    lo = float(d[gold_col].quantile(0.005))
    hi = float(d[gold_col].quantile(0.995))
    grid = np.linspace(lo, hi, 500)
    x_blue = pd.DataFrame({gold_col: grid, "side_blue": 1})
    x_red = pd.DataFrame({gold_col: grid, "side_blue": 0})
    p = (m.predict_proba(x_blue)[:, 1] + m.predict_proba(x_red)[:, 1]) / 2
    curve = pd.DataFrame({"gold_diff": grid, "win_probability": p, "checkpoint": label})

    thresholds: dict[str, float | None] = {}
    for target in [0.80, 0.90, 0.95]:
        idx = np.flatnonzero(p >= target)
        thresholds[f"p{int(target * 100)}"] = float(grid[idx[0]]) if len(idx) else None
    return curve, thresholds


def team_efficiency(pred: pd.DataFrame, gold_col: str, ahead: bool, min_n: int = 20) -> pd.DataFrame:
    mask = pred[gold_col] > 0 if ahead else pred[gold_col] < 0
    x = pred[mask & pred["league"].isin(MAJOR_LEAGUES)].copy()
    grouped = x.groupby(["league", "teamname"], as_index=False).agg(
        n=("result", "size"),
        actual_wins=("result", "sum"),
        expected_wins=("p_state", "sum"),
        mean_expected=("p_state", "mean"),
        mean_residual=("residual", "mean"),
    )
    grouped = grouped[grouped["n"] >= min_n].copy()
    # Simple shrinkage toward zero keeps small samples from dominating rankings.
    grouped["efficiency"] = grouped["mean_residual"] * grouped["n"] / (grouped["n"] + 30)
    grouped["efficiency_pp"] = 100 * grouped["efficiency"]
    return grouped.sort_values("efficiency_pp", ascending=False)


def threshold_for_subset(d: pd.DataFrame, gold_col: str, target: float = 0.90) -> float | None:
    if len(d) < 200 or d["result"].nunique() < 2:
        return None
    _, thresholds = fit_gold_curve(d, gold_col, "subset")
    return thresholds[f"p{int(target * 100)}"]


def save_chart(fig: go.Figure, name: str) -> None:
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif"),
        margin=dict(l=55, r=35, t=65, b=55),
    )
    fig.write_html(ASSETS / name, include_plotlyjs="cdn", full_html=True)


def main() -> None:
    download_data()
    teams = load_team_rows()
    print(f"Team observations: {len(teams):,}; games: {teams['gameid'].nunique():,}")

    summary: dict[str, object] = {
        "games": int(teams["gameid"].nunique()),
        "team_observations": int(len(teams)),
        "timepoints": {},
    }
    all_curves = []
    perf_rows = []
    preds: dict[int, pd.DataFrame] = {}

    for t in TIMEPOINTS:
        d, full_features, gold_col = state_at(teams, t)
        pred = oof_predictions(d, full_features, gold_col)
        preds[t] = pred
        gold_metrics = metrics(pred["result"], pred["p_gold"])
        state_metrics = metrics(pred["result"], pred["p_state"])
        curve, thresholds = fit_gold_curve(d, gold_col, f"{t} min")
        all_curves.append(curve)

        summary["timepoints"][str(t)] = {
            "games_reaching_checkpoint": int(d["gameid"].nunique()),
            "gold_only": gold_metrics,
            "full_state": state_metrics,
            "thresholds_gold_diff": thresholds,
        }
        perf_rows += [
            {"checkpoint": t, "model": "Gold only", **gold_metrics},
            {"checkpoint": t, "model": "Full state", **state_metrics},
        ]

    curve_df = pd.concat(all_curves, ignore_index=True)
    perf_df = pd.DataFrame(perf_rows)

    # Core chart 1: the dynamic conversion curve.
    fig = px.line(
        curve_df,
        x="gold_diff",
        y="win_probability",
        color="checkpoint",
        labels={"gold_diff": "Gold difference", "win_probability": "Win probability", "checkpoint": "Checkpoint"},
        title="How the Value of a Gold Lead Changes as the Game Progresses",
    )
    fig.add_hline(y=0.90, line_dash="dash", opacity=0.45)
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    save_chart(fig, "dynamic-win-probability.html")

    # Core chart 2: whether state composition adds information beyond gold.
    brier = perf_df.copy()
    fig = px.line(
        brier,
        x="checkpoint",
        y="brier",
        color="model",
        markers=True,
        labels={"checkpoint": "Game minute", "brier": "Brier score (lower is better)", "model": "Model"},
        title="Gold Alone vs. Multidimensional Game State",
    )
    save_chart(fig, "state-vs-gold-brier.html")

    # Use 20 minutes for team-level closing/comeback analysis: late enough for
    # meaningful macro separation while retaining most matches in the sample.
    p20 = preds[20].copy()
    g20 = "golddiffat20"

    closing = team_efficiency(p20, g20, ahead=True)
    comeback = team_efficiency(p20, g20, ahead=False)
    closing.to_csv(OUT / "closing_efficiency.csv", index=False)
    comeback.to_csv(OUT / "comeback_resilience.csv", index=False)

    close_plot = closing.head(12).sort_values("efficiency_pp")
    if not close_plot.empty:
        fig = px.bar(
            close_plot,
            x="efficiency_pp",
            y="teamname",
            color="league",
            orientation="h",
            labels={"efficiency_pp": "Wins above expectation (shrunken, percentage points)", "teamname": "Team"},
            title="Closing Efficiency: Teams That Convert Leads Better Than Expected",
        )
        fig.add_vline(x=0, line_dash="dash", opacity=0.5)
        save_chart(fig, "closing-efficiency.html")

    comeback_plot = comeback.head(12).sort_values("efficiency_pp")
    if not comeback_plot.empty:
        fig = px.bar(
            comeback_plot,
            x="efficiency_pp",
            y="teamname",
            color="league",
            orientation="h",
            labels={"efficiency_pp": "Wins above expectation from deficits (shrunken, pp)", "teamname": "Team"},
            title="Comeback Resilience: Teams That Outperform Their Losing Game States",
        )
        fig.add_vline(x=0, line_dash="dash", opacity=0.5)
        save_chart(fig, "comeback-resilience.html")

    # Lead quality: how much the full state moves win probability away from what
    # the scoreboard's gold difference alone would imply.
    lead20 = p20[p20[g20] > 0].copy()
    sample = lead20.sample(min(len(lead20), 7000), random_state=RANDOM_STATE)
    fig = px.scatter(
        sample,
        x=g20,
        y="state_edge",
        color="result",
        opacity=0.30,
        labels={g20: "Gold lead at 20 minutes", "state_edge": "State edge: P(full state) - P(gold only)", "result": "Won"},
        title="Not All Leads Are Equal: When the Full Game State Disagrees With Gold",
    )
    fig.add_hline(y=0, line_dash="dash", opacity=0.5)
    save_chart(fig, "lead-quality-20.html")

    # Concrete examples of fragile and robust leads.
    example_cols = [
        "gameid", "date", "league", "teamname", "opponent", "result", g20,
        "xpdiffat20", "csdiffat20", "killdiff", "p_gold", "p_state", "state_edge",
    ]
    fragile = lead20.nsmallest(15, "state_edge")[example_cols]
    robust = lead20.nlargest(15, "state_edge")[example_cols]
    fragile.to_csv(OUT / "fragile_leads.csv", index=False)
    robust.to_csv(OUT / "robust_leads.csv", index=False)

    throws = p20[p20["result"] == 0].nlargest(15, "p_state")[example_cols]
    comebacks = p20[p20["result"] == 1].nsmallest(15, "p_state")[example_cols]
    throws.to_csv(OUT / "largest_throws.csv", index=False)
    comebacks.to_csv(OUT / "largest_comebacks.csv", index=False)

    # Major-region 20-minute lead needed for a model-implied 90% win chance.
    league_thresholds = []
    for league in MAJOR_LEAGUES:
        d = p20[p20["league"].eq(league)].copy()
        threshold = threshold_for_subset(d, g20, 0.90)
        if threshold is not None:
            league_thresholds.append({"league": league, "gold_for_90pct": threshold, "observations": len(d)})
    league_df = pd.DataFrame(league_thresholds).sort_values("gold_for_90pct")
    league_df.to_csv(OUT / "league_thresholds.csv", index=False)
    if not league_df.empty:
        fig = px.bar(
            league_df,
            x="league",
            y="gold_for_90pct",
            text_auto=".0f",
            labels={"league": "League", "gold_for_90pct": "20-minute gold lead for 90% win probability"},
            title="How Large Does a 20-Minute Lead Need to Be?",
        )
        save_chart(fig, "league-90-threshold.html")

    # Store compact top-line facts for the website.
    summary["closing_top"] = closing.head(8).to_dict(orient="records")
    summary["comeback_top"] = comeback.head(8).to_dict(orient="records")
    summary["league_thresholds_20"] = league_df.to_dict(orient="records")
    summary["fragile_examples"] = fragile.head(5).assign(date=lambda x: x["date"].astype(str)).to_dict(orient="records")
    summary["robust_examples"] = robust.head(5).assign(date=lambda x: x["date"].astype(str)).to_dict(orient="records")
    summary["largest_throws"] = throws.head(5).assign(date=lambda x: x["date"].astype(str)).to_dict(orient="records")
    summary["largest_comebacks"] = comebacks.head(5).assign(date=lambda x: x["date"].astype(str)).to_dict(orient="records")

    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
    perf_df.to_csv(OUT / "model_performance.csv", index=False)

    print(json.dumps(summary["timepoints"], indent=2))
    print("Analysis complete.")


if __name__ == "__main__":
    main()
