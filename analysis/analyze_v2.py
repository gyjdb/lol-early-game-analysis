from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
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

DATA_URL = "https://raw.githubusercontent.com/twodotone/finalLOL/main/data/csv/2022_LoL_esports_match_data_from_OraclesElixir.csv"
DATA_PATH = Path("/tmp/lol2022.csv")
TIMEPOINTS = [10, 15, 20, 25]
MAJOR_LEAGUES = ["LCK", "LPL", "LEC", "LCS"]
RANDOM_STATE = 42


def download_data():
    if not (DATA_PATH.exists() and DATA_PATH.stat().st_size > 50_000_000):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def columns_needed():
    cols = {"gameid", "datacompleteness", "league", "playoffs", "date", "patch", "participantid", "side", "teamname", "teamid", "gamelength", "result", "turretplates", "opp_turretplates"}
    for t in TIMEPOINTS:
        cols |= {f"goldat{t}", f"xpat{t}", f"csat{t}", f"golddiffat{t}", f"xpdiffat{t}", f"csdiffat{t}", f"killsat{t}", f"opp_killsat{t}"}
    return cols


def load_data():
    wanted = columns_needed()
    df = pd.read_csv(DATA_PATH, usecols=lambda c: c in wanted, low_memory=False)
    df = df[df["participantid"].isin([100, 200])].copy()
    df = df[df["datacompleteness"].eq("complete")].copy()
    df = df[df["result"].isin([0, 1])].copy()
    df["result"] = df["result"].astype(int)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["side_blue"] = df["side"].eq("Blue").astype(int)
    opp = df[["gameid", "side", "teamname"]].copy()
    opp["side"] = opp["side"].map({"Blue": "Red", "Red": "Blue"})
    opp = opp.rename(columns={"teamname": "opponent"})
    return df.merge(opp, on=["gameid", "side"], how="left")


def prepare_checkpoint(df, t):
    gold, xp, cs = f"golddiffat{t}", f"xpdiffat{t}", f"csdiffat{t}"
    d = df[df["gamelength"] >= t * 60].copy()
    d["killdiff"] = pd.to_numeric(d[f"killsat{t}"], errors="coerce") - pd.to_numeric(d[f"opp_killsat{t}"], errors="coerce")
    features = [gold, xp, cs, "killdiff", "side_blue"]
    if t >= 15:
        d["plate_diff"] = pd.to_numeric(d["turretplates"], errors="coerce") - pd.to_numeric(d["opp_turretplates"], errors="coerce")
        features.append("plate_diff")
        prev = t - 5
        d["gold_velocity"] = pd.to_numeric(d[gold], errors="coerce") - pd.to_numeric(d[f"golddiffat{prev}"], errors="coerce")
        d["xp_velocity"] = pd.to_numeric(d[xp], errors="coerce") - pd.to_numeric(d[f"xpdiffat{prev}"], errors="coerce")
        d["cs_velocity"] = pd.to_numeric(d[cs], errors="coerce") - pd.to_numeric(d[f"csdiffat{prev}"], errors="coerce")
        features += ["gold_velocity", "xp_velocity", "cs_velocity"]
    for c in features:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d[gold] = pd.to_numeric(d[gold], errors="coerce")
    return d.dropna(subset=[gold, "result", "gameid"]).copy(), features, gold


def gold_model():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=2500, random_state=RANDOM_STATE))])


def full_state_model():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("poly", PolynomialFeatures(degree=2, include_bias=False)), ("scale", StandardScaler()), ("model", LogisticRegression(C=0.10, max_iter=5000, random_state=RANDOM_STATE))])


def grouped_oof(d, features, gold_col):
    y, groups = d["result"].to_numpy(), d["gameid"].to_numpy()
    Xg, Xf = d[[gold_col, "side_blue"]], d[features]
    pg, pf = np.zeros(len(d)), np.zeros(len(d))
    for train, test in GroupKFold(n_splits=5).split(Xg, y, groups):
        gm, fm = gold_model(), full_state_model()
        gm.fit(Xg.iloc[train], y[train]); fm.fit(Xf.iloc[train], y[train])
        pg[test] = gm.predict_proba(Xg.iloc[test])[:, 1]
        pf[test] = fm.predict_proba(Xf.iloc[test])[:, 1]
    out = d.copy(); out["p_gold"] = pg; out["p_state"] = pf
    out["state_edge"] = pf - pg; out["residual"] = out["result"] - pf
    return out


def score(y, p):
    pv = np.clip(p.to_numpy(), 1e-6, 1 - 1e-6); yv = y.to_numpy()
    return {"brier": float(brier_score_loss(yv, pv)), "log_loss": float(log_loss(yv, pv)), "auc": float(roc_auc_score(yv, pv))}


def fit_curve(d, gold_col, label):
    m = gold_model().fit(d[[gold_col, "side_blue"]], d["result"])
    grid = np.linspace(float(d[gold_col].quantile(.005)), float(d[gold_col].quantile(.995)), 600)
    blue = pd.DataFrame({gold_col: grid, "side_blue": 1}); red = pd.DataFrame({gold_col: grid, "side_blue": 0})
    p = (m.predict_proba(blue)[:, 1] + m.predict_proba(red)[:, 1]) / 2
    thresholds = {}
    for target in (.80, .90, .95):
        hits = np.flatnonzero(p >= target); thresholds[f"p{int(target*100)}"] = float(grid[hits[0]]) if len(hits) else None
    return pd.DataFrame({"gold_diff": grid, "win_probability": p, "checkpoint": label}), thresholds


def team_table(pred, gold_col, ahead, min_n=20):
    mask = pred[gold_col].gt(0) if ahead else pred[gold_col].lt(0)
    x = pred[mask & pred["league"].isin(MAJOR_LEAGUES)]
    g = x.groupby(["league", "teamname"], as_index=False).agg(n=("result", "size"), actual_wins=("result", "sum"), expected_wins=("p_state", "sum"), mean_expected=("p_state", "mean"), residual_mean=("residual", "mean"), residual_sd=("residual", "std"))
    g = g[g["n"] >= min_n].copy(); g["efficiency"] = g["residual_mean"] * g["n"] / (g["n"] + 30)
    g["efficiency_pp"] = 100*g["efficiency"]; g["se_pp"] = 100*g["residual_sd"]/np.sqrt(g["n"])
    return g.sort_values("efficiency_pp", ascending=False)


def chart(fig, filename):
    fig.update_layout(template="plotly_white", font=dict(family="Inter, Arial, sans-serif"), margin=dict(l=55, r=35, t=70, b=55))
    fig.write_html(ASSETS / filename, include_plotlyjs="cdn", full_html=True)


def records(df, n=8):
    x = df.head(n).copy()
    for c in x.columns:
        if pd.api.types.is_datetime64_any_dtype(x[c]): x[c] = x[c].astype(str)
    return x.astype(object).where(pd.notnull(x), None).to_dict(orient="records")


def main():
    download_data(); teams = load_data()
    results = {"games": int(teams["gameid"].nunique()), "team_observations": int(len(teams)), "timepoints": {}}
    preds, curves, perf = {}, [], []
    for t in TIMEPOINTS:
        d, features, gold = prepare_checkpoint(teams, t); p = grouped_oof(d, features, gold); preds[t] = p
        gs, fs = score(p["result"], p["p_gold"]), score(p["result"], p["p_state"])
        curve, th = fit_curve(d, gold, f"{t} min"); curves.append(curve)
        results["timepoints"][str(t)] = {"games_reaching_checkpoint": int(d["gameid"].nunique()), "gold_only": gs, "state_trajectory": fs, "thresholds_gold_diff": th}
        perf += [{"checkpoint": t, "model": "Gold only", **gs}, {"checkpoint": t, "model": "State + trajectory", **fs}]
    perf_df = pd.DataFrame(perf); perf_df.to_csv(OUT / "model_performance.csv", index=False)
    curve_df = pd.concat(curves, ignore_index=True)
    fig = px.line(curve_df, x="gold_diff", y="win_probability", color="checkpoint", labels={"gold_diff":"Gold difference","win_probability":"Win probability","checkpoint":"Checkpoint"}, title="The Point of No Return: How a Gold Lead Matures Over Time")
    fig.add_hline(y=.90, line_dash="dash", opacity=.45); fig.update_yaxes(tickformat=".0%", range=[0,1]); chart(fig, "dynamic-win-probability.html")
    fig = px.line(perf_df, x="checkpoint", y="brier", color="model", markers=True, labels={"checkpoint":"Game minute","brier":"Brier score (lower is better)","model":"Model"}, title="Does Game-State Composition Add Information Beyond Gold?"); chart(fig, "state-vs-gold-brier.html")
    p20, g20 = preds[20].copy(), "golddiffat20"
    p20["trajectory"] = pd.cut(p20["gold_velocity"], [-np.inf,-500,500,np.inf], labels=["Shrinking (>500g lost)","Stable (±500g)","Growing (>500g gained)"])
    p20["gold_bin"] = pd.cut(p20[g20], bins=np.arange(-6000,6500,1000), include_lowest=True)
    traj = p20.groupby(["gold_bin","trajectory"], observed=True).agg(win_rate=("result","mean"), n=("result","size"), mean_gold=(g20,"mean")).reset_index(); traj = traj[traj["n"]>=40]
    traj.to_csv(OUT/"lead_trajectory.csv", index=False)
    fig = px.line(traj, x="mean_gold", y="win_rate", color="trajectory", markers=True, labels={"mean_gold":"20-minute gold difference","win_rate":"Observed win rate","trajectory":"Lead trajectory"}, title="Momentum Matters: Growing vs. Shrinking Game States at 20 Minutes"); fig.update_yaxes(tickformat=".0%", range=[0,1]); chart(fig,"lead-trajectory-20.html")
    closing, comeback = team_table(p20,g20,True), team_table(p20,g20,False); closing.to_csv(OUT/"closing_efficiency.csv",index=False); comeback.to_csv(OUT/"comeback_resilience.csv",index=False)
    for table, fname, title, xlabel in [(closing,"closing-efficiency.html","Closing Efficiency: Who Converts Leads Better Than Expected?","Conversion above expectation (shrunken pp)"),(comeback,"comeback-resilience.html","Comeback Resilience: Who Wins More Often Than Their Deficit Implies?","Comeback performance above expectation (shrunken pp)")]:
        q=table.head(12).sort_values("efficiency_pp")
        if len(q):
            fig=px.bar(q,x="efficiency_pp",y="teamname",color="league",orientation="h",error_x="se_pp",labels={"efficiency_pp":xlabel,"teamname":"Team"},title=title); fig.add_vline(x=0,line_dash="dash",opacity=.5); chart(fig,fname)
    ahead = p20[p20[g20]>0].copy(); sample=ahead.sample(min(7000,len(ahead)),random_state=RANDOM_STATE)
    fig=px.scatter(sample,x=g20,y="state_edge",color="result",opacity=.28,labels={g20:"Gold lead at 20 minutes","state_edge":"Lead quality: P(state + trajectory) − P(gold only)","result":"Won"},title="Not All Leads Are Equal: Lead Quality Beyond the Scoreboard"); fig.add_hline(y=0,line_dash="dash",opacity=.5); chart(fig,"lead-quality-20.html")
    cols=["gameid","date","league","teamname","opponent","result",g20,"xpdiffat20","csdiffat20","killdiff","gold_velocity","p_gold","p_state","state_edge"]
    fragile=ahead.nsmallest(15,"state_edge")[cols]; robust=ahead.nlargest(15,"state_edge")[cols]; throws=p20[p20["result"]==0].nlargest(15,"p_state")[cols]; comebacks=p20[p20["result"]==1].nsmallest(15,"p_state")[cols]
    for df,name in [(fragile,"fragile_leads.csv"),(robust,"robust_leads.csv"),(throws,"largest_throws.csv"),(comebacks,"largest_comebacks.csv")]: df.to_csv(OUT/name,index=False)
    league_rows=[]
    for league in MAJOR_LEAGUES:
        d=p20[p20["league"].eq(league)]
        if len(d)>=200:
            _,th=fit_curve(d,g20,"subset")
            if th["p90"] is not None: league_rows.append({"league":league,"gold_for_90pct":th["p90"],"observations":int(len(d))})
    league_df=pd.DataFrame(league_rows)
    if len(league_df):
        league_df=league_df.sort_values("gold_for_90pct"); fig=px.bar(league_df,x="league",y="gold_for_90pct",text_auto=".0f",labels={"league":"League","gold_for_90pct":"20-minute gold lead for 90% win probability"},title="Regional Conversion: How Much of a Lead Is 'Safe'?"); chart(fig,"league-90-threshold.html")
    league_df.to_csv(OUT/"league_thresholds.csv",index=False)
    moderate=p20[(p20[g20]>=1000)&(p20[g20]<=3000)]; trajectory_summary=moderate.groupby("trajectory",observed=True).agg(n=("result","size"),win_rate=("result","mean"),mean_gold=(g20,"mean"),mean_velocity=("gold_velocity","mean")).reset_index()
    results.update({"closing_top":records(closing),"comeback_top":records(comeback),"league_thresholds_20":records(league_df),"trajectory_20_moderate_leads":records(trajectory_summary,10),"fragile_examples":records(fragile,5),"robust_examples":records(robust,5),"largest_throws":records(throws,5),"largest_comebacks":records(comebacks,5)})
    with open(OUT/"results.json","w",encoding="utf-8") as f: json.dump(results,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(results["timepoints"],indent=2)); print("Analysis complete")


if __name__ == "__main__": main()
