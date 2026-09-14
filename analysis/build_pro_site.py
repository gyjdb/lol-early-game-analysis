"""Publish the current study; preserve the generated 2022 research archive."""
import html
import json
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT/'analysis'
r = json.loads((A/'pro_results.json').read_text())
e = lambda x: html.escape(str(x))
repo = 'https://github.com/gyjdb/lol-early-game-analysis'
metrics = {x['model']:x for x in r['metrics']}
coverage = pd.read_csv(A/'pro_coverage.csv')
teams = pd.read_csv(A/'pro_teams.csv')
lpl = teams[teams.league.eq('LPL')&teams.condition.eq('Ahead')].sort_values('teamname')
eligible_lpl = lpl[lpl.n.ge(15)]
assert (eligible_lpl.low.le(0)&eligible_lpl.high.ge(0)).all(), 'Review LPL uncertainty narrative'
assert r['comparisons'][2]['high'] < 0 and r['comparisons'][3]['low'] > 0, 'Review feature narrative'
blg = lpl[lpl.teamname.eq('Bilibili Gaming')].iloc[0]
lc = coverage[coverage.league.eq('LPL')].iloc[0]
gain = 1-metrics['strength']['brier']/metrics['gold']['brier']

def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join(f'<th scope="col">{e(x)}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{e(x)}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def figure(name,title,caption,csv):
    return f'<figure><div class="figure-header"><h3>{title}</h3><a href="assets/{name}" target="_blank" rel="noopener">Open chart ↗</a></div><iframe src="assets/{name}" title="{e(title)}" loading="lazy"></iframe><figcaption>{caption} <a href="analysis/{csv}">Download CSV ↓</a></figcaption></figure>'

def section(id,label,title,intro,body):
    return f'<section id="{id}" class="shell section"><div class="section-heading"><span class="eyebrow">{label}</span><h2>{title}</h2><p>{intro}</p></div>{body}</section>'

coverage_table = table(['League','Source games','Eligible at 20','Coverage'],[[x.league,x.source_games,x.eligible_games,f'{x.coverage_pct:.1f}%'] for x in coverage.itertuples()])
team_table = table(['LPL team','Season coverage','Leading games*','Wins / expected*','Adjusted pp [95% interval]*'],[[x.teamname,f'{x.eligible_games}/{x.source_games} ({x.coverage_pct:.1f}%)',x.n,f'{x.wins} / {x.expected_wins:.1f}',f'{x.adjusted_pp:+.1f} [{x.low:+.1f}, {x.high:+.1f}]' if x.n>=15 else 'Below display threshold'] for x in lpl.itertuples()])
names = {'gold':'Gold + side','state':'Current state','trajectory':'State + trajectory','strength':'State + prior Elo','full':'State + prior Elo + trajectory'}
model_table = table(['Model','Brier ↓','Log loss ↓','ROC AUC ↑'],[[names[x['model']],f"{x['brier']:.5f}",f"{x['log_loss']:.5f}",f"{x['auc']:.4f}"] for x in r['metrics']])
comparison_table = table(['Comparison','ΔBrier','95% cluster interval'],[[x['comparison'],f"{x['delta_brier']:+.5f}",f"[{x['low']:+.5f}, {x['high']:+.5f}]"] for x in r['comparisons']])
fold_table = table(['Test month','Training ends','Train games','Test games'],[[x['fold'],x['train_end'][:10],x['train_games'],x['test_games']] for x in r['folds']])
cases = ''
for x in r['review_cases'][:2]:
    cases += f'''<article class="case"><div class="eyebrow">LPL · review candidate · {e(x['date'][:10])}</div><h3>{e(x['teamname'])}</h3><p>vs. {e(x['opponent'])} · patch {e(x['patch'])}</p><div class="case-gold">+{x['golddiffat20']:,.0f}<span>gold at 20 · +{x['golddiffat25']:,.0f} at 25 · final loss</span></div><p><strong>Model estimate at 20: {x['p_strength']:.1%}.</strong> This extreme-tail estimate is not a calibrated guarantee.</p><details><summary>Inspect both recorded drafts</summary><p>{e(x['teamname'])}: {e(x['draft'])}</p><p>{e(x['opponent'])}: {e(x['opponent_draft'])}</p></details><small>{e(x['gameid'])}</small></article>'''

body = section('findings','01 / Findings','Team strength changes the benchmark.','A stronger team should already be expected to win more often from the same 20-minute state.',f'''<div class="findings-grid"><article><span class="eyebrow">01 · Prediction</span><h3>{gain:.1%} lower Brier</h3><p>State + prior Elo improves on gold + side in the chronological evaluation. Against state alone, ΔBrier is {r['comparisons'][2]['delta_brier']:+.5f}; the cluster interval is below zero.</p></article><article><span class="eyebrow">02 · Trajectory</span><h3>More features ≠ better forecasts</h3><p>Adding 15→20 changes to state + Elo worsens Brier by {r['comparisons'][3]['delta_brier']:.5f}. This is a result about these linear models and this sample, not proof that momentum never matters.</p></article><article><span class="eyebrow">03 · LPL</span><h3>High conversion can be expected</h3><p>BLG wins {blg.wins}/{blg.n} held-out leading games. The model expected {blg.expected_wins:.1f} wins: an adjusted residual of only {blg.adjusted_pp:+.2f} pp. Raw conversion alone would overstate the surprise.</p></article></div>''')
body += section('lpl','02 / LPL desk','All 14 teams. Coverage made visible.','The current official source restores usable LPL snapshots, but it still does not cover every game.',f'''{team_table}<p class="note">Season coverage uses January–September source games. *Performance uses only April–September chronological predictions with a positive 20-minute gold difference. Every LPL team is listed; charts require at least 15 leading games. Missingness may bias comparisons.</p>{figure('pro-teams.html','Closing relative to state + prior strength','Choose a domestic league. Residual = (wins − expected wins)/(n+30), in percentage points. Error bars resample team-days, holding predictions fixed. They exclude model uncertainty and multiple-comparison adjustment. This is descriptive, not a cross-region power ranking.','pro_teams.csv')}<div class="callout"><strong>No clear LPL closing winner in this evaluation.</strong><p>Every displayed LPL team interval includes zero. The point estimates support a review queue; they do not establish a reliable ordering of coaching quality or macro skill.</p></div>''')
body += section('transition','03 / The next five minutes','What happens to a 20-minute lead?','Track the observable transition before trying to explain the decision that caused it.',figure('pro-next5.html','From 20 minutes to 25','All eligible 20-minute leaders are counted, including games ending before 25. The 500g stability band is descriptive. This chart uses all eligible January–September games, not only evaluation games.','pro_next5.csv')+'''<p class="note">Future state is an outcome here, never an input to the 20-minute prediction. A lead shrinking does not by itself identify a bad Baron call, a lost vision contest, or a draft scaling disadvantage.</p>''')
body += section('validation','04 / Validation','Predict later matches using earlier matches.','Six monthly forward evaluations, with a seven-day training gap. No random train/test split.',model_table+comparison_table+figure('pro-calibration.html','Does the probability match the outcome?','Blue-side games only, equal-width probability bins. Point size shows sample count. This diagnostic does not validate individual 98% predictions.','pro_calibration.csv')+f'<details><summary>Inspect the six evaluation windows</summary>{fold_table}</details><p class="note">Lower ΔBrier is better. Intervals use 2,000 resamples of fixed held-out losses, clustered by league, day and team pair as a series proxy. Model choices and comparisons are exploratory; no untouched final test set or multiplicity adjustment is claimed.</p>')
body += section('review','05 / Match review','A draft-aware review queue.','These are source-identified matches to investigate, not replay-verified explanations.',f'<div class="case-grid">{cases}</div><div class="callout"><strong>The next evidence needed is the replay timeline.</strong><p>Check objective spawn and contest timing, recalls and item completions, vision setup, and teamfight execution against the recorded drafts. The CSV has final objective totals, not checkpoint objective timestamps. Drafts are shown as context and are not modeled as validated scaling scores. No tactical cause is assigned here.</p></div><a href="analysis/pro_lpl_review_queue.csv">Download eight LPL review candidates ↓</a>')
body += section('coverage','06 / Scope & provenance','A frozen 2026 domestic Tier 1 sample.','LPL, LCK, LEC, LCS, LCP and CBLOL. International tournaments, academy and regional leagues are excluded.',coverage_table+f'''<p class="note">{r['eligible_games']:,} of {r['provenance']['raw_games']:,} source games pass paired completeness and 15/20-minute feature checks. LPL: {int(lc.eligible_games)}/{int(lc.source_games)} ({lc.coverage_pct:.1f}%). Coverage refers to the downloaded source, not an independently verified tournament census. Snapshot retrieved 13 September 2026; September is incomplete.</p><div class="method-grid"><article><h3>What the benchmark knows</h3><p>Gold, XP, CS and kill differences at 20 minutes, map side and league-local pre-day Elo. Elo starts every team at 1500 in January, uses fixed K=20, and updates only after the day ends. Available results from excluded snapshot games can update Elo.</p></article><article><h3>What it cannot identify</h3><p>No final objective, vision or plate totals enter the model. No roster adjustment, explicit patch effects, validated draft scaling model or replay annotations. Elo is a rough strength proxy. Missing games, changing rosters and patches limit interpretation.</p></article></div><p>Source: <a href="{e(r['provenance']['source_url'])}">Oracle’s Elixir official 2026 CSV</a> · <a href="https://oracleselixir.com/tools/downloads">Official downloads</a> · <a href="https://lolesports.com/en-GB/season/115547545029543948/handbook">Riot’s Tier 1 league handbook</a>.</p><div class="download-links"><a href="analysis/pro_results.json">Results & provenance</a><a href="analysis/pro_predictions.csv">Held-out predictions</a><a href="analysis/pro_eligibility.csv">Eligibility audit</a><a href="analysis/pro_team_coverage.csv">Team coverage</a><a href="analysis/pro_diagnostics.csv">League & month diagnostics</a><a href="analysis/PRO_METHODOLOGY.md">Full methodology</a><a href="history2022.html">2022 research archive ↗</a></div>''')

page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Beyond the Lead — 2026 Pro LoL Research</title><meta name="description" content="2026 professional League of Legends: LPL coverage, chronological prediction, team strength and 20-to-25-minute lead transitions."><link rel="icon" href="assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="assets/research.css"></head><body><a class="skip" href="#main">Skip to research</a><header><nav><a class="brand" href="#main"><span class="mark">LOL / LAB</span>PRO LEAGUE RESEARCH · 2026</a><div class="nav-links"><a href="#lpl">LPL desk</a><a href="#transition">Next 5 min</a><a href="#validation">Validation</a><a href="#review">Review</a><a class="repo-link" href="{repo}">GitHub ↗</a></div></nav></header><main id="main"><div class="hero"><div class="hero-copy"><div class="eyebrow"><span class="dot"></span>2026 season · six domestic leagues · frozen 13 Sep</div><h1>Beyond<br>the <em>lead.</em></h1><p class="deck">Who converts an advantage beyond what we should already expect?</p><p class="intro">A 20-minute research desk with LPL coverage, earlier-only evaluation and a team-strength baseline. Verified numbers, visible uncertainty, and matches worth reviewing.</p><a class="button primary" href="#lpl">Explore the LPL desk ↗</a></div><div class="hero-panel"><div class="eyebrow">LPL / actual snapshot coverage</div><div class="hero-number">{int(lc.eligible_games)}<span style="font-size:24px;color:#88a6ca"> / {int(lc.source_games)}</span></div><p>games eligible for the 20-minute study</p><p class="intro">All {r['lpl_teams']} teams appear in the coverage audit. Incomplete games remain visible instead of being silently treated as usable.</p><a href="#coverage" style="color:#69deff">Inspect the inclusion rules ↓</a></div></div><div class="shell stats"><div><strong>{r['eligible_games']:,}</strong><span>eligible season games</span></div><div><strong>{r['test_games']:,}</strong><span>chronological test games</span></div><div><strong>6</strong><span>domestic Tier 1 leagues</span></div><div><strong>20 → 25</strong><span>lead transition window</span></div></div>{body}<section class="shell closing"><h2>A stronger research desk.<br><em>Not yet a tactical verdict.</em></h2><p>The evidence supports better benchmarking and targeted review. Explaining why a team lost its advantage still requires the match timeline and replay.</p></section></main><footer class="shell"><span>Research by Ethan Cai · Data: Oracle’s Elixir / Tim Sevenhuysen</span><a href="history2022.html">Original 2022 study</a></footer></body></html>'''
# Called after the historical builder and validator in the workflow.
if 'Beyond the Lead — 2026' not in (ROOT/'index.html').read_text(encoding='utf-8'):
    shutil.copy2(ROOT/'index.html',ROOT/'history2022.html')
    (ROOT/'history2022.html').write_text((ROOT/'history2022.html').read_text(encoding='utf-8').replace('<body>','<body><div class="note" style="margin:0">Historical 2022 study. <a href="index.html">Open the 2026 research desk, including LPL →</a></div>',1),encoding='utf-8')
(ROOT/'index.html').write_text(page,encoding='utf-8')
readme = f'''# Beyond the Lead — 2026 Pro LoL Research

[Public research desk](https://gyjdb.github.io/lol-early-game-analysis/) · [2022 archive](https://gyjdb.github.io/lol-early-game-analysis/history2022.html)

Which teams convert a 20-minute advantage beyond what their state and prior strength predict?

The current study covers six domestic Tier 1 leagues: LPL, LCK, LEC, LCS, LCP and CBLOL. The official Oracle’s Elixir snapshot retrieved 13 September 2026 contains {r['provenance']['raw_games']:,} source games; {r['eligible_games']:,} pass paired 15/20-minute feature checks. **LPL is included: {int(lc.eligible_games)}/{int(lc.source_games)} source games, all 14 teams in the audit.** September is incomplete. This is a frozen study, not a live data feed.

## Verified findings

- Across {r['test_games']:,} April–September forward-evaluated games, gold + side Brier is {metrics['gold']['brier']:.5f}; state + prior Elo is {metrics['strength']['brier']:.5f} ({gain:.1%} lower).
- State + Elo versus state: ΔBrier {r['comparisons'][2]['delta_brier']:+.5f}, 95% cluster interval [{r['comparisons'][2]['low']:+.5f}, {r['comparisons'][2]['high']:+.5f}].
- Adding trajectory to state + Elo worsens Brier: {r['comparisons'][3]['delta_brier']:+.5f}, interval [{r['comparisons'][3]['low']:+.5f}, {r['comparisons'][3]['high']:+.5f}]. This does not establish a universal claim about momentum.
- BLG wins {blg.wins}/{blg.n} held-out games when ahead at 20; expected wins are {blg.expected_wins:.1f}. Its shrunken residual is {blg.adjusted_pp:+.2f} percentage points. All displayed LPL closing intervals include zero; no reliable best-closing team is established.

## Methods and evidence

Monthly expanding training windows use a seven-day gap. All models share the same eligible games and fixed regularized linear specification. Elo uses only earlier days, league-local season-start 1500 and K=20. Final objectives, vision and plates are excluded. Drafts accompany review cases but are not used as scaling predictors.

[Methodology](analysis/PRO_METHODOLOGY.md), [results](analysis/pro_results.json), [predictions](analysis/pro_predictions.csv), [eligibility](analysis/pro_eligibility.csv), [team coverage](analysis/pro_team_coverage.csv), [team residuals](analysis/pro_teams.csv), [next-five-minute outcomes](analysis/pro_next5.csv), [LPL review candidates](analysis/pro_lpl_review_queue.csv).

Intervals resample fixed held-out losses by series proxy (league/day/team-pair); team intervals resample team-days. No model-refit uncertainty, multiple-comparison correction or untouched final test is claimed. Team shrinkage uses fixed n/(n+30); charts require n≥15. Missing data, patch and roster changes can bias comparisons. Review cases are source-identified, not replay-verified tactical explanations.

## Reproduce

```sh
python -m pip install -r analysis/requirements.txt
python analysis/analyze_v2.py
python analysis/build_site.py
python analysis/validate.py
python analysis/analyze_pro.py
python analysis/build_pro_site.py
python analysis/validate_pro.py
python analysis/prepare_pages.py
```

The first three steps preserve and validate the original 2022 study. The 2026 analysis uses the committed compressed six-league team snapshot; it never depends on a mutable daily download. [Snapshot provenance](analysis/data/provenance_2026.json) includes both full-source and subset SHA-256 values. `freeze_2026.py RAW_CSV` reproduces the subset from the matching original file. Source: [Oracle’s Elixir official downloads](https://oracleselixir.com/tools/downloads). Data by Tim Sevenhuysen.

GitHub Actions regenerates and validates both studies, commits results and deploys Pages. The 2022 archive retains its separate 10,522-game sample and historical findings; its missing LPL data does not describe the 2026 source.

Built by Ethan Cai.
'''
(ROOT/'README.md').write_text(readme,encoding='utf-8')
print('Built current homepage and preserved 2022 archive')
