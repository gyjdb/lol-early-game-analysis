"""Build both public narratives from the verified, reproducible outputs."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / 'analysis/results.json').read_text(encoding='utf-8'))
T = R['timepoints']
P = T['20']
REPO = 'https://github.com/gyjdb/lol-early-game-analysis'
LIVE = 'https://gyjdb.github.io/lol-early-game-analysis/'
e = lambda x: html.escape(str(x))
pct = lambda x: f'{x:.1%}'
composition, momentum = P['composition_comparison'], P['trajectory_comparison']
improvement = 1 - P['current_state']['brier'] / P['gold_only']['brier']
trajectory = {x['trajectory'].split()[0]: x for x in R['trajectory_20_moderate_leads']}
growing, shrinking = trajectory['Growing'], trajectory['Shrinking']
cl, cb = R['closing_top'][0], R['comeback_top'][0]
holdout = R['temporal_holdout_20']
assert composition['ci_high'] < 0, 'Review composition narrative against new results.'
assert momentum['ci_low'] < 0 < momentum['ci_high'], 'Review momentum narrative against new results.'
assert cl['teamname'] == 'Counter Logic Gaming' and cb['teamname'] == 'T1', 'Review team narrative.'


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th scope="col">{e(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{e(v)}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def figure(file, title, caption, csv):
    return f'<figure><div class="figure-header"><h3>{title}</h3><a href="assets/{file}" target="_blank" rel="noopener">Open chart ↗</a></div><iframe src="assets/{file}" title="{e(title)}" loading="lazy"></iframe><figcaption>{caption} <a href="analysis/{csv}">Download CSV ↓</a></figcaption></figure>'


def section(id, label, title, intro, body):
    return f'<section id="{id}" class="shell section"><div class="section-heading"><span class="eyebrow">{label}</span><h2>{title}</h2><p>{intro}</p></div>{body}</section>'


def case(x, label):
    return f'<article class="case"><div class="eyebrow">{label} · {e(x["league"])}</div><h3>{e(x["teamname"])}</h3><p>vs. {e(x["opponent"])} · {e(x["date"][:10])}</p><div class="case-gold">+{x["golddiffat20"]:,.0f}<span>gold at 20 minutes</span></div><div class="probability"><span>Gold + side <b>{pct(x["p_gold"])}</b></span><span>Full state <b>{pct(x["p_state"])}</b></span></div><p>XP difference: {x["xpdiffat20"]:+,.0f}. Five-minute gold change: {x["gold_velocity"]:+,.0f}. Final result: <strong>{"win" if x["result"] else "loss"}</strong>.</p><small>{e(x["gameid"])}</small></article>'


threshold_rows = [[f'{t} min', f'{v["games_reaching_checkpoint"]:,}', *[f'+{v["thresholds_gold_diff"][k]:,.0f}' for k in ['p80', 'p90', 'p95']]] for t, v in T.items()]
model_rows = [[name, f'{P[key]["brier"]:.5f}', f'{P[key]["log_loss"]:.5f}', f'{P[key]["auc"]:.5f}'] for name, key in [('Gold + map side', 'gold_only'), ('Current state', 'current_state'), ('State + trajectory', 'state_trajectory')]]
threshold_note = f'Thresholds are approximate crossings on a 600-point fitted grid within the observed 0.5th–99.5th percentile gold range. No threshold uncertainty interval is estimated. At 25 minutes, {T["25"]["games_ended_before_checkpoint"]:,} complete games have already ended and four more have no usable snapshot. The sample has changed.'
comparison_note = f'Current state minus gold: ΔBrier {composition["delta_brier"]:+.5f}, 95% paired game-bootstrap interval [{composition["ci_low"]:+.5f}, {composition["ci_high"]:+.5f}]. Trajectory minus current state: {momentum["delta_brier"]:+.5f}, interval [{momentum["ci_low"]:+.5f}, {momentum["ci_high"]:+.5f}]. The latter includes zero.'
holdout_note = f'The September–December holdout agrees directionally: current state Brier {holdout[1]["brier"]:.5f}, trajectory {holdout[2]["brier"]:.5f}, across {holdout[0]["test_games"]:,} games after training on {holdout[0]["train_games"]:,} January–August games. This does not prove momentum never matters; these features did not establish an additional benefit.'
rank_note = 'Bars show mean residual × n/(n+30), requiring n≥20. Error bars are approximate 95% intervals with the same shrinkage applied. They exclude model uncertainty and multiple-comparison adjustment. The factor 30 is a fixed heuristic, not fitted hierarchical pooling.'

body = section('findings', 'THE FINDINGS', 'The scoreboard is only<br>part of the answer.', '', f'''<div class="findings-grid"><article><span class="index">01</span><h3>A fixed gold lead means less later.</h3><p>The fitted 90% benchmark rises from +{T['10']['thresholds_gold_diff']['p90']:,.0f} at 10 minutes to +{T['25']['thresholds_gold_diff']['p90']:,.0f} at 25. These are different surviving-match samples, not the path of one game.</p></article><article><span class="index">02</span><h3>Current state adds information.</h3><p>At 20 minutes, XP, CS, kills, plates and interactions reduce Brier score from {P['gold_only']['brier']:.5f} to {P['current_state']['brier']:.5f}. Lower is better.</p></article><article><span class="index">03</span><h3>Momentum is not the headline.</h3><p>Growing leads win more often in the raw comparison. Once current state is modeled, five-minute changes give no clear extra improvement in held-out probability quality.</p></article></div>''')
body += section('thresholds', '01 / LEAD VALUE', 'How much gold buys confidence?', '“Up 3K” is incomplete without the clock. Compare the same gold difference at different stages.', figure('dynamic-win-probability.html', 'Gold advantage, four points in time', 'Curves fit the full eligible sample and average across Blue and Red side. These are descriptive model estimates; evaluation below uses unseen games.', 'win_probability_curves.csv') + table(['Checkpoint', 'Eligible games', '80% estimate', '90% estimate', '95% estimate'], threshold_rows) + f'<p class="note">{threshold_note}</p>')
body += section('momentum', '02 / TESTING MOMENTUM', 'A growing lead looks better.<br>Does its history explain why?', 'For teams ahead by 1,000–3,000 gold at 20 minutes, growing leads convert more often than shrinking ones.', f'''<div class="comparison"><div><span>GROWING LEAD</span><strong>{pct(growing['win_rate'])}</strong><small>{growing['n']:,} matches · mean lead +{growing['mean_gold']:,.0f}</small></div><div><span>SHRINKING LEAD</span><strong>{pct(shrinking['win_rate'])}</strong><small>{shrinking['n']:,} matches · mean lead +{shrinking['mean_gold']:,.0f}</small></div><p>The groups still differ in current gold and resources. This <b>{100*(growing['win_rate']-shrinking['win_rate']):.1f}-point gap</b> is descriptive, not an independent momentum effect.</p></div>''' + figure('lead-trajectory-20.html', 'Growing, stable and shrinking game states', 'Five-minute gold-difference change: growing >500g, stable −500g to +500g inclusive, shrinking <−500g. Points show 1,000g bins with at least 40 observations. No causal adjustment or uncertainty bars.', 'lead_trajectory.csv') + '<h3>The stronger test: add history to the same state model.</h3><p>Both richer models use the same regularized interaction pipeline. Trajectory adds five-minute changes in gold, XP and CS. All models use identical game-grouped folds.</p>' + table(['20-minute model', 'Brier ↓', 'Log loss ↓', 'ROC AUC ↑'], model_rows) + f'<div class="callout"><strong>Current state helps. Extra trajectory is inconclusive.</strong><p>{comparison_note}</p><p>{holdout_note}</p></div>' + figure('state-vs-gold-brier.html', 'Probability quality across checkpoints', 'Five-fold out-of-fold scores. Current state and trajectory coincide at 10 minutes because there is no earlier checkpoint. Cross-time scores also reflect changing match populations.', 'model_performance.csv') + figure('calibration-20.html', 'Do probabilities match observed win rates?', '20-minute out-of-fold predictions, Blue side only so each match appears once. The dashed diagonal is ideal calibration. Bins span 10 probability points; hover for sample sizes. This diagnostic is not proof of perfect calibration.', 'calibration_20.csv'))
body += section('quality', '03 / LEAD QUALITY', 'Same scoreboard.<br>Different warning signs.', 'Lead quality is the trajectory model’s probability minus the gold-and-side benchmark. It describes disagreement between two models, not an intrinsic measure of team skill.', '<div class="case-grid">' + case(R['fragile_examples'][0], 'Gold overstates the state') + case(R['robust_examples'][0], 'Gold understates the state') + '</div><p class="note">The fragile example still won. A lower probability does not mean a comeback must happen. These cases were selected after analysis, not issued as prospective scouting calls.</p>' + figure('lead-quality-20.html', 'Where the full model disagrees with gold', 'Positive 20-minute leads, up to 7,000 sampled observations. Above zero: full state rates the lead more highly. Below zero: less highly. Color shows final outcome, which is never used as a feature.', 'predictions_20.csv') + '<details><summary>Model-flagged extreme games and downloadable cases</summary><p>Extreme-game files rank unexpected outcomes by out-of-fold probability, not gold lost or recovered. Unusual records, remakes or data errors may appear; no replay has been reviewed. Treat these as a review queue, not verified “biggest throws.”</p><div class="download-links"><a href="analysis/fragile_leads.csv">Fragile leads ↓</a><a href="analysis/robust_leads.csv">Robust leads ↓</a><a href="analysis/largest_throws.csv">Unexpected losses ↓</a><a href="analysis/largest_comebacks.csv">Unexpected wins ↓</a></div></details>')
body += section('teams', '04 / TEAM CONVERSION', 'Who beat the state-based expectation?', 'Count actual wins, subtract expected wins, then shrink the per-game residual toward zero. At least 20 qualifying matches are required within a league.', f'''<div class="case-grid"><article class="case"><span class="eyebrow">AHEAD AT 20 MINUTES</span><h3>{e(cl['teamname'])}</h3><div class="case-gold">{cl['actual_wins']} / {cl['n']}<span>wins from leads</span></div><p>{cl['expected_wins']:.1f} expected wins. Shrunken residual: <b>{cl['efficiency_pp']:+.2f} percentage points</b>. This is close to G2’s estimate; their ordering should not be treated as decisive.</p></article><article class="case"><span class="eyebrow">BEHIND AT 20 MINUTES</span><h3>{e(cb['teamname'])}</h3><div class="case-gold">{cb['actual_wins']} / {cb['n']}<span>wins from deficits</span></div><p>{cb['expected_wins']:.2f} expected wins. Shrunken residual: <b>{cb['efficiency_pp']:+.2f} percentage points</b>. A retrospective 2022 signal for review, not a current team rating.</p></article></div>''' + figure('closing-efficiency.html', 'Converting positive gold states', 'Eligible LCK, LCS and LEC team-seasons; LPL has no complete records. ' + rank_note, 'closing_efficiency.csv') + figure('comeback-resilience.html', 'Winning from negative gold states', 'Same calculation for teams behind at 20 minutes. Residuals may reflect draft, opponents, objectives and calibration error; they do not isolate closing skill. ' + rank_note, 'comeback_resilience.csv') + figure('league-90-threshold.html', 'Regional benchmarks at 20 minutes', 'Separate descriptive gold-and-side models. Different schedules, team strength and sample sizes prevent causal league comparisons. No LPL estimate is available.', 'league_thresholds.csv'))
body += section('methods', '05 / REPRODUCIBILITY & LIMITS', 'Every headline has a trail.', 'The homepage and README are generated from the same results. The source is pinned to an immutable revision and checked by SHA-256.', f'''<div class="method-grid"><article><h3>Sample & coverage</h3><p>{R['raw_games']:,} source matches → exclude {R['excluded_partial_games']:,} partial matches → {R['games']:,} complete matches. The source includes professional, academy, regional and collegiate competition. “Complete” is a provider flag, not a guarantee that every feature is present. About 0.42% of 20-minute plate values are imputed inside training folds.</p><p>LPL is entirely excluded by the completeness filter. Regional comparisons cover LCK, LCS and LEC only. Later checkpoints condition on games still running.</p></article><article><h3>Validation & leakage</h3><p>Five-fold GroupKFold keeps both sides of each game together. Preprocessing fits inside training folds. Opposing probabilities are reconciled to sum to one. A September–December holdout tests generalization after January–August training.</p><p>Only checkpoint fields, map side, prior snapshots and plates from 15 minutes onward enter models. Final gold, dragons, barons, winner and duration are not features. Duration only determines eligibility.</p></article><article><h3>What the models cannot see</h3><p>Draft, scaling, vision, checkpoint objectives, player strength and series context are absent. Game-grouped folds can share teams, patches and series. These are retrospective 2022 associations.</p><p>Paired bootstrap intervals resample games from fixed out-of-fold losses (1,000 draws). They omit model-refitting uncertainty and are not adjusted for multiple checkpoint comparisons.</p></article></div><div class="download-links"><a href="analysis/results.json">Results JSON ↓</a><a href="analysis/predictions_20.csv">20-minute predictions ↓</a><a href="analysis/league_coverage.csv">Coverage CSV ↓</a><a href="analysis/temporal_holdout_20.csv">Time holdout CSV ↓</a><a href="{REPO}/blob/main/analysis/METHODOLOGY.md">Full methodology ↗</a><a href="{REPO}/actions/workflows/rebuild-analysis.yml">Rebuild workflow ↗</a><a href="{e(R['source_url'])}">Pinned source CSV ↗</a></div>''')
thresholds = {t: v['thresholds_gold_diff'] for t, v in T.items()}
site = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>The Point of No Return · League of Legends Analysis</title><meta name="description" content="10,522 complete competitive League of Legends matches: how lead value changes over time, why current state matters, and where momentum adds little."><meta property="og:title" content="The Point of No Return"><meta property="og:description" content="Gold measures the lead. Game state tests its strength. A reproducible study of 2022 competitive League of Legends."><meta property="og:type" content="website"><meta property="og:url" content="{LIVE}"><link rel="canonical" href="{LIVE}"><link rel="icon" href="assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="assets/research.css"></head><body><a class="skip" href="#findings">Skip to findings</a><header><nav aria-label="Main navigation"><a class="brand" href="#top"><span class="mark">PNR</span> ETHAN CAI <span class="brand-divider">/</span> ESPORTS RESEARCH</a><div class="nav-links"><a href="#thresholds">Lead value</a><a href="#momentum">Momentum</a><a href="#teams">Teams</a><a href="#methods">Methods</a><a href="{REPO}" class="repo-link">GitHub ↗</a></div></nav></header><main id="top"><section class="hero shell"><div class="hero-copy"><div class="eyebrow"><span class="dot"></span> LEAGUE OF LEGENDS · 2022 MATCH STUDY</div><h1>The point of<br><em>no return.</em></h1><p class="deck">When is a lead actually safe?</p><p class="intro">Gold measures the size of an advantage. Its timing and the game state tell a more useful story. An analysis of <strong>{R['games']:,} complete competitive matches</strong> across four checkpoints.</p><a class="button" href="#findings">Explore the findings <span>↓</span></a></div><aside class="hero-panel" aria-label="Explore model-estimated gold thresholds"><div class="panel-top"><span>THE 90% BENCHMARK</span><span class="live-label">FITTED ESTIMATE</span></div><div class="checkpoint-buttons" role="group" aria-label="Choose game minute">{''.join(f'<button type="button" data-minute="{t}" aria-pressed="{str(t=="20").lower()}">{t} min</button>' for t in T)}</div><div class="hero-number" id="threshold-value">+{P['thresholds_gold_diff']['p90']:,.0f}<span>GOLD</span></div><p id="threshold-note">At 20 minutes, the fitted gold model reaches 90% win probability around this lead.</p><div class="prob-bar"><span></span><i>90%</i></div><div class="panel-bottom"><span>Still a chance to lose.</span><span>1 IN 10</span></div><small>Side-averaged logistic estimate, conditional on reaching the checkpoint. A benchmark, not a guarantee.</small></aside></section><section class="stats shell" aria-label="Study at a glance"><div><strong>{R['games']:,}</strong><span>complete matches</span></div><div><strong>10 → 25</strong><span>minutes, four checkpoints</span></div><div><strong>{improvement:.2%}</strong><span>relative Brier reduction from current state at 20′</span></div><div><strong>No clear gain</strong><span>from adding trajectory at 20′</span></div></section>{body}<section class="closing shell"><span class="eyebrow">ANALYST TAKEAWAY</span><h2>Read the clock.<br>Check the resources.<br><em>Don’t overclaim momentum.</em></h2><p>For these 2022 matches, richer current state improves the baseline. The recent trajectory mostly repeats information already on the scoreboard.</p></section></main><footer class="shell"><span>ETHAN CAI · THE POINT OF NO RETURN</span><a href="{REPO}">Source & reproducibility ↗</a><a href="#top">Back to top ↑</a></footer><script>const thresholds={json.dumps(thresholds)};document.querySelectorAll('[data-minute]').forEach(button=>button.addEventListener('click',()=>{{const t=button.dataset.minute;document.querySelectorAll('[data-minute]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));document.getElementById('threshold-value').innerHTML='+'+Math.round(thresholds[t].p90).toLocaleString('en-US')+'<span>GOLD</span>';document.getElementById('threshold-note').textContent='At '+t+' minutes, the fitted gold model reaches 90% win probability around this lead.';}}));</script></body></html>'''
(ROOT / 'index.html').write_text(site, encoding='utf-8')
readme = f'''# The Point of No Return

**When is a League of Legends lead safe—and what does its recent history add?**

[Interactive study]({LIVE}) · [Methodology](analysis/METHODOLOGY.md) · [Results](analysis/results.json) · [Workflow]({REPO}/actions/workflows/rebuild-analysis.yml)

An analysis of **{R['games']:,} complete 2022 competitive matches** from Oracle’s Elixir, with snapshots at 10, 15, 20 and 25 minutes. The sample includes professional, academy, regional and collegiate competition.

## Verified findings

- **A fixed gold lead means less later.** The fitted 90% threshold rises from +{T['10']['thresholds_gold_diff']['p90']:,.0f} at 10 minutes to +{T['25']['thresholds_gold_diff']['p90']:,.0f} at 25. This compares changing surviving-match samples, not one game’s trajectory.
- **Current state adds information.** At 20 minutes, resources and interactions reduce out-of-fold Brier score from **{P['gold_only']['brier']:.5f} to {P['current_state']['brier']:.5f}** ({improvement:.2%} relative reduction).
- **Extra momentum features do not establish further improvement.** Trajectory Brier is {P['state_trajectory']['brier']:.5f}. {comparison_note}
- **The raw momentum gap is confounded.** For +1K–3K leads at 20 minutes, growing leads win {pct(growing['win_rate'])} ({growing['n']:,} matches), shrinking leads {pct(shrinking['win_rate'])} ({shrinking['n']:,}). Their mean current leads differ: +{growing['mean_gold']:,.0f} vs. +{shrinking['mean_gold']:,.0f}. This does not isolate momentum.
- **Team residuals are descriptive.** {cl['teamname']} wins {cl['actual_wins']}/{cl['n']} leads against {cl['expected_wins']:.1f} expected wins. {cb['teamname']} wins {cb['actual_wins']}/{cb['n']} deficits against {cb['expected_wins']:.2f} expected. These are retrospective signals, not causal skill ratings.

## Gold thresholds

| Checkpoint | Eligible games | 80% | 90% | 95% |
| --- | ---: | ---: | ---: | ---: |
''' + '\n'.join('| ' + ' | '.join(row) + ' |' for row in threshold_rows) + f'''

{threshold_note}

## Held-out model quality at 20 minutes

| Model | Brier ↓ | Log loss ↓ | AUC ↑ |
| --- | ---: | ---: | ---: |
''' + '\n'.join('| ' + ' | '.join(row) + ' |' for row in model_rows) + f'''

Models use identical five-fold GroupKFold splits by game ID. Both sides stay together; preprocessing fits inside each training fold. Current state adds XP, CS, kills and plates with second-order interactions. Trajectory adds five-minute gold, XP and CS changes. At 10 minutes there is no trajectory history. “Gold-only” shorthand includes map side.

{holdout_note}

## Reproduce

Python 3.11–3.13:

```sh
python -m pip install -r analysis/requirements.txt
python analysis/analyze_v2.py
python analysis/build_site.py
python analysis/validate.py
python -m http.server 8000
```

Open `http://localhost:8000`. The first run downloads the pinned source to `.cache/lol2022.csv`; `LOL_DATA_PATH` can point to another local copy. SHA-256 enforces the exact dataset. The workflow regenerates CSVs, charts, homepage and README, validates them, commits outputs and explicitly deploys GitHub Pages.

## Evidence

- [All results](analysis/results.json), [20-minute predictions](analysis/predictions_20.csv) with game IDs, folds and opposing probabilities
- [Probability curves](analysis/win_probability_curves.csv), [model performance](analysis/model_performance.csv), [calibration](analysis/calibration_20.csv), [time holdout](analysis/temporal_holdout_20.csv)
- [Trajectory groups](analysis/trajectory_summary.csv), [closing residuals](analysis/closing_efficiency.csv), [comeback residuals](analysis/comeback_resilience.csv), [regional thresholds](analysis/league_thresholds.csv)
- [Fragile](analysis/fragile_leads.csv) and [robust](analysis/robust_leads.csv) model-disagreement cases
- [Unexpected losses](analysis/largest_throws.csv) and [unexpected wins](analysis/largest_comebacks.csv): model-flagged review queues, not replay-verified throw rankings
- [League coverage](analysis/league_coverage.csv)

## Scope and limitations

The source has {R['raw_games']:,} matches; {R['excluded_partial_games']:,} partial matches are excluded. **LPL has no complete matches**, so regional/team comparisons cover LCK, LCS and LEC. Later checkpoints condition on survival. Around 0.42% of 20-minute plate values require fold-local imputation.

No final objective totals enter the models. Plates are used from 15 minutes onward, after their 14-minute expiration in 2022. Draft, scaling, vision, checkpoint objectives, opponent strength and series context are absent. Teams, patches and series can recur across folds. This is retrospective to 2022, not a current live-prediction system.

Lead quality is `P(state + trajectory) − P(gold + side)`. Team residuals are `(actual wins − expected wins)/(n+30)` with n≥20, by league/team and ahead/behind condition. {rank_note} Bootstrap score intervals use 1,000 paired game resamples of fixed out-of-fold losses, not model refits; checkpoint comparisons are not multiplicity-adjusted.

## Provenance

[Oracle’s Elixir](https://oracleselixir.com/) data through an [immutable public mirror]({R['source_url']}).

SHA-256: `{R['source_sha256']}`

The homepage and README are generated by `analysis/build_site.py` from verified outputs. Earlier course visuals remain as historical assets; their old accuracy/fairness results are not evidence for this study.

Built by **Ethan Cai**.
'''
(ROOT / 'README.md').write_text(readme, encoding='utf-8')
print('Built homepage and README from analysis/results.json')
