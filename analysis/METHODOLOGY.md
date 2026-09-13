# Methodology — The Point of No Return

## Data and eligibility

The immutable source URL and SHA-256 are in `results.json`. The 2022 Oracle's Elixir mirror contains 12,415 matches, including professional, academy, regional and collegiate competition. Keep provider-complete team-summary rows (participant IDs 100/200), valid binary results and exactly two opposing sides per game. This leaves 10,522 games; 1,893 partial games are excluded. LPL has no complete games in this file: its absence must not be interpreted as league performance. See `league_coverage.csv`.

At each checkpoint (10, 15, 20, 25 minutes), require duration >= checkpoint and recorded gold difference on both sides. Eligible games: 10,522 / 10,522 / 10,462 / 9,442. At 25 minutes 1,076 complete games have ended, and four surviving games lack snapshots. Across-time comparisons therefore condition on different surviving populations. About 0.42% of 20-minute plate differences are missing despite the completeness flag.

## Features and leakage

Gold baseline: checkpoint gold difference plus map side. Current state: gold, XP, CS and kill differences plus side; turret-plate difference from 15 minutes onward. In the 2022 ruleset plates expire at 14 minutes, so their final count is available at later checkpoints. Trajectory: add changes in gold, XP and CS difference since the previous five-minute checkpoint. These changes are raw five-minute differences, not per-minute rates. At 10 minutes current state and trajectory are identical.

No end-of-game gold, objective totals, duration, team identity or final result is a predictor. Duration only gates checkpoint eligibility. Draft, scaling, vision, objective timing and opponent strength are unobserved.

## Models and validation

Gold: training-fold median imputation, standardization and logistic regression (C=1, max_iter=2500). Richer models: training-fold median imputation, degree-two polynomial features including squares and pairwise interactions, standardization and logistic regression (C=0.1, max_iter=5000). Parameters are fixed, not chosen against the reported outcomes.

Five-fold GroupKFold groups by game ID, so mirrored Blue/Red observations never straddle training and validation. All models share the same folds. Preprocessing fits only on training data. To reconcile two model perspectives, set P(team) = [raw P(team) + 1 - raw P(opponent)]/2. Thus opposing probabilities sum to one. Published Brier, log loss and ROC AUC use these out-of-fold probabilities. Log loss clips to [1e-6, 1-1e-6].

Compare current state against gold to measure the joint value of added resources and model flexibility. Compare trajectory against the same current-state pipeline to isolate the predictive contribution of history. This is not a causal test, and the former comparison does not isolate any single resource.

A prespecified temporal sensitivity check trains 20-minute models on January–August and evaluates September–December (9,007 training, 1,455 test games), with the same probability reconciliation. Neither grouping scheme prevents the same team, patch or series appearing in both groups. No external-year validation is claimed.

## Uncertainty and calibration

For Brier differences, average paired loss differences within each game, then bootstrap games 1,000 times (seed 42). Percentile 95% intervals condition on fixed out-of-fold predictions; models are not refitted. These do not capture all training uncertainty and are not corrected for multiple checkpoint comparisons. Negative delta favors the added features.

Calibration uses one Blue-side prediction per game at 20 minutes, binned in ten equal-width probability bands, reporting mean prediction, observed win fraction and n. The diagonal is a diagnostic reference, not evidence of guaranteed calibration.

## Gold thresholds

Fit a gold-and-side logistic model to the full eligible sample separately at each checkpoint. Average predicted probabilities for Blue and Red across a 600-point gold grid spanning the observed 0.5th–99.5th percentiles. Report the first grid point crossing 80%, 90%, or 95%, or null when unsupported. These are approximate descriptive model crossings, not out-of-fold estimates, guarantees or causal tipping points. No threshold confidence interval is estimated. Regional thresholds repeat this within eligible leagues with at least 200 team rows.

## Trajectory and lead quality

At 20 minutes, growing means gold difference increased by >500 since minute 15; shrinking means it decreased by >500; stable includes exactly -500 and +500. Observed curves use 1,000g current-gold bins with at least 40 observations. The +1K–3K summary includes both endpoints. These broad groups retain differences in current resources and are not a controlled momentum experiment.

Lead quality = P(state + trajectory) - P(gold + side), using out-of-fold values. Positive/negative numbers measure model disagreement. They do not directly measure an intrinsic property or cause. Case selection is retrospective and outcome-labeled. Extreme unexpected-loss/win files rank model probability, not magnitude of gold reversal; unreviewed records may include data errors or remakes. Replay review is required before making specific gameplay claims.

## Team residuals

At 20 minutes, split positive and negative gold differences; exclude exact ties. Within league/team, sum actual wins and expected wins from the trajectory model. Require n>=20. Shrunken residual in percentage points = 100*(actual wins - expected wins)/(n+30). This uses a fixed heuristic shrinkage constant, not estimated hierarchical partial pooling.

Approximate plotted 95% error half-width = 1.96*100*sample_sd(result-p)/sqrt(n)*n/(n+30). This describes sampling variability conditional on fixed predictions and shrinkage, excluding model uncertainty, within-series dependence and selection/multiple-comparison effects. Rankings may reflect team strength, draft, opponents, region and calibration error. They do not identify causal closing or comeback skill. Only eligible LCK, LCS and LEC games contribute; international tournament games are not silently pooled into league rankings.

## Reproduction and publication

Run `analyze_v2.py`, `build_site.py`, then `validate.py` with dependencies in `requirements.txt`. The validator independently recomputes headline scores, team counts/residuals and threshold crossings, checks opposing probabilities and fold membership, and checks local page links and chart assets. Source checksum failures or narrative-condition changes stop publication. The workflow commits generated outputs and explicitly uploads/deploys Pages because a bot commit alone does not reliably trigger a subsequent Pages workflow.
