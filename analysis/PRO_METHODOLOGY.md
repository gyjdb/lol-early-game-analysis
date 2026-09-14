# 2026 domestic Tier 1 research protocol

## Source and population

Official Oracle’s Elixir CSV downloaded 13 September 2026. The exact full-source hash and compressed subset hash are in `data/provenance_2026.json`. Freeze the source rather than downloading a different file on each rebuild. The subset contains team rows for LPL, LCK, LEC, LCS, LCP and CBLOL and champion labels joined from player rows. International, academy and regional tournaments are excluded. Coverage is relative to the source, not a separately audited schedule.

The unit is a game pair. Require two opposing team rows, one winner, complete provider labels, duration >=20 minutes and nonmissing gold/XP/CS/kill differences at 20 plus gold/XP/CS changes from 15. Both rows must pass. We do not impute missing LPL snapshots or mistake the provider's complete label for sufficient evidence. Differences must reconcile across opposing sides. Eligibility audit includes excluded rows. Seasonal per-team coverage and held-out performance have different denominators.

## Forecasting

Five fixed logistic models, each StandardScaler + L2 logistic regression C=0.1: gold + side; current state (gold/XP/CS/kill differences + side); state + 15-to-20 differences; state + prior Elo; and state + prior Elo + trajectory. Scaling is fitted only on each training window. Opposing probabilities are reconciled as (p_team + 1 - p_opponent)/2. No final objective, ward, plate or other post-checkpoint totals are features.

League-local Elo starts all teams at 1500 in January; expected score = 1/(1+10^(-difference/400)), K=20 per game. All same-day results are applied after the day, preventing same-day series results from entering a prediction. Available results from games with missing snapshots can update Elo. This is a rough prior strength proxy, not a roster-aware rating, international strength scale, or causal control sufficient to identify macro skill.

Evaluation runs monthly April through September with expanding earlier training data and a seven-day gap before each test month. Tests contain both sides of each game together; metrics use only Blue rows so each game counts once. The seven-day gap reduces series overlap at boundaries; it does not establish an exact series split. Test-month Elo can incorporate earlier-day results available before the prediction, while model coefficients remain fixed within the month. September ends at the snapshot date.

Report Brier, log loss, ROC AUC, calibration counts, all model comparisons, and regional/monthly diagnostics. There is no hyperparameter search here, but analysis choices are exploratory and no untouched final test set is claimed. Do not infer superiority from a rounded score alone.

## Uncertainty and teams

Score intervals use 2,000 bootstrap resamples of fixed held-out loss differences clustered by league/date/unordered team-pair (a series proxy). No model refits or multiple-comparison adjustment. Team residual = 100*(wins-expected wins)/(n+30), separately ahead and behind at 20; 30 is a fixed heuristic. Intervals resample team-days with the same formula. Charts display n>=15; all teams appear in coverage tables. These intervals omit fitted-model uncertainty. They are descriptive and do not justify definitive rankings, including across leagues. State + prior Elo is the reference for the team desk, with trajectory retained as an explicit ablation.

## Lead transitions and cases

For all eligible January–September 20-minute leaders, assign exactly one outcome: win/loss before 25; missing 25 snapshot; no longer ahead at 25; growth >500 gold; shrinkage >500 gold; or within 500. Final results and 25-minute state are outcomes only, never 20-minute predictors. Report counts to expose survival conditioning. The 500 gold band is a descriptive convention.

LPL review candidates are held-out losses while leading at 20, ordered by the state + Elo estimate. Game ID, date, patch, 20/25 gold and both recorded drafts are provided. No replay was reviewed and no objective decision is inferred. Extreme predicted probabilities are not separately tail-calibrated. Drafts are context, not validated champion scaling scores. A tactical explanation needs timestamped objective/vision/item events and VOD annotation; final totals cannot substitute for that evidence.

## Boundaries

Nonrandom missingness, patch changes, roster changes, small team samples, source errors and model misspecification remain. The historical 2022 study is a separate archive with different coverage and models; its estimates are not pooled into this study.
