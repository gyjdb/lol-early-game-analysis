# The Point of No Return

**When is a League of Legends lead safe—and what does its recent history add?**

[Interactive study](https://gyjdb.github.io/lol-early-game-analysis/) · [Methodology](analysis/METHODOLOGY.md) · [Results](analysis/results.json) · [Workflow](https://github.com/gyjdb/lol-early-game-analysis/actions/workflows/rebuild-analysis.yml)

An analysis of **10,522 complete 2022 competitive matches** from Oracle’s Elixir, with snapshots at 10, 15, 20 and 25 minutes. The sample includes professional, academy, regional and collegiate competition.

## Verified findings

- **A fixed gold lead means less later.** The fitted 90% threshold rises from +2,843 at 10 minutes to +5,148 at 25. This compares changing surviving-match samples, not one game’s trajectory.
- **Current state adds information.** At 20 minutes, resources and interactions reduce out-of-fold Brier score from **0.14587 to 0.14175** (2.82% relative reduction).
- **Extra momentum features do not establish further improvement.** Trajectory Brier is 0.14175. Current state minus gold: ΔBrier -0.00412, 95% paired game-bootstrap interval [-0.00522, -0.00305]. Trajectory minus current state: +0.00000, interval [-0.00019, +0.00018]. The latter includes zero.
- **The raw momentum gap is confounded.** For +1K–3K leads at 20 minutes, growing leads win 73.6% (1,967 matches), shrinking leads 65.2% (425). Their mean current leads differ: +2,051 vs. +1,800. This does not isolate momentum.
- **Team residuals are descriptive.** Counter Logic Gaming wins 20/22 leads against 16.0 expected wins. T1 wins 13/27 deficits against 7.45 expected. These are retrospective signals, not causal skill ratings.

## Gold thresholds

| Checkpoint | Eligible games | 80% | 90% | 95% |
| --- | ---: | ---: | ---: | ---: |
| 10 min | 10,522 | +1,804 | +2,843 | +3,808 |
| 15 min | 10,522 | +2,662 | +4,197 | +5,640 |
| 20 min | 10,462 | +3,109 | +4,892 | +6,545 |
| 25 min | 9,442 | +3,253 | +5,148 | +6,941 |

Thresholds are approximate crossings on a 600-point fitted grid within the observed 0.5th–99.5th percentile gold range. No threshold uncertainty interval is estimated. At 25 minutes, 1,076 complete games have already ended and four more have no usable snapshot. The sample has changed.

## Held-out model quality at 20 minutes

| Model | Brier ↓ | Log loss ↓ | AUC ↑ |
| --- | ---: | ---: | ---: |
| Gold + map side | 0.14587 | 0.44686 | 0.87163 |
| Current state | 0.14175 | 0.43619 | 0.87843 |
| State + trajectory | 0.14175 | 0.43612 | 0.87846 |

Models use identical five-fold GroupKFold splits by game ID. Both sides stay together; preprocessing fits inside each training fold. Current state adds XP, CS, kills and plates with second-order interactions. Trajectory adds five-minute gold, XP and CS changes. At 10 minutes there is no trajectory history. “Gold-only” shorthand includes map side.

The September–December holdout agrees directionally: current state Brier 0.12650, trajectory 0.12695, across 1,455 games after training on 9,007 January–August games. This does not prove momentum never matters; these features did not establish an additional benefit.

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

The source has 12,415 matches; 1,893 partial matches are excluded. **LPL has no complete matches**, so regional/team comparisons cover LCK, LCS and LEC. Later checkpoints condition on survival. Around 0.42% of 20-minute plate values require fold-local imputation.

No final objective totals enter the models. Plates are used from 15 minutes onward, after their 14-minute expiration in 2022. Draft, scaling, vision, checkpoint objectives, opponent strength and series context are absent. Teams, patches and series can recur across folds. This is retrospective to 2022, not a current live-prediction system.

Lead quality is `P(state + trajectory) − P(gold + side)`. Team residuals are `(actual wins − expected wins)/(n+30)` with n≥20, by league/team and ahead/behind condition. Bars show mean residual × n/(n+30), requiring n≥20. Error bars are approximate 95% intervals with the same shrinkage applied. They exclude model uncertainty and multiple-comparison adjustment. The factor 30 is a fixed heuristic, not fitted hierarchical pooling. Bootstrap score intervals use 1,000 paired game resamples of fixed out-of-fold losses, not model refits; checkpoint comparisons are not multiplicity-adjusted.

## Provenance

[Oracle’s Elixir](https://oracleselixir.com/) data through an [immutable public mirror](https://raw.githubusercontent.com/twodotone/finalLOL/bec30f4031cfb11aa3424240a427997480134f79/data/csv/2022_LoL_esports_match_data_from_OraclesElixir.csv).

SHA-256: `cd14d6e3500c08e4a48d737144aa2d2dce72111f4dcf51028949a3b811832a93`

The homepage and README are generated by `analysis/build_site.py` from verified outputs. Earlier course visuals remain as historical assets; their old accuracy/fairness results are not evidence for this study.

Built by **Ethan Cai**.
