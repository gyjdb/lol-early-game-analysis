# Early Gold, Final Victory

A portfolio data science case study on whether the first 15 minutes of a professional League of Legends match are enough to predict the winner.

**Live project:** https://gyjdb.github.io/lol-early-game-analysis/

## What this project does

Using the 2022 Oracle's Elixir professional match dataset, I analyze early-game advantages and build a binary classifier that predicts whether the Blue-side team will win using only information available by minute 15.

The project covers:

- exploratory analysis of gold, XP, CS, and match outcomes;
- missingness analysis for timeline features;
- a permutation test for the relationship between early gold advantage and victory;
- an interpretable baseline model;
- feature engineering and cross-validated model selection;
- a fairness check comparing playoff and regular-season performance.

## Key results

| Result | Value |
| --- | ---: |
| Raw dataset size | 148,980 rows |
| Match-level observations | 12,415 |
| Gold-lead permutation test | p = 0.0001 |
| Majority-class benchmark | 52.40% |
| Baseline test accuracy | 72.25% |
| Final test accuracy | **72.69%** |
| Improvement over baseline | +0.44 percentage points |
| Playoff-vs-regular fairness test | p = 0.6845 |

The final model was a regularized Logistic Regression selected through 5-fold cross-validation after comparison with an RBF SVM and Random Forest.

## Final engineered features

- `killdiffat15`: Blue kills minus Red kills at minute 15
- `plate_diff`: Blue turret plates minus Red turret plates
- `relative_gold_lead`: 15-minute gold differential relative to Blue's current gold

These complement the original baseline features `golddiffat15`, `xpdiffat15`, and `csdiffat15`.

## Main takeaway

A 15-minute gold lead is strongly associated with eventual victory, but it is not deterministic. The outcome distributions still overlap substantially, and combining multiple early-game signals produces a modest improvement over a simple three-feature baseline.

The model also showed no evidence of worse performance in playoff matches than in regular-season matches under the chosen accuracy-based fairness test.

## Project site

The GitHub Pages site presents the full analysis as an interactive case study with Plotly visualizations:

**https://gyjdb.github.io/lol-early-game-analysis/**

## Data source

2022 professional League of Legends match data from **Oracle's Elixir**.

## Notes

This repository contains the presentation layer and interactive visual assets for the analysis. The course notebook/code is intentionally not published here.

---

Built by **Ethan Cai**.
