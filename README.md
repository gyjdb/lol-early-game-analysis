# Beyond the Lead — 2026 Pro LoL Research

[Public research desk](https://gyjdb.github.io/lol-early-game-analysis/) · [2022 archive](https://gyjdb.github.io/lol-early-game-analysis/history2022.html)

Which teams convert a 20-minute advantage beyond what their state and prior strength predict?

The current study covers six domestic Tier 1 leagues: LPL, LCK, LEC, LCS, LCP and CBLOL. The official Oracle’s Elixir snapshot retrieved 13 September 2026 contains 2,302 source games; 2,161 pass paired 15/20-minute feature checks. **LPL is included: 535/674 source games, all 14 teams in the audit.** September is incomplete. This is a frozen study, not a live data feed.

## Verified findings

- Across 1,515 April–September forward-evaluated games, gold + side Brier is 0.16050; state + prior Elo is 0.15523 (3.3% lower).
- State + Elo versus state: ΔBrier -0.00305, 95% cluster interval [-0.00558, -0.00052].
- Adding trajectory to state + Elo worsens Brier: +0.00195, interval [+0.00066, +0.00318]. This does not establish a universal claim about momentum.
- BLG wins 53/62 held-out games when ahead at 20; expected wins are 52.9. Its shrunken residual is +0.09 percentage points. All displayed LPL closing intervals include zero; no reliable best-closing team is established.

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
