# Research Methodology — The Point of No Return

## Research question

**When does a professional League of Legends lead become statistically safe, and which teams close games or recover from deficits better than the game state alone would predict?**

The analysis does not treat 15 minutes as a privileged timestamp. It models the game at **10, 15, 20, and 25 minutes** to study how the meaning of a lead changes through time.

## Why this is more useful than “does an early gold lead predict winning?”

A gold lead and a win are mechanically related, so simply showing that positive gold difference predicts victory has limited analytical value. The harder questions are:

1. How much gold is needed for an 80%, 90%, or 95% win probability at each stage of the game?
2. Does the composition of a lead — XP, CS, kills, structural pressure — add information beyond gold alone?
3. Does trajectory matter? Is a growing +2,000 gold lead different from a shrinking +2,000 gold lead?
4. After controlling for game state, which teams convert advantages better than expected?
5. Which teams win from losing states more often than expected?
6. Which apparent leads are fragile, and which are stronger than the scoreboard suggests?

## Leakage control

The data contains two team-summary rows per game. All out-of-fold predictions use **GroupKFold grouped by `gameid`**, so both sides of the same match always remain in the same fold.

## Dynamic checkpoints

Models are estimated at 10, 15, 20, and 25 minutes. A match is included at a checkpoint only if it reached that time and has a recorded gold difference.

## Gold-only benchmark

A regularized logistic model uses gold difference plus map side and estimates P(win). The fitted curve is used to estimate gold leads associated with **80%, 90%, and 95% win probability**.

## State + trajectory model

The richer model uses information known by the checkpoint:

- gold difference
- XP difference
- CS difference
- kill difference
- map side
- turret-plate difference from 15 minutes onward
- five-minute changes in gold, XP, and CS difference

Second-order interactions let the model distinguish game states with the same gold lead but different resource composition and momentum.

The primary metrics are **Brier score**, log loss, and ROC AUC because this is fundamentally a probability-estimation problem rather than a hard win/loss classification problem.

## Lead Quality

`Lead Quality = P(win | state + trajectory) - P(win | gold only)`

Positive values mean the full game state is stronger than gold alone suggests. Negative values identify potentially fragile leads.

## Momentum analysis

At 20 minutes, gold trajectory is classified as growing (>500g relative gain since 15), stable (±500g), or shrinking (>500g relative loss). Win rates are compared within similar current gold-difference bins.

## Closing efficiency

At 20 minutes, for teams that are ahead:

`residual = actual result - out-of-fold expected win probability`

Residuals are aggregated by team. A positive value means the team won more often than its game states predicted. Rankings are partially pooled toward zero using `mean residual × n / (n + 30)` and require at least 20 qualifying observations.

## Comeback resilience

The same expected-wins framework is applied to teams that are behind at 20 minutes. Positive residual performance means a team wins more often from losing positions than expected.

## Regional conversion

LCK, LPL, LEC, and LCS receive separate descriptive 20-minute gold-to-win-probability curves. These comparisons are descriptive rather than causal because schedules, team strength, opponents, and meta differ.

## Case studies

The analysis surfaces the largest model-implied throws, most improbable comebacks, most fragile positive leads, and most robust leads that the gold scoreboard understates.

## Data

2022 professional League of Legends match data from Oracle’s Elixir. The source provides game-state snapshots at 10, 15, 20, and 25 minutes.
