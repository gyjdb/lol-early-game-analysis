---
layout: default
title: Early Gold, Final Victory
---

<style>
.wrapper {
  width: 1120px;
}

header {
  width: 240px;
}

section {
  width: 820px;
}

iframe {
  width: 100%;
  max-width: 100%;
  height: 500px;
  border: 0;
}

table {
  display: block;
  width: 100%;
  overflow-x: auto;
  white-space: nowrap;
  font-size: 14px;
}

@media screen and (max-width: 960px) {
  .wrapper {
    width: auto;
    margin: 0 20px;
  }

  header,
  section,
  footer {
    float: none;
    position: static;
    width: auto;
  }

  section {
    padding: 20px 0;
    border: 0;
  }
}
</style>

# Early Gold, Final Victory

# Early Gold, Final Victory

**Ethan Cai**

An analysis of 2022 professional League of Legends matches from Oracle's Elixir, focused on early-game advantages at 15 minutes and match outcomes.

## Introduction

League of Legends is a five-versus-five game in which teams build advantages through gold, XP, CS, kills, and map pressure.

**Question:** How strongly is a gold advantage at 15 minutes associated with whether the Blue-side team eventually wins?

The raw dataset contains **148,980 rows**. I use one Blue team-summary row per game for the main analysis, leaving **12,415 match observations**.

Relevant columns include `result`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `opp_killsat15`, `turretplates`, `opp_turretplates`, `gamelength`, and `playoffs`.

## Data Cleaning and Exploratory Data Analysis

I kept team-summary rows and then one Blue-side row per match, converted dates/numeric values, converted `playoffs` to Boolean, created an `outcome` label, and converted game duration to minutes. Missing timeline values were retained until a specific analysis required complete values.

A sample of the cleaned data:

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>gameid</th>
      <th>league</th>
      <th>playoffs</th>
      <th>teamname</th>
      <th>game_minutes</th>
      <th>outcome</th>
      <th>golddiffat15</th>
      <th>xpdiffat15</th>
      <th>csdiffat15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ESPORTSTMNT01_2690210</td>
      <td>LCKC</td>
      <td>False</td>
      <td>HANJIN BRION Challengers</td>
      <td>28.55</td>
      <td>Loss</td>
      <td>107.0</td>
      <td>-1617.0</td>
      <td>-23.0</td>
    </tr>
    <tr>
      <td>ESPORTSTMNT01_2690219</td>
      <td>LCKC</td>
      <td>False</td>
      <td>T1 Esports Academy</td>
      <td>35.23</td>
      <td>Loss</td>
      <td>-1763.0</td>
      <td>-906.0</td>
      <td>-22.0</td>
    </tr>
    <tr>
      <td>ESPORTSTMNT01_2690227</td>
      <td>LCKC</td>
      <td>False</td>
      <td>KT Rolster Challengers</td>
      <td>32.87</td>
      <td>Win</td>
      <td>1191.0</td>
      <td>2298.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <td>ESPORTSTMNT01_2690255</td>
      <td>LCKC</td>
      <td>False</td>
      <td>Dplus Kia Challengers</td>
      <td>41.47</td>
      <td>Loss</td>
      <td>550.0</td>
      <td>-1259.0</td>
      <td>-40.0</td>
    </tr>
    <tr>
      <td>ESPORTSTMNT01_2690264</td>
      <td>LCKC</td>
      <td>False</td>
      <td>DN SOOPers Challengers</td>
      <td>33.67</td>
      <td>Win</td>
      <td>1478.0</td>
      <td>204.0</td>
      <td>-9.0</td>
    </tr>
  </tbody>
</table>

### Univariate Analysis

<iframe src="assets/gold-diff-distribution.html" width="800" height="500" frameborder="0"></iframe>

The distribution shows the range of early gold leads and deficits at 15 minutes.

<iframe src="assets/game-length-distribution.html" width="800" height="500" frameborder="0"></iframe>

Game length matters because later timeline values cannot exist after a match has ended.

### Bivariate Analysis

<iframe src="assets/gold-diff-by-outcome.html" width="800" height="500" frameborder="0"></iframe>

Blue-side wins are shifted toward more positive 15-minute gold differences, but the overlap shows that an early lead does not determine the result.

<iframe src="assets/gold-vs-xp.html" width="800" height="500" frameborder="0"></iframe>

Gold and XP advantages are related but not identical; combining several early-game signals is useful for prediction.

### Interesting Aggregates

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>outcome</th>
      <th>Matches</th>
      <th>Mean_Gold_Diff_15</th>
      <th>Median_Gold_Diff_15</th>
      <th>Mean_XP_Diff_15</th>
      <th>Mean_CS_Diff_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Loss</td>
      <td>5026</td>
      <td>-1533.13</td>
      <td>-1384.5</td>
      <td>-1076.08</td>
      <td>-18.66</td>
    </tr>
    <tr>
      <td>Win</td>
      <td>5496</td>
      <td>1874.78</td>
      <td>1687.5</td>
      <td>974.30</td>
      <td>15.77</td>
    </tr>
  </tbody>
</table>

## Assessment of Missingness

I do **not** believe `goldat25` missingness is MNAR: if a match ends before 25 minutes the value cannot exist, and observed `gamelength` explains that mechanism. Additional source/API metadata could help explain timeline values missing despite sufficiently long games.

For game length, the permutation p-value is **0.0002**. The test indicates that `goldat25` missingness depends on game length.

<iframe src="assets/missingness-game-length.html" width="800" height="500" frameborder="0"></iframe>

For side, using TVD, the permutation p-value is **1.0000**. The test provides no evidence that `goldat25` missingness depends on side.

## Hypothesis Testing

**Null hypothesis:** mean Blue-side gold difference at 15 is the same for wins and losses.

**Alternative hypothesis:** the mean is higher for Blue-side wins.

I used the difference in means and a one-sided permutation test with 10,000 permutations at alpha = 0.05. The observed difference was **3407.92 gold** and the p-value was **0.0001**. Because the p-value is below 0.05, I reject the null hypothesis and find statistically significant evidence that Blue-side wins tend to have larger 15-minute gold advantages. This is evidence of association, not causation.

## Framing a Prediction Problem

I predict `result` (Blue win/loss) from information available at 15 minutes. This is binary classification. I use accuracy because the classes are close to balanced and compare against the majority-class benchmark.

## Baseline Model

The baseline is a single sklearn Pipeline with median imputation, standardization, and logistic regression. Its three quantitative features are `golddiffat15`, `xpdiffat15`, and `csdiffat15`; no categorical encoding is needed.

- Training accuracy: **0.7123**
- Test accuracy: **0.7225**
- Majority-class test accuracy: **0.5240**

The baseline substantially exceeds the majority benchmark and has similar train/test performance.

## Final Model

The final model adds `killdiffat15`, `plate_diff`, and `relative_gold_lead`. These capture combat advantage, early structural pressure, and the proportional size of the gold lead.

I compared Logistic Regression, an RBF SVM, and Random Forest using five-fold GridSearchCV on the training data. The selected classifier is **LogisticRegression** with **C = 0.01**.

- Final training accuracy: **0.7149**
- Final test accuracy: **0.7269**
- Change from baseline: **+0.0044**

<iframe src="assets/final-confusion-matrix.html" width="700" height="500" frameborder="0"></iframe>

## Fairness Analysis

Group X is playoff matches and Group Y is regular-season matches. The metric is accuracy. The null says the two accuracies are approximately equal; the alternative says playoff accuracy is lower. The test statistic is playoff accuracy minus regular-season accuracy, with alpha = 0.05.

Playoff accuracy is **0.7353**, regular-season accuracy is **0.7251**, the observed difference is **+0.0102**, and the one-sided permutation p-value is **0.6845**. I fail to reject the null hypothesis; there is not sufficient evidence that the model performs worse on playoff matches.

<iframe src="assets/fairness-null.html" width="800" height="500" frameborder="0"></iframe>
