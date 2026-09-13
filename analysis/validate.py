"""Fail publication on inconsistent predictions, scores, coverage or local links."""
import json
import math
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / 'analysis/results.json').read_text(encoding='utf-8'))
p = pd.read_csv(ROOT / 'analysis/predictions_20.csv')
assert not p.duplicated(['gameid', 'side']).any()
assert p.groupby('gameid').size().eq(2).all()
assert p.groupby('gameid').fold.nunique().eq(1).all(), 'Mirror rows cross folds'
assert p.groupby('gameid').result.sum().eq(1).all()
assert p.gameid.nunique() == r['timepoints']['20']['games_reaching_checkpoint']
assert len(p) == 2 * p.gameid.nunique()
for col, key in [('p_gold', 'gold_only'), ('p_snapshot', 'current_state'), ('p_state', 'state_trajectory')]:
    assert p[col].between(0, 1).all()
    assert np.allclose(p.groupby('gameid')[col].sum(), 1)
    scores = {'brier': brier_score_loss(p.result, p[col]), 'log_loss': log_loss(p.result, np.clip(p[col], 1e-6, 1-1e-6)), 'auc': roc_auc_score(p.result, p[col])}
    for metric, value in scores.items():
        assert math.isclose(value, r['timepoints']['20'][key][metric], abs_tol=1e-7), (key, metric)
assert np.allclose(p.state_edge, p.p_state - p.p_gold)
assert np.allclose(p.residual, p.result - p.p_state)
assert p.loc[p.gold_velocity.eq(-500), 'trajectory'].eq('Stable (within 500g)').all()
assert p.loc[p.gold_velocity.eq(500), 'trajectory'].eq('Stable (within 500g)').all()
coverage = pd.read_csv(ROOT / 'analysis/league_coverage.csv')
assert coverage.complete.sum() == r['games']
assert coverage['partial'].sum() == r['excluded_partial_games']
assert coverage.loc[coverage.league.eq('LPL'), 'complete'].eq(0).all()
for file, ahead in [('closing_efficiency.csv', True), ('comeback_resilience.csv', False)]:
    teams = pd.read_csv(ROOT / 'analysis' / file)
    for row in teams.itertuples():
        mask = p.golddiffat20.gt(0) if ahead else p.golddiffat20.lt(0)
        q = p[mask & p.teamname.eq(row.teamname) & p.league.eq(row.league)]
        assert len(q) == row.n and q.result.sum() == row.actual_wins
        assert math.isclose(q.p_state.sum(), row.expected_wins, abs_tol=1e-7)
        assert math.isclose(row.efficiency_pp, 100*(row.actual_wins-row.expected_wins)/(row.n+30), abs_tol=1e-7)
curves = pd.read_csv(ROOT / 'analysis/win_probability_curves.csv')
for t, stats in r['timepoints'].items():
    q = curves[curves.checkpoint.eq(t + ' min')]
    assert len(q) == 600 and q.win_probability.is_monotonic_increasing
    for target in [80, 90, 95]:
        hit = q[q.win_probability >= target/100].iloc[0].gold_diff
        assert math.isclose(hit, stats['thresholds_gold_diff']['p'+str(target)], abs_tol=1e-7)


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []; self.ids = set(); self.frames = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            assert a['id'] not in self.ids, 'Duplicate HTML id'
            self.ids.add(a['id'])
        for key in ['href', 'src']:
            if key in a: self.links.append(a[key])
        if tag == 'iframe':
            assert a.get('title') and a.get('loading') == 'lazy'
            self.frames.append(a['src'])


page = Links(); text = (ROOT / 'index.html').read_text(encoding='utf-8'); page.feed(text)
assert len(page.frames) == 8
for link in page.links:
    parsed = urlparse(link)
    if parsed.scheme or parsed.netloc: continue
    if parsed.path: assert (ROOT / unquote(parsed.path)).is_file(), link
    elif parsed.fragment: assert parsed.fragment in page.ids, link
for frame in page.frames:
    chart = (ROOT / frame).read_text(encoding='utf-8')
    assert 'Plotly.newPlot' in chart and 'plotly.min.js' in chart
assert (ROOT / 'assets/plotly.min.js').stat().st_size > 1000000
for file in ['index.html', 'README.md']:
    text = (ROOT / file).read_text(encoding='utf-8')
    assert f"{r['games']:,}" in text and 'LPL' in text
    assert '72.69%' not in text and 'p = 0.0001' not in text
print(f'Validated {p.gameid.nunique():,} held-out game pairs, scores, thresholds, team residuals, coverage and 8 chart links.')
