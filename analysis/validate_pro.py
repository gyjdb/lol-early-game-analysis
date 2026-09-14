"""Independently reconcile published counts, held-out losses and site artifacts."""
import json, math
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
root = Path(__file__).resolve().parents[1]; a = root/'analysis'
r = json.loads((a/'pro_results.json').read_text())
p = pd.read_csv(a/'pro_predictions.csv',parse_dates=['date','train_end'])
assert not p.duplicated(['gameid','side']).any()
assert p.groupby('gameid').size().eq(2).all()
assert p.groupby('gameid').fold.nunique().eq(1).all()
assert (p.date-p.train_end).gt(pd.Timedelta(days=7)).all()
assert p.groupby('gameid').result.sum().eq(1).all()
blue = p[p.side.eq('Blue')]
assert len(blue)==r['test_games']
raw=pd.read_csv(a/'data/tier1_2026.csv.gz',parse_dates=['date'])
rating={}; expected={}
for day,g in raw.groupby(raw.date.dt.normalize(),sort=True):
    changes={}
    for game,pair in g.groupby('gameid'):
        b=pair[pair.side.eq('Blue')].iloc[0]; red=pair[pair.side.eq('Red')].iloc[0]
        kb,kr=(b.league,b.teamname),(red.league,red.teamname)
        diff=rating.get(kb,1500)-rating.get(kr,1500)
        expected[game]=diff
        delta=20*(b.result-1/(1+10**(-diff/400)))
        changes[kb]=changes.get(kb,0)+delta;changes[kr]=changes.get(kr,0)-delta
    for key,value in changes.items():rating[key]=rating.get(key,1500)+value
assert np.allclose(blue.elo_diff,blue.gameid.map(expected)), 'Prior ratings must exclude same-day outcomes'
for m in r['metrics']:
    c='p_'+m['model']; assert p[c].between(0,1).all()
    assert np.allclose(p.groupby('gameid')[c].sum(),1)
    scores=[brier_score_loss(blue.result,blue[c]),log_loss(blue.result,blue[c]),roc_auc_score(blue.result,blue[c])]
    for key,val in zip(['brier','log_loss','auc'],scores): assert math.isclose(m[key],val,abs_tol=1e-10)
for row in pd.read_csv(a/'pro_teams.csv').itertuples():
    q=p[p.league.eq(row.league)&p.teamname.eq(row.teamname)]
    q=q[q.golddiffat20.gt(0) if row.condition=='Ahead' else q.golddiffat20.lt(0)]
    assert len(q)==row.n and q.result.sum()==row.wins
    assert math.isclose(q.p_strength.sum(),row.expected_wins,abs_tol=1e-8)
    assert math.isclose(100*(q.result-q.p_strength).sum()/(len(q)+30),row.adjusted_pp,abs_tol=1e-8)
audit=pd.read_csv(a/'pro_eligibility.csv')
assert audit[audit.eligible20].gameid.nunique()==r['eligible_games']
coverage=pd.read_csv(a/'pro_coverage.csv')
assert coverage.source_games.sum()==r['provenance']['raw_games']
for row in coverage.itertuples():
    q=audit[audit.league.eq(row.league)]
    assert row.source_games==q.gameid.nunique()
    assert row.eligible_games==q[q.eligible20].gameid.nunique()
flow=pd.read_csv(a/'pro_next5_games.csv')
assert len(flow)==len(audit[audit.eligible20&audit.golddiffat20.gt(0)])
assert flow.gameid.is_unique
assert pd.read_csv(a/'pro_next5.csv').games.sum()==len(flow)
assert not any('25' in col or col in ['result','gamelength'] for cols in r['features'].values() for col in cols)
class Page(HTMLParser):
    def __init__(self): super().__init__();self.links=[];self.ids=set();self.frames=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if 'id' in d: assert d['id'] not in self.ids;self.ids.add(d['id'])
        for k in ['href','src']:
            if k in d:self.links.append(d[k])
        if tag=='iframe': assert d.get('title');self.frames.append(d['src'])
for file,count in [('index.html',3),('history2022.html',8)]:
    page=Page();page.feed((root/file).read_text(encoding='utf-8'));assert len(page.frames)==count
    for link in page.links:
        u=urlparse(link)
        if u.scheme or u.netloc:continue
        if u.path:assert (root/u.path).is_file(),link
        elif u.fragment:assert u.fragment in page.ids,link
    for frame in page.frames:assert 'Plotly.newPlot' in (root/frame).read_text(encoding='utf-8')
for file in ['index.html','README.md']:
    text=(root/file).read_text(encoding='utf-8');assert '535' in text and '1,515' in text and '2026' in text
print('Validated chronological pairs, scores, coverage, residuals, transitions and both public studies')
