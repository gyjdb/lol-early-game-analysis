"""Frozen 2026 domestic Tier 1 study; chronological predictions, audited coverage."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis'
meta = json.loads((OUT / 'data/provenance_2026.json').read_text())
assert hashlib.sha256((OUT / 'data/tier1_2026.csv.gz').read_bytes()).hexdigest() == meta['snapshot_sha256']
d = pd.read_csv(OUT / 'data/tier1_2026.csv.gz')
d['date'] = pd.to_datetime(d.date)
d['day'] = d.date.dt.normalize()
assert not d.duplicated(['gameid','side']).any()
assert d.groupby('gameid').size().eq(2).all()
assert d.groupby('gameid').result.sum().eq(1).all()
assert d.groupby('gameid').side.nunique().eq(2).all()
opp = d[['gameid','side','teamname','draft']].copy()
opp['side'] = opp.side.map({'Blue':'Red','Red':'Blue'})
d = d.merge(opp.rename(columns={'teamname':'opponent','draft':'opponent_draft'}),on=['gameid','side'])
d['blue'] = d.side.eq('Blue').astype(int)
d['killdiff20'] = d.killsat20-d.opp_killsat20
d['gold_change15_20'] = d.golddiffat20-d.golddiffat15
d['xp_change15_20'] = d.xpdiffat20-d.xpdiffat15
d['cs_change15_20'] = d.csdiffat20-d.csdiffat15
state = ['golddiffat20','xpdiffat20','csdiffat20','killdiff20','blue']
trajectory = ['gold_change15_20','xp_change15_20','cs_change15_20']
# Same-day results never enter a same-day rating. Ratings are league-local,
# equal at season start, with fixed K=20 and no tuning against test outcomes.
ratings = {}
for day, g in d.groupby('day',sort=True):
    changes = {}
    for row in g[g.side.eq('Blue')].itertuples():
        a,b = (row.league,row.teamname),(row.league,row.opponent)
        diff = ratings.get(a,1500.)-ratings.get(b,1500.)
        mask = d.gameid.eq(row.gameid)
        d.loc[mask,'elo_diff'] = np.where(d.loc[mask,'blue'].eq(1),diff,-diff)
        expected = 1/(1+10**(-diff/400))
        delta = 20*(row.result-expected)
        changes[a] = changes.get(a,0)+delta
        changes[b] = changes.get(b,0)-delta
    for team,delta in changes.items(): ratings[team] = ratings.get(team,1500.)+delta

d['eligible20'] = d.datacompleteness.eq('complete') & d.gamelength.ge(1200) & d[state+trajectory].notna().all(axis=1)
d['eligible20'] = d.groupby('gameid').eligible20.transform('all')
coverage = d.groupby('league').agg(raw_rows=('gameid','size'),eligible_rows=('eligible20','sum'))/2
coverage.columns = ['source_games','eligible_games']
coverage = coverage.astype(int).reset_index()
coverage['coverage_pct'] = 100*coverage.eligible_games/coverage.source_games
coverage.to_csv(OUT/'pro_coverage.csv',index=False)
tc = d.groupby(['league','teamname']).agg(source_games=('gameid','size'),eligible_games=('eligible20','sum')).reset_index()
tc['coverage_pct'] = 100*tc.eligible_games/tc.source_games
tc.to_csv(OUT/'pro_team_coverage.csv',index=False)
d[['gameid','league','date','side','teamname','datacompleteness','gamelength','eligible20',*state,*trajectory]].to_csv(OUT/'pro_eligibility.csv',index=False)
q = d[d.eligible20].copy()
for col in state[:-1]+trajectory:
    assert np.allclose(q.groupby('gameid')[col].sum(),0), col
models = {'gold':['golddiffat20','blue'],'state':state,'trajectory':state+trajectory,'strength':state+['elo_diff'],'full':state+trajectory+['elo_diff']}
predictions = []; folds = []
for month in sorted(q.date.dt.to_period('M').unique()):
    if str(month) < '2026-04': continue
    start = month.start_time
    train = q[q.date < start-pd.Timedelta(days=7)]
    test = q[q.date.dt.to_period('M').eq(month)].copy()
    assert train.date.max() < test.date.min()-pd.Timedelta(days=7)
    test['fold'] = str(month)
    test['train_end'] = train.date.max()
    for name,features in models.items():
        model = make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000))
        model.fit(train[features],train.result)
        raw = pd.Series(model.predict_proba(test[features])[:,1],index=test.index)
        other = raw.groupby(test.gameid).transform('sum')-raw
        test['p_'+name] = (raw+1-other)/2
    predictions.append(test)
    folds.append({'fold':str(month),'train_end':str(train.date.max()),'train_games':train.gameid.nunique(),'test_games':test.gameid.nunique()})
p = pd.concat(predictions).sort_values(['date','gameid','side'])
p.to_csv(OUT/'pro_predictions.csv',index=False)
blue = p[p.side.eq('Blue')].copy()
metrics = []
for name in models:
    prob = blue['p_'+name]
    metrics.append({'model':name,'games':len(blue),'brier':brier_score_loss(blue.result,prob),'log_loss':log_loss(blue.result,prob),'auc':roc_auc_score(blue.result,prob)})
pd.DataFrame(metrics).to_csv(OUT/'pro_metrics.csv',index=False)
diagnostics = []
for grouping in ['league','fold']:
    for label,g in blue.groupby(grouping):
        for name in models:
            diagnostics.append({'grouping':grouping,'group':label,'model':name,'games':len(g),'brier':brier_score_loss(g.result,g['p_'+name]),'log_loss':log_loss(g.result,g['p_'+name],labels=[0,1])})
pd.DataFrame(diagnostics).to_csv(OUT/'pro_diagnostics.csv',index=False)
# Cluster fixed prediction losses by league/date/team-pair (series proxy).
# This is not a model-refit interval or a definitive series identifier.
blue['cluster'] = blue.apply(lambda r: r.league+'|'+str(r.day)+'|'+'|'.join(sorted([r.teamname,r.opponent])),axis=1)
intervals = []
rng = np.random.default_rng(2026)
for a,b in [('state','gold'),('trajectory','state'),('strength','state'),('full','strength')]:
    blue['delta'] = (blue.result-blue['p_'+a])**2-(blue.result-blue['p_'+b])**2
    clusters = blue.groupby('cluster').delta.agg(['sum','count']).to_numpy()
    ix = rng.integers(0,len(clusters),(2000,len(clusters)))
    samples = clusters[ix].sum(axis=1)
    low,high = np.quantile(samples[:,0]/samples[:,1],[.025,.975])
    intervals.append({'comparison':a+' minus '+b,'delta_brier':blue.delta.mean(),'low':low,'high':high})

rows = []
for (league,team),g in p.groupby(['league','teamname']):
    for condition,mask in [('Ahead',g.golddiffat20.gt(0)),('Behind',g.golddiffat20.lt(0))]:
        s = g[mask]; n = len(s)
        if not n: continue
        residual = s.result-s.p_strength
        grouped = pd.DataFrame({'day':s.day,'residual':residual}).groupby('day').residual.agg(['sum','count']).to_numpy()
        ix = rng.integers(0,len(grouped),(2000,len(grouped)))
        boot = grouped[ix].sum(axis=1)
        low,high = np.quantile(100*boot[:,0]/(boot[:,1]+30),[.025,.975])
        rows.append({'league':league,'teamname':team,'condition':condition,'n':n,'wins':int(s.result.sum()),'expected_wins':s.p_strength.sum(),'win_rate':s.result.mean(),'adjusted_pp':100*residual.sum()/(n+30),'low':low,'high':high})
teams = pd.DataFrame(rows).merge(tc,on=['league','teamname'])
teams.to_csv(OUT/'pro_teams.csv',index=False)
# Every 20-minute leader has a status: never silently drop short games at 25.
lead = q[q.golddiffat20.gt(0)].copy()
lead['next5'] = np.select([lead.gamelength.lt(1500)&lead.result.eq(1),lead.gamelength.lt(1500)&lead.result.eq(0),lead.golddiffat25.isna(),lead.golddiffat25.le(0),lead.golddiffat25.gt(lead.golddiffat20+500),lead.golddiffat25.lt(lead.golddiffat20-500)],['Won before 25','Lost before 25','Missing 25 snapshot','Lead lost by 25','Grew by >500g','Shrank by >500g'],default='Within 500g')
flow = lead.groupby(['league','next5']).agg(games=('gameid','size'),win_rate=('result','mean')).reset_index()
flow.to_csv(OUT/'pro_next5.csv',index=False)
lead.to_csv(OUT/'pro_next5_games.csv',index=False)
cases = p[p.league.eq('LPL')&p.golddiffat20.gt(0)&p.result.eq(0)].nlargest(8,'p_strength')
cases.to_csv(OUT/'pro_lpl_review_queue.csv',index=False)

def chart(fig,name):
    fig.update_layout(template='plotly_white',font={'family':'IBM Plex Sans, Arial','color':'#152133'},paper_bgcolor='#ffffff',plot_bgcolor='#ffffff',margin={'l':55,'r':20,'t':45,'b':65},legend_title_text='',autosize=True)
    fig.write_html(ROOT/'assets'/name,include_plotlyjs='plotly.min.js',full_html=True,config={'responsive':True,'displaylogo':False})
chart(px.bar(flow,x='league',y='games',color='next5',color_discrete_sequence=['#155bea','#008c91','#ba632d','#7448ae','#8197b4','#bc3a50','#5378a4'],labels={'games':'20-minute leaders','league':''}),'pro-next5.html')
rank = teams[teams.condition.eq('Ahead')&teams.n.ge(15)].sort_values(['league','adjusted_pp'])
# Keep separate regions; the menu defaults to LPL, not an international ranking.
import plotly.graph_objects as go
fig = go.Figure()
for league in meta['leagues']:
    z = rank[rank.league.eq(league)]
    fig.add_trace(go.Bar(x=z.adjusted_pp,y=z.teamname,orientation='h',name=league,visible=league=='LPL',marker_color='#155bea',error_x={'type':'data','symmetric':False,'array':np.maximum(0,z.high-z.adjusted_pp),'arrayminus':np.maximum(0,z.adjusted_pp-z.low)},customdata=z[['n','wins','expected_wins','coverage_pct']],hovertemplate='%{y}<br>Adjusted residual: %{x:.1f} pp<br>Leading games: %{customdata[0]}<br>Wins: %{customdata[1]} / expected %{customdata[2]:.1f}<br>Season coverage: %{customdata[3]:.1f}%<extra></extra>'))
fig.update_layout(updatemenus=[{'buttons':[{'label':l,'method':'update','args':[{'visible':[j==i for j in range(6)]}]} for i,l in enumerate(meta['leagues'])],'x':0,'y':1.15}],xaxis_title='Above expectation (shrunken pp)',showlegend=False)
chart(fig,'pro-teams.html')
cal = blue.copy();cal['bin'] = pd.cut(cal.p_strength,np.linspace(0,1,9),include_lowest=True)
cal = cal.groupby('bin',observed=True).agg(predicted=('p_strength','mean'),observed=('result','mean'),games=('gameid','size')).reset_index()
cal.to_csv(OUT/'pro_calibration.csv',index=False)
fig = px.scatter(cal,x='predicted',y='observed',size='games',hover_data=['games'],range_x=[0,1],range_y=[0,1],labels={'predicted':'Predicted Blue win probability','observed':'Observed Blue win rate'})
fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line={'dash':'dot','color':'#8197b4'})
chart(fig,'pro-calibration.html')
result = {'provenance':meta,'eligible_games':q.gameid.nunique(),'test_games':len(blue),'folds':folds,'metrics':metrics,'comparisons':intervals,'coverage':coverage.to_dict('records'),'lpl_teams':int(tc[tc.league.eq('LPL')].shape[0]),'features':models,'review_cases':cases[['gameid','teamname','opponent','date','patch','golddiffat20','golddiffat25','p_strength','draft','opponent_draft']].astype({'date':str}).to_dict('records')}
(OUT/'pro_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['eligible_games','test_games','metrics','comparisons','coverage']},indent=2))
