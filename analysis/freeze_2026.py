"""Reproduce the committed subset from the matching official original CSV."""
import sys, gzip, hashlib, json
from pathlib import Path
import pandas as pd
out = Path(__file__).resolve().parent/'data'
meta = json.loads((out/'provenance_2026.json').read_text())
raw = Path(sys.argv[1])
assert hashlib.sha256(raw.read_bytes()).hexdigest() == meta['source_sha256']
d = pd.read_csv(raw,low_memory=False)
d = d[d.league.isin(meta['leagues'])]
cols = ['gameid','datacompleteness','league','split','playoffs','date','game','patch','participantid','side','teamname','teamid','gamelength','result']+[f'{s}{t}' for t in [15,20,25] for s in ['golddiffat','xpdiffat','csdiffat','killsat','opp_killsat']]
t = d[d.participantid.isin([100,200])][cols].copy()
p = d[d.position.isin(['top','jng','mid','bot','sup'])].copy()
p['draft'] = p.position+': '+p.champion.fillna('?')
t = t.merge(p.groupby(['gameid','side']).draft.agg(' / '.join),on=['gameid','side'],how='left')
b = gzip.compress(t.to_csv(index=False).encode(),mtime=0)
assert hashlib.sha256(b).hexdigest() == meta['snapshot_sha256'], 'Serialization differs; inspect before replacing the frozen snapshot'
(out/'tier1_2026.csv.gz').write_bytes(b)
print('Verified original and reproduced frozen subset')
