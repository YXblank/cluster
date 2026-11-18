"""Evaluate a CSV with one row per query, including failed estimates."""
import argparse,csv,json,numpy as np
p=argparse.ArgumentParser(); p.add_argument('csv'); a=p.parse_args()
rows=list(csv.DictReader(open(a.csv))); assert rows

def num(x):
    try:return float(x)
    except (ValueError,TypeError):return float('nan')
t=np.array([num(r['te_m']) for r in rows]); r=np.array([num(r['re_deg']) for r in rows]); v=np.isfinite(t)&np.isfinite(r)&(t>=0)&(r>=0)
out={'queries':len(t),'valid':int(v.sum()),'mean_TE':float(t[v].mean()) if v.any() else None,'mean_RE':float(r[v].mean()) if v.any() else None}
for x,y in [(0.25,2),(.5,5),(5,10)]:out[f'recall_{x}m_{y}deg_pct']=float(100*np.mean(v&(t<=x)&(r<=y)))
print(json.dumps(out,indent=2))
