from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from src.data_loader import load_raw_data, clean_and_merge_data
raw = load_raw_data('data')
m = clean_and_merge_data(raw['pipeline'], raw['accounts'], raw['products'], raw['teams'])
closed = m[m.target.notnull()].copy()
# Intrinsic signal: can static features separate Won vs Lost at all? (closed deals, temporal split)
tr = closed[closed.close_date < '2017-10-01']; te = closed[closed.close_date >= '2017-10-01']
gw = tr.target.mean()
for col in ['sales_agent','product','account','sector','regional_office','manager']:
    s = tr.groupby(col).target.agg(['sum','count'])
    enc = ((s['sum']+10*gw)/(s['count']+10))
    p = te[col].map(enc).fillna(gw)
    print(f"AUC single target-encoded {col:16s}: {roc_auc_score(te.target,p):.3f}")
# duration bimodality
print("\nLost duration quantiles:", np.quantile(closed[closed.target==0].deal_age_days,[.25,.5,.75]))
print("Won  duration quantiles:", np.quantile(closed[closed.target==1].deal_age_days,[.25,.5,.75]))
# engage day-of-week / close at quarter end
closed['close_qe'] = closed.close_date.dt.month.isin([3,6,9,12])
print("\nWin rate by closes-in-quarter-end-month:", closed.groupby('close_qe').target.mean().round(3).to_dict())
