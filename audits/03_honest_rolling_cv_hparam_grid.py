from pathlib import Path
import sys, warnings, itertools
warnings.filterwarnings("ignore")
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from src.data_loader import load_raw_data, clean_and_merge_data
from src.features import create_as_of_dataset, FeatureEngineer

raw = load_raw_data('data')
m = clean_and_merge_data(raw['pipeline'], raw['accounts'], raw['products'], raw['teams'])
def pit(snap, t):
    hist = m[(m.close_date.notnull()) & (m.close_date < t)]
    d = FeatureEngineer().fit(hist).transform(snap)
    d['agent_open_load'] = d.sales_agent.map(snap.sales_agent.value_counts()).astype(float)
    ah = hist.groupby('account').target.agg(['count','mean'])
    d['acc_prior_deals'] = d.account.map(ah['count']).fillna(0)
    d['acc_prior_wr'] = d.account.map(ah['mean']).fillna(hist.target.mean())
    rec = hist[hist.close_date >= t - pd.Timedelta(days=90)]
    d['agent_recent_wr'] = d.sales_agent.map(rec.groupby('sales_agent').target.mean()).fillna(hist.target.mean())
    qe = pd.Timestamp(t).to_period('Q').end_time.normalize()
    d['days_to_quarter_end'] = (qe - pd.Timestamp(t)).days
    return d
dates = pd.date_range('2017-03-01','2017-10-01',freq='MS')
snaps = {dt: pit(create_as_of_dataset(m, dt, 90), dt) for dt in dates}
base = [c for c in FeatureEngineer().get_feature_columns() if c not in ('is_account_assigned','has_parent_company')]
HON = base + ['agent_open_load','acc_prior_deals','acc_prior_wr','agent_recent_wr','days_to_quarter_end']

def cv(cols, fn):
    a=[];b=[]
    for test_dt in dates[4:]:
        tr = pd.concat([snaps[d] for d in dates if d+pd.Timedelta(days=90)<=test_dt]); te = snaps[test_dt]
        tr=tr[tr.is_account_assigned==1]; te=te[te.is_account_assigned==1]
        p = fn().fit(tr[cols],tr.target).predict_proba(te[cols])[:,1]
        a.append(roc_auc_score(te.target,p)); b.append(brier_score_loss(te.target,p))
    return np.mean(a), np.std(a), np.mean(b)

print("LogReg honest features:", "AUC=%.3f±%.3f Brier=%.3f" % cv(HON, lambda: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))))
best=[]
for nl, mcs, lr_, ne, cs in itertools.product([3,7,15,31],[20,80,200],[0.02,0.05],[100,300],[0.6,1.0]):
    f = lambda: lgb.LGBMClassifier(num_leaves=nl,min_child_samples=mcs,learning_rate=lr_,n_estimators=ne,colsample_bytree=cs,subsample=0.8,subsample_freq=1,reg_lambda=2,verbose=-1,random_state=42)
    r = cv(HON, f); best.append((r[0],r[1],r[2],nl,mcs,lr_,ne,cs))
best.sort(reverse=True)
print(f"\nGrid de {len(best)} combinações LightGBM (rolling-origin, 4 folds):")
print("Top-5:"); [print("  AUC=%.3f±%.3f Brier=%.3f | leaves=%d min_child=%d lr=%.2f n_est=%d colsample=%.1f"%x) for x in best[:5]]
print("Pior: AUC=%.3f"%best[-1][0], "| Mediana: AUC=%.3f"%best[len(best)//2][0])
