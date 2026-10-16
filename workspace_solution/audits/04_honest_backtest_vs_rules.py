import sys, warnings
warnings.filterwarnings("ignore")
sys.path.append('workspace_solution')
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
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
    return d
dates = pd.date_range('2017-03-01','2017-10-01',freq='MS')
snaps = {dt: pit(create_as_of_dataset(m, dt, 90), dt) for dt in dates}
T = pd.Timestamp('2017-10-01')
te_all = snaps[T]
# 1) Proxy reconstruction check
print("AUC de 'acc_prior_deals==0' prevendo AUSÊNCIA de conta:", round(roc_auc_score(1-te_all.is_account_assigned, (te_all.acc_prior_deals==0).astype(int)),3))
print("AUC de 'sector==Unknown' prevendo ausência de conta:", round(roc_auc_score(1-te_all.is_account_assigned, (te_all.sector=='Unknown').astype(int)),3))
# 2) Live pipeline: how many deals would be condemned by the artifact
live = m[m.deal_stage.isin(['Engaging','Prospecting'])]
print(f"Pipeline vivo (2017-12-31): {len(live)} deals, sem conta: {(live.is_account_assigned==0).sum()} ({(live.is_account_assigned==0).mean():.1%}) -> recebem P≈0 no modelo atual")

# 3) Honest backtest on qualified population, with bootstrap
HON = [c for c in FeatureEngineer().get_feature_columns() if c not in ('is_account_assigned','has_parent_company')] + ['agent_open_load','acc_prior_wr','agent_recent_wr']
tr = pd.concat([snaps[d] for d in dates if d+pd.Timedelta(days=90)<=T]); tr = tr[tr.is_account_assigned==1]
te = te_all[te_all.is_account_assigned==1].copy()
P = dict(num_leaves=15,min_child_samples=20,learning_rate=0.05,n_estimators=100,subsample=0.8,subsample_freq=1,reg_lambda=2,verbose=-1,random_state=42)
cal = CalibratedClassifierCV(lgb.LGBMClassifier(**P), method='isotonic', cv=5).fit(tr[HON], tr.target)
te['p'] = cal.predict_proba(te[HON])[:,1]
print(f"\nPopulação qualificada Q4: n={len(te)}, WR real={te.target.mean():.3f}, P média prevista={te.p.mean():.3f}")
print(f"Modelo honesto: AUC={roc_auc_score(te.target,te.p):.3f}  Brier={brier_score_loss(te.target,te.p):.3f}  (naive Brier={brier_score_loss(te.target,np.full(len(te),tr.target.mean())):.3f}; naive com WR do treino={tr.target.mean():.3f})")
K = int(0.2*len(te))
pr = te.p.clip(0,1)
pol = {
 "ML honesto: EV = P × ticket": pr*te.sales_price,
 "Regra: ticket, penaliza >90d": te.sales_price*np.where(te.deal_age_days>90,0.1,1),
 "Regra: ticket × fator idade empírico": te.sales_price*np.where(te.deal_age_days<=45,1.0,np.where(te.deal_age_days<=90,0.5,0.05)),
 "Só ticket": te.sales_price.astype(float),
 "Só P(Won)": pr,
}
rng = np.random.RandomState(0); tot = te.realized_value.sum()
print(f"\nBacktest top-{K} (20%) — receita capturada, IC95% bootstrap:")
for name, s in pol.items():
    vals=[]
    for _ in range(300):
        idx = rng.randint(0,len(te),len(te)); sub = te.iloc[idx]; ss = s.iloc[idx]
        top = sub.iloc[np.argsort(-ss.values)[:K]]
        vals.append(top.realized_value.sum()/sub.realized_value.sum())
    top = te.iloc[np.argsort(-s.values)[:K]].realized_value.sum()/tot
    print(f"  {name:38s} {top:.1%}  [{np.percentile(vals,2.5):.1%}, {np.percentile(vals,97.5):.1%}]")
print("  Aleatório                              20.0%")
