import sys
sys.path.append('workspace_solution')
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from src.data_loader import load_raw_data, clean_and_merge_data
from src.features import create_as_of_dataset, FeatureEngineer
from src.modeling import LeadScorerEngine
raw = load_raw_data('data')
m = clean_and_merge_data(raw['pipeline'], raw['accounts'], raw['products'], raw['teams'])
t = create_as_of_dataset(m,'2017-10-01',90)
# Who are the no-account deals in the Q4 snapshot? final stage
print(pd.crosstab(t.is_account_assigned, t.deal_stage))
# single-feature AUCs on full Q4 cohort
print("AUC is_account_assigned alone:", round(roc_auc_score(t.target, t.is_account_assigned),4))
rule = t.is_account_assigned * (t.deal_age_days<=138)
print("AUC rule (account & age<=138):", round(roc_auc_score(t.target, rule),4))
rule2 = t.is_account_assigned * np.where(t.deal_age_days<=60, 2, np.where(t.deal_age_days<=138,1,0))
print("AUC rule2 (account x age bucket):", round(roc_auc_score(t.target, rule2),4))
