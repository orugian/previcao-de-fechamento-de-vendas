import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
    f1_score,
    classification_report
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import lightgbm as lgb

from feature_engineering import load_and_clean_data, build_features

def train_and_evaluate():
    print("=== TREINAMENTO, CALIBRAÇÃO & COMPARAÇÃO DE MODELOS ===")
    
    # 1. Carregar e enriquecer
    df = load_and_clean_data()
    split_date = '2017-11-01'
    df_feat, global_win_rate = build_features(df, split_date=split_date)
    
    # 2. Filtrar histórico fechado
    closed = df_feat[df_feat['deal_stage'].isin(['Won', 'Lost'])].copy()
    
    train_mask = closed['close_date'] < pd.to_datetime(split_date)
    test_mask = closed['close_date'] >= pd.to_datetime(split_date)
    
    train_df = closed[train_mask].copy()
    test_df = closed[test_mask].copy()
    
    print(f"Conjunto de Treino (Out-of-Time < {split_date}): {train_df.shape[0]} deals (Taxa Won: {train_df['target_won'].mean()*100:.2f}%)")
    print(f"Conjunto de Teste (Out-of-Time >= {split_date}): {test_df.shape[0]} deals (Taxa Won: {test_df['target_won'].mean()*100:.2f}%)")
    
    # Features selecionadas
    numeric_features = [
        'days_in_pipeline',
        'sales_price',
        'log_sales_price',
        'revenue_clean',
        'log_revenue',
        'employees_clean',
        'log_employees',
        'revenue_per_employee',
        'account_age',
        'is_subsidiary',
        'has_account',
        'agent_historical_win_rate',
        'product_historical_win_rate',
        'sector_historical_win_rate',
        'regional_win_rate'
    ]
    
    categorical_features = [
        'series',
        'regional_office',
        'sector_clean'
    ]
    
    X_train_raw = train_df[numeric_features + categorical_features]
    y_train = train_df['target_won'].astype(int)
    
    X_test_raw = test_df[numeric_features + categorical_features]
    y_test = test_df['target_won'].astype(int)
    
    # Preprocessor para baseline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    
    # --- MODELO 1: BASELINE NAIVE (Probabilidade Constante da Média) ---
    p_naive_train = y_train.mean()
    p_naive_test = np.full_like(y_test, p_naive_train, dtype=float)
    brier_naive = brier_score_loss(y_test, p_naive_test)
    
    # --- MODELO 2: REGRESSÃO LOGÍSTICA ---
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_proc, y_train)
    p_lr_test = lr.predict_proba(X_test_proc)[:, 1]
    
    # --- MODELO 3: LIGHTGBM (Base) ---
    lgb_train_data = lgb.Dataset(X_train_proc, label=y_train)
    lgb_test_data = lgb.Dataset(X_test_proc, label=y_test, reference=lgb_train_data)
    
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_clf.fit(X_train_proc, y_train)
    p_lgb_test = lgb_clf.predict_proba(X_test_proc)[:, 1]
    
    # --- MODELO 4: LIGHTGBM CALIBRADO (Isotonic Regression) ---
    # Usamos CalibratedClassifierCV com cv=5 sobre os dados de treino para aprender calibração isotônica sem vazamento
    calibrated_lgb = CalibratedClassifierCV(
        estimator=lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=5,
            num_leaves=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        ),
        method='isotonic',
        cv=5
    )
    calibrated_lgb.fit(X_train_proc, y_train)
    p_cal_test = calibrated_lgb.predict_proba(X_test_proc)[:, 1]
    
    # Avaliação Comparativa
    def compute_metrics(y_true, p_pred):
        y_pred = (p_pred >= 0.5).astype(int)
        return {
            'roc_auc': float(roc_auc_score(y_true, p_pred)),
            'pr_auc': float(average_precision_score(y_true, p_pred)),
            'brier_score': float(brier_score_loss(y_true, p_pred)),
            'log_loss': float(log_loss(y_true, p_pred)),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred))
        }
    
    metrics = {
        'Naive Baseline': {
            'brier_score': float(brier_naive),
            'log_loss': float(log_loss(y_test, p_naive_test)),
            'roc_auc': 0.5,
            'pr_auc': float(y_test.mean())
        },
        'Logistic Regression': compute_metrics(y_test, p_lr_test),
        'LightGBM (Uncalibrated)': compute_metrics(y_test, p_lgb_test),
        'LightGBM (Isotonic Calibrated)': compute_metrics(y_test, p_cal_test)
    }
    
    print("\n=== RESULTADOS COMPARATIVOS NO CONJUNTO DE TESTE OUT-OF-TIME ===")
    results_df = pd.DataFrame(metrics).T
    print(results_df.round(4).to_string())
    
    # Salvar artefatos
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    joblib.dump(lgb_clf, "models/lightgbm_base.joblib")
    joblib.dump(calibrated_lgb, "models/lightgbm_calibrated.joblib")
    
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("\nModelos e artefatos salvos com sucesso em models/!")
    return metrics

if __name__ == '__main__':
    train_and_evaluate()
