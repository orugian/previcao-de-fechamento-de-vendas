import pandas as pd
import numpy as np
import joblib
from feature_engineering import load_and_clean_data, build_features
from scorer_engine import calculate_urgency_decay

def run_backtest_simulation(split_date='2017-11-01'):
    df = load_and_clean_data()
    df_feat, _ = build_features(df, split_date=split_date)
    
    # Conjunto de teste (deals fechados a partir de split_date)
    test_df = df_feat[(df_feat['deal_stage'].isin(['Won', 'Lost'])) & (df_feat['close_date'] >= pd.to_datetime(split_date))].copy()
    
    # Carregar modelos
    preprocessor = joblib.load("models/preprocessor.joblib")
    calibrated_model = joblib.load("models/lightgbm_calibrated.joblib")
    
    numeric_features = [
        'days_in_pipeline', 'sales_price', 'log_sales_price', 'revenue_clean',
        'log_revenue', 'employees_clean', 'log_employees', 'revenue_per_employee',
        'account_age', 'is_subsidiary', 'has_account', 'agent_historical_win_rate',
        'product_historical_win_rate', 'sector_historical_win_rate', 'regional_win_rate'
    ]
    categorical_features = ['series', 'regional_office', 'sector_clean']
    
    X_raw = test_df[numeric_features + categorical_features]
    X_proc = preprocessor.transform(X_raw)
    
    # Predição de probabilidade e cálculo de score
    test_df['pred_prob'] = calibrated_model.predict_proba(X_proc)[:, 1]
    
    # Calcular expected value
    test_df['expected_value'] = test_df['pred_prob'] * test_df['sales_price']
    
    # Receita real capturada
    test_df['actual_revenue'] = np.where(test_df['deal_stage'] == 'Won', test_df['close_value'], 0.0)
    
    total_realized_revenue = test_df['actual_revenue'].sum()
    total_deals = len(test_df)
    
    # Ordenar por Score / EV (Modelo Inteligente)
    model_sorted = test_df.sort_values(by='expected_value', ascending=False).copy()
    model_sorted['cum_revenue_model'] = model_sorted['actual_revenue'].cumsum()
    model_sorted['pct_deals'] = np.arange(1, total_deals + 1) / total_deals * 100
    model_sorted['pct_revenue_model'] = model_sorted['cum_revenue_model'] / total_realized_revenue * 100
    
    # Baseline Aleatório / Intuição (Média esperada de seleção aleatória)
    model_sorted['pct_revenue_random'] = model_sorted['pct_deals']
    
    # Calcular métricas nos decis (Top 10%, 20%, 30%)
    deciles = [10, 20, 30, 50]
    benchmarks = []
    for d in deciles:
        cutoff_idx = int(total_deals * (d / 100.0))
        rev_model = model_sorted.iloc[:cutoff_idx]['actual_revenue'].sum()
        rev_random = total_realized_revenue * (d / 100.0)
        lift_dollars = rev_model - rev_random
        lift_pct = (rev_model / rev_random - 1.0) * 100
        
        benchmarks.append({
            'decil': f'Top {d}% dos Deals',
            'n_deals': cutoff_idx,
            'receita_modelo': rev_model,
            'receita_random': rev_random,
            'lift_dolares': lift_dollars,
            'lift_percentual': lift_pct
        })
        
    df_bench = pd.DataFrame(benchmarks)
    return model_sorted[['pct_deals', 'pct_revenue_model', 'pct_revenue_random']], df_bench, total_realized_revenue

if __name__ == '__main__':
    curve, bench, total_rev = run_backtest_simulation()
    print(f"=== BACKTEST ECONÔMICO & SIMULAÇÃO DE LIFT FINANCEIRO ===")
    print(f"Receita Total Realizada no Teste: ${total_rev:,.2f}")
    print("\nResultados por Ponto de Corte:")
    for _, row in bench.iterrows():
        print(f"[{row['decil']} - {row['n_deals']} deals]:")
        print(f"  - Receita Modelo: ${row['receita_modelo']:,.2f}")
        print(f"  - Receita Baseline Aleatório/Feeling: ${row['receita_random']:,.2f}")
        print(f"  - LIFT Adicional em Vendas: +${row['lift_dolares']:,.2f} (+{row['lift_percentual']:.1f}% de eficiência!)")
