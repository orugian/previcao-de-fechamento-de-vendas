import pandas as pd
import numpy as np

def load_and_clean_data(data_dir="data"):
    # 1. Carregar tabelas
    df_pipeline = pd.read_csv(f"{data_dir}/sales_pipeline.csv")
    df_accounts = pd.read_csv(f"{data_dir}/accounts.csv")
    df_products = pd.read_csv(f"{data_dir}/products.csv")
    df_teams = pd.read_csv(f"{data_dir}/sales_teams.csv")

    # 2. Corrigir inconsistência de nomenclatura em produtos ('GTXPro' -> 'GTX Pro')
    df_pipeline['product'] = df_pipeline['product'].replace({'GTXPro': 'GTX Pro'})

    # 3. Converter datas
    df_pipeline['engage_date'] = pd.to_datetime(df_pipeline['engage_date'])
    df_pipeline['close_date'] = pd.to_datetime(df_pipeline['close_date'])

    # 4. Enriquecer com tabelas dimensionais
    df = df_pipeline.merge(df_products, on='product', how='left')
    df = df.merge(df_teams, on='sales_agent', how='left')
    df = df.merge(df_accounts, on='account', how='left')

    return df

def build_features(df, as_of_date='2017-12-31', train_mode=True, split_date='2017-11-01'):
    as_of = pd.to_datetime(as_of_date)
    split_dt = pd.to_datetime(split_date)

    # Identificar conjuntos
    is_closed = df['deal_stage'].isin(['Won', 'Lost'])
    
    # Target
    df['target_won'] = np.where(df['deal_stage'] == 'Won', 1, np.where(df['deal_stage'] == 'Lost', 0, np.nan))

    # Features de Conta
    df['has_account'] = df['account'].notnull().astype(int)
    median_rev = df[df['has_account'] == 1]['revenue'].median()
    median_emp = df[df['has_account'] == 1]['employees'].median()
    
    df['revenue_clean'] = df['revenue'].fillna(median_rev)
    df['employees_clean'] = df['employees'].fillna(median_emp)
    df['log_revenue'] = np.log1p(df['revenue_clean'])
    df['log_employees'] = np.log1p(df['employees_clean'])
    df['revenue_per_employee'] = df['revenue_clean'] / (df['employees_clean'] + 1)
    df['is_subsidiary'] = df['subsidiary_of'].notnull().astype(int)
    df['account_age'] = np.where(df['year_established'].notnull(), 2017 - df['year_established'], 0)
    df['sector_clean'] = df['sector'].fillna('Unknown')
    
    # Features de Produto
    df['log_sales_price'] = np.log1p(df['sales_price'])
    
    # Features Temporais / Estágio
    # Para deals abertos: dias desde engage_date até as_of_date (se nulo, 0)
    # Para deals fechados: ciclo realizado (close_date - engage_date)
    df['days_in_pipeline'] = np.where(
        is_closed,
        (df['close_date'] - df['engage_date']).dt.days.fillna(0),
        (as_of - df['engage_date']).dt.days.fillna(0)
    )
    df['days_in_pipeline'] = df['days_in_pipeline'].clip(lower=0)
    df['is_prospecting'] = (df['deal_stage'] == 'Prospecting').astype(int)

    # Taxas históricas de conversão com Out-Of-Time e Laplace Smoothing
    # Para evitar leakage, calculamos as taxas usando estritamente o conjunto de treino histórico (close_date < split_date)
    train_closed = df[is_closed & (df['close_date'] < split_dt)]
    global_win_rate = train_closed['target_won'].mean()

    # 1. Taxa do Vendedor (Laplace shrinkage com peso 10)
    agent_stats = train_closed.groupby('sales_agent')['target_won'].agg(['count', 'sum'])
    agent_wr = ((agent_stats['sum'] + 10 * global_win_rate) / (agent_stats['count'] + 10)).to_dict()
    df['agent_historical_win_rate'] = df['sales_agent'].map(agent_wr).fillna(global_win_rate)

    # 2. Taxa do Produto (Laplace shrinkage com peso 10)
    prod_stats = train_closed.groupby('product')['target_won'].agg(['count', 'sum'])
    prod_wr = ((prod_stats['sum'] + 10 * global_win_rate) / (prod_stats['count'] + 10)).to_dict()
    df['product_historical_win_rate'] = df['product'].map(prod_wr).fillna(global_win_rate)

    # 3. Taxa do Setor (Laplace shrinkage com peso 10)
    sector_stats = train_closed.groupby('sector_clean')['target_won'].agg(['count', 'sum'])
    sec_wr = ((sector_stats['sum'] + 10 * global_win_rate) / (sector_stats['count'] + 10)).to_dict()
    df['sector_historical_win_rate'] = df['sector_clean'].map(sec_wr).fillna(global_win_rate)

    # 4. Taxa do Escritório Regional
    reg_stats = train_closed.groupby('regional_office')['target_won'].agg(['count', 'sum'])
    reg_wr = ((reg_stats['sum'] + 10 * global_win_rate) / (reg_stats['count'] + 10)).to_dict()
    df['regional_win_rate'] = df['regional_office'].map(reg_wr).fillna(global_win_rate)

    return df, global_win_rate

if __name__ == '__main__':
    df = load_and_clean_data()
    df_feat, g_wr = build_features(df)
    print(f"Dataset processado: {df_feat.shape}")
    print(f"Global Win Rate (Treino histórico): {g_wr:.3f}")
    print("\nExemplo de features geradas:")
    cols = ['opportunity_id', 'sales_agent', 'product', 'deal_stage', 'days_in_pipeline', 
            'agent_historical_win_rate', 'product_historical_win_rate', 'sector_historical_win_rate', 'target_won']
    print(df_feat[cols].head(5))
