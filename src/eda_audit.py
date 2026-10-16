import pandas as pd
import numpy as np

def run_eda():
    print("=== EDA & AUDITORIA DE DADOS ===")
    
    # 1. Carregar tabelas
    df_pipeline = pd.read_csv("data/sales_pipeline.csv")
    df_accounts = pd.read_csv("data/accounts.csv")
    df_products = pd.read_csv("data/products.csv")
    df_teams = pd.read_csv("data/sales_teams.csv")
    
    print(f"\n1. DIMENSÕES:")
    print(f"- sales_pipeline: {df_pipeline.shape}")
    print(f"- accounts: {df_accounts.shape}")
    print(f"- products: {df_products.shape}")
    print(f"- sales_teams: {df_teams.shape}")
    
    print(f"\n2. DEAL STAGE DISTRIBUTION:")
    print(df_pipeline['deal_stage'].value_counts(dropna=False))
    
    print(f"\n3. VALORES NULOS NO PIPELINE:")
    print(df_pipeline.isnull().sum())
    
    print(f"\n4. VALORES NULOS EM ACCOUNTS:")
    print(df_accounts.isnull().sum())
    
    # Datas
    df_pipeline['engage_date'] = pd.to_datetime(df_pipeline['engage_date'])
    df_pipeline['close_date'] = pd.to_datetime(df_pipeline['close_date'])
    
    print(f"\n5. INTERVALOS DE DATAS:")
    print(f"- engage_date min: {df_pipeline['engage_date'].min()}, max: {df_pipeline['engage_date'].max()}")
    print(f"- close_date min: {df_pipeline['close_date'].min()}, max: {df_pipeline['close_date'].max()}")
    
    # Status por presença de datas
    print(f"\n6. DATAS POR STAGE:")
    for stage, group in df_pipeline.groupby('deal_stage'):
        print(f"Stage: {stage} (n={len(group)})")
        print(f"  - engage_date nulls: {group['engage_date'].isnull().sum()}")
        print(f"  - close_date nulls: {group['close_date'].isnull().sum()}")
        print(f"  - close_value nulls/zeros: {(group['close_value'].isnull() | (group['close_value'] == 0)).sum()}")
        if stage in ['Won', 'Lost']:
            print(f"  - close_value mean: {group['close_value'].mean():.1f}, min: {group['close_value'].min()}, max: {group['close_value'].max()}")

    # Integridade Referencial
    print(f"\n7. INTEGRIDADE REFERENCIAL:")
    orphan_accounts = df_pipeline[~df_pipeline['account'].isin(df_accounts['account']) & df_pipeline['account'].notnull()]
    null_accounts = df_pipeline[df_pipeline['account'].isnull()]
    print(f"- Deals com conta órfã (não está em accounts.csv): {len(orphan_accounts)}")
    print(f"- Deals com conta NULA: {len(null_accounts)} (Stages: {null_accounts['deal_stage'].value_counts().to_dict()})")
    
    orphan_products = df_pipeline[~df_pipeline['product'].isin(df_products['product'])]
    print(f"- Deals com produto órfão: {len(orphan_products)}")
    
    orphan_agents = df_pipeline[~df_pipeline['sales_agent'].isin(df_teams['sales_agent'])]
    print(f"- Deals com agente órfão: {len(orphan_agents)}")

    # Produtos
    print(f"\n8. PRODUTOS & PREÇOS:")
    print(df_products)
    
    # Taxa de conversão histórica geral e por produto
    closed_deals = df_pipeline[df_pipeline['deal_stage'].isin(['Won', 'Lost'])].copy()
    closed_deals['won'] = (closed_deals['deal_stage'] == 'Won').astype(int)
    print(f"\n9. TAXA DE CONVERSÃO GERAL (Won / (Won + Lost)): {closed_deals['won'].mean()*100:.2f}% (Total fechados: {len(closed_deals)})")
    
    print("\nTaxa de Conversão por Produto:")
    prod_stats = closed_deals.groupby('product').agg(
        total=('won', 'count'),
        won=('won', 'sum'),
        win_rate=('won', 'mean'),
        avg_close_value=('close_value', lambda x: x[x > 0].mean())
    ).reset_index()
    prod_stats = prod_stats.merge(df_products, on='product')
    print(prod_stats.to_string(index=False))

    # Duração média do ciclo de venda (dias entre engage_date e close_date)
    closed_deals['cycle_days'] = (closed_deals['close_date'] - closed_deals['engage_date']).dt.days
    print(f"\n10. CICLO MÉDIO DE FECHAMENTO (dias):")
    print(f"Won deals - Média: {closed_deals[closed_deals['won']==1]['cycle_days'].mean():.1f} dias, Mediana: {closed_deals[closed_deals['won']==1]['cycle_days'].median():.1f} dias, Max: {closed_deals[closed_deals['won']==1]['cycle_days'].max()} dias")
    print(f"Lost deals - Média: {closed_deals[closed_deals['won']==0]['cycle_days'].mean():.1f} dias, Mediana: {closed_deals[closed_deals['won']==0]['cycle_days'].median():.1f} dias, Max: {closed_deals[closed_deals['won']==0]['cycle_days'].max()} dias")

if __name__ == "__main__":
    run_eda()
