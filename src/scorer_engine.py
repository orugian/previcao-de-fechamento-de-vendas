import os
import joblib
import numpy as np
import pandas as pd
import shap
from feature_engineering import load_and_clean_data, build_features

FEATURE_LABELS = {
    'num__days_in_pipeline': 'Velocidade / Tempo no Pipeline',
    'num__agent_historical_win_rate': 'Track Record do Vendedor',
    'num__sales_price': 'Ticket / Valor do Produto',
    'num__log_sales_price': 'Faixa de Preço do Produto',
    'num__revenue_clean': 'Faturamento Anual da Conta',
    'num__log_revenue': 'Escala da Conta Cliente',
    'num__employees_clean': 'Porte da Conta (Funcionários)',
    'num__log_employees': 'Estrutura da Conta',
    'num__revenue_per_employee': 'Produtividade / Margem da Conta',
    'num__account_age': 'Maturidade da Empresa Cliente',
    'num__is_subsidiary': 'Pertence a Grupo / Holding',
    'num__has_account': 'Conta Qualificada no CRM',
    'num__product_historical_win_rate': 'Atratividade Histórica do Produto',
    'num__sector_historical_win_rate': 'Fit do Setor Econômico',
    'num__regional_win_rate': 'Performance da Regional',
    'cat__series_GTK': 'Linha de Produto: GTK (Enterprise)',
    'cat__series_GTX': 'Linha de Produto: GTX (Core)',
    'cat__series_MG': 'Linha de Produto: MG (Econômico)',
    'cat__regional_office_Central': 'Região: Central',
    'cat__regional_office_East': 'Região: East',
    'cat__regional_office_West': 'Região: West'
}

def calculate_urgency_decay(days):
    """
    Decaimento temporal baseado na análise de sobrevivência histórica:
    - <= 60 dias: Janela de ouro (decay = 1.0)
    - 60 a 138 dias: Zona de risco / esfriamento (decaimento exponencial suave)
    - > 138 dias: Deal zumbi (nunca na história houve fechamento após 138 dias)
    """
    if days <= 60:
        return 1.0
    elif days <= 138:
        return float(np.exp(-(days - 60) / 50.0))
    else:
        return float(0.05 * np.exp(-(days - 138) / 100.0))

class LeadScorerEngine:
    def __init__(self, models_dir="models"):
        self.preprocessor = joblib.load(f"{models_dir}/preprocessor.joblib")
        self.base_model = joblib.load(f"{models_dir}/lightgbm_base.joblib")
        self.calibrated_model = joblib.load(f"{models_dir}/lightgbm_calibrated.joblib")
        self.feature_names = self.preprocessor.get_feature_names_out()
        self.explainer = shap.TreeExplainer(self.base_model)
        
    def score_pipeline(self, df_pipeline):
        numeric_features = [
            'days_in_pipeline', 'sales_price', 'log_sales_price', 'revenue_clean',
            'log_revenue', 'employees_clean', 'log_employees', 'revenue_per_employee',
            'account_age', 'is_subsidiary', 'has_account', 'agent_historical_win_rate',
            'product_historical_win_rate', 'sector_historical_win_rate', 'regional_win_rate'
        ]
        categorical_features = ['series', 'regional_office', 'sector_clean']
        
        X_raw = df_pipeline[numeric_features + categorical_features]
        X_proc = self.preprocessor.transform(X_raw)
        
        # 1. Probabilidade calibrada
        probs = self.calibrated_model.predict_proba(X_proc)[:, 1]
        
        # 2. SHAP Values
        shap_values = self.explainer.shap_values(X_proc)
        # Em classificação binária, o LightGBM TreeExplainer retorna ou lista de 2 arrays ou matriz única
        if isinstance(shap_values, list):
            shap_matrix = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_matrix = shap_values[:, :, 1]
        else:
            shap_matrix = shap_values
            
        results = []
        for i in range(len(df_pipeline)):
            row = df_pipeline.iloc[i]
            p = float(probs[i])
            price = float(row['sales_price']) if pd.notnull(row['sales_price']) else 500.0
            days = float(row['days_in_pipeline'])
            decay = calculate_urgency_decay(days)
            
            # Valor esperado bruto
            expected_value = p * price
            
            # Score de priorização acionável (0 a 100)
            # Combina probabilidade, relevância do ticket e urgência
            # Normalização logarítmica suave para balancear tickets pequenos e gigantes
            ticket_weight = np.log1p(price) / np.log1p(26768) # 0 a 1
            raw_score = (0.5 * p + 0.3 * ticket_weight + 0.2 * decay) * 100
            
            # Se for deal zumbi (>138 dias), teto máximo de score cai substancialmente
            if days > 138:
                raw_score = min(raw_score, 35.0)
                status_urgency = "Zumbi / Estagnado"
            elif days > 60:
                status_urgency = "Atenção / Esfriando"
            else:
                status_urgency = "Quente / Ativo"

            # Top SHAP prós e contras
            lead_shaps = shap_matrix[i]
            top_pos_idx = np.argsort(lead_shaps)[-3:][::-1]
            top_neg_idx = np.argsort(lead_shaps)[:3]
            
            pos_factors = [
                {"feature": FEATURE_LABELS.get(self.feature_names[idx], self.feature_names[idx]), "impact": float(lead_shaps[idx])}
                for idx in top_pos_idx if lead_shaps[idx] > 0
            ]
            neg_factors = [
                {"feature": FEATURE_LABELS.get(self.feature_names[idx], self.feature_names[idx]), "impact": float(lead_shaps[idx])}
                for idx in top_neg_idx if lead_shaps[idx] < 0
            ]

            results.append({
                'opportunity_id': row['opportunity_id'],
                'sales_agent': row['sales_agent'],
                'manager': row['manager'],
                'regional_office': row['regional_office'],
                'product': row['product'],
                'series': row['series'],
                'sales_price': price,
                'account': row['account'] if pd.notnull(row['account']) else 'Conta Não Identificada',
                'sector': row['sector_clean'],
                'deal_stage': row['deal_stage'],
                'engage_date': str(row['engage_date'])[:10] if pd.notnull(row['engage_date']) else 'N/A',
                'days_in_pipeline': int(days),
                'urgency_status': status_urgency,
                'calibrated_probability': round(p * 100, 1),
                'expected_value': round(expected_value, 2),
                'lead_score': round(raw_score, 1),
                'top_positive_factors': pos_factors,
                'top_negative_factors': neg_factors
            })
            
        df_scored = pd.DataFrame(results)
        return df_scored

if __name__ == '__main__':
    df = load_and_clean_data()
    df_feat, _ = build_features(df)
    
    # Filtrar oportunidades abertas
    open_pipeline = df_feat[df_feat['deal_stage'].isin(['Engaging', 'Prospecting'])].copy()
    print(f"Total de oportunidades abertas a ranquear: {len(open_pipeline)}")
    
    engine = LeadScorerEngine()
    scored = engine.score_pipeline(open_pipeline)
    
    print("\nTop 5 Oportunidades Prioritárias no Pipeline Geral:")
    cols_view = ['opportunity_id', 'sales_agent', 'account', 'product', 'days_in_pipeline', 
                 'calibrated_probability', 'expected_value', 'lead_score', 'urgency_status']
    print(scored.sort_values(by='lead_score', ascending=False)[cols_view].head(5).to_string(index=False))
