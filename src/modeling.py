"""
Model Training, Probability Calibration, Economic Scoring & TreeSHAP Engine
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

FEATURE_LABELS_PT = {
    "deal_age_days": "Tempo em Negociação (Dias)",
    "velocity_decay": "Índice de Vitalidade Temporal",
    "is_zombie_deal": "Alerta de Estagnação (>90 dias)",
    "in_sweet_spot": "Janela Ideal de Conversão (7-45 dias)",
    "is_account_assigned": "Conta Cliente Qualificada",
    "has_parent_company": "Subsidiária de Grupo Corporativo",
    "sales_price": "Ticket de Tabela do Produto ($)",
    "log_sales_price": "Escala de Ticket (Log $)",
    "revenue": "Receita Anual do Cliente ($M)",
    "log_revenue": "Porte Financeiro do Cliente (Log)",
    "employees": "Quadro de Funcionários",
    "log_employees": "Escala de Funcionários (Log)",
    "revenue_per_employee": "Eficiência Financeira por Funcionário",
    "account_age": "Maturidade da Empresa Cliente (Anos)",
    "age_to_price_ratio": "Desgaste Temporal vs. Ticket",
    "agent_win_rate": "Taxa de Conversão Histórica do Vendedor",
    "product_win_rate": "Taxa de Conversão Histórica do Produto",
    "sector_win_rate": "Conversão Histórica do Setor",
    "region_win_rate": "Performance da Regional",
    "agent_product_win_rate": "Domínio Vendedor × Produto",
}


class LeadScorerEngine:
    """
    Encapsulates the raw LightGBM model (for TreeSHAP exact attribution),
    the Isotonic Calibrated Classifier (for calibrated P(Won)), and economic EV scoring.
    """

    def __init__(
        self,
        n_estimators: int = 120,
        learning_rate: float = 0.03,
        num_leaves: int = 15,
        min_child_samples: int = 30,
        random_state: int = 42,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "random_state": random_state,
            "verbose": -1,
        }
        self.raw_lgbm = lgb.LGBMClassifier(**self.params)
        self.calibrated_model = CalibratedClassifierCV(
            estimator=lgb.LGBMClassifier(**self.params),
            method="isotonic",
            cv=5,
        )
        self.logistic_baseline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(random_state=random_state, max_iter=1000)),
            ]
        )
        self.explainer = None
        self.feature_cols = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.feature_cols = list(X_train.columns)
        self.logistic_baseline.fit(X_train, y_train)
        self.raw_lgbm.fit(X_train, y_train)
        self.calibrated_model.fit(X_train, y_train)
        self.explainer = shap.TreeExplainer(self.raw_lgbm)
        return self

    def predict_calibrated_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrated_model.predict_proba(X[self.feature_cols])[:, 1]

    def score_deals(self, df_transformed: pd.DataFrame) -> pd.DataFrame:
        """
        Computes calibrated win probabilities, urgency multiplier, Expected Value ($),
        normalized Lead Priority Score (0-100), and top-3 positive/negative TreeSHAP factors.
        """
        res = df_transformed.copy()
        X = res[self.feature_cols]

        probs = self.predict_calibrated_proba(X)
        res["prob_won"] = probs

        # Urgency multiplier: boosts deals in the active sweet spot (7-45 days) and penalizes zombies
        urgency_mult = np.where(
            res["in_sweet_spot"] == 1,
            1.15,
            np.where(res["is_zombie_deal"] == 1, 0.65, 1.0),
        )
        res["urgency_multiplier"] = urgency_mult
        res["expected_value_usd"] = res["prob_won"] * res["sales_price"]
        res["risk_adjusted_ev"] = res["expected_value_usd"] * res["urgency_multiplier"]

        # Priority Score (0 to 100) combining calibrated probability and log expected value
        ev_percentile = pd.Series(res["risk_adjusted_ev"], index=res.index).rank(pct=True)
        res["lead_score"] = np.round(
            (0.55 * res["prob_won"] + 0.45 * ev_percentile) * 100.0, 1
        )

        # Assign Priority Tier
        res["priority_tier"] = pd.cut(
            res["lead_score"],
            bins=[-1, 35, 65, 100],
            labels=["Baixa (Nutrição/Descarte)", "Média (Acompanhamento)", "Alta (Ataque Imediato)"],
        )

        # Compute TreeSHAP local explanations
        shap_exp = self.explainer(X)
        shap_vals = shap_exp.values
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]

        top_pos_list = []
        top_neg_list = []

        for i in range(len(res)):
            row_shap = shap_vals[i]
            # Sort indices
            pos_idx = np.argsort(row_shap)[::-1]
            neg_idx = np.argsort(row_shap)

            pos_factors = [
                f"{FEATURE_LABELS_PT.get(self.feature_cols[idx], self.feature_cols[idx])} (+{row_shap[idx]:.2f})"
                for idx in pos_idx[:3]
                if row_shap[idx] > 0
            ]
            neg_factors = [
                f"{FEATURE_LABELS_PT.get(self.feature_cols[idx], self.feature_cols[idx])} ({row_shap[idx]:.2f})"
                for idx in neg_idx[:3]
                if row_shap[idx] < 0
            ]

            top_pos_list.append(" | ".join(pos_factors) if pos_factors else "Nenhum fator positivo dominante")
            top_neg_list.append(" | ".join(neg_factors) if neg_factors else "Sem riscos críticos identificados")

        res["top_positive_shap"] = top_pos_list
        res["top_negative_shap"] = top_neg_list
        return res

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> "LeadScorerEngine":
        return joblib.load(filepath)
