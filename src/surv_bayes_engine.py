"""
Surv-Bayes ERV Engine (Expected Revenue Velocity & Decision Matrix)
Frente 1: Lead Scorer & Decision Engine para RevOps
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


class SurvBayesERVEngine:
    """
    Motor de Decisão e Priorização baseado em Análise de Sobrevivência,
    Encolhimento Empírico Bayesiano (James-Stein) e Throughput Econômico (ERV).
    
    Elimina 100% do data leakage do campo cadastral 'account' através do
    Princípio da Ortogonalidade Cadastral (neutralidade basal + bônus de expansão).
    """

    def __init__(
        self,
        m_product: float = 30.0,
        m_agent: float = 20.0,
        m_interaction: float = 10.0,
        tau_ramp: float = 15.0,
        tau_decay_half: float = 45.0,
        tau_quarantine: float = 110.0,
        tau_max: float = 138.0,
        median_duration: float = 48.0,
    ):
        self.m_product = m_product
        self.m_agent = m_agent
        self.m_interaction = m_interaction
        self.tau_ramp = tau_ramp
        self.tau_decay_half = tau_decay_half
        self.tau_quarantine = tau_quarantine
        self.tau_max = tau_max
        self.median_duration = median_duration

        self.global_win_rate = 0.6315
        self.product_priors = {}
        self.agent_priors = {}
        self.agent_product_priors = {}
        self.repeat_accounts = set()
        self.fitted = False

    def fit(self, historical_closed_df: pd.DataFrame):
        """
        Ajusta os priors empíricos Bayesianos estritamente em negócios fechados
        disponíveis até a data de corte (zero lookahead leakage).
        """
        closed = historical_closed_df[historical_closed_df["target"].notnull()].copy()
        self.global_win_rate = float(closed["target"].mean())

        # 1. Encolhimento Bayesiano por Produto (M_product = 30)
        prod_stats = closed.groupby("product")["target"].agg(["count", "sum"])
        self.product_priors = (
            (prod_stats["sum"] + self.m_product * self.global_win_rate)
            / (prod_stats["count"] + self.m_product)
        ).to_dict()

        # 2. Encolhimento Bayesiano por Vendedor (M_agent = 20)
        agent_stats = closed.groupby("sales_agent")["target"].agg(["count", "sum"])
        self.agent_priors = (
            (agent_stats["sum"] + self.m_agent * self.global_win_rate)
            / (agent_stats["count"] + self.m_agent)
        ).to_dict()

        # 3. Encolhimento Bayesiano para Interação Vendedor x Produto (M_interaction = 10)
        ap_stats = closed.groupby(["sales_agent", "product"])["target"].agg(["count", "sum"])
        ap_priors = {}
        for (a, p), row in ap_stats.iterrows():
            prod_base = self.product_priors.get(p, self.global_win_rate)
            ap_priors[(a, p)] = (row["sum"] + self.m_interaction * prod_base) / (row["count"] + self.m_interaction)
        self.agent_product_priors = ap_priors

        # 4. Mapeamento de Contas Recorrentes (para bônus positivo de relacionamento)
        won_accounts = closed[closed["target"] == 1]["account"].dropna().value_counts()
        self.repeat_accounts = set(won_accounts[won_accounts >= 2].index)

        self.fitted = True
        return self

    def _calc_survival(self, tau: float) -> float:
        """
        Função Contínua de Sobrevivência e Vitalidade Temporal S(tau).
        - 0 a 15 dias: maturação inicial (0.50 -> 0.70)
        - 15 a 110 dias: decaimento exponencial por atrito
        - > 110 dias: quarentena operacional (0.00)
        """
        if tau < 0:
            return 0.50
        if tau <= self.tau_ramp:
            return 0.50 + 0.20 * (tau / self.tau_ramp)
        if tau <= self.tau_quarantine:
            return 0.70 * np.exp(-(tau - self.tau_ramp) / self.tau_decay_half)
        return 0.0

    def _calc_quarter_modulator(self, ref_date: pd.Timestamp, tau: float) -> float:
        """
        Modulador de Sazonalidade Trimestral Omega_quarter.
        Meses de fim de trimestre (3, 6, 9, 12) têm sprint comercial (Win Rate sobe para ~80%).
        """
        month = ref_date.month
        is_quarter_end = month in [3, 6, 9, 12]
        is_quarter_start = month in [1, 4, 7, 10]

        if is_quarter_end:
            if 20 <= tau <= self.tau_quarantine:
                return 1.30  # Aceleração máxima de fechamento
            return 1.10
        if is_quarter_start:
            if tau > 30:
                return 0.80  # Penalidade para deals herdados do trimestre anterior
            return 1.00
        return 1.00

    def score_pipeline(
        self,
        df: pd.DataFrame,
        reference_date: str = "2017-12-31"
    ) -> pd.DataFrame:
        """
        Pontua todas as oportunidades do pipeline ativo (Engaging e Prospecting)
        com o Motor Surv-Bayes ERV e classificação em Matriz 4Q.
        """
        if not self.fitted:
            raise ValueError("O motor SurvBayesERVEngine precisa ser ajustado com fit() antes de pontuar.")

        ref_dt = pd.to_datetime(reference_date)
        res = df.copy()

        # 1. Base Probability (Encolhimento Bayesiano Produto + Vendedor)
        p_prod = res["product"].map(self.product_priors).fillna(self.global_win_rate)
        p_agent = res["sales_agent"].map(self.agent_priors).fillna(self.global_win_rate)
        
        ap_list = []
        for a, p in zip(res["sales_agent"], res["product"]):
            ap_list.append(self.agent_product_priors.get((a, p), self.product_priors.get(p, self.global_win_rate)))
        p_ap = pd.Series(ap_list, index=res.index)

        # Probabilidade intrínseca basal do par (Produto, Vendedor)
        p_base = 0.50 * p_prod + 0.30 * p_ap + 0.20 * p_agent
        res["prob_base"] = np.clip(p_base, 0.40, 0.85)

        # 2. Account Expansion Multiplier (AEM) — Neutralidade Cadastral
        # Contas sem cadastro ou novas = 1.00 (sem penalização).
        # Contas com histórico prévio de recompras = 1.15 (bônus de expansão).
        aem = np.where(res["account"].isin(self.repeat_accounts), 1.15, 1.00)
        res["account_multiplier"] = aem

        # 3. Dynamic Survival & Quarter Modulator por Deal
        survival_vals = []
        quarter_mods = []
        p_win_vals = []
        erv_vals = []
        matrix_quadrants = []
        action_playbooks = []
        positive_drivers = []
        risk_bottlenecks = []

        is_prospecting_mask = res["deal_stage"] == "Prospecting"

        for idx, row in res.iterrows():
            stage = row["deal_stage"]
            tau = float(row.get("deal_age_days", 0.0))
            price = float(row.get("sales_price", 550.0))
            prod_name = str(row.get("product", ""))
            agent_name = str(row.get("sales_agent", ""))
            acc_name = str(row.get("account", ""))
            base_p = float(res.loc[idx, "prob_base"])
            acc_mult = float(res.loc[idx, "account_multiplier"])

            pos_factors = []
            risk_factors = []

            # ==============================================================
            # ESTEIRA B: PROSPECTING (Leads novos sem contato ativo ainda)
            # ==============================================================
            if stage == "Prospecting":
                s_val = 0.60  # Sobrevivência inicial neutra
                q_mod = 1.00
                p_win = np.clip(base_p * s_val, 0.25, 0.60)
                # Índice de Potencial de Ativação (IPA)
                erv = (price * (1.0 + (base_p - self.global_win_rate))) / 14.0  # ciclo de 14 dias para primeiro contato

                quadrant = "🌱 Prospecção / Ativação Inicial"
                playbook = "Realizar 1º contato de sondagem para qualificar dor, timing e orçamento."
                
                pos_factors.append(f"Oportunidade Nova no Topo de Funil (Ticket: ${price:,.0f})")
                if price >= 4000:
                    pos_factors.append("Ticket Alto Estratégico (Priorizar contato)")
                if acc_mult > 1.0:
                    pos_factors.append(f"Conta com histórico prévio no CRM ({acc_name})")
                else:
                    risk_factors.append("Lead sem contato inicial estabelecido")

            # ==============================================================
            # ESTEIRA A: ENGAGING (Negociações em andamento com engage_date)
            # ==============================================================
            else:
                s_val = self._calc_survival(tau)
                q_mod = self._calc_quarter_modulator(ref_dt, tau)
                p_win = np.clip(base_p * s_val * q_mod, 0.0, 0.95)

                # Dias restantes de esforço útil (com piso protetor de 7 dias)
                remaining_cycle = max(7.0, self.median_duration - tau)
                # Expected Revenue Velocity ($/dia)
                erv = (p_win * price * acc_mult) / remaining_cycle

                # Classificação na Matriz 4Q
                if tau > self.tau_quarantine:
                    quadrant = "🛑 Q4: Quarentena / Zumbis"
                    playbook = "Arquivar ou enviar para nutrição de marketing. Não queimar tempo de vendedor."
                    risk_factors.append(f"Estagnação Crítica ({int(tau)} dias em aberto — Taxa de Fechamento Histórica: 0%)")
                    risk_factors.append("Corte Sanitário Ativado (>110 dias)")
                    if price >= 4000:
                        pos_factors.append(f"Ticket Elevado (${price:,.0f}), mas probabilidade colapsada por tempo")
                
                elif price >= 4000 and (15 <= tau <= 45):
                    quadrant = "🔥 Q1: Hot Closers (Ataque Imediato)"
                    playbook = "Ligar hoje para apresentar proposta executiva e cravar data de fechamento."
                    pos_factors.append(f"Ticket Enterprise/Premium (${price:,.0f})")
                    pos_factors.append(f"Janela de Ouro de Fechamento ({int(tau)} dias)")
                    if q_mod > 1.10:
                        pos_factors.append("Aceleração de Fim de Trimestre (Sprint de Metas)")
                    if acc_mult > 1.0:
                        pos_factors.append(f"Cliente Recorrente de Carteira ({acc_name})")

                elif price >= 3000 and (46 <= tau <= 90):
                    quadrant = "⚠️ Q2: High-Value Stalling (Alerta Executivo)"
                    playbook = "Envolver gerente regional (visita conjunta ou condição executiva) para destravar."
                    pos_factors.append(f"Alto Ticket em Jogo (${price:,.0f})")
                    risk_factors.append(f"Negociação esfriando ({int(tau)} dias — risco iminente de perda)")
                    if q_mod > 1.10:
                        pos_factors.append("Pressão positiva de fechamento trimestral")

                elif price < 1500 and (tau <= 45):
                    quadrant = "⚡ Q3: Quick Wins (Giro Rápido)"
                    playbook = "Disparo de proposta comercial simplificada e follow-up ágil para volume."
                    pos_factors.append(f"Produto de Giro Rápido (${price:,.0f})")
                    pos_factors.append(f"Lead fresco ({int(tau)} dias de conversa)")
                
                else:
                    quadrant = "💼 Cadência Normal de Acompanhamento"
                    playbook = "Manter cadência consultiva semanal padrão sem concessões extraordinárias."
                    pos_factors.append(f"Negociação ativa regular ({int(tau)} dias)")
                    if tau > 60:
                        risk_factors.append(f"Tempo avançado ({int(tau)} dias — acelerar próximos passos)")

            survival_vals.append(round(s_val, 4))
            quarter_mods.append(round(q_mod, 2))
            p_win_vals.append(round(p_win, 4))
            erv_vals.append(round(erv, 2))
            matrix_quadrants.append(quadrant)
            action_playbooks.append(playbook)
            positive_drivers.append(" | ".join(pos_factors) if pos_factors else "Negociação em parâmetros normais")
            risk_bottlenecks.append(" | ".join(risk_factors) if risk_factors else "Sem risco crítico de estagnação")

        res["survival_factor"] = survival_vals
        res["quarter_modulator"] = quarter_mods
        res["calibrated_prob_won"] = p_win_vals
        res["erv_usd_per_day"] = erv_vals
        res["matrix_quadrant"] = matrix_quadrants
        res["action_playbook"] = action_playbooks
        res["positive_drivers"] = positive_drivers
        res["risk_bottlenecks"] = risk_bottlenecks

        # 4. Cálculo do Lead Score Normalizado (0 a 100)
        # Deals em Quarentena recebem teto de 15 pontos
        # Deals ativos são ranqueados pela combinação equilibrada de P(Win) e ERV
        active_mask = res["matrix_quadrant"] != "🛑 Q4: Quarentena / Zumbis"
        
        erv_series = res["erv_usd_per_day"]
        # Percentil do ERV entre os deals ativos
        erv_rank = pd.Series(0.0, index=res.index)
        if active_mask.sum() > 0:
            erv_rank[active_mask] = erv_series[active_mask].rank(pct=True)

        lead_scores = np.zeros(len(res))
        for i, idx in enumerate(res.index):
            if not active_mask.iloc[i]:
                # Pontuação de quarentena proporcional ao ticket residual
                lead_scores[i] = min(15.0, round(float(res.loc[idx, "sales_price"]) / 2000.0, 1))
            else:
                prob = float(res.loc[idx, "calibrated_prob_won"])
                er_p = float(erv_rank.iloc[i])
                score = (0.40 * prob + 0.60 * er_p) * 100.0
                lead_scores[i] = round(np.clip(score, 16.0, 99.5), 1)

        res["lead_score"] = lead_scores

        # 5. Faixa Executiva de Prioridade
        res["priority_tier"] = pd.cut(
            res["lead_score"],
            bins=[-1, 20, 55, 80, 100],
            labels=[
                "Quarentena / Descarte",
                "Baixa Prioridade",
                "Média (Acompanhamento)",
                "Alta (Ataque Imediato)"
            ]
        )

        return res

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> "SurvBayesERVEngine":
        return joblib.load(filepath)
