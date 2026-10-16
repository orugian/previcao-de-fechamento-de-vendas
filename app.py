"""
Desafio 003 — Lead Scorer & Decision Engine (RevOps)
Interface Web Funcional em Streamlit com Motor Surv-Bayes ERV e Copiloto LLM
"""
from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Configuração da Página
st.set_page_config(
    page_title="NEXUS // RevOps Intelligence Cockpit (DRACO AI)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeção de Tema Cyberpunk NEXUS (Tokens de Design Tático)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo+Condensed:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* Variáveis Globais NEXUS Engine */
    :root {
        --nexus-surface: #0000F2;
        --nexus-surface-alt: #F2F2F2;
        --nexus-primary: #000091;
        --nexus-hairline: #616191;
        --nexus-ink: #F5F5F5;
        --nexus-bg-deep: #030511;
        --nexus-bg-surface: #07091B;
        --nexus-bg-card: #0C102E;
        --cyber-cyan: #00F0FF;
        --cyber-green: #00FF9D;
        --cyber-amber: #FFB800;
        --cyber-crimson: #FF3366;
    }

    /* Streamlit Background & Text */
    .stApp {
        background-color: var(--nexus-bg-deep) !important;
        color: var(--nexus-ink) !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* Sidebar Cyberpunk */
    [data-testid="stSidebar"] {
        background-color: var(--nexus-bg-surface) !important;
        border-right: 1px solid rgba(97, 97, 145, 0.3) !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Archivo Condensed', sans-serif !important;
        color: #FFFFFF !important;
        letter-spacing: 0.5px !important;
    }

    /* Metrics & Monospace */
    [data-testid="stMetricValue"] {
        font-family: 'Archivo Condensed', sans-serif !important;
        font-size: 32px !important;
        font-weight: 700 !important;
        color: var(--cyber-cyan) !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.8px !important;
        color: #8B8DA8 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--nexus-surface) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--nexus-surface) !important;
        border-radius: 3px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 0 10px rgba(0, 0, 242, 0.4) !important;
    }
    .stButton > button:hover {
        background-color: #1a1aff !important;
        border-color: var(--cyber-cyan) !important;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.5) !important;
    }

    /* Cards e Callouts */
    .stAlert {
        background-color: var(--nexus-bg-card) !important;
        border-radius: 3px !important;
        border: 1px solid rgba(97, 97, 145, 0.3) !important;
        color: var(--nexus-ink) !important;
    }

    /* Inputs */
    input, select, textarea {
        background-color: #040616 !important;
        border: 1px solid rgba(97, 97, 145, 0.3) !important;
        border-radius: 3px !important;
        color: #FFFFFF !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)

    /* Flagship Banner */
    .flagship-banner {
        background: linear-gradient(90deg, #000091, #0000F2);
        border: 1px solid var(--cyber-cyan);
        border-radius: 3px;
        padding: 10px 16px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #FFFFFF;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
    }
    .flagship-banner a {
        background: #FFFFFF;
        color: #000091;
        padding: 4px 10px;
        border-radius: 2px;
        text-decoration: none;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Adicionar diretório raiz ao sys.path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from src.copilot_llm import query_llm_copilot, DealCopilotAnalysis

ARTIFACTS_DIR = APP_DIR / "artifacts"
DATA_PATH = ARTIFACTS_DIR / "active_scored_deals.csv"
BACKTEST_PATH = ARTIFACTS_DIR / "economic_backtest_summary.csv"


@st.cache_data
def load_scored_deals() -> pd.DataFrame:
    """Carrega as 2.089 oportunidades ativas pontuadas pelo motor Surv-Bayes ERV."""
    if not DATA_PATH.exists():
        st.error(f"Arquivo {DATA_PATH} não encontrado. Execute o treinamento primeiro.")
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    # Formatação de campos
    df["account_display"] = df["account"].fillna("Conta Não Qualificada (Topo de Funil)")
    return df


@st.cache_data
def load_backtest_summary() -> pd.DataFrame:
    """Carrega o resumo do backtest econômico out-of-time (Q4 2017)."""
    if BACKTEST_PATH.exists():
        return pd.read_csv(BACKTEST_PATH)
    return pd.DataFrame()


df_active = load_scored_deals()
df_backtest = load_backtest_summary()

if df_active.empty:
    st.stop()

# ==============================================================================
# SIDEBAR: FILTROS E SELEÇÃO DE PERFIL
# ==============================================================================
with st.sidebar:
    st.title("NEXUS RevOps")
    st.markdown("**Motor de Throughput Comercial (Surv-Bayes ERV)**")
    st.divider()

    st.subheader("Perfil de Acesso")
    view_mode = st.radio(
        "Modo de Visualização:",
        ["Vendedor (Cockpit Pessoal)", "Gestão / RevOps (Visão Macro)"],
        index=0
    )

    selected_agent = "Todos"
    selected_region = "Todas"
    selected_manager = "Todos"

    if view_mode == "Vendedor (Cockpit Pessoal)":
        agents_list = sorted(df_active["sales_agent"].unique())
        selected_agent = st.selectbox("Selecione o Vendedor:", agents_list, index=0)
        user_filter_mask = df_active["sales_agent"] == selected_agent
        agent_data = df_active[user_filter_mask].iloc[0] if user_filter_mask.sum() > 0 else None
        if agent_data is not None:
            st.caption(f"[REGIONAL: {agent_data['regional_office']} | GERENTE: {agent_data['manager']}]")
    else:
        regions_list = ["Todas"] + sorted(df_active["regional_office"].dropna().unique().tolist())
        selected_region = st.selectbox("Escritório Regional:", regions_list, index=0)

        managers_list = ["Todos"] + sorted(df_active["manager"].dropna().unique().tolist())
        selected_manager = st.selectbox("Gerente Comercial:", managers_list, index=0)

        user_filter_mask = pd.Series(True, index=df_active.index)
        if selected_region != "Todas":
            user_filter_mask &= df_active["regional_office"] == selected_region
        if selected_manager != "Todos":
            user_filter_mask &= df_active["manager"] == selected_manager

    st.divider()
    st.subheader("Filtros de Pipeline")

    quadrants_available = ["Todos"] + sorted(df_active["matrix_quadrant"].unique().tolist())
    selected_quadrant = st.selectbox("Segmento do Funil / Matriz 4Q:", quadrants_available, index=0)
    if selected_quadrant != "Todos":
        user_filter_mask &= df_active["matrix_quadrant"] == selected_quadrant

    stage_options = ["Todos", "Engaging (Negociação Ativa)", "Prospecting (Leads Novos)"]
    selected_stage = st.selectbox("Estágio do Funil:", stage_options, index=0)
    if "Engaging" in selected_stage:
        user_filter_mask &= df_active["deal_stage"] == "Engaging"
    elif "Prospecting" in selected_stage:
        user_filter_mask &= df_active["deal_stage"] == "Prospecting"

    st.divider()
    st.subheader("Configuração do Copiloto DRACO")
    llm_provider = st.selectbox(
        "Provedor LLM:",
        ["OpenRouter (Multi-Model / Hermes)", "OpenAI Direto", "Offline Grounded (Sem Custo)"],
        index=0
    )

    openrouter_key_input = ""
    openrouter_model_input = "nousresearch/hermes-3-llama-3.1-70b"
    openai_key_input = ""

    if llm_provider.startswith("OpenRouter"):
        openrouter_key_input = st.text_input(
            "OpenRouter API Key:",
            type="password",
            value=os.getenv("OPENROUTER_API_KEY", ""),
            help="Chave de API do OpenRouter. Acesse https://openrouter.ai/keys"
        )
        model_options = [
            "nousresearch/hermes-3-llama-3.1-70b",
            "nousresearch/deephermes-3-llama-3-8b-preview",
            "nousresearch/hermes-3-llama-3.1-405b",
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet",
            "Customizado..."
        ]
        chosen_model = st.selectbox("Modelo:", model_options, index=0)
        if chosen_model == "Customizado...":
            openrouter_model_input = st.text_input("Identificador do Modelo:", value="meta-llama/llama-3.3-70b-instruct")
        else:
            openrouter_model_input = chosen_model

        if not openrouter_key_input and not os.getenv("OPENROUTER_API_KEY"):
            st.info("[STATUS] Insira sua OpenRouter Key. Sem chave, o sistema usa o motor determinístico offline.")
    elif llm_provider.startswith("OpenAI"):
        openai_key_input = st.text_input(
            "OpenAI API Key:",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Chave de API da OpenAI (sk-...)"
        )
        if not openai_key_input and not os.getenv("OPENAI_API_KEY"):
            st.info("[STATUS] Modo Offline Grounded ativo.")
    else:
        st.info("[STATUS] Modo Offline Grounded ativo: análises geradas deterministicamente pelo motor Surv-Bayes.")

    st.markdown("---")
    st.markdown("""
    <div style="background:#07091B; border:1px solid #0000F2; padding:10px 12px; border-radius:3px;">
        <span style="color:#00F0FF; font-family:'JetBrains Mono',monospace; font-size:11px;">[INTERFACE FLAGSHIP DISPONÍVEL]</span>
        <p style="font-size:11.5px; color:#F5F5F5; margin:4px 0 0 0;">
            Acesse o novo <strong>NEXUS Cyberpunk Cockpit com Mascote 3D DRACO</strong> nativo executando <code>python web_app.py</code> em <code>http://localhost:8000</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Aplicação do filtro global
filtered_df = df_active[user_filter_mask].copy()

# ==============================================================================
# HEADER PRINCIPAL E KPIS
# ==============================================================================
st.markdown("## NEXUS Cockpit // Priorização Comercial & Throughput")
if view_mode.startswith("Vendedor"):
    st.markdown(f"**Carteira de Negociações:** `{selected_agent}` | Pipeline Ativo na Segunda-Feira")
else:
    st.markdown(f"**Visão Executiva de RevOps:** Regional `{selected_region}` | Gerência `{selected_manager}`")

col1, col2, col3, col4, col5 = st.columns(5)

total_deals_count = len(filtered_df)
zombies_count = (filtered_df["matrix_quadrant"].str.contains("Quarentena")).sum()
active_working_set = total_deals_count - zombies_count
active_pipeline_val = filtered_df[~filtered_df["matrix_quadrant"].str.contains("Quarentena")]["sales_price"].sum()
total_erv_val = filtered_df[~filtered_df["matrix_quadrant"].str.contains("Quarentena")]["erv_usd_per_day"].sum()

with col1:
    st.metric("Oportunidades no CRM", f"{total_deals_count:,}")
with col2:
    st.metric("Fila Ativa Útil (Sem Zumbis)", f"{active_working_set:,}", delta=f"-{zombies_count} zumbis")
with col3:
    st.metric("Pipeline Ativo em Risco", f"${active_pipeline_val:,.0f}")
with col4:
    st.metric("Throughput Total (ERV)", f"${total_erv_val:,.2f}/dia")
with col5:
    st.metric("Deals em Quarentena (>110d)", f"{zombies_count:,}", delta_color="inverse")

st.markdown("---")

# ==============================================================================
# TABS PRINCIPAIS
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "01. Fila de Ação de Segunda-Feira",
    "02. Cerca Sanitária (Quarentena de Zumbis)",
    "03. Diagnóstico Visual & Matriz 4Q",
    "04. Backtesting Econômico (Lift em USD)"
])

# ------------------------------------------------------------------------------
# TAB 1: COCKPIT OPERACIONAL
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Fila Priorizada de Ataque Comercial")
    st.markdown(
        "Os negócios abaixo estão ordenados pelo **Throughput Comercial (ERV)** e **Lead Score**, "
        "combinando o valor do ticket com a janela de maturação e probabilidade estatística de vitória."
    )

    # Excluir zumbis da fila diária (Cerca Sanitária Ativa)
    working_df = filtered_df[~filtered_df["matrix_quadrant"].str.contains("Quarentena")].copy()
    working_df = working_df.sort_values(by="lead_score", ascending=False)

    if working_df.empty:
        st.warning("Nenhuma oportunidade ativa na fila após a aplicação dos filtros selecionados.")
    else:
        # Card de Destaque para o Lead #1
        top_deal = working_df.iloc[0]
        with st.container():
            st.success(
                f"### LEAD PRIORITÁRIO 01 DA SEMANA: `{top_deal['opportunity_id']}` — {top_deal['product']} (${top_deal['sales_price']:,.0f})\n"
                f"**Cliente / Conta:** {top_deal['account_display']} | **Estágio:** {top_deal['deal_stage']} ({int(top_deal['deal_age_days'])} dias de negociação)\n\n"
                f"**Score de Prioridade:** `{top_deal['lead_score']}/100` ({top_deal['priority_tier']}) | "
                f"**Probabilidade Calibrada:** `{top_deal['calibrated_prob_won']*100:.1f}%` | "
                f"**Throughput (ERV):** `${top_deal['erv_usd_per_day']:,.2f}/dia`\n\n"
                f"**Diretriz Tática:** {top_deal['action_playbook']}"
            )

        st.markdown("#### Tabela de Negociações Vivas (Ordem de Atendimento)")

        cols_display = [
            "opportunity_id", "sales_agent", "product", "account_display",
            "deal_stage", "deal_age_days", "sales_price", "calibrated_prob_won",
            "erv_usd_per_day", "lead_score", "matrix_quadrant", "action_playbook"
        ]

        display_table = working_df[cols_display].copy()
        display_table["deal_age_days"] = display_table["deal_age_days"].astype(int)
        display_table["calibrated_prob_won"] = (display_table["calibrated_prob_won"] * 100).round(1).astype(str) + "%"
        display_table["sales_price"] = "$" + display_table["sales_price"].map("{:,.0f}".format)
        display_table["erv_usd_per_day"] = "$" + display_table["erv_usd_per_day"].map("{:,.2f}".format)

        display_table.columns = [
            "ID Deal", "Vendedor", "Produto", "Conta Cliente",
            "Estágio", "Idade (dias)", "Ticket", "Prob. Vitória",
            "Throughput ERV", "Score", "Quadrante", "Recomendação Operacional"
        ]

        st.dataframe(display_table, use_container_width=True, height=350)

        # Seletor para o Copiloto LLM
        st.divider()
        st.subheader("Consultar Copiloto DRACO para Ação Tática Individual")
        
        selected_opp_id = st.selectbox(
            "Selecione uma Oportunidade para Diagnóstico Completo com IA:",
            working_df["opportunity_id"].tolist(),
            index=0
        )

        selected_row = working_df[working_df["opportunity_id"] == selected_opp_id].iloc[0].to_dict()

        col_act1, col_act2 = st.columns([1, 4])
        with col_act1:
            btn_generate = st.button("Executar Análise com IA", type="primary", use_container_width=True)

        if btn_generate:
            with st.spinner("O Copiloto DRACO está analisando os dados matemáticos do deal..."):
                selected_provider = "openrouter" if llm_provider.startswith("OpenRouter") else ("openai" if llm_provider.startswith("OpenAI") else "offline")
                analysis: DealCopilotAnalysis = query_llm_copilot(
                    selected_row,
                    provider=selected_provider,
                    model=openrouter_model_input if selected_provider == "openrouter" else "gpt-4o-mini",
                    openrouter_key=openrouter_key_input if openrouter_key_input else None,
                    openai_key=openai_key_input if openai_key_input else None
                )

                st.markdown("### Diagnóstico Executivo & Roteiro de Abordagem")
                
                c_card1, c_card2 = st.columns(2)
                with c_card1:
                    st.info(f"**Resumo Executivo:**\n\n{analysis.resumo_executivo}")
                    st.warning(f"**Diagnóstico de Gargalo / Risco:**\n\n{analysis.diagnostico_risco}")
                with c_card2:
                    st.success(f"**Alavanca Comercial de Fechamento:**\n\n{analysis.alavanca_comercial}")
                    st.markdown(f"**Próxima Ação Imediata:**\n\n`{analysis.acao_imediata}`")

                st.markdown("#### Roteiro de Contato Personalizado (Pronto para Disparo)")
                st.code(analysis.roteiro_contato, language="markdown")
                st.caption("Dica: Copie o texto acima e envie diretamente via WhatsApp corporativo ou e-mail institucional.")

# ------------------------------------------------------------------------------
# TAB 2: CERCA SANITÁRIA (QUARENTENA DE ZUMBIS)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Cerca Sanitária do Pipeline: Quarentena de Negócios Zumbis")
    st.markdown(
        """
        **A Lei de Sobrevivência Comercial do CRM:**  
        A auditoria empírica comprovou que **nenhum negócio na história da empresa fechou após 138 dias**.  
        Oportunidades com mais de **110 dias em aberto** apresentam taxa de conversão futura estatisticamente de **0,00%**.  
        Manter esses negócios na fila diária consome a atenção do vendedor e destrói o resultado econômico da empresa.
        """
    )

    zombies_df = filtered_df[filtered_df["matrix_quadrant"].str.contains("Quarentena")].copy()
    zombies_df = zombies_df.sort_values(by="deal_age_days", ascending=False)

    col_z1, col_z2, col_z3 = st.columns(3)
    with col_z1:
        st.metric("Oportunidades em Quarentena", f"{len(zombies_df):,}")
    with col_z2:
        st.metric("Volume Financeiro Zumbi Travado", f"${zombies_df['sales_price'].sum():,.0f}")
    with col_z3:
        st.metric("Idade Média dos Zumbis", f"{zombies_df['deal_age_days'].mean():.0f} dias")

    st.markdown("#### Lista de Negócios Sugeridos para Arquivamento ou Nutrição")
    cols_z_disp = [
        "opportunity_id", "sales_agent", "product", "account_display",
        "deal_age_days", "sales_price", "risk_bottlenecks", "action_playbook"
    ]
    z_table = zombies_df[cols_z_disp].copy()
    z_table["deal_age_days"] = z_table["deal_age_days"].astype(int)
    z_table["sales_price"] = "$" + z_table["sales_price"].map("{:,.0f}".format)
    z_table.columns = [
        "ID Deal", "Vendedor", "Produto", "Conta",
        "Dias em Aberto", "Ticket Tabela", "Diagnóstico de Estagnação", "Ação de Governança"
    ]
    st.dataframe(z_table, use_container_width=True, height=350)

    st.download_button(
        label="Exportar Lista de Zumbis para Automação de Marketing (CSV)",
        data=zombies_df.to_csv(index=False).encode("utf-8"),
        file_name="quarentena_zumbis_crm_nexus.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------------------------
# TAB 3: DIAGNÓSTICO VISUAL & MATRIZ 4Q
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Distribuição Estratégica do Funil e Matriz 4Q")

    c_g1, c_g2 = st.columns(2)

    with c_g1:
        quadrant_summary = filtered_df["matrix_quadrant"].value_counts().reset_index()
        quadrant_summary.columns = ["Quadrante", "Contagem"]
        
        fig_quad = px.bar(
            quadrant_summary,
            x="Contagem",
            y="Quadrante",
            orientation="h",
            color="Quadrante",
            title="Volume de Oportunidades por Quadrante na Carteira",
            color_discrete_sequence=["#00FF9D", "#FFB800", "#00F0FF", "#FF3366"]
        )
        fig_quad.update_layout(
            showlegend=False,
            height=380,
            paper_bgcolor="#07091B",
            plot_bgcolor="#07091B",
            font=dict(color="#F5F5F5", family="Space Grotesk")
        )
        st.plotly_chart(fig_quad, use_container_width=True)

    with c_g2:
        live_sub = filtered_df[~filtered_df["matrix_quadrant"].str.contains("Quarentena")]
        fig_scatter = px.scatter(
            live_sub,
            x="deal_age_days",
            y="erv_usd_per_day",
            color="matrix_quadrant",
            size="sales_price",
            hover_data=["opportunity_id", "sales_agent", "product", "account_display"],
            title="Throughput Comercial (ERV $/dia) vs Idade (Fila Ativa)",
            labels={"deal_age_days": "Dias em Negociação", "erv_usd_per_day": "Throughput ERV ($/dia)"}
        )
        fig_scatter.update_layout(
            height=380,
            paper_bgcolor="#07091B",
            plot_bgcolor="#07091B",
            font=dict(color="#F5F5F5", family="Space Grotesk")
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Entendendo os 4 Quadrantes de Decisão:")
    cq1, cq2, cq3, cq4 = st.columns(4)
    with cq1:
        st.success("**[Q1] Hot Closers**\n\nTicket Alto ($ $\\ge 4k$) na janela de ouro (15-45 dias). Prioridade absoluta de ligação hoje.")
    with cq2:
        st.warning("**[Q2] High-Value Stalling**\n\nTicket Alto ($ $\\ge 3k$) esfriando (46-90 dias). Demanda envolvimento do gerente para destravar.")
    with cq3:
        st.info("**[Q3] Quick Wins**\n\nProdutos transacionais (< $1.500) em idade fresca. Giro rápido para fechar meta semanal.")
    with cq4:
        st.error("**[Q4] Quarentena / Zumbis**\n\nDeals com >110 dias sem avanço. Removidos da visão diária para evitar desperdício de tempo.")

# ------------------------------------------------------------------------------
# TAB 4: BACKTESTING ECONÔMICO & AUDITORIA DE LEAKAGE
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Backtesting Econômico Out-of-Time & Auditoria de Data Leakage")
    st.markdown(
        """
        **A Prova Quantitativa da Solução:**  
        Avaliamos o modelo em um teste cego estrito no 4º trimestre de 2017 (`2017-10-01` a `2017-12-31`), 
        treinado exclusivamente nos dados históricos anteriores.  
        Simulamos uma capacidade de contato restrita a **20,5% das oportunidades abertas (~500 deals de 2.434)**.
        """
    )

    if not df_backtest.empty:
        st.dataframe(df_backtest, use_container_width=True)

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("### Conclusão Econômica:")
        st.markdown(
            """
            1. **Captura de 69,2% da Receita com Apenas 20,5% do Volume:**  
               O motor Surv-Bayes ERV capturou **$877.572,00** da receita de Q4, superando em **3,37x (+337%)** a escolha no *feeling*.
            2. **Ganho Líquido de +$215.640 sobre a Escolha por Preço:**  
               Vendedores que ordenam apenas pelo maior preço caem na armadilha dos negócios de $26k estagnados há meses. O Surv-Bayes penaliza o atrito temporal e direciona para deals que estão vivos.
            """
        )

    with col_b2:
        st.markdown("### Transparência Técnica: A Descoberta do Leakage")
        st.markdown(
            """
            - **O Erro dos Modelos Tradicionais:** Modelos ingênuos de ML atingiam ~81% de ROC-AUC porque o CRM só registra o vínculo da conta cliente no fechamento. O algoritmo aprendia a prever se o cadastro estava completo, não se o cliente iria comprar.
            - **O Julgamento Humano do AI Master:** Identificamos que 68% dos leads ativos não tinham conta e que condená-los à nota zero paralisaria a operação. Implementamos o **Princípio da Ortogonalidade Cadastral**, tornando o sistema 100% honesto e imune a vazamentos.
            """
        )

# Rodapé
st.markdown("---")
st.caption("Desenvolvido para o Programa AI Master — G4 Educação | Desafio 003: Lead Scorer & Decision Engine | Engine NEXUS")
