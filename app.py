import os
import sys

# Garantir que o diretório src/ seja encontrado independentemente de como o streamlit foi invocado
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Importar motores locais
from feature_engineering import load_and_clean_data, build_features
from scorer_engine import LeadScorerEngine
from llm_copilot import generate_llm_action_plan
from backtest_engine import run_backtest_simulation

# Configuração da página Streamlit
st.set_page_config(
    page_title="AI Lead Scorer — RevOps Cockpit",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS refinada
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .badge-hot {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-warn {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-zombie {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Carregando e processando inteligência do pipeline...")
def get_scored_data():
    df = load_and_clean_data()
    df_feat, _ = build_features(df)
    open_pipeline = df_feat[df_feat['deal_stage'].isin(['Engaging', 'Prospecting'])].copy()
    engine = LeadScorerEngine()
    scored_df = engine.score_pipeline(open_pipeline)
    return scored_df

@st.cache_data(show_spinner="Calculando simulação de backtest...")
def get_backtest_data():
    return run_backtest_simulation()

def main():
    # Sidebar
    st.sidebar.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=300&q=80", use_container_width=True)
    st.sidebar.title("🎯 AI Lead Scorer")
    st.sidebar.caption("RevOps Decision Cockpit — G4 Educação")

    scored_df = get_scored_data()

    # Chave de API OpenAI opcional
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configurações de IA")
    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key_input = st.sidebar.text_input("OpenAI API Key (Opcional)", value=env_key, type="password", help="Se deixada em branco, o copiloto opera via motor heurístico fundamentado em SHAP.")
    
    if api_key_input:
        st.sidebar.success("🔑 Copiloto Generativo Ativo")
    else:
        st.sidebar.info("⚡ Modo Heurístico Ativo (SHAP)")

    # Perfil de Acesso
    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 Perfil de Acesso")
    view_mode = st.sidebar.radio(
        "Selecione sua visão:",
        ["🧑‍💼 Vendedor (Minha Segunda-feira)", "👔 Gestão & RevOps", "📈 Backtest & Lift Financeiro"]
    )

    if view_mode == "🧑‍💼 Vendedor (Minha Segunda-feira)":
        render_seller_view(scored_df, api_key_input)
    elif view_mode == "👔 Gestão & RevOps":
        render_manager_view(scored_df)
    else:
        render_backtest_view()

def render_seller_view(df: pd.DataFrame, api_key: str):
    st.title("🧑‍💼 Minha Segunda-feira — Priorização de Carteira")
    st.markdown("Bem-vindo! Esta ferramenta analisa matematicamente o seu pipeline e indica **exatamente onde focar** para maximizar fechamentos e evitar perda de tempo em negócios estagnados.")

    # Filtro de Vendedor
    agents = sorted(df['sales_agent'].unique().tolist())
    selected_agent = st.selectbox("Selecione seu nome (Vendedor):", agents, index=0)

    agent_df = df[df['sales_agent'] == selected_agent].copy()
    agent_df = agent_df.sort_values(by='lead_score', ascending=False).reset_index(drop=True)

    # Métricas de Cabeçalho do Vendedor
    c1, c2, c3, c4 = st.columns(4)
    total_deals = len(agent_df)
    hot_deals = len(agent_df[agent_df['lead_score'] >= 70])
    zombie_deals = len(agent_df[agent_df['urgency_status'] == 'Zumbi / Estagnado'])
    ev_total = agent_df['expected_value'].sum()

    with c1:
        st.metric("Oportunidades Ativas", f"{total_deals} deals")
    with c2:
        st.metric("Deals Quentes (Score ≥ 70)", f"{hot_deals} deals", delta=f"{hot_deals/max(total_deals,1)*100:.0f}% da carteira")
    with c3:
        st.metric("Pipeline Ponderado (EV)", f"${ev_total:,.2f}", help="Soma da (Probabilidade Calibrada × Ticket) de cada oportunidade")
    with c4:
        st.metric("Deals Zumbis (>138 dias)", f"{zombie_deals} deals", delta=f"-{zombie_deals} estagnados", delta_color="inverse", help="Deals abertos há mais tempo que o ciclo máximo histórico de fechamento")

    st.markdown("---")

    # Banner de Destaque - Top 3 Ações Imediatas
    st.subheader("🔥 Top Oportunidades Prioritárias para Agir Hoje")
    top_3 = agent_df.head(3)

    cols = st.columns(3)
    for idx, (_, deal) in enumerate(top_3.iterrows()):
        with cols[idx]:
            with st.container(border=True):
                st.markdown(f"### #{idx+1} — {deal['product']}")
                st.markdown(f"**Cliente:** `{deal['account']}`")
                st.markdown(f"**Score:** `{deal['lead_score']}/100` | **Prob:** `{deal['calibrated_probability']}%`")
                st.markdown(f"**Ticket:** `${deal['sales_price']:,.2f}` | **Dias:** `{deal['days_in_pipeline']} dias`")
                
                status_color = "badge-hot" if deal['urgency_status'] == 'Quente / Ativo' else ("badge-warn" if deal['urgency_status'] == 'Atenção / Esfriando' else "badge-zombie")
                st.markdown(f"<span class='{status_color}'>{deal['urgency_status']}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # Tabela Geral do Vendedor com Filtro de Urgência
    st.subheader("📋 Todas as Oportunidades da Carteira")
    
    col_f1, col_f2 = st.columns([2, 2])
    with col_f1:
        filter_urgency = st.multiselect(
            "Filtrar por Status de Urgência:",
            options=['Quente / Ativo', 'Atenção / Esfriando', 'Zumbi / Estagnado'],
            default=['Quente / Ativo', 'Atenção / Esfriando', 'Zumbi / Estagnado']
        )
    with col_f2:
        min_score = st.slider("Score Mínimo:", 0.0, 100.0, 0.0, step=5.0)

    filtered_df = agent_df[
        (agent_df['urgency_status'].isin(filter_urgency)) &
        (agent_df['lead_score'] >= min_score)
    ]

    # Exibição tabular amigável
    display_cols = ['opportunity_id', 'account', 'product', 'sales_price', 'days_in_pipeline', 
                    'calibrated_probability', 'expected_value', 'lead_score', 'urgency_status']
    
    st.dataframe(
        filtered_df[display_cols].rename(columns={
            'opportunity_id': 'ID Deal',
            'account': 'Cliente',
            'product': 'Produto',
            'sales_price': 'Ticket ($)',
            'days_in_pipeline': 'Dias em Aberto',
            'calibrated_probability': 'Prob. Fechamento (%)',
            'expected_value': 'Valor Esperado ($)',
            'lead_score': 'Score Final (0-100)',
            'urgency_status': 'Status'
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Seção do Copiloto de IA & Explainability
    st.subheader("🤖 Copiloto de Vendas & Explicabilidade SHAP")
    st.markdown("Selecione um deal abaixo para inspecionar os **fatores matemáticos de decisão** e gerar o **plano tático de abordagem**:")

    deal_options = agent_df['opportunity_id'] + " — " + agent_df['account'] + " (" + agent_df['product'] + ")"
    selected_deal_str = st.selectbox("Escolha a oportunidade:", deal_options)

    if selected_deal_str:
        sel_id = selected_deal_str.split(" — ")[0]
        lead_row = agent_df[agent_df['opportunity_id'] == sel_id].iloc[0].to_dict()

        col_exp1, col_exp2 = st.columns([1, 1])

        with col_exp1:
            st.markdown("#### 📊 Por que esta pontuação? (Atribuição SHAP)")
            st.markdown(f"**Score:** `{lead_row['lead_score']}/100` | **Probabilidade Estimada:** `{lead_row['calibrated_probability']}%`")

            # Prós
            st.markdown("**🟢 Fatores Positivos (Alavancas):**")
            pos_factors = lead_row.get('top_positive_factors', [])
            if pos_factors:
                for f in pos_factors:
                    st.success(f"**{f['feature']}**: contribuição de +{f['impact']:.2f} no modelo")
            else:
                st.write("Nenhum fator positivo predominante.")

            # Contras
            st.markdown("**🔴 Fatores de Risco (Gargalos):**")
            neg_factors = lead_row.get('top_negative_factors', [])
            if neg_factors:
                for f in neg_factors:
                    st.error(f"**{f['feature']}**: impacto negativo de {f['impact']:.2f}")
            else:
                st.write("Nenhum gargalo expressivo detectado.")

        with col_exp2:
            st.markdown("#### ⚡ Ação Recomendada & Roteiro de Contato")
            
            if st.button("🚀 Gerar / Atualizar Plano com Copiloto IA", type="primary"):
                with st.spinner("Analisando atributos matemáticos e redigindo plano tático..."):
                    plan = generate_llm_action_plan(lead_row, api_key)
                    st.session_state[f"plan_{sel_id}"] = plan

            # Recuperar plano salvo na sessão
            plan = st.session_state.get(f"plan_{sel_id}", None)
            if not plan:
                # Gera o plano inicial automático
                plan = generate_llm_action_plan(lead_row, api_key)
                st.session_state[f"plan_{sel_id}"] = plan

            st.caption(f"Modo: **{plan.mode}**")
            st.markdown(f"**🚨 Diagnóstico de Risco:** {plan.risk_diagnosis}")
            st.markdown(f"**💡 Alavanca de Fechamento:** {plan.closing_hook}")
            st.markdown(f"**🎯 Ação Imediata:** `{plan.action_today}`")
            st.markdown("**💬 Roteiro de Mensagem Pronto para Copiar:**")
            st.code(plan.outreach_template, language="text")

def render_manager_view(df: pd.DataFrame):
    st.title("👔 Gestão Comercial & Visão RevOps")
    st.markdown("Painel executivo para liderança comercial: diagnóstico de gargalos, distribuição regional e auditoria de saúde do pipeline.")

    # Filtros Globais
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        offices = ['Todas'] + sorted(df['regional_office'].unique().tolist())
        sel_office = st.selectbox("Filtrar Regional:", offices)
    with c_f2:
        managers = ['Todos'] + sorted(df['manager'].unique().tolist())
        sel_manager = st.selectbox("Filtrar Gerente:", managers)

    filtered_df = df.copy()
    if sel_office != 'Todas':
        filtered_df = filtered_df[filtered_df['regional_office'] == sel_office]
    if sel_manager != 'Todos':
        filtered_df = filtered_df[filtered_df['manager'] == sel_manager]

    # KPIs de Liderança
    k1, k2, k3, k4 = st.columns(4)
    total_pipe = len(filtered_df)
    nominal_val = (filtered_df['sales_price']).sum()
    weighted_val = filtered_df['expected_value'].sum()
    zombie_cnt = len(filtered_df[filtered_df['urgency_status'] == 'Zumbi / Estagnado'])
    zombie_pct = (zombie_cnt / max(total_pipe, 1)) * 100

    with k1:
        st.metric("Total de Deals em Aberto", f"{total_pipe:,.0f}")
    with k2:
        st.metric("Pipeline Nominal (Preço de Lista)", f"${nominal_val:,.2f}", help="Soma simples dos preços de tabela")
    with k3:
        st.metric("Forecast Calibrado (Realista)", f"${weighted_val:,.2f}", delta=f"-{(1 - weighted_val/max(nominal_val,1))*100:.1f}% vs Nominal", delta_color="normal", help="Valor ponderado pelas probabilidades calibradas")
    with k4:
        st.metric("Deals Zumbis (>138 dias)", f"{zombie_cnt} ({zombie_pct:.1f}%)", delta="Gargalo Crítico", delta_color="inverse")

    st.markdown("---")

    # Gráficos Analíticos
    g1, g2 = st.columns(2)

    with g1:
        st.subheader("🗺️ Concentração por Status de Urgência")
        status_counts = filtered_df['urgency_status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Total']
        fig_pie = px.pie(
            status_counts, 
            values='Total', 
            names='Status', 
            color='Status',
            color_discrete_map={
                'Quente / Ativo': '#2E7D32',
                'Atenção / Esfriando': '#FB8C00',
                'Zumbi / Estagnado': '#C62828'
            },
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with g2:
        st.subheader("👥 Saúde do Pipeline por Vendedor")
        agent_agg = filtered_df.groupby('sales_agent').agg(
            total_deals=('opportunity_id', 'count'),
            zombie_deals=('urgency_status', lambda x: (x == 'Zumbi / Estagnado').sum()),
            ev=('expected_value', 'sum')
        ).reset_index().sort_values(by='zombie_deals', ascending=False).head(10)

        fig_bar = px.bar(
            agent_agg,
            x='sales_agent',
            y=['zombie_deals', 'total_deals'],
            barmode='group',
            labels={'value': 'Qtd Deals', 'sales_agent': 'Vendedor'},
            title="Top 10 Vendedores com Maior Fila de Deals Estagnados",
            color_discrete_map={'zombie_deals': '#C62828', 'total_deals': '#1E88E5'}
        )
        fig_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Tabela de Performance por Gerente / Vendedor
    st.subheader("📊 Auditoria Completa por Vendedor")
    table_agg = filtered_df.groupby(['regional_office', 'manager', 'sales_agent']).agg(
        total=('opportunity_id', 'count'),
        hot=('urgency_status', lambda x: (x == 'Quente / Ativo').sum()),
        zombie=('urgency_status', lambda x: (x == 'Zumbi / Estagnado').sum()),
        avg_score=('lead_score', 'mean'),
        expected_rev=('expected_value', 'sum')
    ).reset_index().sort_values(by='expected_rev', ascending=False)

    st.dataframe(
        table_agg.rename(columns={
            'regional_office': 'Regional',
            'manager': 'Gerente',
            'sales_agent': 'Vendedor',
            'total': 'Total Deals',
            'hot': 'Deals Quentes',
            'zombie': 'Deals Zumbis',
            'avg_score': 'Score Médio',
            'expected_rev': 'Forecast Ponderado ($)'
        }),
        use_container_width=True,
        hide_index=True
    )

def render_backtest_view():
    st.title("📈 Backtest Econômico & Comprovação de Lift")
    st.markdown("""
    Para comprovar à diretoria que o modelo não é uma "teoria de laboratório", foi executado um **Backtesting Cego Fora do Tempo (Out-of-Time)** 
    no último bimestre dos dados históricos (**Novembro e Dezembro de 2017**, com **1.419 negociações reais**).
    
    A pergunta de negócio respondida foi:  
    > *"Se os vendedores tivessem focado rigorosamente nos leads priorizados pelo modelo em vez de seguir o feeling, quanta receita a mais teria sido capturada com a mesma capacidade de trabalho?"*
    """)

    curve_df, bench_df, total_rev = get_backtest_data()

    # Cards de Lift Econômico
    m1, m2, m3 = st.columns(3)
    top20 = bench_df[bench_df['decil'] == 'Top 20% dos Deals'].iloc[0]
    top30 = bench_df[bench_df['decil'] == 'Top 30% dos Deals'].iloc[0]

    with m1:
        st.metric("Receita Histórica no Teste", f"${total_rev:,.2f}", help="Total faturado no bimestre de validação")
    with m2:
        st.metric(
            "Lift no Top 20% da Fila", 
            f"+${top20['lift_dolares']:,.2f}", 
            delta=f"+{top20['lift_percentual']:.1f}% de Eficiência", 
            help="Receita extra capturada no Top 20% vs baseline aleatório"
        )
    with m3:
        st.metric(
            "Lift no Top 30% da Fila", 
            f"+${top30['lift_dolares']:,.2f}", 
            delta=f"+{top30['lift_percentual']:.1f}% de Eficiência", 
            help="Receita extra capturada no Top 30% vs baseline aleatório"
        )

    st.markdown("---")

    # Gráfico de Ganhos Cumulativos (Cumulative Gains Chart)
    st.subheader("🎯 Curva de Ganhos Cumulativos (Lift Financeiro)")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve_df['pct_deals'],
        y=curve_df['pct_revenue_model'],
        mode='lines',
        name='Modelo AI Lead Scorer (Priorização Inteligente)',
        line=dict(color='#1E88E5', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=curve_df['pct_deals'],
        y=curve_df['pct_revenue_random'],
        mode='lines',
        name='Baseline "No Feeling" / Seleção Aleatória',
        line=dict(color='#757575', width=2, dash='dash')
    ))

    # Ponto de destaque no Top 20%
    fig.add_annotation(
        x=20, 
        y=top20['receita_modelo'] / total_rev * 100,
        text=f"Top 20% dos Deals capturam {(top20['receita_modelo']/total_rev)*100:.1f}% da receita!",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#2E7D32",
        ax=50,
        ay=-40
    )

    fig.update_layout(
        title="Curva de Captura de Receita por % da Fila Trabalhada",
        xaxis_title="% de Oportunidades Trabalhadas pelos Vendedores",
        yaxis_title="% da Receita Total Capturada ($)",
        template="plotly_white",
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95),
        margin=dict(l=30, r=30, t=50, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Tabela Detalhada de Decis
    st.subheader("📋 Tabela Comparativa de Impacto nos Decis")
    st.dataframe(
        bench_df.rename(columns={
            'decil': 'Ponto de Corte',
            'n_deals': 'Qtd de Deals',
            'receita_modelo': 'Receita com o Modelo ($)',
            'receita_random': 'Receita no Feeling ($)',
            'lift_dolares': 'LIFT Adicional ($)',
            'lift_percentual': 'Ganho de Eficiência (%)'
        }),
        use_container_width=True,
        hide_index=True
    )

if __name__ == '__main__':
    main()
