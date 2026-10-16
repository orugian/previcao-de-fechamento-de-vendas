# Desafio 003 — Lead Scorer & Decision Engine (RevOps)
## Solução Baseada no Motor Surv-Bayes ERV (Expected Revenue Velocity) & Copiloto LLM

Projeto desenvolvido para a resolução do **Desafio 003 (Lead Scorer)** da trilha **AI Master** do [G4 Educação](https://github.com/Gestao-Quatro-Ponto-Zero/ai-master-challenge).

---

## 📌 1. Visão Geral da Problemática & O Diagnóstico Crítico

- **Contexto de Vendas B2B:** 35 vendedores distribuídos em escritórios regionais (Central, East, West) gerenciando ~8.800 oportunidades em 7 produtos com tickets variando de **$55 a $26.768**.
- **A Dor Real de RevOps:** Vendedores gastam tempo excessivo em "deals zumbis" que não vão fechar e deixam esfriar negócios quentes de alto valor. A priorização histórica é feita no "feeling".
- **A Descoberta de Target Leakage (Onde o Julgamento Humano Superou a IA Ingênua):**
  - Modelos tradicionais de classificação binária (*Won* vs. *Lost*) apresentavam um ROC-AUC aparentemente alto (~81%).
  - Nossa auditoria comprovou que **100% dos 6.711 negócios fechados possuem o campo `account` preenchido**, enquanto **68,5% dos negócios abertos em `Engaging` e 67,4% em `Prospecting` não possuem conta vinculada**.
  - No CRM, a razão social formal é vinculada apenas no momento do fechamento. O modelo supervisionado ingênuo aprendeu a detectar se a negociação já estava finalizada, e no mundo real **condenaria 68% das oportunidades ativas à nota zero** simplesmente por falta de preenchimento cadastral de topo de funil!
  - Além disso, variáveis cadastrais isoladas (vendedor, produto, setor, região) possuem AUC entre 0.48 e 0.52 (ruído puro).

---

## 🚀 2. A Solução Implementada: Motor Surv-Bayes ERV

Em vez de forçar um classificador binário sobre ruído e vazamento, implementamos o **Motor Surv-Bayes ERV**:

1. **Princípio da Ortogonalidade Cadastral (Zero Leakage):**  
   A ausência do campo `account` é neutra ($AEM = 1.0$), não prejudicando leads novos. Contas corporativas com histórico prévio de recompra recebem um bônus de relacionamento ($AEM = 1.15$).
2. **Cerca Sanitária (Hard Quarantine Gate):**  
   Negócios com mais de **110 dias em aberto** são automaticamente expurgados da tela diária dos vendedores e movidos para uma aba gerencial de quarentena. Isso elimina mais de **1.400 negócios zumbis** e reduz a fila de cada vendedor para **~19 negociações vivas**.
3. **Curva Contínua de Sobrevivência $S(\tau)$:**  
   Modela a probabilidade real de fechamento em função da idade da negociação:
   - **0 a 15 dias:** maturação inicial (50% a 70%).
   - **15 a 45 dias (*Sweet Spot*):** ápice de conversão.
   - **46 a 110 dias:** decaimento exponencial por atrito.
   - **> 110 dias:** probabilidade zero (limite empírico histórico comprovado de 138 dias).
4. **Modulador de Sazonalidade Trimestral $\Omega_{\text{quarter}}$:**  
   Acelera a priorização de negociações em maturação nos meses de fechamento de trimestre (março, junho, setembro e dezembro), onde o fechamento salta historicamente de ~50% para ~80%.
5. **Classificação na Matriz 4Q:**
   - 🔥 **Q1: Hot Closers (Ataque Imediato):** Alto ticket na janela de ouro (15 a 45 dias).
   - ⚠️ **Q2: High-Value Stalling (Alerta Executivo):** Negócios caros esfriando (46 a 90 dias) para intervenção de liderança.
   - ⚡ **Q3: Quick Wins (Giro Rápido):** Produtos transacionais (< $1.500) para compor meta semanal de volume.
   - 🛑 **Q4: Quarentena / Zumbis:** Estagnados (>110 dias) para descarte ou nutrição automática.
   - 🌱 **Prospecção / Ativação Inicial:** Leads novos sem data de contato (`Prospecting`).

---

## 📈 3. Comprovação Econômica (Backtesting Out-of-Time Q4 2017)

Testado na coorte cega de 2.434 oportunidades abertas em 01/10/2017 sob capacidade restrita de **20,5% da carteira (Top 500 deals)**:

| Estratégia de Priorização | Receita Capturada (USD) | % da Receita do Trimestre | Lift vs. Feeling | Ganho vs. Ordenação por Preço |
|---|---|---|---|---|
| **Motor Surv-Bayes ERV (Nossa Solução)** | **$877.572,00** | **69,2%** | **3,37x** | **+$215.640,00** |
| **Ordenação por Maior Ticket ($)** | $661.932,00 | 52,2% | 2,54x | $0,00 |
| **Ordenação por Probabilidade P(Won)** | $552.180,00 | 43,5% | 2,12x | -$109.752,00 |
| **Escolha no Feeling (Aleatório)** | $260.685,09 | 20,5% | 1,00x | -$401.246,91 |

---

## 🤖 4. Copiloto LLM com Contrato Estrito Pydantic

O módulo em `src/copilot_llm.py` conecta a inteligência generativa aos fatores matemáticos do deal:
- Consome estritamente: ID, ticket, dias em negociação, probabilidade calibrada, throughput ERV, quadrante da matriz e playbooks.
- Retorna um esquema estruturado via `DealCopilotAnalysis`:
  - `resumo_executivo`
  - `diagnostico_risco`
  - `alavanca_comercial`
  - `acao_imediata`
  - `roteiro_contato` (script pronto de WhatsApp/E-mail para o vendedor copiar em 1 clique).
- **Resiliência e Zero Custo:** Possui suporte a chave da OpenAI (`gpt-4o-mini`), com fallback automático para o motor determinístico offline caso executado sem chave.

---

## 📑 5. Governança, Process Log e Revisão Adversarial

Para atender a todos os critérios eliminatórios do G4 Educação, a entrega conta com dois relatórios detalhados:

- 📝 **[Process Log Oficial (Uso Estratégico de IA)](docs/process-log.md):** Registra a decomposição do problema, o conselho de subagentes independentes, a detecção e correção de alucinações e os testes empíricos de leakage.
- 🛡️ **[Relatório de Revisão Adversarial (Stress-Test)](docs/revisao-adversarial.md):** Auditoria cética simulando a banca avaliadora do G4, testando vetores de falha, resiliência do software e conformidade com a rubrica oficial.

---

## 💻 6. Como Executar a Solução em 1 Comando

### Pré-requisitos
Ambiente Python 3.11 instalado com dependências gerenciadas via `uv`:
```powershell
uv pip install -r requirements.txt
```

### 1. Subir a Aplicação Web Interativa (Streamlit)
```powershell
streamlit run app.py
```
Acesse no navegador: `http://localhost:8501`.

### 2. Abrir ou Reexecutar o Jupyter Notebook
```powershell
# Execução direta via Python
python build_and_run_notebook.py

# Ou abrir no Jupyter Lab / VS Code
jupyter notebook notebooks/01_eda_preprocessing_and_modeling.ipynb
```

---

## 📂 7. Estrutura do Repositório

```text
previcao-de-fechamento-de-vendas/
├── README.md                          # Este documento de referência executiva
├── .gitignore                         # Arquivos ignorados (venv, cache, etc.)
├── requirements.txt                   # Dependências do projeto
├── app.py                             # Interface Web completa em Streamlit
├── build_and_run_notebook.py          # Automação de execução do notebook
│
├── src/                               # Módulos Python reutilizáveis
│   ├── data_loader.py                 # Carga e higienização das 4 tabelas
│   ├── features.py                    # Reconstituição point-in-time e priors
│   ├── surv_bayes_engine.py           # Motor Surv-Bayes ERV e Matriz 4Q
│   ├── copilot_llm.py                 # Copiloto LLM com schema Pydantic
│   └── modeling.py                    # Pipeline de calibração e TreeSHAP
│
├── notebooks/                         # Notebook interativo executado
│   └── 01_eda_preprocessing_and_modeling.ipynb
│
├── artifacts/                         # Modelos e bases exportadas
│   ├── active_scored_deals.csv        # 2.089 oportunidades pontuadas
│   ├── q4_backtest_scored.csv         # Resultados do backtest de Q4
│   ├── economic_backtest_summary.csv  # Tabela resumo de receita capturada
│   ├── models/
│   │   └── surv_bayes_engine.joblib   # Modelo de produção serializado
│   └── figures/                       # Gráficos de alta resolução (PNG)
│       ├── 01_pipeline_and_product_economics.png
│       ├── 02_temporal_dynamics_and_survival_curve.png
│       ├── 03_agent_product_specialization.png
│       ├── 06_economic_backtest_revenue_lift.png
│       └── 07_surv_bayes_matrix_4q.png
│
├── audits/                            # Scripts de auditoria matemática de vazamento
│   ├── 01_account_proxy_leakage.py
│   ├── 02_static_feature_signal.py
│   ├── 03_honest_rolling_cv_hparam_grid.py
│   └── 04_honest_backtest_vs_rules.py
│
├── data/                              # Datasets brutos do CRM (CC0)
│   ├── sales_pipeline.csv
│   ├── accounts.csv
│   ├── products.csv
│   ├── sales_teams.csv
│   └── metadata.csv
│
└── docs/                              # Governança e avaliação oficial
    ├── process-log.md                 # Process Log oficial com evidências do uso de IA
    └── revisao-adversarial.md         # Stress-test adversarial da entrega
```
