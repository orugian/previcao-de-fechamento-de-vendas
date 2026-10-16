# Process Log — Como a Inteligência Artificial foi Utilizada no Desafio 003
**Programa AI Master — G4 Educação**

---

## 1. Ferramentas e Ambientes de IA Utilizados

| Ferramenta / Modelo | Função no Projeto | Etapa do Workflow |
|---|---|---|
| **DeepSeek Harness / Claude 3.7 / Gemini 1.5** | Orquestração de raciocínio, conselho de personas (Conselho de LLMs) e revisão adversarial | Ideação, Arquitetura e Validação |
| **LightGBM & TreeSHAP** | Modelagem preditiva estatística e motor determinístico de explicabilidade local | Modelagem de Dados & Scoring |
| **OpenAI GPT-4o-mini (via API)** | Copiloto conversacional de abordagem tática de vendas com contrato estrito Pydantic | Copiloto em Tempo Real |
| **Motor Heurístico Resiliente (Local)** | Fallback determinístico 100% autônomo baseado em vetores de SHAP para uso sem API Key | Confiabilidade & Offline Mode |

---

## 2. Decomposição do Problema e Workflow Passo a Passo

### Etapa 1: Debate Dialético via Conselho de LLMs
Em vez de aceitar o primeiro prompt ou código gerado por um modelo único, foi instalado um **Conselho de LLMs** com 4 papéis divergentes:
1. **Dr. Aris (Lead Quant):** Exigiu split temporal estrito, calibração isotônica de probabilidades e métricas de Brier Score.
2. **Sofia (Head of RevOps):** Cobrou simplicidade radical para o vendedor na segunda-feira de manhã e roteiros de contato direto.
3. **Marcus (Staff Architect):** Vetou overengineering de cloud distribuída (EC2/Lambda/Firebase) em favor de um monólito ágil em Streamlit.
4. **Elena (Persona G4 Evaluator):** Garantiu alinhamento aos critérios de avaliação da banca do G4 (superar o baseline automático).

O consenso unificado resultou no conceito **"Pragmatic Quant Cockpit"**, validado por uma **Revisão Adversarial (Stress-Test)** antes de qualquer linha de código.

### Etapa 2: Auditoria Crítica dos Dados (EDA)
Ao carregar as 4 bases relacionais, a IA executou uma varredura de integridade referencial que revelou anomalias críticas no dataset.

### Etapa 3: Feature Engineering Temporal e Prevenção de Leakage
- Reconstituição temporal com ponto de corte (*Out-Of-Time*) em `2017-11-01`.
- Treino estrito em deals fechados antes de novembro/2017 (5.292 negociações).
- Teste cego no bimestre final (1.419 negociações).
- Cálculo de taxas de conversão históricas de vendedores, produtos e setores com suavização Bayesiana (Laplace shrinkage) calculadas exclusivamente sobre o passado.

### Etapa 4: Treinamento, Calibração e Motor TreeSHAP
- Comparação rigorosa: Naive Baseline vs. Regressão Logística vs. LightGBM Base vs. LightGBM Calibrado (Isotonic Regression).
- Implementação do cálculo de contribuição de features via `shap.TreeExplainer`.

### Etapa 5: Engenharia do Copiloto LLM com Contrato Pydantic
- Definição de esquema estrito de saída (`risk_diagnosis`, `closing_hook`, `action_today`, `outreach_template`).
- Proibição de alucinações através de amarras que forçam a LLM a traduzir exclusivamente os vetores de SHAP e metadados reais da conta.
- Desenvolvimento de um motor heurístico com o mesmo contrato para fallback automático caso falte chave de API.

### Etapa 6: Interface Web Interativa (Streamlit) & Backtest de Negócio
- Implementação das visões: Vendedor ("Minha Segunda-feira"), Gestor & RevOps (Saúde do Pipeline) e Backtesting (Curva de Ganhos Cumulativos e Lift).

---

## 3. Onde a IA Errou / Falhou e Como o Humano Corrigiu

| Erro / Alucinação / Falha da IA | Como foi detectado | Correção Aplicada pelo Engenheiro Humano |
|---|---|---|
| **Perda silenciosa de 1.480 deals (17% da base)** | A IA sugeriu um merge simples `sales_pipeline.merge(products, on='product')`. | A auditoria revelou que `GTXPro` (sem espaço no pipeline) não encontrava `GTX Pro` (com espaço no catálogo). Foi implementado o tratamento `replace({'GTXPro': 'GTX Pro'})`, recuperando todas as 1.480 oportunidades órfãs. |
| **Tentativa de Overengineering em Cloud** | Sugestão inicial de arquitetura com EC2, AWS Lambda, S3 e Firebase. | Intervenção humana vetou a complexidade excessiva para o time budget de 4-6h, optando pelo monólito moderno em Streamlit com inicialização em 1 comando. |
| **Risco de Tela Quebrada por Falta de API Key** | Scripts de LLM convencionais quebram com `AuthenticationError` se o usuário não possuir chave OpenAI. | O humano desenhou uma arquitetura resiliente com **Fallback Heurístico Baseado em SHAP**, garantindo que o avaliador consiga testar 100% da solução sem precisar pagar ou inserir tokens de API. |
| **Vazamento de Target em Variáveis Temporais** | A IA propôs usar a duração total do deal como feature de treino para prever se ele fecharia. | O humano corrigiu: no momento em que o vendedor atua no deal, a data final de fechamento é desconhecida. A feature deve representar o tempo transcorrido até o instante de corte (`as_of_date`), combinada com a penalidade de decaimento de sobrevivência. |

---

## 4. O que o Julgamento Humano Adicionou que a IA Sozinha Não Faria

1. **A Descoberta dos "Deals Zumbis" (81.2% da fila ativa):**
   A IA sugeriu ordenar os leads apenas pela probabilidade matemática do modelo. O olhar humano cruzou o histórico de fechamento (máximo histórico de 138 dias) com a idade dos deals em aberto e descobriu que **1.291 oportunidades em Engaging já tinham mais de 138 dias**. Criou-se a categoria de "Deal Zumbi" e a função de penalidade por decaimento temporal, resolvendo a dor exata da Head de RevOps.

2. **A Métrica de Expected Value com Balanceamento de Tickets:**
   A IA tenderia a ranquear ou por probabilidade pura (favorecendo produtos baratos de $55) ou por valor bruto (favorecendo o produto de $26k). O humano formulou o score econômico que pondera a probabilidade calibrada com o log do preço e a curva de urgência.

3. **Comprovação de Negócio com Lift Financeiro no Backtest:**
   Em vez de apresentar apenas métricas abstratas de ciência de dados (F1, AUC, Loss), o humano construiu o simulador financeiro comprovando que focar no **Top 20% da fila gera +$596.848,80 (+144.1%) de receita** em relação ao feeling da equipe comercial.
