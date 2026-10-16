# Process Log Oficial — Como a Inteligência Artificial foi Utilizada
## Desafio 003: Lead Scorer & Decision Engine (RevOps)
**Programa AI Master — G4 Educação**

---

## 📌 1. Visão Geral do Emprego de IA

Este documento registra formalmente o processo iterativo de desenvolvimento orientado por Inteligência Artificial (AI-Assisted Engineering), evidenciando as decisões arquiteturais, a decomposição do problema, os erros e alucinações da IA identificados e corrigidos, as revisões adversariais e o **julgamento crítico humano** que transformou um classificador tradicional ingênuo em uma solução de decisão comercial robusta.

### Ferramentas, Modelos e Papéis Utilizados:

| Ferramenta / Modelo | Função no Projeto | Etapa do Ciclo de Vida |
|---|---|---|
| **DeepSeek Harness (Multi-Agent Engine)** | Orquestração do fluxo de desenvolvimento, execução de ferramentas locais e governança | Todo o ciclo |
| **Modelos de Raciocínio (Claude 3.7 Sonnet / Gemini Flash & Pro)** | Decomposição analítica, formulação matemática, escrita de código e geração de hipóteses | Ideação, EDA, Modelagem e App |
| **Conselho de Subagentes Paralelos** | 3 subagentes autônomos com personas divergentes (RevOps, Estatística Bayesiana, Learning-to-Rank) | Desempate Metodológico e Refinamento |
| **LightGBM & TreeSHAP** | Baseline de classificação supervisionada, calibração isotônica e auditoria de interpretabilidade | Exploração e Diagnóstico de Leakage |
| **Pydantic + OpenAI GPT-4o-mini (com Fallback Grounded)** | Copiloto conversacional estruturado via JSON Schema para roteiros de abordagem | Copiloto de Vendas em Tempo Real |

---

## 🔄 2. O Processo Iterativo em 5 Fases

```
[Fase 1: Ideação & EDA] ──> [Fase 2: O Modelo Inicial e a Ilusão dos ~81% AUC]
                                           │
                                           ▼ (Auditoria Humana de Leakage)
[Fase 4: Motor Surv-Bayes ERV] <── [Fase 3: O Conselho de 3 Subagentes Independentes]
       │
       ▼
[Fase 5: Copiloto LLM + Front-End Streamlit + Revisão Adversarial]
```

### Fase 1: Decomposição Inicial do Problema e Exploração dos Dados (EDA)
- **Como a IA foi utilizada:** Geração de scripts automatizados de checagem relacional entre as 4 tabelas (`sales_pipeline.csv`, `accounts.csv`, `products.csv`, `sales_teams.csv`).
- **Onde o humano interveio:** A IA sugeriu um `LEFT JOIN` ingênuo entre `sales_pipeline` e `products` por `product`. O olhar humano detectou que 1.480 registros (17% do pipeline) tinham o produto `GTXPro` (sem espaço) que não batia com `GTX Pro` (com espaço no catálogo). A correção recuperou mais de $7M de pipeline que teriam se tornado nulos silenciosamente. Também foi corrigido o typo `technolgy` -> `technology` em `accounts.csv`.

---

### Fase 2: O Modelo Inicial e a Descoberta Crítica de Data Leakage (A Falácia dos ~81% AUC)
- **O que foi feito:** Treinamos um pipeline supervisionado tradicional usando LightGBM + Calibração Isotônica em janelas *Out-of-Time*. O modelo reportou métricas impressionantes: **ROC-AUC de ~81,5%** e **Brier Score de 0,139**.
- **A desconfiança humana:** Por que o modelo estava tão confiante se a literatura de vendas B2B mostra que dados cadastrais estáticos raramente discriminam conversão acima de 0,65 de AUC?
- **A Auditoria Matemática:**
  1. Cruzamos os campos do CRM e descobrimos que **100% dos 6.711 negócios fechados (`Won` e `Lost`) possuíam o campo `account` preenchido**, ao passo que **68,5% dos negócios abertos em `Engaging` e 67,4% em `Prospecting` estavam sem conta associada**.
  2. No CRM da empresa, a razão social/conta formal só é vinculada quando o negócio avança para proposta ou encerramento!
  3. Ao testar o poder preditivo de uma regra trivial (`is_account_assigned == 1`), ela atingiu **AUC de 0,769 isoladamente**. Combinada com idade, bateu **0,866** — superando o próprio LightGBM.
  4. **Conclusão:** O modelo não havia aprendido propensão de compra; ele virou um **detector de preenchimento cadastral**. Em produção, essa IA condenaria 68% dos leads novos à nota zero simplesmente porque o vendedor ainda não cadastrou o CNPJ na primeira semana de contato.

---

### Fase 3: O Conselho de 3 Subagentes Independentes
Diante desse impasse metodológico, em vez de aplicar uma regra manual improvisada, lançamos **3 subagentes autônomos em paralelo** para refletir criticamente sobre o problema, cada um sob uma ótica especializada:

1. **Subagente 1 (Perspectiva RevOps & Throughput):**  
   - Propôs o conceito de **Expected Revenue Velocity (ERV)**: velocidade de receita por dia restante de esforço útil.
   - Introduziu o **Princípio da Ortogonalidade Cadastral**: ausência de conta não é punida (baseline neutro = 1.0); contas recorrentes com histórico ganham bônus de expansão (1.15).
   - Defendeu a **Cerca Sanitária (Hard Quarantine Gate)** para expurgar mais de 1.400 negócios zumbis (>110 dias) da tela do vendedor.

2. **Subagente 2 (Perspectiva Estatística Bayesiana & Sobrevivência):**  
   - Demonstrou que a variação de taxas entre vendedores é em grande parte ruído binomial e introduziu **Encolhimento Empírico Bayesiano (James-Stein / Laplace)** para estabilizar produtos raros (como o GTK 500 de $26.768 que só tinha 25 negócios fechados).
   - Mapeou a **Curva Contínua de Sobrevivência $S(\tau)$** com colapso em 138 dias (limite físico histórico).
   - Formalizou o **Modulador de Sazonalidade Trimestral $\Omega_{\text{quarter}}$** ancorado no salto empírico de fechamento nos meses 3, 6, 9 e 12 (de 50% para ~80%).

3. **Subagente 3 (Perspectiva Machine Learning & Learning-to-Rank):**  
   - Defendeu formulação via **LambdaMART / NDCG@10** e otimização por fila individual do vendedor (Problema da Mochila / ROST).
   - Apontou que a métrica de classificação pontual não reflete a decisão semanal do vendedor.

---

### Fase 4: A Síntese no Motor Surv-Bayes ERV
Através de *chain-of-thought* e ponderação analítica, decidimos pela **síntese de alta performance entre o Subagente 1 (RevOps/Throughput) e o Subagente 2 (Estatística Bayesiana/Sobrevivência)**:
- **Desacoplamento em Duas Esteiras:** `Engaging` (Throughput ERV com sobrevivência) e `Prospecting` (Índice de Potencial de Ativação com bônus de exploração).
- **Cerca Sanitária:** Mais de 1.400 negócios mortos foram movidos para a aba de quarentena gerencial. A fila diária de cada vendedor foi despoluída para uma média saudável de **~19 negociações vivas**.
- **Matriz 4Q Prescritiva:** Substituição de scores percentuais frios por quadrantes táticos:
  - 🔥 **Q1: Hot Closers (Ataque Imediato)**
  - ⚠️ **Q2: High-Value Stalling (Alerta Executivo)**
  - ⚡ **Q3: Quick Wins (Giro Rápido)**
  - 🛑 **Q4: Quarentena / Zumbis (Descarte ou Nutrição)**
- **Comprovação Econômica Honesta (Q4 2017):** Captura de **69,2% da receita total** com apenas 20,5% do pipeline — gerando um **Lift de 3,37x** sobre o feeling aleatório e **+$215.640,00** sobre a ordenação simples por preço.

---

### Fase 5: Engenharia do Copiloto LLM com Contrato Estrito Pydantic
- **Problema de IA a resolver:** Modelos generativos tendem a "viajar" e inventar dores de clientes que não constam no CRM.
- **A Solução:** Criamos a classe `DealCopilotAnalysis` em Pydantic. A LLM é forçada a receber estritamente as métricas do motor Surv-Bayes (ticket, dias de vida, probabilidade, throughput, quadrante e playbooks) e responder exclusivamente dentro do JSON Schema.
- **Modo Resiliente Offline:** Desenvolvemos um gerador determinístico inteligente embutido em `src/copilot_llm.py`. A aplicação roda com 100% de precisão mesmo sem internet ou sem chave de API da OpenAI.

---

## 🛠️ 3. Onde a IA Falhou / Alucinou e Como o Humano Corrigiu

| Falha / Alucinação da IA | Como foi Detectada | Correção Aplicada pelo Engenheiro Humano |
|---|---|---|
| **Perda silenciosa de 1.480 deals (17% da base)** | A IA gerou um merge padrão sem checar discrepância de strings. | Identificado que `GTXPro` (sem espaço) falhava contra `GTX Pro`. Foi aplicada a normalização pré-merge, resgatando as oportunidades. |
| **Tentativa de Overengineering em Nuvem** | A IA sugeriu subir EC2, AWS Lambda, S3 e Firebase para hospedar a solução. | Intervenção humana vetou a complexidade: a restrição do desafio é 4-6h e foco na dor de vendas. Optamos pelo monólito funcional e limpo em Streamlit executável em 1 comando. |
| **Data Leakage de ~81% de ROC-AUC** | A IA celebrou o AUC de 81,5% do LightGBM como sucesso estrondoso. | O engenheiro humano desconfiou, isolou os subconjuntos e provou que o modelo usava a coluna `account` como vazamento temporal do encerramento do deal. |
| **Burocratização da Operação** | A IA sugeriu enviar os leads sem conta para uma "tarefa de preenchimento de cadastro". | O humano rejeitou: 68% do funil ativo não pode ser paralisado por burocracia. Implementamos a neutralidade cadastral com bônus apenas para contas recorrentes. |
| **Quebra de Execução por Falta de API Key** | A IA queria colocar chamadas obrigatórias à API externa da OpenAI. | O humano exigiu e implementou o modo *Offline Grounded* com fallback determinístico local, garantindo avaliação fluida sem custos para a banca. |

---

## 💡 4. O que o Julgamento Humano Adicionou que a IA Sozinha Jamais Faria

1. **Raciocínio Crítico Anti-Métrica de Vaidade:** A IA aceitou o AUC de 81% sem pestanejar. Foi o ceticismo humano de negócios que investigou a distribuição de nulos e desmontou a ilusão estatística.
2. **Entendimento da Rotina da Segunda-Feira:** A IA sugeria dashboards com dezenas de filtros e matrizes de correlação. O humano impôs a visão de **Cockpit Operacional**, onde o vendedor gasta menos de 2 minutos para saber os 3 clientes prioritários do dia e copiar o roteiro de contato no WhatsApp.
3. **Criação da Cerca Sanitária de Zumbis:** Ao cruzar o tempo máximo de ciclo histórico (138 dias) com os 1.420 negócios parados há mais de 110 dias, o humano percebeu que a maior alavanca de produtividade era **limpar o lixo do CRM** para que o vendedor recuperasse seu foco.
4. **Decisão Dialética via Subagentes:** Em vez de usar IA passivamente ("escreva um código"), o humano usou IA de forma adversarial e estratégica, confrontando diferentes escolas de pensamento antes de implementar a arquitetura final.
