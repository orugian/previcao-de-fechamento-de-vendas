# Guia Consolidado de Entrega e Avaliação — Desafio 003 (Lead Scorer)
**Programa AI Master — G4 Educação**

---

## 1. Visão Geral da Problemática (Desafio 003)

### O Cenário de Negócio
- **Área:** Vendas / Revenue Operations (RevOps).
- **Time comercial:** 35 vendedores em escritórios regionais, gerenciados por sales managers.
- **Pipeline:** ~8.800 oportunidades com ~85 contas clientes e 7 produtos.
- **Dor central:** A priorização atual é feita "no feeling" (intuição). Vendedores perdem tempo em deals improváveis e deixam esfriar oportunidades com alto potencial de conversão.
- **A Demanda da Head de RevOps:**
  > *"Nossos vendedores gastam tempo demais em deals que não vão fechar e deixam oportunidades boas esfriar. Preciso de algo funcional — não um modelo no Jupyter Notebook que ninguém vai usar. Quero uma ferramenta que o vendedor abra, veja o pipeline, e saiba onde focar. Pode ser simples, mas precisa funcionar."*

---

## 2. O Que se Espera Entregar como Produto Final

O Desafio 003 é categorizado como **Build**. O principal deliverable é **software funcional**, não apenas relatórios ou teoria.

### Pilares Obrigatórios da Entrega:

### A. Solução Funcional (Software em Execução)
1. **Precisa rodar de verdade:** Não é aceito mockup estático, wireframe ou apresentação em slides. Deve haver uma aplicação utilizável (ex.: aplicação web em Streamlit, FastAPI + React/HTML, dashboard interativo, etc.).
2. **Dados reais integrados:** A ferramenta deve ingerir e cruzar as 4 tabelas do CRM (`sales_pipeline.csv`, `accounts.csv`, `products.csv`, `sales_teams.csv`).
3. **Lógica de Scoring / Priorização explicável:**
   - Não pode ser apenas ordenar por valor monetário (`close_value`).
   - Deve considerar múltiplos fatores (ex.: tempo no pipeline, velocidade de engajamento, histórico do vendedor, setor/tamanho da conta, histórico de conversão por produto, etc.).
4. **Explainability (O "Porquê" do Score):**
   - O vendedor precisa entender claramente a razão de um lead ter nota 85 vs nota 20 (fatores positivos e de risco).
5. **Usabilidade para o vendedor:**
   - Filtros úteis (por vendedor, por manager, por escritório/região, por estágio).
   - Visão prática de "ação para hoje/segunda-feira de manhã".

### B. Documentação Mínima de Entrega
1. **Setup & Execução:** Instruções passo a passo claras e reproduzíveis para subir a aplicação localmente ou link de deploy público ativo.
2. **Lógica de Negócio e Algoritmo:** Explicação transparente dos critérios de priorização adotados e o racional por trás de cada peso/variável.
3. **Limitações & Próximos Passos:** O que a versão atual não cobre e o que seria necessário para produção enterprise.

### C. Process Log (Obrigatório — Eliminatório)
Sem evidência de uso de IA, a submissão é **desclassificada**. O G4 avalia como você trabalha com IA:
- Quais ferramentas de IA foram usadas e em quais etapas.
- Como o problema foi decomposto antes dos prompts.
- Onde a IA falhou/alucinou e como você corrigiu com raciocínio crítico.
- Qual foi o julgamento humano adicionado que a IA não geraria espontaneamente.
- Histórico de commits, prints ou exports de chat que comprovem a evolução.

---

## 3. Critérios de Avaliação e Baseline do G4

O G4 já executou este desafio utilizando múltiplos modelos (Claude 3.5 Sonnet, GPT-4o, Gemini 1.5 Pro) gerando soluções de referência automáticas (**o baseline**). 

### O que NÃO é avaliado:
- Sintaxe decorada ou linguagem de programação específica.
- Frameworks da moda por pura vaidade técnica.
- Documentos prolixos de 40 páginas sem foco prático.

### O que É avaliado:
| Critério | Peso / Importância | O que os avaliadores buscam |
|---|---|---|
| **Resolução Real do Problema** | Alta | A ferramenta atende ao pedido da Head de RevOps? Ajuda o vendedor a vender mais? |
| **Qualidade da Solução (Build)** | Alta | O software roda limpo? A interface é intuitiva? A priorização faz sentido econômico? |
| **Explainability** | Crítica | A justificativa do score é acionável pelo vendedor ou é uma caixa preta? |
| **Uso Estratégico de IA** | Crítica | Você usou IA como alavanca e aplicou julgamento crítico para superar o baseline? |
| **Clareza de Comunicação** | Média-Alta | O README e as recomendações são claras tanto para um dev quanto para a Head de RevOps? |

---

## 4. Estrutura Recomendada para a Submissão Oficial

Seguindo o padrão do `CONTRIBUTING.md` do repositório original para o Pull Request:

```text
submissions/seu-nome/
├── README.md               # Preenchido com o template oficial (Executive summary, solução, process log)
├── solution/               # Código-fonte da aplicação funcional
│   ├── app.py / src/
│   ├── requirements.txt / package.json
│   └── ...
├── process-log/            # Evidências do uso de IA
│   ├── prompts-and-iterations.md
│   ├── chat-exports/ / screenshots/
│   └── ...
└── docs/                   # Documentação arquitetural ou técnica adicional (se houver)
```

---

## 5. Como Realizar com Sucesso (Roadmap de Execução Recomendado)

1. **Exploração e Sanidade dos Dados (EDA):**
   - Entender taxas de conversão (`Won` vs `Lost`), tempo médio de ciclo de venda por produto e setor, performance por agente e estágio.
2. **Modelagem do Score (Heurística Inteligente + Machine Learning):**
   - Combinar probabilidade de conversão estatística com valor esperado do deal ($) e urgência (tempo estagnado).
   - Gerar fatores explicativos claros (+15 por conta Enterprise com alta conversão, -20 por deal estagnado há mais de 60 dias).
3. **Construção da Aplicação (UI/UX Focada em Vendas):**
   - Interface rápida (ex.: Streamlit ou web dashboard moderna).
   - Visão macro (KPIs do pipeline) + Visão micro (lista de deals priorizados com filtros por vendedor/região).
   - "Call to action" e sugestão de próxima ação para cada lead.
4. **Documentação e Process Log:**
   - Registrar as decisões de prompt, as correções arquiteturais e as lições do processo.
