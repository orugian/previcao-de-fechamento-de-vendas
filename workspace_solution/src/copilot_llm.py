"""
Copiloto LLM para Vendas e RevOps com Contrato Estrito Pydantic
Frente 2: IA Generativa Conectada ao Motor Matemático Surv-Bayes ERV
"""
import os
import json
import urllib.request
import urllib.error
from pydantic import BaseModel, Field


class DealCopilotAnalysis(BaseModel):
    """Contrato estrito de saída do Copiloto LLM para tomada de decisão comercial."""
    resumo_executivo: str = Field(
        ...,
        description="Diagnóstico executivo da negociação em 1 a 2 frases claras."
    )
    diagnostico_risco: str = Field(
        ...,
        description="Identificação precisa do maior gargalo temporal ou cadastral."
    )
    alavanca_comercial: str = Field(
        ...,
        description="Ponto forte ou argumento de valor para destravar a negociação."
    )
    acao_imediata: str = Field(
        ...,
        description="Ação tática recomendada para o vendedor executar na segunda-feira."
    )
    roteiro_contato: str = Field(
        ...,
        description="Script pronto e personalizado de mensagem (WhatsApp/Email) para abordagem."
    )


def generate_deterministic_analysis(deal_dict: dict) -> DealCopilotAnalysis:
    """
    Motor determinístico de alta fidelidade ancorado no Surv-Bayes ERV.
    Utilizado como fallback sem alucinação quando não houver chave de API externa.
    """
    opp_id = deal_dict.get("opportunity_id", "N/A")
    agent = deal_dict.get("sales_agent", "Vendedor")
    product = deal_dict.get("product", "Produto")
    price = float(deal_dict.get("sales_price", 0.0))
    account = deal_dict.get("account")
    account_str = str(account) if pd_not_null(account) else "Conta em Prospecção (Não Qualificada)"
    sector = deal_dict.get("sector", "Geral")
    age = int(float(deal_dict.get("deal_age_days", 0)))
    stage = deal_dict.get("deal_stage", "Engaging")
    quadrant = deal_dict.get("matrix_quadrant", "Geral")
    score = float(deal_dict.get("lead_score", 50.0))
    prob = float(deal_dict.get("calibrated_prob_won", 0.5)) * 100
    erv = float(deal_dict.get("erv_usd_per_day", 0.0))
    playbook = deal_dict.get("action_playbook", "")
    positive_drivers = deal_dict.get("positive_drivers", "")
    risk_bottlenecks = deal_dict.get("risk_bottlenecks", "")

    # 1. Resumo Executivo
    if "Quarentena" in quadrant:
        resumo = (
            f"Oportunidade {opp_id} ({product}) está estagnada há {age} dias no estágio {stage}. "
            f"Com probabilidade de fechamento estatisticamente nula ({prob:.1f}%), o deal encontra-se em Quarentena Sanitária."
        )
    elif "Hot Closers" in quadrant:
        resumo = (
            f"Oportunidade prioritária de alto valor ({product} a ${price:,.0f}) para {account_str}. "
            f"Encontra-se no ponto ótimo de conversão ({age} dias de negociação), com Score {score:.1f} e ERV de ${erv:,.2f}/dia."
        )
    elif "High-Value Stalling" in quadrant:
        resumo = (
            f"Negociação estratégica de alto ticket (${price:,.0f}) que entrou na zona de alerta temporal ({age} dias). "
            f"Exige intervenção executiva antes que ultrapasse o limiar crítico de estagnação."
        )
    elif "Quick Wins" in quadrant:
        resumo = (
            f"Oportunidade transacional de giro rápido ({product} a ${price:,.0f}) com lead fresco ({age} dias). "
            f"Excelente para compor meta de volume da semana com baixo custo de transação."
        )
    else:
        resumo = (
            f"Oportunidade {opp_id} ({product}) com potencial de ${price:,.0f} associada a {account_str}. "
            f"Apresenta probabilidade calibrada de {prob:.1f}% e cadência consultiva ativa."
        )

    # 2. Diagnóstico de Risco
    if age > 110:
        risco = (
            f"Necrose temporal crítica: o negócio já acumula {age} dias sem desfecho. "
            f"No histórico da empresa, nenhum deal fechou após 138 dias; insistir consome horas que deveriam ir para o pipeline novo."
        )
    elif age > 45:
        risco = (
            f"Risco de esfriamento iminente ({age} dias em aberto). "
            f"Negociações entre 46 e 90 dias sofrem queda de mais de 50% na probabilidade de vitória se não houver avanço nos próximos dias."
        )
    elif not pd_not_null(account):
        risco = (
            "Ausência de cadastro formal da entidade corporativa (Conta) no CRM. "
            "Embora não penalizada pelo modelo, indica que o decisor econômico ou o CNPJ ainda não foram formalmente validados."
        )
    else:
        risco = f"Fatores de atenção mapeados: {risk_bottlenecks}."

    # 3. Alavanca Comercial
    if price >= 4000:
        alavanca = (
            f"Ticket Enterprise (${price:,.0f}) com alto retorno esperado por hora investida. "
            f"Afinidade sólida do catálogo e foco em ROI para o setor {sector}."
        )
    elif "Quick Wins" in quadrant:
        alavanca = (
            f"Ciclo de decisão enxuto para o produto {product} (${price:,.0f}). "
            f"Objeções de preço são mínimas; a alavanca é agilidade de implantação imediata."
        )
    else:
        alavanca = f"Drivers identificados pelo motor Surv-Bayes: {positive_drivers}."

    # 4. Ação Imediata
    acao = playbook if playbook else "Contatar decisor para validar cronograma de fechamento."

    # 5. Roteiro de Contato Personalizado
    if "Hot Closers" in quadrant:
        roteiro = (
            f"Olá! Conforme alinhamos em nossas conversas sobre o {product}, preparei as condições finais para iniciarmos "
            f"o projeto sem travas. Você teria 15 minutos hoje às 14h ou 16h para revisarmos a minuta e confirmarmos o início?"
        )
    elif "High-Value Stalling" in quadrant:
        roteiro = (
            f"Olá! Percebi que nossa negociação sobre a implementação do {product} ficou em pausa nas últimas semanas. "
            f"Nosso diretor comercial autorizou uma condição especial de homologação válida apenas para este mês. "
            f"Faz sentido retomarmos ou prefere que eu congele a proposta por enquanto?"
        )
    elif "Quick Wins" in quadrant:
        roteiro = (
            f"Olá! Tudo pronto para liberarmos o seu acesso ao {product}. "
            f"Enviei o link direto da proposta no seu e-mail para darmos sequência ainda hoje. Qualquer dúvida, estou por aqui!"
        )
    elif "Quarentena" in quadrant:
        roteiro = (
            f"Olá! Como não tivemos retorno recente a respeito da proposta do {product}, estou assumindo que o timing atual "
            f"mudou de prioridade na empresa e vou encerrar o acompanhamento deste chamado no CRM. "
            f"Quando voltarem a avaliar essa frente, basta me chamar. Um abraço!"
        )
    else:
        roteiro = (
            f"Olá! Gostaria de entender como evoluiu a avaliação técnica do {product} pela sua equipe. "
            f"Qual seria o próximo passo ideal para apoiá-los nessa decisão?"
        )

    return DealCopilotAnalysis(
        resumo_executivo=resumo,
        diagnostico_risco=risco,
        alavanca_comercial=alavanca,
        acao_imediata=acao,
        roteiro_contato=roteiro
    )


def pd_not_null(val) -> bool:
    """Verifica se o valor é não-nulo de forma segura."""
    if val is None:
        return False
    if str(val).strip().lower() in ["nan", "none", "", "unknown"]:
        return False
    return True


def query_llm_copilot(
    deal_dict: dict,
    api_key: str = None,
    model: str = "gpt-4o-mini"
) -> DealCopilotAnalysis:
    """
    Executa a análise do deal via API OpenAI (se houver chave configurada)
    ou aciona o motor determinístico inteligente como fallback garantido.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return generate_deterministic_analysis(deal_dict)

    # Chamada com Structured Outputs via JSON Schema
    prompt = f"""Você é o Head de Revenue Operations e Especialista Comercial Sênior.
Analise a oportunidade de venda abaixo com base nos dados do CRM e no Motor Surv-Bayes ERV.

DADOS DO DEAL:
- ID Oportunidade: {deal_dict.get('opportunity_id')}
- Vendedor: {deal_dict.get('sales_agent')} ({deal_dict.get('regional_office')})
- Gerente: {deal_dict.get('manager')}
- Produto: {deal_dict.get('product')} (Ticket: ${deal_dict.get('sales_price')})
- Empresa / Conta: {deal_dict.get('account')} (Setor: {deal_dict.get('sector')})
- Tempo em Negociação: {deal_dict.get('deal_age_days')} dias (Estágio: {deal_dict.get('deal_stage')})
- Probabilidade Calibrada de Vitória: {float(deal_dict.get('calibrated_prob_won', 0.5))*100:.1f}%
- Throughput Econômico (ERV): ${deal_dict.get('erv_usd_per_day')}/dia
- Score de Prioridade: {deal_dict.get('lead_score')} / 100 (Faixa: {deal_dict.get('priority_tier')})
- Quadrante na Matriz: {deal_dict.get('matrix_quadrant')}
- Playbook Recomendado: {deal_dict.get('action_playbook')}
- Drivers Positivos: {deal_dict.get('positive_drivers')}
- Gargalos e Riscos: {deal_dict.get('risk_bottlenecks')}

DIRETRIZES:
1. Respeite os dados estritamente: não invente informações corporativas que não constem acima.
2. Seja acionável e direto para o vendedor aplicar na manhã de segunda-feira.
3. Gere o script de contato de forma natural, profissional e persuasiva.
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }

    schema = DealCopilotAnalysis.model_json_schema()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Você é um diretor de RevOps de elite que responde estritamente no esquema JSON solicitado."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "DealCopilotAnalysis",
                "strict": True,
                "schema": schema
            }
        },
        "temperature": 0.2
    }

    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
            raw_content = data["choices"][0]["message"]["content"]
            parsed_json = json.loads(raw_content)
            return DealCopilotAnalysis(**parsed_json)
    except Exception as e:
        # Fallback resiliente em caso de timeout, chave inválida ou erro de rede
        return generate_deterministic_analysis(deal_dict)
