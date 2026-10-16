"""
Copiloto LLM para Vendas e RevOps com Contrato Estrito Pydantic
Frente 2: IA Generativa Conectada ao Motor Matemático Surv-Bayes ERV
"""
import os
import json
import re
import urllib.request
import urllib.error
from typing import Optional
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
    execution_source: Optional[str] = Field(
        default="deterministic:offline",
        description="Origem real da análise executada."
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


def _extract_json_payload(text: str) -> dict:
    """Extrai e faz o parse de JSON de forma resiliente, inclusive se envolto em blocos markdown ou <think>."""
    text = text.strip()
    # Remove blocos de raciocínio de modelos como DeepHermes ou Qwen (<think>...</think>)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Tenta parse direto
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Procura por blocos ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Procura por primeiro '{' e último '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError(f"Não foi possível extrair JSON válido do texto da LLM: {text[:200]}")


def query_llm_copilot(
    deal_dict: dict,
    api_key: Optional[str] = None,
    provider: str = "auto",
    model: Optional[str] = None,
    openrouter_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    temperature: float = 0.2
) -> DealCopilotAnalysis:
    """
    Executa a análise do deal via OpenRouter (com modelos Nous Hermes ou outros)
    ou OpenAI, com fallback determinístico garantido caso nenhuma chave esteja configurada
    ou ocorra erro de conexão/timeout.
    """
    # Resolução de chave e provedor
    resolved_or_key = openrouter_key or os.getenv("OPENROUTER_API_KEY")
    resolved_oa_key = openai_key or os.getenv("OPENAI_API_KEY")

    # Suporte ao parâmetro genérico api_key
    if api_key:
        api_key_clean = api_key.strip()
        if api_key_clean.startswith("sk-or-"):
            resolved_or_key = api_key_clean
            if provider == "auto":
                provider = "openrouter"
        elif api_key_clean.startswith("sk-"):
            resolved_oa_key = api_key_clean
            if provider == "auto":
                provider = "openai"
        else:
            # Caso seja chave genérica sem prefixo específico
            if provider == "openrouter":
                resolved_or_key = api_key_clean
            else:
                resolved_oa_key = api_key_clean

    # Determinar provedor ativo
    active_provider = provider.lower()
    if active_provider == "auto":
        if resolved_or_key:
            active_provider = "openrouter"
        elif resolved_oa_key:
            active_provider = "openai"
        else:
            return generate_deterministic_analysis(deal_dict)

    prompt = f"""Você é o Head de Revenue Operations e Especialista Comercial Sênior no ecossistema NEXUS DRACO de inteligência comercial.
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
2. Seja acionável, direto e estratégico para o vendedor aplicar na manhã de segunda-feira.
3. Gere o script de contato de forma natural, profissional e persuasiva.
4. Responda ESTRITAMENTE em formato JSON com os campos:
   - "resumo_executivo": string (1 a 2 frases com diagnóstico de valor e estado)
   - "diagnostico_risco": string (maior gargalo de tempo ou cadastral)
   - "alavanca_comercial": string (ponto forte ou argumento para destravar)
   - "acao_imediata": string (ação tática recomendada para hoje)
   - "roteiro_contato": string (mensagem pronta para WhatsApp ou Email)
"""

    if active_provider == "openrouter":
        if not resolved_or_key:
            return generate_deterministic_analysis(deal_dict)
        
        target_model = model or "nousresearch/hermes-3-llama-3.1-70b"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resolved_or_key}",
            "HTTP-Referer": "https://nexus-revops.ai",
            "X-Title": "NEXUS RevOps Cockpit - DRACO AI"
        }
        payload = {
            "model": target_model,
            "messages": [
                {
                    "role": "system",
                    "content": "Você é o diretor de RevOps de elite da NEXUS (Copiloto DRACO). Responda EXCLUSIVAMENTE em formato JSON com as 5 chaves solicitadas sem formatação markdown extra."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }
        url = "https://openrouter.ai/api/v1/chat/completions"

    elif active_provider == "openai":
        if not resolved_oa_key:
            return generate_deterministic_analysis(deal_dict)

        target_model = model or "gpt-4o-mini"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resolved_oa_key}"
        }
        schema = DealCopilotAnalysis.model_json_schema()
        payload = {
            "model": target_model,
            "messages": [
                {
                    "role": "system",
                    "content": "Você é um diretor de RevOps de elite que responde estritamente no esquema JSON solicitado."
                },
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
            "temperature": temperature
        }
        url = "https://api.openai.com/v1/chat/completions"
    else:
        return generate_deterministic_analysis(deal_dict)

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
            raw_content = data["choices"][0]["message"]["content"]
            parsed_json = _extract_json_payload(raw_content)
            res = DealCopilotAnalysis(**parsed_json)
            res.execution_source = f"{active_provider}:{target_model}"
            return res
    except Exception as e:
        # Fallback resiliente garantido em caso de timeout, falha de rede ou chave expirada
        fallback_res = generate_deterministic_analysis(deal_dict)
        fallback_res.execution_source = f"fallback:{active_provider}_failed"
        return fallback_res


def query_draco_chat(
    user_message: str,
    context_summary: Optional[str] = None,
    openrouter_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    provider: str = "auto",
    model: Optional[str] = None,
    temperature: float = 0.3
) -> dict:
    """
    Interação conversacional direta com o mascote DRACO (Copiloto RevOps AI).
    Suporta OpenRouter, OpenAI e motor analítico grounded offline.
    """
    resolved_or_key = openrouter_key or os.getenv("OPENROUTER_API_KEY")
    resolved_oa_key = openai_key or os.getenv("OPENAI_API_KEY")

    active_provider = provider.lower()
    if active_provider == "auto":
        if resolved_or_key:
            active_provider = "openrouter"
        elif resolved_oa_key:
            active_provider = "openai"
        else:
            active_provider = "offline"

    system_instruction = (
        "Você é o DRACO, o dragão cibernético e guardião de inteligência de Revenue Operations do cockpit NEXUS. "
        "Você é ultra-técnico, assertivo, cirúrgico e focado em throughput financeiro (Surv-Bayes ERV). "
        "Não use emojis. Use linguagem profissional de diretoria comercial, métricas quantitativas e diretrizes táticas. "
        "Responda em parágrafos claros com tópicos objetivos."
    )

    if context_summary:
        system_instruction += f"\nCONTEXTO ATUAL DO CRM E PIPELINE:\n{context_summary}"

    if active_provider == "openrouter" and resolved_or_key:
        target_model = model or "nousresearch/hermes-3-llama-3.1-70b"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resolved_or_key}",
            "HTTP-Referer": "https://nexus-revops.ai",
            "X-Title": "NEXUS RevOps Cockpit - DRACO AI"
        }
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature
        }
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
                raw_answer = data["choices"][0]["message"]["content"]
                # Sanitiza tags de raciocínio se houver
                clean_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
                return {
                    "answer": clean_answer,
                    "source": f"openrouter:{target_model}",
                    "is_fallback": False
                }
        except Exception:
            pass

    # Fallback Analítico Grounded Offline
    msg_low = user_message.lower()
    if "zumbi" in msg_low or "quarentena" in msg_low or "110" in msg_low:
        ans = (
            "DRACO INTELLIGENCE // CERCA SANITÁRIA:\n\n"
            "Nossa auditoria histórica demonstrou taxa zero de fechamento para negócios acima de 138 dias. "
            "Ao estabelecer a cerca sanitária estrita em 110 dias, expurgamos mais de 1.400 negócios zumbis que drenavam "
            "68% da capacidade comercial. A recomendação tática é descarte imediato do CRM ou transferência para cadência "
            "automática de nutrição de marketing, liberando cada vendedor para focar em sua fila viva de ~19 negócios."
        )
    elif "leakage" in msg_low or "vazamento" in msg_low or "conta" in msg_low:
        ans = (
            "DRACO INTELLIGENCE // AUDITORIA DE DATA LEAKAGE:\n\n"
            "Modelos tradicionais de ML apresentavam AUC artificial de 81% porque 100% dos negócios fechados possuíam o campo "
            "'account' cadastrado, enquanto 68% dos leads novos em prospecção não tinham razão social vinculada. "
            "Implementamos o Princípio da Ortogonalidade Cadastral (AEM = 1.0 para leads sem conta, e 1.15 para clientes com recompra), "
            "garantindo que oportunidades novas não sejam penalizadas por falta de preenchimento de topo de funil."
        )
    elif "backtest" in msg_low or "resultado" in msg_low or "lift" in msg_low or "receita" in msg_low:
        ans = (
            "DRACO INTELLIGENCE // PERFORMANCE ECONÔMICA (Q4 2017):\n\n"
            "Na coorte cega de 2.434 negócios, sob restrição de capacidade de 20,5% (Top 500 deals):\n"
            "- Motor Surv-Bayes ERV capturou $877.572,00 (69,2% da receita total).\n"
            "- Lift de 3,37x (+237%) em relação à escolha aleatória (feeling).\n"
            "- Ganho líquido de +$215.640,00 sobre a ordenação ingênua por maior preço de tabela."
        )
    elif "prioriz" in msg_low or "segunda" in msg_low or "atacar" in msg_low or "foco" in msg_low:
        ans = (
            "DRACO INTELLIGENCE // DIRETRIZ DE ATAQUE COMERCIAL:\n\n"
            "1. Ataque Imediato (Q1 Hot Closers): Negócios de alto ticket (>= $4.000) entre 15 e 45 dias de maturação. "
            "Este é o sweet spot estatístico de fechamento.\n"
            "2. Intervenção Executiva (Q2 Stalling): Negócios caros entre 46 e 90 dias que demandam envolvimento da gerência.\n"
            "3. Giro Rápido (Q3 Quick Wins): Produtos transacionais (< $1.500) para compor meta de volume semanal.\n"
            "4. Prospecção Ativa: Realizar primeiro contato de qualificação com foco em dor e cronograma de decisão."
        )
    else:
        ans = (
            "DRACO INTELLIGENCE // ADVISOR REVOPS:\n\n"
            f"Consulta recebida: '{user_message}'.\n"
            "O motor Surv-Bayes ERV avalia continuamente a probabilidade calibrada de vitória ponderada pelo decaimento "
            "temporal S(tau), modulador sazonal de trimestre e multiplicador cadastral ortogonal. "
            "Recomendo selecionar uma oportunidade específica na tabela ou acionar o diagnóstico detalhado para gerar o roteiro de abordagem personalizado."
        )

    return {
        "answer": ans,
        "source": "deterministic:draco_offline",
        "is_fallback": True
    }
