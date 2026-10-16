import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import requests

class LeadActionPlan(BaseModel):
    opportunity_id: str
    account: str
    lead_score: float
    urgency_status: str
    risk_diagnosis: str = Field(description="Diagnóstico conciso do principal risco ou gargalo do deal")
    closing_hook: str = Field(description="A principal alavanca de valor para fechar a oportunidade")
    action_today: str = Field(description="Ação tática imediata recomendada para o vendedor executar hoje")
    outreach_template: str = Field(description="Sugestão de mensagem curta de contato (WhatsApp ou e-mail) para o decisor")
    mode: str = Field(default="LLM Generative", description="Modo de geração: LLM Generative ou Heuristic Fallback")

def generate_heuristic_plan(lead_data: Dict[str, Any]) -> LeadActionPlan:
    """
    Fallback determinístico e inteligente baseado estritamente nos vetores de SHAP e regras de negócio.
    Garante que a ferramenta seja 100% funcional mesmo sem API Key configurada.
    """
    opp_id = lead_data['opportunity_id']
    account = lead_data['account']
    product = lead_data['product']
    score = lead_data['lead_score']
    prob = lead_data['calibrated_probability']
    days = lead_data['days_in_pipeline']
    urgency = lead_data['urgency_status']
    top_pos = lead_data.get('top_positive_factors', [])
    top_neg = lead_data.get('top_negative_factors', [])

    # Diagnóstico de risco
    if days > 138:
        risk = f"Deal estagnado há {days} dias (superior ao ciclo máximo histórico de 138 dias). Alto risco de desengajamento definitivo."
    elif days > 60:
        risk = f"Deal esfriando: {days} dias em andamento. Está ultrapassando a média histórica de 55 dias e precisa de reativação rápida."
    elif top_neg:
        risk = f"Principal ponto de atenção matemático: {top_neg[0]['feature']} com peso restritivo na probabilidade."
    else:
        risk = "Oportunidade recente com boa tração inicial e baixo atrito identificado."

    # Alavanca comercial
    if top_pos:
        main_lever = f"Forte tração impulsionada por: {top_pos[0]['feature']} (probabilidade estimada em {prob}%)."
    else:
        main_lever = f"Ticket atrativo ({product} a ${lead_data.get('sales_price', 0):,.2f}) com alinhamento comercial."

    # Ação tática para hoje
    if days > 138:
        action = f"Enviar mensagem de 'breakup' / desqualificação formal. Se o cliente não responder em 48h, mover para Lost para liberar capacidade."
    elif days > 60:
        action = f"Ligar hoje para o sponsor e propor uma reunião de alinhamento com oferta por tempo limitado para a série {product}."
    else:
        action = f"Agendar demonstração técnica de fechamento ou enviar minuta de proposta com foco na dor do setor {lead_data.get('sector', 'geral')}."

    # Template de contato
    client_name = account if account != 'Conta Não Identificada' else 'Parceiro'
    outreach = (
        f"Olá, time da {client_name}! Acompanhando a implementação do {product}, "
        f"gostaria de alinhar os últimos detalhes para desbloquearmos o início do projeto nesta semana. "
        f"Você teria 15 minutos hoje às 15h?"
    )

    return LeadActionPlan(
        opportunity_id=opp_id,
        account=account,
        lead_score=score,
        urgency_status=urgency,
        risk_diagnosis=risk,
        closing_hook=main_lever,
        action_today=action,
        outreach_template=outreach,
        mode="Heuristic Engine (SHAP Grounded)"
    )

def generate_llm_action_plan(lead_data: Dict[str, Any], api_key: Optional[str] = None) -> LeadActionPlan:
    """
    Gera o plano tático acionável via LLM com ancoragem matemática rígida nos fatores SHAP.
    Se a API Key não for informada ou a requisição falhar, faz fallback transparente para o motor heurístico.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key or key.strip() == "":
        return generate_heuristic_plan(lead_data)

    system_prompt = """Você é o Head de RevOps & Especialista Comercial do G4 Educação.
Sua missão é capacitar o vendedor com um diagnóstico acionável e direto sobre uma oportunidade comercial.

REGRAS CRÍTICAS DE GROUNDING (ZERO ALUCINAÇÃO):
1. Ancore seu diagnóstico ESTRITAMENTE nos fatores estatísticos e SHAP fornecidos.
2. Não invente nomes de pessoas ou fatos que não constam nos dados.
3. Seja conciso, pragmático e direto — o vendedor vai ler isso em 20 segundos antes de ligar para o cliente.
4. Responda ESTRITAMENTE em formato JSON compatível com o esquema solicitado.
"""

    pos_str = ", ".join([f"{f['feature']} (+{f['impact']:.2f})" for f in lead_data.get('top_positive_factors', [])])
    neg_str = ", ".join([f"{f['feature']} ({f['impact']:.2f})" for f in lead_data.get('top_negative_factors', [])])

    user_prompt = f"""DADOS DO DEAL NO CRM:
- ID: {lead_data['opportunity_id']}
- Vendedor: {lead_data['sales_agent']} (Regional: {lead_data['regional_office']})
- Cliente: {lead_data['account']} (Setor: {lead_data.get('sector', 'N/A')})
- Produto: {lead_data['product']} (Ticket: ${lead_data.get('sales_price', 0):,.2f})
- Dias no Pipeline: {lead_data['days_in_pipeline']} dias (Status: {lead_data['urgency_status']})
- Probabilidade Estimada: {lead_data['calibrated_probability']}% | Score: {lead_data['lead_score']}/100
- Fatores SHAP Positivos (Alavancas): {pos_str or 'N/A'}
- Fatores SHAP Negativos (Gargalos): {neg_str or 'N/A'}

Retorne um JSON com os campos:
{{
  "risk_diagnosis": "Diagnóstico claro em 1-2 frases do maior risco real",
  "closing_hook": "A principal alavanca comercial para conduzir o fechamento",
  "action_today": "1 ação tática ultra-específica para hoje",
  "outreach_template": "Roteiro de 2-3 frases de mensagem via WhatsApp ou e-mail"
}}
"""

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            parsed = json.loads(content)
            return LeadActionPlan(
                opportunity_id=lead_data['opportunity_id'],
                account=lead_data['account'],
                lead_score=lead_data['lead_score'],
                urgency_status=lead_data['urgency_status'],
                risk_diagnosis=parsed.get('risk_diagnosis', ''),
                closing_hook=parsed.get('closing_hook', ''),
                action_today=parsed.get('action_today', ''),
                outreach_template=parsed.get('outreach_template', ''),
                mode="LLM Copilot (GPT-4o-mini)"
            )
        else:
            return generate_heuristic_plan(lead_data)
    except Exception:
        return generate_heuristic_plan(lead_data)

if __name__ == '__main__':
    # Teste unitário do gerador com dados simulados
    sample_lead = {
        'opportunity_id': 'HN6TZPAF',
        'sales_agent': 'Wilburn Farren',
        'regional_office': 'West',
        'account': 'Acme Corporation',
        'product': 'GTX Plus Pro',
        'sales_price': 5482.0,
        'days_in_pipeline': 61,
        'urgency_status': 'Atenção / Esfriando',
        'calibrated_probability': 74.3,
        'lead_score': 82.1,
        'sector': 'technology',
        'top_positive_factors': [{'feature': 'Ticket / Valor do Produto', 'impact': 0.42}],
        'top_negative_factors': [{'feature': 'Velocidade / Tempo no Pipeline', 'impact': -0.31}]
    }
    plan = generate_llm_action_plan(sample_lead)
    print("=== PLANO DE AÇÃO GERADO PELO COPILOTO ===")
    print(plan.model_dump_json(indent=2))
