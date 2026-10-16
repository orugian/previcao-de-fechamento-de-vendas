"""
Web App Backend — Cockpit de Decisão RevOps (Surv-Bayes ERV + Copiloto DRACO AI)
Arquitetura FastAPI com endpoints de alta performance para a interface Cyberpunk.
"""
from pathlib import Path
import os
import io
import sys
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Adicionar raiz ao path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.copilot_llm import query_llm_copilot, query_draco_chat, DealCopilotAnalysis

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_PATH = ARTIFACTS_DIR / "active_scored_deals.csv"
BACKTEST_PATH = ARTIFACTS_DIR / "economic_backtest_summary.csv"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
WEB_DIR = ROOT_DIR / "web"

app = FastAPI(
    title="NEXUS RevOps Intelligence Cockpit API",
    description="Motor de Throughput Comercial Surv-Bayes ERV & Copiloto DRACO AI",
    version="2.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache em memória do DataFrame
_DF_ACTIVE: Optional[pd.DataFrame] = None
_DF_BACKTEST: Optional[pd.DataFrame] = None

def clean_record_for_json(record: dict) -> dict:
    """Higieniza tipos numpy/NaN para compatibilidade estrita com JSON."""
    cleaned = {}
    for k, v in record.items():
        if pd.isna(v) or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            cleaned[k] = None
        elif isinstance(v, (np.floating, float)):
            cleaned[k] = round(float(v), 4)
        elif isinstance(v, (np.integer, int)):
            cleaned[k] = int(v)
        else:
            cleaned[k] = str(v)
    return cleaned


def get_df_active() -> pd.DataFrame:
    global _DF_ACTIVE
    if _DF_ACTIVE is None:
        if not DATA_PATH.exists():
            raise RuntimeError(f"Arquivo {DATA_PATH} não encontrado.")
        df = pd.read_csv(DATA_PATH)
        df["account_display"] = df["account"].fillna("Conta Não Qualificada (Topo de Funil)")
        df["account"] = df["account"].fillna("")
        df["deal_age_days"] = df["deal_age_days"].fillna(0).astype(int)
        df["sales_price"] = df["sales_price"].fillna(0).astype(float)
        df["calibrated_prob_won"] = df["calibrated_prob_won"].fillna(0).astype(float)
        df["erv_usd_per_day"] = df["erv_usd_per_day"].fillna(0).astype(float)
        df["lead_score"] = df["lead_score"].fillna(0).astype(float)
        _DF_ACTIVE = df
    return _DF_ACTIVE

def get_df_backtest() -> pd.DataFrame:
    global _DF_BACKTEST
    if _DF_BACKTEST is None:
        if BACKTEST_PATH.exists():
            _DF_BACKTEST = pd.read_csv(BACKTEST_PATH)
        else:
            _DF_BACKTEST = pd.DataFrame()
    return _DF_BACKTEST


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "engine": "Surv-Bayes ERV",
        "version": "2.4.0",
        "design_system": "Cyberpunk High-Tech (NEXUS Engine)",
        "copilot": "DRACO AI",
        "total_deals_loaded": len(get_df_active())
    }


@app.get("/api/metadata")
def get_metadata():
    df = get_df_active()
    agents = sorted([str(x) for x in df["sales_agent"].dropna().unique().tolist()])
    regional_offices = sorted([str(x) for x in df["regional_office"].dropna().unique().tolist()])
    managers = sorted([str(x) for x in df["manager"].dropna().unique().tolist()])
    products = sorted([str(x) for x in df["product"].dropna().unique().tolist()])
    quadrants = sorted([str(x) for x in df["matrix_quadrant"].dropna().unique().tolist()])
    stages = sorted([str(x) for x in df["deal_stage"].dropna().unique().tolist()])

    total_deals = len(df)
    zombies = int(df["matrix_quadrant"].str.contains("Quarentena", na=False).sum())
    active_deals = total_deals - zombies
    active_mask = ~df["matrix_quadrant"].str.contains("Quarentena", na=False)
    active_val = float(df[active_mask]["sales_price"].sum())
    total_erv = float(df[active_mask]["erv_usd_per_day"].sum())

    return {
        "agents": agents,
        "regional_offices": regional_offices,
        "managers": managers,
        "products": products,
        "quadrants": quadrants,
        "stages": stages,
        "stats_global": {
            "total_deals": total_deals,
            "zombies": zombies,
            "active_deals": active_deals,
            "active_pipeline_val": round(active_val, 2),
            "total_erv_per_day": round(total_erv, 2)
        }
    }


@app.get("/api/deals")
def get_deals(
    agent: Optional[str] = Query(None),
    region: Optional[str] = Query(None),
    manager: Optional[str] = Query(None),
    quadrant: Optional[str] = Query(None),
    stage: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    is_zombie: Optional[bool] = Query(None),
    sort_by: str = Query("lead_score"),
    sort_asc: bool = Query(False),
    limit: int = Query(500, le=2500),
    offset: int = Query(0)
):
    df = get_df_active()
    mask = pd.Series(True, index=df.index)

    if agent and agent != "Todos":
        mask &= df["sales_agent"] == agent
    if region and region != "Todas":
        mask &= df["regional_office"] == region
    if manager and manager != "Todos":
        mask &= df["manager"] == manager
    if quadrant and quadrant != "Todos":
        mask &= df["matrix_quadrant"] == quadrant
    if stage and stage != "Todos":
        mask &= df["deal_stage"] == stage

    if is_zombie is not None:
        if is_zombie:
            mask &= df["matrix_quadrant"].str.contains("Quarentena", na=False)
        else:
            mask &= ~df["matrix_quadrant"].str.contains("Quarentena", na=False)

    if search:
        s = search.lower().strip()
        search_mask = (
            df["opportunity_id"].str.lower().str.contains(s, na=False) |
            df["account_display"].str.lower().str.contains(s, na=False) |
            df["sales_agent"].str.lower().str.contains(s, na=False) |
            df["product"].str.lower().str.contains(s, na=False)
        )
        mask &= search_mask

    filtered = df[mask].copy()

    # Métricas agregadas sobre o filtro corrente
    total_in_filter = len(filtered)
    zombies_in_filter = int(filtered["matrix_quadrant"].str.contains("Quarentena", na=False).sum())
    active_in_filter = total_in_filter - zombies_in_filter
    live_mask = ~filtered["matrix_quadrant"].str.contains("Quarentena", na=False)
    active_val = float(filtered[live_mask]["sales_price"].sum())
    erv_val = float(filtered[live_mask]["erv_usd_per_day"].sum())

    # Ordenação segura
    valid_sort_cols = [
        "lead_score", "erv_usd_per_day", "sales_price",
        "deal_age_days", "calibrated_prob_won", "opportunity_id",
        "sales_agent", "product", "account", "account_display",
        "deal_stage", "matrix_quadrant"
    ]
    actual_sort = sort_by if sort_by in valid_sort_cols else "lead_score"
    filtered = filtered.sort_values(by=actual_sort, ascending=sort_asc)

    # Paginação
    sliced = filtered.iloc[offset:offset + limit]

    cols = [
        "opportunity_id", "sales_agent", "product", "account", "account_display",
        "deal_stage", "deal_age_days", "sales_price", "calibrated_prob_won",
        "erv_usd_per_day", "lead_score", "matrix_quadrant", "priority_tier",
        "action_playbook", "positive_drivers", "risk_bottlenecks",
        "regional_office", "manager", "sector"
    ]

    raw_records = sliced[cols].to_dict(orient="records")
    records = [clean_record_for_json(r) for r in raw_records]

    return {
        "total": total_in_filter,
        "offset": offset,
        "limit": limit,
        "stats": {
            "total_deals": total_in_filter,
            "zombies": zombies_in_filter,
            "active_deals": active_in_filter,
            "active_pipeline_val": round(active_val, 2),
            "total_erv_per_day": round(erv_val, 2)
        },
        "deals": records
    }


@app.get("/api/deal/{opp_id}")
def get_deal_by_id(opp_id: str):
    df = get_df_active()
    row = df[df["opportunity_id"].str.upper() == opp_id.upper()]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Oportunidade {opp_id} não encontrada.")
    return clean_record_for_json(row.iloc[0].to_dict())


@app.get("/api/backtest")
def get_backtest():
    df_b = get_df_backtest()
    if df_b.empty:
        return {"data": [], "summary": "Sem dados de backtest carregados."}
    return {
        "data": df_b.to_dict(orient="records"),
        "key_takeaways": [
            "Captura de 69,2% da receita de Q4 com apenas 20,5% do volume (Top 500 deals).",
            "Lift de 3,37x (+337%) sobre a seleção aleatória (feeling).",
            "Ganho líquido de +$215.640,00 sobre a ordenação ingênua por maior preço de tabela.",
            "Imunidade a data leakage comprovada pelo Princípio da Ortogonalidade Cadastral."
        ]
    }


@app.get("/api/quadrants-summary")
def get_quadrants_summary(
    agent: Optional[str] = Query(None),
    region: Optional[str] = Query(None),
    manager: Optional[str] = Query(None)
):
    df = get_df_active()
    mask = pd.Series(True, index=df.index)
    if agent and agent != "Todos":
        mask &= df["sales_agent"] == agent
    if region and region != "Todas":
        mask &= df["regional_office"] == region
    if manager and manager != "Todos":
        mask &= df["manager"] == manager

    sub = df[mask]
    counts = sub["matrix_quadrant"].value_counts().to_dict()
    sums = sub.groupby("matrix_quadrant")["sales_price"].sum().to_dict()

    items = []
    for q, count in counts.items():
        items.append({
            "quadrant": q,
            "count": int(count),
            "volume_usd": float(sums.get(q, 0.0))
        })

    return {"quadrants": items}


class CopilotRequest(BaseModel):
    opportunity_id: str
    provider: str = "auto"
    openrouter_key: Optional[str] = None
    openai_key: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.2


@app.post("/api/copilot/analyze")
def run_copilot(req: CopilotRequest):
    df = get_df_active()
    row = df[df["opportunity_id"].str.upper() == req.opportunity_id.upper()]
    if row.empty:
        raise HTTPException(status_code=404, detail="Oportunidade não encontrada no CRM.")
    
    deal_dict = row.iloc[0].to_dict()
    analysis = query_llm_copilot(
        deal_dict=deal_dict,
        provider=req.provider,
        model=req.model,
        openrouter_key=req.openrouter_key,
        openai_key=req.openai_key,
        temperature=req.temperature
    )
    source = analysis.execution_source or "deterministic:offline"
    is_fallback = source.startswith("fallback:") or source.startswith("deterministic:")
    return {
        "opportunity_id": req.opportunity_id,
        "analysis": analysis.model_dump(),
        "provider_used": req.provider,
        "model_used": source,
        "is_fallback": is_fallback
    }


class DracoChatRequest(BaseModel):
    message: str
    context_summary: Optional[str] = None
    openrouter_key: Optional[str] = None
    openai_key: Optional[str] = None
    provider: str = "auto"
    model: Optional[str] = None
    temperature: float = 0.3


@app.post("/api/copilot/chat")
def draco_chat(req: DracoChatRequest):
    return query_draco_chat(
        user_message=req.message,
        context_summary=req.context_summary,
        openrouter_key=req.openrouter_key,
        openai_key=req.openai_key,
        provider=req.provider,
        model=req.model,
        temperature=req.temperature
    )


@app.get("/api/export/zombies")
def export_zombies():
    df = get_df_active()
    zombies = df[df["matrix_quadrant"].str.contains("Quarentena", na=False)].copy()
    stream = io.StringIO()
    zombies.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=quarentena_zumbis_crm_nexus.csv"}
    )


@app.get("/api/export/active")
def export_active():
    df = get_df_active()
    live = df[~df["matrix_quadrant"].str.contains("Quarentena", na=False)].copy()
    live = live.sort_values(by="lead_score", ascending=False)
    stream = io.StringIO()
    live.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=fila_ativa_revops_nexus.csv"}
    )


# Servir figuras de diagnóstico estáticas
if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

# Servir frontend estático
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    print("\n[START] NEXUS RevOps Intelligence Cockpit ativo em http://localhost:8000 ...")
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False)
