"""
API HTTP mínima para o sistema multiagente.
Uso: uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
  POST /chat          → Processa pergunta com roteamento automático
  POST /chat/agente   → Força agente específico
  GET  /stats         → Métricas de performance
  GET  /health        → Healthcheck
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agentes.coordenador import rotear
from src.agentes.executor import SistemaAgentes


# ══════════════════════════════════════════════════════════════
# LIFECYCLE
# ══════════════════════════════════════════════════════════════

_sistema: SistemaAgentes | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _sistema
    _sistema = SistemaAgentes()
    yield
    if _sistema:
        _sistema.fechar()


app = FastAPI(
    title="Neuron API",
    description="Sistema Multiagente Local — 3 Níveis de Performance",
    version="1.0.0",
    lifespan=lifespan,
)


# ══════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════


class PerguntaRequest(BaseModel):
    pergunta: str = Field(..., min_length=1, max_length=2000)
    nivel: int | None = Field(None, ge=1, le=3, description="Forçar nível (1/2/3)")


class PerguntaAgenteRequest(PerguntaRequest):
    agente: str = Field(..., description="Nome do agente: generalista, programador, pesquisador, analista")


class RespostaChat(BaseModel):
    resposta: str
    agente: str
    nivel_usado: int | None = None


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=RespostaChat)
async def chat(req: PerguntaRequest):
    """Processa pergunta com roteamento automático."""
    assert _sistema is not None

    if req.nivel:
        _sistema.forcar_nivel(req.nivel)

    nome_agente = rotear(req.pergunta)
    resposta = _sistema.executar(nome_agente, req.pergunta)

    return RespostaChat(resposta=resposta, agente=nome_agente)


@app.post("/chat/agente", response_model=RespostaChat)
async def chat_com_agente(req: PerguntaAgenteRequest):
    """Processa pergunta com agente forçado."""
    assert _sistema is not None

    if req.nivel:
        _sistema.forcar_nivel(req.nivel)

    resposta = _sistema.executar(req.agente, req.pergunta)

    return RespostaChat(resposta=resposta, agente=req.agente)


@app.get("/stats")
async def stats():
    """Métricas de performance."""
    assert _sistema is not None
    return _sistema.estatisticas()
