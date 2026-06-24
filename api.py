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

import os
import threading
from asyncio import to_thread
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.agentes.executor import SistemaAgentes

# ══════════════════════════════════════════════════════════════
# AUTENTICAÇÃO
# ══════════════════════════════════════════════════════════════

_API_KEY = os.environ.get("NEURON_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verificar_api_key(api_key: str | None = Security(_api_key_header)):
    """
    Middleware de autenticação via header X-API-Key.
    Se NEURON_API_KEY não está definida no ambiente, autenticação é desabilitada.
    """
    if not _API_KEY:
        # Sem chave configurada → acesso livre (dev local)
        return None

    if not api_key or api_key != _API_KEY:
        raise HTTPException(
            status_code=401,
            detail="API key inválida ou ausente. Envie header X-API-Key.",
        )
    return api_key


# ══════════════════════════════════════════════════════════════
# LIFECYCLE
# ══════════════════════════════════════════════════════════════

_sistema: SistemaAgentes | None = None
_lock = threading.Lock()  # Thread-safety para estado compartilhado


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _sistema
    _sistema = SistemaAgentes()
    yield
    if _sistema:
        _sistema.fechar()


app = FastAPI(
    title="Potato-Claw API",
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
    nivel: int
    fonte: str


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=RespostaChat)
async def chat(req: PerguntaRequest, _: str | None = Depends(verificar_api_key)):
    """Processa pergunta com roteamento automático."""
    if not _sistema:
        raise HTTPException(status_code=503, detail="Sistema não inicializado")

    def executar():
        with _lock:
            if req.nivel:
                _sistema.forcar_nivel(req.nivel)
            resposta = _sistema.executar("generalista", req.pergunta)
            return resposta, _sistema.ultimo_agente, _sistema.ultimo_nivel, _sistema.ultima_fonte

    resposta, agente, nivel, fonte = await to_thread(executar)
    return RespostaChat(resposta=resposta, agente=agente, nivel=nivel, fonte=fonte)


@app.post("/chat/agente", response_model=RespostaChat)
async def chat_com_agente(req: PerguntaAgenteRequest, _: str | None = Depends(verificar_api_key)):
    """Processa pergunta com agente forçado."""
    if not _sistema:
        raise HTTPException(status_code=503, detail="Sistema não inicializado")

    def executar():
        with _lock:
            if req.nivel:
                _sistema.forcar_nivel(req.nivel)
            resposta = _sistema.executar(req.agente, req.pergunta)
            return resposta, _sistema.ultimo_agente, _sistema.ultimo_nivel, _sistema.ultima_fonte

    resposta, agente, nivel, fonte = await to_thread(executar)
    return RespostaChat(resposta=resposta, agente=agente, nivel=nivel, fonte=fonte)


@app.get("/stats")
async def stats(_: str | None = Depends(verificar_api_key)):
    """Métricas de performance."""
    if not _sistema:
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    return _sistema.estatisticas()
