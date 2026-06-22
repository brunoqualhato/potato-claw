"""
Analisador de Intenção via LLM.

Em vez de comparar palavras-chave, a LLM coordenadora analisa
semanticamente a pergunta e decide:
  - Qual agente usar
  - Se precisa buscar na web
  - Se pode resolver com ferramenta local (cálculo, hora, arquivo)
  - Parâmetros extraídos (local, expressão, etc)

Uma única chamada de ~100ms substitui toda a lógica de keyword matching.
"""

from __future__ import annotations

import json

import ollama

from src.core.config import COORDENADOR_MODELO


# ══════════════════════════════════════════════════════════════
# PROMPT DO ANALISADOR
# ══════════════════════════════════════════════════════════════

_PROMPT_ANALISAR = """Você é um classificador de intenções. Analise a pergunta e retorne APENAS um JSON válido.

Campos obrigatórios:
- "agente": um de [generalista, programador, pesquisador, analista]
- "precisa_web": true/false (precisa de dados atuais da internet?)
- "ferramenta": null ou um de [calculo, data_hora, saudacao, arquivo, comando]
- "parametros": objeto com dados extraídos (pode ser vazio {})

Regras para "precisa_web":
- true se a resposta CORRETA depende de informações que mudam com o tempo (clima, cotação, notícias, versões atuais, eventos)
- true se pede dados de um local/país específico que não é conhecimento geral
- true se pede documentação, tutorial ou referência de tecnologia
- false se é conhecimento geral estável (o que é python, como funciona recursão)
- false se pode ser resolvido com ferramenta local

Regras para "ferramenta":
- "calculo" se é expressão matemática. parametros: {"expressao": "..."}
- "data_hora" se pergunta hora/data/dia. parametros: {"local": "nome_do_local"} ou {} se local
- "saudacao" se é apenas oi/hello/bom dia. parametros: {}
- "arquivo" se pede ler/criar/listar arquivo. parametros: {"acao": "ler|criar|listar", "caminho": "..."}
- "comando" se pede executar algo. parametros: {"comando": "..."}
- null se nenhuma ferramenta se aplica (precisa de LLM)

Exemplos:
Pergunta: "que horas são no Congo"
{"agente":"generalista","precisa_web":false,"ferramenta":"data_hora","parametros":{"local":"congo"}}

Pergunta: "como está o tempo em São Paulo"
{"agente":"pesquisador","precisa_web":true,"ferramenta":null,"parametros":{}}

Pergunta: "quanto é 15% de 300"
{"agente":"generalista","precisa_web":false,"ferramenta":"calculo","parametros":{"expressao":"300*0.15"}}

Pergunta: "crie uma API rest em python com autenticação JWT"
{"agente":"programador","precisa_web":false,"ferramenta":null,"parametros":{}}

Pergunta: "qual a última versão do node.js"
{"agente":"pesquisador","precisa_web":true,"ferramenta":null,"parametros":{}}

Pergunta: "analise a viabilidade de um SaaS de voip"
{"agente":"analista","precisa_web":false,"ferramenta":null,"parametros":{}}

Pergunta: "oi"
{"agente":"generalista","precisa_web":false,"ferramenta":"saudacao","parametros":{}}

Responda APENAS o JSON, sem explicação."""


# ══════════════════════════════════════════════════════════════
# TIPOS
# ══════════════════════════════════════════════════════════════

class IntencaoAnalisada:
    """Resultado da análise de intenção."""

    __slots__ = ("agente", "precisa_web", "ferramenta", "parametros", "_raw")

    def __init__(
        self,
        agente: str = "generalista",
        precisa_web: bool = False,
        ferramenta: str | None = None,
        parametros: dict | None = None,
        raw: str = "",
    ):
        self.agente = agente
        self.precisa_web = precisa_web
        self.ferramenta = ferramenta
        self.parametros = parametros or {}
        self._raw = raw

    def __repr__(self) -> str:
        return (
            f"Intencao(agente={self.agente}, web={self.precisa_web}, "
            f"ferramenta={self.ferramenta}, params={self.parametros})"
        )


# ══════════════════════════════════════════════════════════════
# ANALISADOR
# ══════════════════════════════════════════════════════════════

_AGENTES_VALIDOS = {"generalista", "programador", "pesquisador", "analista"}
_FERRAMENTAS_VALIDAS = {"calculo", "data_hora", "saudacao", "arquivo", "comando", None}


def analisar_intencao(pergunta: str, modelo: str = COORDENADOR_MODELO) -> IntencaoAnalisada:
    """
    Usa a LLM coordenadora para analisar semanticamente a pergunta.
    Retorna IntencaoAnalisada com agente, web, ferramenta e parâmetros.

    Performance: ~80-150ms com modelo 1.2B, 20 tokens de saída.
    """
    try:
        response = ollama.chat(
            model=modelo,
            messages=[
                {"role": "system", "content": _PROMPT_ANALISAR},
                {"role": "user", "content": pergunta},
            ],
            options={
                "temperature": 0.05,
                "num_predict": 120,
            },
        )
        raw = response["message"]["content"].strip()
        return _parse_resposta(raw)

    except Exception:
        # Fallback conservador: manda pro generalista com web
        return IntencaoAnalisada(agente="generalista", precisa_web=True)


def _parse_resposta(raw: str) -> IntencaoAnalisada:
    """Parseia o JSON da LLM com tolerância a erros."""
    # Tenta extrair JSON balanceado (suporta 1 nível de aninhamento)
    # Procura a primeira { e encontra a } que fecha balanceadamente
    inicio = raw.find("{")
    if inicio == -1:
        return IntencaoAnalisada(raw=raw)

    depth = 0
    fim = -1
    for i in range(inicio, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                fim = i + 1
                break

    if fim == -1:
        return IntencaoAnalisada(raw=raw)

    json_str = raw[inicio:fim]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return IntencaoAnalisada(raw=raw)

    # Valida campos
    agente = data.get("agente", "generalista")
    if agente not in _AGENTES_VALIDOS:
        agente = "generalista"

    precisa_web = bool(data.get("precisa_web", False))

    ferramenta = data.get("ferramenta")
    if ferramenta not in _FERRAMENTAS_VALIDAS:
        ferramenta = None

    parametros = data.get("parametros", {})
    if not isinstance(parametros, dict):
        parametros = {}

    return IntencaoAnalisada(
        agente=agente,
        precisa_web=precisa_web,
        ferramenta=ferramenta,
        parametros=parametros,
        raw=raw,
    )
