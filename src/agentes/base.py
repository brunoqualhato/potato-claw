"""
Classe base para agentes do sistema.
Arquitetura extensível sem overhead de performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass(slots=True)
class ConfigAgente:
    """Configuração imutável de um agente — acesso O(1) sem dict lookup."""

    nome: str
    modelo_rapido: str
    modelo_profundo: str
    system_prompt: str
    palavras_chave: list[str] = field(default_factory=list)
    nivel_preferido: int = 2
    usa_web: bool = False


class Agente:
    """
    Classe base para agentes. Cada agente pode sobrescrever hooks
    do pipeline sem alterar o executor principal.

    Métodos de hook (override opcional):
    - pre_execucao: ajuste de query/nível antes do pipeline
    - pos_execucao: pós-processamento da resposta
    - preparar_contexto_extra: injeta contexto adicional no prompt
    """

    __slots__ = ("config",)

    def __init__(self, config: ConfigAgente):
        self.config = config

    @property
    def nome(self) -> str:
        return self.config.nome

    def pre_execucao(self, pergunta: str, nivel: int) -> tuple[str, int]:
        """Hook pré-execução. Retorna (pergunta_ajustada, nivel_ajustado)."""
        return pergunta, nivel

    def pos_execucao(self, pergunta: str, resposta: str) -> str:
        """Hook pós-execução. Retorna resposta processada."""
        return resposta

    def preparar_contexto_extra(self, pergunta: str) -> str:
        """Retorna contexto extra para injetar no prompt (vazio por padrão)."""
        return ""


class AgenteProgramador(Agente):
    """Agente especializado em código — prioriza nível 3 para tarefas longas."""

    def pre_execucao(self, pergunta: str, nivel: int) -> tuple[str, int]:
        # Perguntas de código com mais de 10 palavras merecem modelo profundo
        if nivel == 2 and len(pergunta.split()) > 10:
            return pergunta, 3
        return pergunta, nivel


class AgentePesquisador(Agente):
    """Agente de pesquisa — sempre tenta web."""

    def pre_execucao(self, pergunta: str, nivel: int) -> tuple[str, int]:
        # Pesquisador funciona bem no nível 2 com dados web
        return pergunta, nivel


class AgenteAnalista(Agente):
    """Agente analista — prefere contexto profundo."""

    def pre_execucao(self, pergunta: str, nivel: int) -> tuple[str, int]:
        if nivel == 2 and len(pergunta.split()) > 12:
            return pergunta, 3
        return pergunta, nivel


class AgenteGeneralista(Agente):
    """Agente genérico — respostas rápidas e econômicas."""
    pass


# ══════════════════════════════════════════════════════════════
# REGISTRO DE AGENTES
# ══════════════════════════════════════════════════════════════

_CLASSES_AGENTE: dict[str, type[Agente]] = {
    "programador": AgenteProgramador,
    "pesquisador": AgentePesquisador,
    "analista": AgenteAnalista,
    "generalista": AgenteGeneralista,
}


def criar_agente(nome: str, config: ConfigAgente) -> Agente:
    """Factory que instancia o agente correto baseado no nome."""
    classe = _CLASSES_AGENTE.get(nome, Agente)
    return classe(config)


def registrar_agente(nome: str, classe: type[Agente]):
    """Permite registrar agentes customizados (plugin system)."""
    _CLASSES_AGENTE[nome] = classe
