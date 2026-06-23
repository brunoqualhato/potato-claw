"""Scheduler leve e testavel: o tempo entra como parametro (sem sleep)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Tarefa:
    nome: str
    intervalo_s: float
    callback: Callable[[], None]
    proximo: float


class Scheduler:
    def __init__(self) -> None:
        self._tarefas: list[Tarefa] = []

    def agendar(
        self,
        nome: str,
        intervalo_s: float,
        callback: Callable[[], None],
        agora: float = 0.0,
    ) -> Tarefa:
        tarefa = Tarefa(nome, intervalo_s, callback, agora + intervalo_s)
        self._tarefas.append(tarefa)
        return tarefa

    def vencidas(self, agora: float) -> list[Tarefa]:
        return [t for t in self._tarefas if agora >= t.proximo]

    def marcar_executada(self, tarefa: Tarefa, agora: float) -> None:
        tarefa.proximo = agora + tarefa.intervalo_s
