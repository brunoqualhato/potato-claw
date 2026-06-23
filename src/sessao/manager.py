"""Sessao por (canal, usuario): historico isolado com trimming."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Mensagem:
    role: str
    conteudo: str


class Session:
    def __init__(self, chave: str, limite: int = 20) -> None:
        self.chave = chave
        self.limite = limite
        self.mensagens: list[Mensagem] = []

    def adicionar(self, role: str, conteudo: str) -> None:
        self.mensagens.append(Mensagem(role, conteudo))
        if len(self.mensagens) > self.limite:
            self.mensagens = self.mensagens[-self.limite:]

    def historico(self) -> list[dict]:
        return [{"role": m.role, "content": m.conteudo} for m in self.mensagens]


class SessionManager:
    def __init__(self, limite: int = 20) -> None:
        self._sessoes: dict[str, Session] = {}
        self._limite = limite

    def obter(self, canal: str, usuario: str) -> Session:
        chave = f"{canal}:{usuario}"
        if chave not in self._sessoes:
            self._sessoes[chave] = Session(chave, self._limite)
        return self._sessoes[chave]
