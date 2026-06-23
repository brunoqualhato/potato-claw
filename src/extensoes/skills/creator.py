"""skill-creator: gera o esqueleto de uma nova skill."""
from __future__ import annotations

from pathlib import Path

_TEMPLATE = """---
name: {nome}
description: {descricao}
---

# {nome}

Descreva aqui as instrucoes da skill: quando usar e como agir.
"""


def criar_skill(raiz: Path, nome: str, descricao: str) -> Path:
    raiz = Path(raiz)
    destino = raiz / nome
    if destino.exists():
        raise FileExistsError(f"Skill '{nome}' ja existe em {destino}")
    destino.mkdir(parents=True)
    (destino / "SKILL.md").write_text(
        _TEMPLATE.format(nome=nome, descricao=descricao), encoding="utf-8"
    )
    return destino
