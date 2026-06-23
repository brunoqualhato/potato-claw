"""Descoberta e carregamento de skills em pasta/markdown."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    nome: str
    descricao: str
    conteudo: str
    caminho: Path


def _parse_frontmatter(texto: str) -> tuple[dict, str]:
    if not texto.startswith("---"):
        return {}, texto
    partes = texto.split("---", 2)
    if len(partes) < 3:
        return {}, texto
    meta: dict = {}
    for linha in partes[1].strip().splitlines():
        if ":" in linha:
            chave, _, valor = linha.partition(":")
            meta[chave.strip()] = valor.strip()
    return meta, partes[2].lstrip("\n")


def carregar_skills(raiz: Path) -> list[Skill]:
    raiz = Path(raiz)
    if not raiz.is_dir():
        return []
    skills: list[Skill] = []
    for sub in sorted(p for p in raiz.iterdir() if p.is_dir()):
        arq = sub / "SKILL.md"
        if not arq.is_file():
            continue
        meta, corpo = _parse_frontmatter(arq.read_text(encoding="utf-8"))
        skills.append(
            Skill(
                nome=meta.get("name", sub.name),
                descricao=meta.get("description", ""),
                conteudo=corpo,
                caminho=sub,
            )
        )
    return skills


def resumo_skills(skills: list[Skill]) -> str:
    if not skills:
        return ""
    linhas = ["Skills disponiveis:"]
    linhas += [f"- {s.nome}: {s.descricao}" for s in skills]
    return "\n".join(linhas)
