"""Onboarding leve: resumo do setup atual (perfil, modelos, canais)."""
from __future__ import annotations


def resumo_setup(
    perfil_ativo: str,
    modelos: dict[str, str],
    canais_disponiveis: list[str],
) -> str:
    linhas = [
        "Setup do potato-claw",
        f"Perfil ativo: {perfil_ativo}",
        "Modelos:",
    ]
    for funcao, modelo in modelos.items():
        linhas.append(f"  - {funcao}: {modelo}")
    linhas.append("Canais disponiveis: " + ", ".join(canais_disponiveis))
    linhas.append("Dica: canais sao opt-in. Sem configurar, roda 100% offline no CLI.")
    return "\n".join(linhas)
