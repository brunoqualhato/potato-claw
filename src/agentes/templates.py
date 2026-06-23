"""
Template Library para o Agent Loop.

Modelos pequenos (1.2B) são fracos em planejamento do zero mas bons em
completar padrões. Esta biblioteca fornece esqueletos pré-construídos
que reduzem a carga cognitiva na LLM — ela só precisa preencher conteúdo.

Uso:
    template = selecionar_template("API REST com FastAPI e banco de dados")
    if template:
        # Usa o plano do template em vez de pedir à LLM para planejar
        sessao.plano = template.gerar_plano()
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(slots=True)
class TemplateStep:
    descricao: str
    arquivo: str
    dependencias: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TemplateProjeto:
    """Template pré-definido para um tipo de projeto."""
    nome: str
    palavras_chave: list[str]
    stack: str
    steps: list[TemplateStep]
    esqueletos: dict[str, str] = field(default_factory=dict)  # arquivo → esqueleto parcial

    def gerar_plano(self) -> list[TemplateStep]:
        return self.steps.copy()


# ══════════════════════════════════════════════════════════════
# TEMPLATES PRÉ-DEFINIDOS
# ══════════════════════════════════════════════════════════════

TEMPLATES: list[TemplateProjeto] = [
    TemplateProjeto(
        nome="cli_python",
        palavras_chave=[
            "cli", "terminal", "menu", "interativo", "lista", "contato", "tarefa",
            "todo", "cadastro", "gerenciador", "calculadora", "conversor", "jogo",
            "quiz", "agenda", "inventário", "estoque", "registro", "controle",
            "sistema", "ferramenta", "utilitário", "app", "aplicativo",
            "galeria", "fotos", "foto", "álbum", "imagem", "imagens",
            "notas", "nota", "diário", "receita", "receitas", "biblioteca",
            "catálogo", "catalogo", "organizador", "player", "música",
        ],
        stack="Python",
        steps=[
            TemplateStep("Arquivo de dependências", "requirements.txt"),
            TemplateStep("Modelos de dados e lógica de negócio", "models.py"),
            TemplateStep("Persistência de dados (JSON/SQLite)", "storage.py"),
            TemplateStep("Ponto de entrada CLI interativo com menu", "main.py", ["models.py", "storage.py"]),
            TemplateStep("Documentação com instruções de execução", "README.md"),
        ],
        esqueletos={
            "main.py": (
                "#!/usr/bin/env python3\n"
                "# Ponto de entrada — CLI interativa com menu\n"
                "# Imports dos módulos do projeto\n"
                "# Função menu() com loop while True e opções numeradas\n"
                "# if __name__ == '__main__': menu()\n"
            ),
        },
    ),
    TemplateProjeto(
        nome="api_fastapi",
        palavras_chave=["api", "rest", "fastapi", "endpoint", "servidor", "http", "backend", "jwt", "auth"],
        stack="Python + FastAPI",
        steps=[
            TemplateStep("Arquivo de dependências", "requirements.txt"),
            TemplateStep("Modelos Pydantic e schemas", "models.py"),
            TemplateStep("Lógica de negócio e persistência", "services.py"),
            TemplateStep("Rotas/endpoints da API", "routes.py", ["models.py", "services.py"]),
            TemplateStep("Servidor FastAPI com startup", "app.py", ["routes.py"]),
            TemplateStep("Documentação com instruções e exemplos curl", "README.md"),
        ],
        esqueletos={
            "app.py": (
                "#!/usr/bin/env python3\n"
                "from fastapi import FastAPI\n"
                "from routes import router\n"
                "app = FastAPI(title='...')\n"
                "app.include_router(router)\n"
                "# uvicorn app:app --reload\n"
            ),
        },
    ),
    TemplateProjeto(
        nome="api_flask",
        palavras_chave=["flask", "web", "servidor", "site"],
        stack="Python + Flask",
        steps=[
            TemplateStep("Arquivo de dependências", "requirements.txt"),
            TemplateStep("Modelos de dados", "models.py"),
            TemplateStep("Rotas e views", "routes.py", ["models.py"]),
            TemplateStep("Aplicação Flask principal", "app.py", ["routes.py"]),
            TemplateStep("Documentação com instruções", "README.md"),
        ],
    ),
    TemplateProjeto(
        nome="script_automacao",
        palavras_chave=["script", "automação", "automatizar", "bot", "scraper", "cron", "etl"],
        stack="Python",
        steps=[
            TemplateStep("Arquivo de dependências", "requirements.txt"),
            TemplateStep("Utilitários e helpers", "utils.py"),
            TemplateStep("Lógica principal do script", "core.py", ["utils.py"]),
            TemplateStep("Ponto de entrada com argparse/CLI", "main.py", ["core.py"]),
            TemplateStep("Documentação com exemplos de uso", "README.md"),
        ],
    ),
    TemplateProjeto(
        nome="cli_node",
        palavras_chave=["node", "nodejs", "javascript", "npm"],
        stack="Node.js",
        steps=[
            TemplateStep("Manifesto do projeto", "package.json"),
            TemplateStep("Módulos de lógica de negócio", "src/models.js"),
            TemplateStep("Persistência de dados", "src/storage.js"),
            TemplateStep("CLI interativa com readline/inquirer", "index.js", ["src/models.js", "src/storage.js"]),
            TemplateStep("Documentação com instruções", "README.md"),
        ],
        esqueletos={
            "index.js": (
                "#!/usr/bin/env node\n"
                "const readline = require('readline');\n"
                "// Menu interativo com rl.question()\n"
                "// Loop principal com opções numeradas\n"
            ),
        },
    ),
    TemplateProjeto(
        nome="fullstack_simples",
        palavras_chave=["fullstack", "frontend", "html", "tela", "interface", "formulário", "dashboard"],
        stack="Python + HTML",
        steps=[
            TemplateStep("Arquivo de dependências", "requirements.txt"),
            TemplateStep("Backend com Flask/FastAPI servindo HTML", "app.py"),
            TemplateStep("Template HTML principal", "templates/index.html", ["app.py"]),
            TemplateStep("Estilos CSS", "static/style.css"),
            TemplateStep("Documentação com instruções", "README.md"),
        ],
    ),
]


# ══════════════════════════════════════════════════════════════
# SELETOR DE TEMPLATE
# ══════════════════════════════════════════════════════════════

def selecionar_template(objetivo: str) -> TemplateProjeto | None:
    """
    Seleciona o template mais adequado baseado no objetivo.
    Retorna None se nenhum template é confiável o suficiente.
    """
    objetivo_lower = objetivo.lower()
    pontuacoes: list[tuple[TemplateProjeto, int]] = []

    for template in TEMPLATES:
        score = 0
        for kw in template.palavras_chave:
            if kw in objetivo_lower:
                score += 2
            # Match parcial
            elif any(kw in palavra for palavra in objetivo_lower.split()):
                score += 1
        if score > 0:
            pontuacoes.append((template, score))

    if not pontuacoes:
        return None

    pontuacoes.sort(key=lambda x: x[1], reverse=True)
    melhor, melhor_score = pontuacoes[0]

    # Só retorna se confiança mínima (pelo menos 2 keywords bateram)
    if melhor_score >= 4:
        return melhor

    # Score 2-3: retorna mas pode ser overridden pelo LLM
    if melhor_score >= 2:
        return melhor

    return None


def obter_esqueleto(template: TemplateProjeto, arquivo: str) -> str:
    """Retorna esqueleto parcial do arquivo para injetar no prompt."""
    return template.esqueletos.get(arquivo, "")
