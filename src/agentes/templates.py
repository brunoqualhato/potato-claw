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
            "cli", "terminal", "menu", "interativo", "calculadora", "conversor", "jogo",
            "quiz", "utilitário", "python",
            "player", "música",
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
        palavras_chave=["flask", "servidor flask", "api flask", "backend flask"],
        stack="Python + Flask (API)",
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
        palavras_chave=[
            "node", "nodejs", "javascript", "npm", "js",
            "cli em javascript", "cli em node", "terminal node",
            "app node", "app javascript", "app js",
        ],
        stack="Node.js",
        steps=[
            TemplateStep("Manifesto do projeto", "package.json"),
            TemplateStep("Módulos de lógica de negócio", "src/models.js"),
            TemplateStep("Persistência de dados (JSON file)", "src/storage.js"),
            TemplateStep("CLI interativa com readline", "index.js", ["src/models.js", "src/storage.js"]),
            TemplateStep("Documentação com instruções", "README.md"),
        ],
        esqueletos={
            "package.json": (
                '{\n'
                '  "name": "meu-projeto",\n'
                '  "version": "1.0.0",\n'
                '  "main": "index.js",\n'
                '  "scripts": {\n'
                '    "start": "node index.js"\n'
                '  },\n'
                '  "dependencies": {}\n'
                '}\n'
            ),
            "index.js": (
                "#!/usr/bin/env node\n"
                "'use strict';\n\n"
                "const readline = require('readline');\n"
                "const { /* imports */ } = require('./src/models');\n"
                "const storage = require('./src/storage');\n\n"
                "const rl = readline.createInterface({\n"
                "  input: process.stdin,\n"
                "  output: process.stdout\n"
                "});\n\n"
                "function menu() {\n"
                "  console.log('\\n=== Menu ===');\n"
                "  console.log('1. ...');\n"
                "  console.log('0. Sair');\n"
                "  rl.question('Opção: ', (opcao) => {\n"
                "    if (opcao === '0') { rl.close(); return; }\n"
                "    // processar opção\n"
                "    menu(); // loop\n"
                "  });\n"
                "}\n\n"
                "menu();\n"
            ),
            "src/storage.js": (
                "const fs = require('fs');\n"
                "const path = require('path');\n\n"
                "const DATA_FILE = path.join(__dirname, '..', 'data.json');\n\n"
                "function carregar() {\n"
                "  if (!fs.existsSync(DATA_FILE)) return [];\n"
                "  return JSON.parse(fs.readFileSync(DATA_FILE, 'utf-8'));\n"
                "}\n\n"
                "function salvar(dados) {\n"
                "  fs.writeFileSync(DATA_FILE, JSON.stringify(dados, null, 2));\n"
                "}\n\n"
                "module.exports = { carregar, salvar };\n"
            ),
        },
    ),
    TemplateProjeto(
        nome="api_express",
        palavras_chave=[
            "express", "api node", "api javascript", "api js", "backend node",
            "rest node", "rest javascript", "servidor node", "servidor js",
            "node", "nodejs", "api", "rest",
        ],
        stack="Node.js + Express",
        steps=[
            TemplateStep("Manifesto do projeto com dependências", "package.json"),
            TemplateStep("Modelos de dados e validação", "src/models.js"),
            TemplateStep("Lógica de negócio e persistência", "src/services.js", ["src/models.js"]),
            TemplateStep("Rotas/endpoints da API", "src/routes.js", ["src/services.js"]),
            TemplateStep("Servidor Express com startup", "index.js", ["src/routes.js"]),
            TemplateStep("Documentação com exemplos curl", "README.md"),
        ],
        esqueletos={
            "package.json": (
                '{\n'
                '  "name": "meu-api",\n'
                '  "version": "1.0.0",\n'
                '  "main": "index.js",\n'
                '  "scripts": {\n'
                '    "start": "node index.js",\n'
                '    "dev": "node --watch index.js"\n'
                '  },\n'
                '  "dependencies": {\n'
                '    "express": "^4.18.0"\n'
                '  }\n'
                '}\n'
            ),
            "index.js": (
                "const express = require('express');\n"
                "const routes = require('./src/routes');\n\n"
                "const app = express();\n"
                "app.use(express.json());\n"
                "app.use('/api', routes);\n\n"
                "const PORT = process.env.PORT || 3000;\n"
                "app.listen(PORT, () => {\n"
                "  console.log(`Servidor rodando em http://localhost:${PORT}`);\n"
                "});\n"
            ),
        },
    ),
    TemplateProjeto(
        nome="react_app",
        palavras_chave=[
            "react", "frontend react", "spa", "single page", "vite",
            "componente react", "interface react", "tela react",
        ],
        stack="React + Vite",
        steps=[
            TemplateStep("Manifesto do projeto", "package.json"),
            TemplateStep("Configuração Vite", "vite.config.js"),
            TemplateStep("HTML base", "index.html"),
            TemplateStep("Componente principal App", "src/App.jsx", ["src/components/"]),
            TemplateStep("Entry point React", "src/main.jsx"),
            TemplateStep("Estilos CSS", "src/App.css"),
            TemplateStep("Documentação com instruções", "README.md"),
        ],
        esqueletos={
            "package.json": (
                '{\n'
                '  "name": "meu-react-app",\n'
                '  "version": "1.0.0",\n'
                '  "scripts": {\n'
                '    "dev": "vite",\n'
                '    "build": "vite build"\n'
                '  },\n'
                '  "dependencies": {\n'
                '    "react": "^18.2.0",\n'
                '    "react-dom": "^18.2.0"\n'
                '  },\n'
                '  "devDependencies": {\n'
                '    "vite": "^5.0.0",\n'
                '    "@vitejs/plugin-react": "^4.0.0"\n'
                '  }\n'
                '}\n'
            ),
            "src/main.jsx": (
                "import React from 'react';\n"
                "import ReactDOM from 'react-dom/client';\n"
                "import App from './App';\n"
                "import './App.css';\n\n"
                "ReactDOM.createRoot(document.getElementById('root')).render(\n"
                "  <React.StrictMode>\n"
                "    <App />\n"
                "  </React.StrictMode>\n"
                ");\n"
            ),
        },
    ),
    TemplateProjeto(
        nome="typescript_cli",
        palavras_chave=[
            "typescript", "ts", "cli typescript", "cli ts",
        ],
        stack="TypeScript + Node.js",
        steps=[
            TemplateStep("Manifesto do projeto", "package.json"),
            TemplateStep("Configuração TypeScript", "tsconfig.json"),
            TemplateStep("Módulos de lógica de negócio", "src/models.ts"),
            TemplateStep("Persistência de dados", "src/storage.ts", ["src/models.ts"]),
            TemplateStep("CLI interativa com readline", "src/index.ts", ["src/models.ts", "src/storage.ts"]),
            TemplateStep("Documentação com instruções", "README.md"),
        ],
        esqueletos={
            "package.json": (
                '{\n'
                '  "name": "meu-projeto-ts",\n'
                '  "version": "1.0.0",\n'
                '  "scripts": {\n'
                '    "build": "tsc",\n'
                '    "start": "node dist/index.js",\n'
                '    "dev": "ts-node src/index.ts"\n'
                '  },\n'
                '  "dependencies": {},\n'
                '  "devDependencies": {\n'
                '    "typescript": "^5.0.0",\n'
                '    "ts-node": "^10.9.0",\n'
                '    "@types/node": "^20.0.0"\n'
                '  }\n'
                '}\n'
            ),
            "tsconfig.json": (
                '{\n'
                '  "compilerOptions": {\n'
                '    "target": "ES2020",\n'
                '    "module": "commonjs",\n'
                '    "outDir": "./dist",\n'
                '    "rootDir": "./src",\n'
                '    "strict": true,\n'
                '    "esModuleInterop": true\n'
                '  },\n'
                '  "include": ["src/**/*"]\n'
                '}\n'
            ),
        },
    ),
    TemplateProjeto(
        nome="next_app",
        palavras_chave=[
            "next", "nextjs", "next.js", "ssr", "fullstack react",
            "fullstack javascript", "fullstack js",
        ],
        stack="Next.js",
        steps=[
            TemplateStep("Manifesto do projeto", "package.json"),
            TemplateStep("Configuração Next.js", "next.config.js"),
            TemplateStep("Layout principal", "app/layout.jsx"),
            TemplateStep("Página principal", "app/page.jsx", ["app/layout.jsx"]),
            TemplateStep("Rota API (se necessário)", "app/api/route.js"),
            TemplateStep("Estilos globais", "app/globals.css"),
            TemplateStep("Documentação com instruções", "README.md"),
        ],
        esqueletos={
            "package.json": (
                '{\n'
                '  "name": "meu-next-app",\n'
                '  "version": "1.0.0",\n'
                '  "scripts": {\n'
                '    "dev": "next dev",\n'
                '    "build": "next build",\n'
                '    "start": "next start"\n'
                '  },\n'
                '  "dependencies": {\n'
                '    "next": "^14.0.0",\n'
                '    "react": "^18.2.0",\n'
                '    "react-dom": "^18.2.0"\n'
                '  }\n'
                '}\n'
            ),
        },
    ),
    TemplateProjeto(
        nome="fullstack_simples",
        palavras_chave=[
            "site", "fullstack", "frontend", "html", "tela", "interface web",
            "formulário", "dashboard", "painel", "página web", "pagina",
            "site simples", "landing", "webapp", "web",
            "lista", "contato", "tarefa", "todo", "cadastro", "gerenciador",
            "agenda", "inventário", "estoque", "registro", "controle",
            "sistema", "galeria", "fotos", "foto", "álbum", "imagem",
            "notas", "nota", "diário", "receita", "receitas", "biblioteca",
            "catálogo", "catalogo", "organizador", "aplicativo",
        ],
        stack="Python + Flask + HTML",
        steps=[
            TemplateStep("Arquivo de dependências", "requirements.txt"),
            TemplateStep("Modelos de dados e lógica de negócio", "models.py"),
            TemplateStep("Backend Flask com rotas e persistência", "app.py", ["models.py"]),
            TemplateStep("Template HTML principal com formulários", "templates/index.html", ["app.py"]),
            TemplateStep("Estilos CSS", "static/style.css"),
            TemplateStep("JavaScript do frontend (interatividade)", "static/script.js"),
            TemplateStep("Documentação com instruções de execução", "README.md"),
        ],
        esqueletos={
            "requirements.txt": "flask>=3.0\n",
            "app.py": (
                "from flask import Flask, render_template, request, jsonify\n"
                "import models  # Importar modelos do projeto\n\n"
                "app = Flask(__name__)\n\n"
                "# Inicializar dados/storage\n\n"
                "@app.route('/')\n"
                "def index():\n"
                "    # Carregar dados e renderizar template\n"
                "    return render_template('index.html')\n\n"
                "# Rotas CRUD: adicionar, editar, deletar\n\n"
                "if __name__ == '__main__':\n"
                "    app.run(debug=True, port=5000)\n"
            ),
        },
    ),
]


# ══════════════════════════════════════════════════════════════
# SELETOR DE TEMPLATE
# ══════════════════════════════════════════════════════════════

def selecionar_template(objetivo: str) -> TemplateProjeto | None:
    """
    Seleciona o template mais adequado baseado no objetivo.
    Retorna None se nenhum template é confiável o suficiente.

    Lógica de prioridade:
    1. Palavras de INTENÇÃO DE ENTREGA (site, web, api, cli) — peso 10 (define o tipo de projeto)
    2. Keywords de stack/framework (node, react, flask) — peso 5
    3. Keywords de domínio (lista, contato, tarefa) — peso 1 (NÃO determinam o tipo)
    4. Match parcial — peso 0.5
    """
    objetivo_lower = objetivo.lower()
    pontuacoes: list[tuple[TemplateProjeto, float]] = []
    # ─── KEYWORDS DE INTENÇÃO DE ENTREGA ───
    # Estas palavras indicam O QUE o usuário quer (site vs cli vs api)
    # e devem ter peso dominante sobre palavras de domínio
    _INTENT_KEYWORDS = {
        # Web/Site → prioriza templates web
        "site", "web", "página", "pagina", "webapp", "interface web",
        "frontend", "dashboard", "painel", "formulário", "formulario",
        "navegador", "browser", "html", "landing",
        # API → prioriza templates API
        "api", "rest", "endpoint", "servidor", "http", "backend",
        # CLI → prioriza templates CLI
        "cli", "terminal", "menu", "linha de comando",
    }

    # Keywords que indicam stack/framework explicitamente
    _STACK_KEYWORDS = {
        "node", "nodejs", "javascript", "js", "npm",
        "react", "vite", "next", "nextjs", "next.js",
        "express", "typescript", "ts",
        "flask", "fastapi", "django",
        "python", "py",
    }

    # ─── EXCLUSÃO CRUZADA ───
    # Se o objetivo contém palavras web, penaliza templates CLI e vice-versa
    _WEB_INTENT_WORDS = {"site", "web", "página", "pagina", "webapp", "frontend",
                         "dashboard", "painel", "formulário", "formulario",
                         "navegador", "browser", "html", "landing"}
    _CLI_INTENT_WORDS = {"cli", "terminal", "linha de comando"}

    tem_intencao_web = any(w in objetivo_lower for w in _WEB_INTENT_WORDS)
    tem_intencao_cli = any(w in objetivo_lower for w in _CLI_INTENT_WORDS)

    for template in TEMPLATES:
        score = 0

        # Penalização cruzada: se quer web, templates CLI perdem pontos
        eh_template_cli = template.nome in ("cli_python", "cli_node", "typescript_cli")
        eh_template_web = template.nome in (
            "api_flask", "api_fastapi", "api_express",
            "fullstack_simples", "react_app", "next_app",
        )

        if tem_intencao_web and eh_template_cli:
            score -= 20  # Forte penalização
        if tem_intencao_cli and eh_template_web:
            score -= 20

        for kw in template.palavras_chave:
            if kw in objetivo_lower:
                if kw in _INTENT_KEYWORDS:
                    score += 10  # Intenção de entrega = peso dominante
                elif kw in _STACK_KEYWORDS:
                    score += 5   # Stack explícita = peso alto
                else:
                    score += 1   # Domínio (lista, contato) = peso mínimo
            # Match parcial (keyword dentro de uma palavra)
            elif any(kw in palavra for palavra in objetivo_lower.split()):
                score += 0.5

        pontuacoes.append((template, score))

    if not pontuacoes:
        return None

    pontuacoes.sort(key=lambda x: x[1], reverse=True)
    melhor, melhor_score = pontuacoes[0]

    # Só retorna se confiança mínima (score ≥ 2)
    if melhor_score >= 2:
        return melhor

    return None


def obter_esqueleto(template: TemplateProjeto, arquivo: str) -> str:
    """Retorna esqueleto parcial do arquivo para injetar no prompt."""
    return template.esqueletos.get(arquivo, "")
