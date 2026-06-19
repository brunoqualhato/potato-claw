#!/usr/bin/env python3
"""
Sistema Multiagente Local — 3 Níveis de Performance
Otimizado para Mac M1 com 8GB RAM.

Uso:
    python main.py
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.core.config import AGENTES, MODELOS, NIVEIS, EMBEDDING_MODEL, PERFIL_ATIVO, PERFIS
from src.core.llm import verificar_modelo_disponivel
from src.agentes.coordenador import rotear
from src.agentes.executor import SistemaAgentes

console = Console()


def exibir_banner():
    banner = Text()
    banner.append("🤖 Sistema Multiagente Local\n", style="bold blue")
    banner.append("   Mac M1 • 8GB RAM • 3 Níveis de Performance\n", style="dim")
    banner.append(f"   Perfil: {PERFIL_ATIVO}\n", style="bold cyan")
    banner.append("   ⚡ Turbo  🚀 Rápido  🧠 Profundo\n\n", style="dim")
    banner.append("   /ajuda para comandos • /sair para encerrar", style="italic dim")
    console.print(Panel(banner, border_style="blue"))


def verificar_dependencias():
    console.print("\n[dim]Verificando dependências...[/dim]")

    # Ollama
    try:
        import ollama
        ollama.list()
        console.print("  ✅ Ollama conectado")
    except Exception as e:
        console.print(f"  [red]❌ Ollama não está rodando: {e}[/red]")
        console.print("  [yellow]Execute: ollama serve[/yellow]")
        return False

    # Modelos LLM
    modelos_unicos = set(MODELOS.values())
    for modelo in modelos_unicos:
        if verificar_modelo_disponivel(modelo):
            console.print(f"  ✅ {modelo}")
        else:
            console.print(f"  [yellow]⚠️  {modelo} — ollama pull {modelo}[/yellow]")

    # Embedding
    if verificar_modelo_disponivel(EMBEDDING_MODEL):
        console.print(f"  ✅ {EMBEDDING_MODEL} (embeddings)")
    else:
        console.print(f"  [yellow]⚠️  {EMBEDDING_MODEL} — ollama pull {EMBEDDING_MODEL}[/yellow]")

    # ChromaDB
    try:
        import chromadb
        console.print("  ✅ ChromaDB disponível")
    except ImportError:
        console.print("  [red]❌ pip install chromadb[/red]")
        return False

    console.print()
    return True


def processar_comando(comando: str, sistema: SistemaAgentes) -> bool | str:
    partes = comando.split(maxsplit=1)
    cmd = partes[0].lower()

    if cmd == "/sair":
        return False

    elif cmd == "/ajuda":
        table = Table(title="Comandos Disponíveis", border_style="blue")
        table.add_column("Comando", style="cyan")
        table.add_column("Descrição")
        table.add_row("/nivel <1|2|3>", "Forçar nível (1=turbo, 2=rápido, 3=profundo)")
        table.add_row("/agente <nome>", "Forçar agente (programador, pesquisador, analista)")
        table.add_row("/stats", "Métricas de performance por nível")
        table.add_row("/knowledge <texto>", "Adicionar conhecimento ao ChromaDB")
        table.add_row("/ingest <arquivo>", "Ingerir arquivo de texto no ChromaDB")
        table.add_row("/contexto", "Ver histórico recente")
        table.add_row("/resumo", "Ver último resumo da conversa")
        table.add_row("/limpar", "Limpar histórico de mensagens")
        table.add_row("/modelos", "Ver modelos configurados")
        table.add_row("/ajuda", "Esta lista")
        table.add_row("/sair", "Encerrar")
        console.print(table)

    elif cmd == "/modelos":
        table = Table(title=f"Perfil Ativo: {PERFIL_ATIVO}", border_style="blue")
        table.add_column("Função", style="cyan")
        table.add_column("Modelo")
        table.add_column("Status")
        for funcao, modelo in MODELOS.items():
            status = "✅" if verificar_modelo_disponivel(modelo) else "❌"
            table.add_row(funcao, modelo, status)
        console.print(table)

        console.print("\n[dim]Perfis disponíveis (edite PERFIL_ATIVO em src/core/config.py):[/dim]")
        for nome, perfil in PERFIS.items():
            marker = " ←" if nome == PERFIL_ATIVO else ""
            console.print(f"  [cyan]{nome}[/cyan]: {perfil['rapido']} / {perfil['completo']}{marker}")

    elif cmd == "/nivel":
        if len(partes) < 2 or partes[1].strip() not in ("1", "2", "3"):
            console.print("[yellow]Use: /nivel 1, /nivel 2 ou /nivel 3[/yellow]")
            for n, info in NIVEIS.items():
                console.print(f"  [dim]{n}: {info['descricao']}[/dim]")
        else:
            nivel = int(partes[1].strip())
            sistema.forcar_nivel(nivel)
            console.print(f"[green]Próxima → Nível {nivel}: {NIVEIS[nivel]['descricao']}[/green]")

    elif cmd == "/stats":
        stats = sistema.estatisticas()
        table = Table(title="📊 Performance por Nível", border_style="green")
        table.add_column("Nível", style="cyan")
        table.add_column("Chamadas")
        table.add_column("Tempo Médio")
        for nivel, info in stats["metricas"].items():
            nome = NIVEIS.get(nivel, {}).get("nome", "?")
            table.add_row(f"{nivel} ({nome})", str(info["total"]), f"{info['avg_ms']}ms")
        console.print(table)
        console.print(f"\n  📋 Cache: {stats['cache']['entradas']} entradas, {stats['cache']['hits_total']} hits")
        console.print(f"  🧲 ChromaDB: {stats['chromadb']['documentos']} documentos")
        console.print(f"  💬 Mensagens: {stats['mensagens_total']}")

    elif cmd == "/knowledge":
        if len(partes) < 2:
            console.print("[yellow]Use: /knowledge <texto>[/yellow]")
        else:
            sistema.ingerir_conhecimento(partes[1].strip())

    elif cmd == "/ingest":
        if len(partes) < 2:
            console.print("[yellow]Use: /ingest <caminho_do_arquivo>[/yellow]")
        else:
            caminho = Path(partes[1].strip())
            if caminho.exists():
                texto = caminho.read_text(encoding="utf-8")
                chunks = [texto[i:i+500] for i in range(0, len(texto), 450)]
                for chunk in chunks:
                    if chunk.strip():
                        sistema.ingerir_conhecimento(chunk, fonte=str(caminho))
                console.print(f"[green]📚 Ingerido: {len(chunks)} chunks de {caminho.name}[/green]")
            else:
                console.print(f"[red]Arquivo não encontrado: {caminho}[/red]")

    elif cmd == "/contexto":
        msgs = sistema.memoria.ultimas_mensagens(5)
        if msgs:
            for m in msgs:
                papel = "👤" if m["role"] == "user" else "🤖"
                agente_info = f" [{m['agente']}]" if m.get("agente") else ""
                console.print(f"  {papel}{agente_info} {m['content'][:100]}...")
        else:
            console.print("[dim]Nenhum histórico ainda.[/dim]")

    elif cmd == "/resumo":
        resumo = sistema.memoria.ultimo_resumo()
        if resumo:
            console.print(Panel(resumo, title="📝 Último Resumo", border_style="yellow"))
        else:
            console.print("[dim]Nenhum resumo gerado ainda.[/dim]")

    elif cmd == "/limpar":
        sistema.memoria.limpar_historico()
        console.print("[green]Histórico limpo.[/green]")

    elif cmd == "/agente":
        if len(partes) < 2:
            console.print("[yellow]Use: /agente <nome>[/yellow]")
            console.print(f"[dim]Disponíveis: {', '.join(AGENTES.keys())}[/dim]")
        else:
            nome = partes[1].strip().lower()
            if nome in AGENTES:
                console.print(f"[green]Próxima → {nome}[/green]")
                return f"agente:{nome}"
            else:
                console.print(f"[red]Agente '{nome}' não existe.[/red]")

    else:
        console.print(f"[red]Comando desconhecido: {cmd}[/red]")
        console.print("[dim]/ajuda para ver comandos[/dim]")

    return True


def main():
    exibir_banner()

    if not verificar_dependencias():
        console.print("[yellow]Continuar mesmo assim? (s/n)[/yellow] ", end="")
        try:
            resp = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            sys.exit(1)
        if resp != "s":
            sys.exit(1)

    sistema = SistemaAgentes()
    agente_forcado = None

    try:
        while True:
            console.print("\n[bold blue]Você:[/bold blue] ", end="")
            try:
                entrada = input().strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Encerrando...[/dim]")
                break

            if not entrada:
                continue

            if entrada.startswith("/"):
                resultado = processar_comando(entrada, sistema)
                if resultado is False:
                    break
                elif isinstance(resultado, str) and resultado.startswith("agente:"):
                    agente_forcado = resultado.split(":", 1)[1]
                continue

            # Pipeline
            if agente_forcado:
                nome_agente = agente_forcado
                agente_forcado = None
                console.print(f"[dim]🎯 Forçado: {nome_agente}[/dim]")
            else:
                nome_agente = rotear(entrada)
                console.print(f"[dim]🎯 {nome_agente}[/dim]")

            console.print(f"[bold green]{nome_agente.capitalize()}:[/bold green] ", end="")
            sistema.executar(nome_agente, entrada)

    finally:
        sistema.fechar()
        console.print("\n[dim]👋 Até logo![/dim]")


if __name__ == "__main__":
    main()
