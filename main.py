#!/usr/bin/env python3
"""
Sistema Multiagente Local — 3 Níveis de Performance
Otimizado para Mac M1 com 8GB RAM.

Uso:
    python main.py                       # Modo interativo
    python main.py --query "pergunta"    # Modo batch (resposta única)
    python main.py --json --query "..."  # Saída JSON estruturada
"""

import argparse
import contextlib
import io
import json as json_lib
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.agentes.coordenador import validar_prompt
from src.agentes.executor import SistemaAgentes
from src.agentes.sessao_codigo import (
    _exportar_disco,
    avancar_sessao,
    editar_arquivo,
    executar_completo,
    gc_chromadb,
    metricas_qualidade,
    obter_sessao,
    rerun_step,
)
from src.core.config import (
    AGENTES,
    EMBEDDING_MODEL,
    MODELOS,
    NIVEIS,
    PERFIL_ATIVO,
    PERFIS,
    WARMUP_HABILITADO,
    WARMUP_MODELOS,
)
from src.core.llm import verificar_modelo_disponivel, warmup_modelos
from src.core.logging_config import setup_logging
from src.core.utils import normalizar

# Prefixos que indicam continuidade conversacional
_PREFIXOS_FOLLOWUP = (
    "e ", "mas ", "também ", "e em ", "e no ", "e na ",
    "e de ", "e o ", "e a ", "e as ", "e os ",
    "e quanto", "e qual", "e como", "e quando", "e onde",
)

# Padrão de localidade após preposição
_RE_LOC = re.compile(
    r"\b(em|no|na|nos|nas|de|do|da|dos|das|para)\s+([A-Za-zÀ-ú][A-Za-zÀ-ú ]{1,30}?)(?=[,?]|\s*$)",
    re.IGNORECASE,
)


def _resolver_followup(entrada: str, sistema: "SistemaAgentes") -> str:
    """
    Se a entrada parece um follow-up (começa com conectivo ou é muito curta),
    recupera a última pergunta do histórico e constrói uma query enriquecida.
    Tenta substituir entidade de lugar quando possível.
    Retorna a entrada original se não for follow-up.
    """
    entrada_norm = normalizar(entrada.strip())
    num_tokens = len(entrada_norm.split())

    eh_followup = any(entrada_norm.startswith(p) for p in _PREFIXOS_FOLLOWUP) or (
        num_tokens <= 3 and any(c in entrada_norm for c in [" em ", " no ", " na ", " de ", " do ", " da "])
    )

    if not eh_followup:
        return entrada

    historico = sistema.memoria.ultimas_mensagens(4)
    ultima_user = next(
        (m["content"] for m in reversed(historico) if m["role"] == "user"),
        None,
    )
    if not ultima_user:
        return entrada

    # Tenta substituir entidade de lugar: "e em Goiânia" → troca local na query original
    nova_loc_m = _RE_LOC.search(entrada)
    antiga_loc_m = _RE_LOC.search(ultima_user)
    if nova_loc_m and antiga_loc_m:
        nova_loc = re.sub(r"^e\s+", "", nova_loc_m.group(0).strip(), flags=re.IGNORECASE)
        query_substituida = ultima_user.replace(antiga_loc_m.group(0).strip(), nova_loc.strip())
        return query_substituida

    # Fallback: concatena
    return f"{ultima_user.rstrip('?.')} {entrada}"


console = Console()


def exibir_banner():
    banner = Text()
    banner.append("🥔 Potato-Claw\n", style="bold blue")
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
    modelos_unicos.discard(EMBEDDING_MODEL)
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
        import chromadb  # noqa: F401
        console.print("  ✅ ChromaDB disponível")
    except ImportError:
        console.print("  [red]❌ pip install chromadb[/red]")
        return False

    if WARMUP_HABILITADO:
        funcoes = tuple(f.strip() for f in WARMUP_MODELOS.split(",") if f.strip())
        console.print(f"  [dim]🔥 Pré-carregando: {', '.join(funcoes)}...[/dim]")
        warmup_modelos(MODELOS, funcoes=funcoes)
    else:
        console.print("  [dim]🔥 Warm-up desativado (carregamento sob demanda)[/dim]")

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
        table.add_row("/projeto <desc>", "Executar projeto completo (agent loop autônomo)")
        table.add_row("/projeto -i <desc>", "Projeto interativo (pede feedback a cada step)")
        table.add_row("/continuar", "Executar próximo step manualmente")
        table.add_row("/rerun <N>", "Re-executar step N do projeto")
        table.add_row("/editar <arq> <instrução>", "Editar arquivo do projeto com instrução")
        table.add_row("/exportar", "Exportar arquivos gerados para disco")
        table.add_row("/gc", "Garbage collection do ChromaDB (remove antigos)")
        table.add_row("/qualidade", "Métricas de qualidade da sessão")
        table.add_row("/feedback <texto>", "Salvar preferência/correção para aprendizado")
        table.add_row("/contexto", "Ver histórico recente")
        table.add_row("/resumo", "Ver último resumo da conversa")
        table.add_row("/limpar", "Limpar histórico de mensagens")
        table.add_row("/limparcache", "Limpar cache de respostas")
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
            table.add_row(
                f"{nivel} ({nome})",
                str(info["total"]),
                f"{info['avg_ms']}ms (p95 {info['p95_ms']}ms)",
            )
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
                # Chunks de 500 chars com 10% overlap (step=450 era 90% — excessivo)
                chunk_size = 500
                step_size = chunk_size  # Sem overlap — cada chunk é independente
                chunks = [texto[i:i+chunk_size] for i in range(0, len(texto), step_size)]
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

    elif cmd == "/limparcache":
        sistema.cache.limpar()
        console.print("[green]Cache limpo.[/green]")

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

    elif cmd == "/projeto":
        if len(partes) < 2:
            # Mostra status da sessão ativa
            sessao = obter_sessao()
            if sessao:
                console.print(Panel(
                    f"Objetivo: {sessao.objetivo}\n"
                    f"Progresso: {sessao.progresso}\n"
                    f"Arquivos: {', '.join(sessao.scratchpad.keys()) or 'nenhum ainda'}\n"
                    f"Tempo: {sessao.tempo_total_ms / 1000:.1f}s",
                    title="📂 Sessão Ativa",
                    border_style="green",
                ))
                for step in sessao.plano:
                    icon = "✅" if step.concluido else "⬜"
                    console.print(f"  {icon} {step.numero}. {step.descricao} → {step.arquivo}")
            else:
                console.print("[yellow]Use: /projeto <descrição do que quer construir>[/yellow]")
                console.print("[dim]O agent loop executa tudo automaticamente até finalizar.[/dim]")
                console.print("[dim]Exemplo: /projeto API REST de tarefas com FastAPI e SQLite[/dim]")
        else:
            objetivo = partes[1].strip()
            interativo = False
            if objetivo.startswith("-i "):
                interativo = True
                objetivo = objetivo[3:].strip()
            modo = "interativo " if interativo else ""
            console.print(f"\n[bold cyan]🚀 Agent loop {modo}para:[/bold cyan] {objetivo}\n")
            sessao = executar_completo(objetivo, salvar_disco=True, interativo=interativo)
            # Salva no histórico do sistema
            sistema.memoria.salvar_mensagem("user", f"/projeto {objetivo}")
            sistema.memoria.salvar_mensagem(
                "assistant",
                f"Projeto {'concluído' if sessao.concluida else 'gerado com pendências'}: "
                f"{sessao.progresso} steps, "
                f"{len(sessao.scratchpad)} arquivos gerados em {sessao.tempo_total_ms/1000:.1f}s",
                "programador",
                3,
            )

    elif cmd == "/continuar":
        sessao = obter_sessao()
        if not sessao:
            console.print("[yellow]Nenhuma sessão ativa. Use /projeto <objetivo> para iniciar.[/yellow]")
        elif sessao.concluida:
            console.print("[green]✅ Projeto já concluído![/green]")
        else:
            step = sessao.step_pendente()
            console.print(f"[dim]⚙️  Step {step.numero}/{len(sessao.plano)}: {step.descricao}...[/dim]")
            step_exec, codigo = avancar_sessao()
            if step_exec and codigo:
                titulo = f"Step {step_exec.numero} • {step_exec.arquivo or step_exec.descricao}"
                console.print(Panel(codigo, title=titulo, border_style="green"))
                console.print(f"[dim]Progresso: {sessao.progresso}[/dim]")

    elif cmd == "/exportar":
        sessao = obter_sessao()
        if not sessao or not sessao.scratchpad:
            console.print("[yellow]Nenhum arquivo para exportar.[/yellow]")
        else:
            caminho = _exportar_disco(sessao)
            console.print(f"[green]✅ Exportado para: {caminho}[/green]")

    elif cmd == "/rerun":
        if len(partes) < 2 or not partes[1].strip().isdigit():
            console.print("[yellow]Use: /rerun <número_do_step>[/yellow]")
        else:
            num = int(partes[1].strip())
            step, codigo = rerun_step(num)
            if step:
                console.print(Panel(codigo[:2000], title=f"🔄 Re-run Step {num}", border_style="cyan"))
            else:
                console.print(f"[red]Step {num} não encontrado.[/red]")

    elif cmd == "/editar":
        sessao = obter_sessao()
        if not sessao:
            console.print("[yellow]Nenhuma sessão ativa.[/yellow]")
        elif len(partes) < 2 or " " not in partes[1]:
            console.print("[yellow]Use: /editar <arquivo> <instrução>[/yellow]")
            console.print(f"[dim]Arquivos: {', '.join(sessao.scratchpad.keys())}[/dim]")
        else:
            resto = partes[1].strip()
            arquivo = resto.split(" ", 1)[0]
            instrucao = resto.split(" ", 1)[1] if " " in resto else ""
            resultado = editar_arquivo(sessao, arquivo, instrucao)
            console.print(Panel(resultado[:2000], title=f"✏️  {arquivo}", border_style="green"))

    elif cmd == "/gc":
        console.print("[dim]🗑️  Executando garbage collection...[/dim]")
        removidos = gc_chromadb(dias_expiracao=30)
        console.print(f"[green]Removidos: {removidos} documentos antigos[/green]")

    elif cmd == "/qualidade":
        sessao = obter_sessao()
        if not sessao:
            console.print("[yellow]Nenhuma sessão para analisar.[/yellow]")
        else:
            m = metricas_qualidade(sessao)
            console.print(Panel(
                f"Taxa de sucesso: {m['taxa_sucesso']}\n"
                f"Steps pulados: {m['pulados']}\n"
                f"Média tentativas/step: {m['media_tentativas']}\n"
                f"Tempo médio/step: {m['tempo_por_step_ms']}ms\n"
                f"Total chars gerados: {m['total_chars_gerados']}",
                title="📊 Qualidade", border_style="blue",
            ))

    elif cmd == "/feedback":
        if len(partes) < 2:
            console.print("[yellow]Use: /feedback <sua correção ou preferência>[/yellow]")
            console.print("[dim]Exemplo: /feedback quando peço API, prefiro FastAPI com SQLAlchemy[/dim]")
        else:
            # Pega a última pergunta do histórico para associar
            msgs = sistema.memoria.ultimas_mensagens(2)
            ultima_pergunta = next(
                (m["content"] for m in reversed(msgs) if m["role"] == "user"), "geral"
            )
            sistema.salvar_feedback(ultima_pergunta, partes[1].strip())
            console.print("[green]✅ Preferência salva! Será usada em interações futuras.[/green]")

    else:
        console.print(f"[red]Comando desconhecido: {cmd}[/red]")
        console.print("[dim]/ajuda para ver comandos[/dim]")

    return True


def main():
    parser = argparse.ArgumentParser(description="Sistema Multiagente Local")
    parser.add_argument("--query", "-q", type=str, help="Pergunta para modo batch (não-interativo)")
    parser.add_argument("--json", action="store_true", help="Saída em formato JSON")
    parser.add_argument("--agente", "-a", type=str, help="Forçar agente específico")
    parser.add_argument("--nivel", "-n", type=int, choices=[1, 2, 3], help="Forçar nível")
    parser.add_argument("--debug", action="store_true", help="Ativar logging DEBUG")
    parser.add_argument("--serve", action="store_true",
                        help="Modo servidor: sobe os canais configurados (Telegram etc) e fica no ar")
    args = parser.parse_args()

    # Configura logging antes de qualquer outra coisa
    setup_logging(force_level="DEBUG" if args.debug else None)

    # ─── Modo servidor de canais (sempre no ar) ───
    if args.serve:
        import asyncio

        from src.conexoes.servidor import servir
        if args.nivel:
            console.print("[dim]Nivel forcado nao se aplica ao modo servidor[/dim]")
        try:
            asyncio.run(servir())
        except KeyboardInterrupt:
            console.print("\n[dim]👋 Servidor encerrado.[/dim]")
        return

    # ─── Modo batch ───
    if args.query:
        sistema = SistemaAgentes()
        try:
            if args.nivel:
                sistema.forcar_nivel(args.nivel)

            if args.agente:
                nome_agente = args.agente
            else:
                nome_agente = "generalista"

            if args.json:
                # Toda saída operacional fica fora do stdout para preservar JSON válido.
                with contextlib.redirect_stdout(io.StringIO()):
                    resposta = sistema.executar(nome_agente, args.query)
            else:
                resposta = sistema.executar(nome_agente, args.query)

            if args.json:
                print(json_lib.dumps({
                    "agente": sistema.ultimo_agente,
                    "nivel": sistema.ultimo_nivel,
                    "fonte": sistema.ultima_fonte,
                    "resposta": resposta,
                }, ensure_ascii=False, indent=2))
        finally:
            sistema.fechar()
        return

    # ─── Modo interativo ───
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

            # Pipeline — roteamento unificado via analisador no executor
            query_roteamento = entrada
            if agente_forcado:
                nome_agente = agente_forcado
                agente_forcado = None
                console.print(f"[dim]🎯 Forçado: {nome_agente}[/dim]")
            else:
                query_roteamento = _resolver_followup(entrada, sistema)
                if query_roteamento != entrada:
                    console.print(f"[dim]🔗 Contexto: {query_roteamento[:80]}[/dim]")

                # Validação leve (sem LLM) — só rejeita input vazio/curto
                valido, motivo = validar_prompt(query_roteamento)
                if not valido:
                    console.print(f"[yellow]⚠️ {motivo}[/yellow]")
                    continue

                # Passa como "generalista" — o executor via analisar_intencao()
                # redireciona para o agente correto (uma única chamada LLM)
                nome_agente = "generalista"

            console.print("[bold green]Potato-Claw:[/bold green] ", end="")
            sistema.executar(nome_agente, query_roteamento)

    finally:
        sistema.fechar()
        console.print("\n[dim]👋 Até logo![/dim]")


if __name__ == "__main__":
    main()
