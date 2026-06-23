from src.extensoes.skills.loader import (
    carregar_skills,
    resumo_skills,
    _parse_frontmatter,
)


def test_parse_frontmatter():
    texto = "---\nname: eco\ndescription: repete\n---\nCorpo da skill\n"
    meta, corpo = _parse_frontmatter(texto)
    assert meta["name"] == "eco"
    assert meta["description"] == "repete"
    assert corpo.strip() == "Corpo da skill"


def test_carregar_skills(tmp_path):
    d = tmp_path / "eco"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: eco\ndescription: repete o texto\n---\nInstrucoes\n",
        encoding="utf-8",
    )
    skills = carregar_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].nome == "eco"
    assert skills[0].descricao == "repete o texto"
    assert "Instrucoes" in skills[0].conteudo


def test_resumo_skills(tmp_path):
    d = tmp_path / "eco"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: eco\ndescription: repete\n---\nx\n", encoding="utf-8"
    )
    resumo = resumo_skills(carregar_skills(tmp_path))
    assert "eco" in resumo and "repete" in resumo


def test_pasta_sem_skill_md_ignorada(tmp_path):
    (tmp_path / "vazia").mkdir()
    assert carregar_skills(tmp_path) == []
