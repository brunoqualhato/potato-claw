import pytest
from src.extensoes.skills.creator import criar_skill
from src.extensoes.skills.loader import carregar_skills


def test_criar_skill_gera_esqueleto(tmp_path):
    caminho = criar_skill(tmp_path, "tradutor", "traduz textos")
    assert (caminho / "SKILL.md").is_file()
    skills = carregar_skills(tmp_path)
    assert skills[0].nome == "tradutor"
    assert skills[0].descricao == "traduz textos"


def test_criar_skill_duplicada_levanta(tmp_path):
    criar_skill(tmp_path, "x", "y")
    with pytest.raises(FileExistsError):
        criar_skill(tmp_path, "x", "z")
