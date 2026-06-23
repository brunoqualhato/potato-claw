from src.sessao.manager import Session, SessionManager


def test_sessoes_isoladas_por_canal_usuario():
    mgr = SessionManager()
    s1 = mgr.obter("telegram", "u1")
    s2 = mgr.obter("telegram", "u2")
    s3 = mgr.obter("telegram", "u1")
    assert s1 is s3
    assert s1 is not s2


def test_adicionar_e_historico():
    s = Session("cli:local")
    s.adicionar("user", "oi")
    s.adicionar("assistant", "ola")
    assert s.historico() == [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": "ola"},
    ]


def test_trimming_respeita_limite():
    s = Session("cli:local", limite=2)
    s.adicionar("user", "1")
    s.adicionar("assistant", "2")
    s.adicionar("user", "3")
    hist = s.historico()
    assert len(hist) == 2
    assert hist[0]["content"] == "2"
    assert hist[1]["content"] == "3"
