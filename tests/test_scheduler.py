from src.extensoes.cron.scheduler import Scheduler


def test_tarefa_nao_vence_antes_do_intervalo():
    sch = Scheduler()
    sch.agendar("ping", intervalo_s=10, callback=lambda: None, agora=0.0)
    assert sch.vencidas(agora=5.0) == []


def test_tarefa_vence_apos_intervalo():
    sch = Scheduler()
    sch.agendar("ping", intervalo_s=10, callback=lambda: None, agora=0.0)
    vencidas = sch.vencidas(agora=10.0)
    assert len(vencidas) == 1
    assert vencidas[0].nome == "ping"


def test_marcar_executada_reagenda():
    sch = Scheduler()
    sch.agendar("ping", intervalo_s=10, callback=lambda: None, agora=0.0)
    t = sch.vencidas(agora=10.0)[0]
    sch.marcar_executada(t, agora=10.0)
    assert sch.vencidas(agora=15.0) == []
    assert len(sch.vencidas(agora=20.0)) == 1
