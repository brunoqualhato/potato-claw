import io
import json

from src.extensoes.mcp.stdio import StdioTransport


class FakeProc:
    def __init__(self, resposta: dict):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(json.dumps(resposta) + "\n")
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


def test_request_encoda_e_decoda():
    t = StdioTransport(["echo"])
    t._proc = FakeProc({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
    resp = t.request({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}})
    assert resp["result"] == {"ok": True}
    enviado = t._proc.stdin.getvalue().strip()
    assert json.loads(enviado)["method"] == "ping"
