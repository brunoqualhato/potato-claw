import pytest

from src.extensoes.mcp.client import MCPClient, MCPError


class FakeTransport:
    def __init__(self, respostas):
        self.respostas = respostas
        self.enviados = []

    def request(self, payload):
        self.enviados.append(payload)
        return self.respostas[payload["method"]]


def test_list_tools():
    t = FakeTransport({
        "tools/list": {"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "soma"}]}}
    })
    c = MCPClient(t)
    tools = c.list_tools()
    assert tools == [{"name": "soma"}]
    assert t.enviados[0]["method"] == "tools/list"
    assert t.enviados[0]["jsonrpc"] == "2.0"


def test_call_tool():
    t = FakeTransport({
        "tools/call": {"jsonrpc": "2.0", "id": 1, "result": {"content": "4"}}
    })
    c = MCPClient(t)
    assert c.call_tool("soma", {"a": 2, "b": 2}) == {"content": "4"}
    assert t.enviados[0]["params"]["name"] == "soma"
    assert t.enviados[0]["params"]["arguments"] == {"a": 2, "b": 2}


def test_erro_rpc_levanta():
    t = FakeTransport({
        "tools/list": {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "x"}}
    })
    with pytest.raises(MCPError):
        MCPClient(t).list_tools()
