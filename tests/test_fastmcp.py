"""Testes unitários para o servidor FastMCPServer."""

import pytest

from app.mcp import FastMCPServer
from app.schemas.mcp import HybridSearchInput, HybridSearchOutput


@pytest.fixture
def mcp_server() -> FastMCPServer:
    server = FastMCPServer(name="test-server", version="1.0.0")

    async def mock_hybrid_search(params: HybridSearchInput) -> HybridSearchOutput:
        return HybridSearchOutput(
            query=params.query,
            results=[],
            total_found=0,
            execution_time_ms=1.5,
        )

    server.register_tool(
        name="hybrid_search",
        description="Busca híbrida combinando dense e sparse",
        input_model=HybridSearchInput,
        handler=mock_hybrid_search,
    )
    return server


def test_server_capabilities(mcp_server: FastMCPServer) -> None:
    caps = mcp_server.get_capabilities()
    assert caps["serverInfo"]["name"] == "test-server"
    assert caps["serverInfo"]["version"] == "1.0.0"
    assert "tools" in caps


def test_list_tools(mcp_server: FastMCPServer) -> None:
    tools = mcp_server.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "hybrid_search"
    assert "inputSchema" in tools[0]
    assert tools[0]["inputSchema"]["properties"]["query"]["type"] == "string"


@pytest.mark.asyncio
async def test_call_tool_success(mcp_server: FastMCPServer) -> None:
    resp = await mcp_server.call_tool(
        name="hybrid_search",
        arguments={"query": "algoritmos de roteamento", "top_k": 3},
    )
    assert resp["isError"] is False
    assert resp["structuredContent"]["query"] == "algoritmos de roteamento"
    assert resp["structuredContent"]["execution_time_ms"] == 1.5


@pytest.mark.asyncio
async def test_call_tool_validation_error(mcp_server: FastMCPServer) -> None:
    # query é obrigatória e min_length=1
    resp = await mcp_server.call_tool(
        name="hybrid_search",
        arguments={"query": ""},
    )
    assert resp["isError"] is True
    assert "Argumentos inválidos" in resp["content"][0]["text"]


@pytest.mark.asyncio
async def test_call_unknown_tool(mcp_server: FastMCPServer) -> None:
    resp = await mcp_server.call_tool(
        name="inexistent_tool",
        arguments={},
    )
    assert resp["isError"] is True
    assert "Ferramenta desconhecida" in resp["content"][0]["text"]


@pytest.mark.asyncio
async def test_jsonrpc_lifecycle(mcp_server: FastMCPServer) -> None:
    # 1. Initialize
    init_res = await mcp_server.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": "1", "method": "initialize"}
    )
    assert init_res["result"]["serverInfo"]["name"] == "test-server"

    # 2. Tools list
    list_res = await mcp_server.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": "2", "method": "tools/list"}
    )
    assert len(list_res["result"]["tools"]) == 1

    # 3. Tools call
    call_res = await mcp_server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": "3",
            "method": "tools/call",
            "params": {"name": "hybrid_search", "arguments": {"query": "teste rpc"}},
        }
    )
    assert call_res["result"]["isError"] is False

    # 4. Ping
    ping_res = await mcp_server.handle_jsonrpc({"jsonrpc": "2.0", "id": "4", "method": "ping"})
    assert ping_res["result"] == {}

    # 5. Unknown method
    err_res = await mcp_server.handle_jsonrpc({"jsonrpc": "2.0", "id": "5", "method": "unknown"})
    assert "error" in err_res
    assert err_res["error"]["code"] == -32601


def test_duplicate_registration_error(mcp_server: FastMCPServer) -> None:
    with pytest.raises(ValueError, match="já foi registrada"):
        mcp_server.register_tool(
            name="hybrid_search",
            description="duplicada",
            input_model=HybridSearchInput,
            handler=lambda x: x,
        )
