"""Servidor e roteador base para o protocolo MCP (Model Context Protocol)."""

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ValidationError

ToolHandler = Callable[..., Awaitable[Any] | Any]


class ToolRegistration(BaseModel):
    """Metadados e contrato de uma ferramenta registrada no servidor MCP."""

    name: str
    description: str
    input_model: type[BaseModel]

    model_config = {"arbitrary_types_allowed": True}


class FastMCPServer:
    """Implementação leve de servidor MCP em Python para exposição de ferramentas RAG.

    Suporta descoberta de recursos, validação automática de esquemas via Pydantic e
    despacho padronizado sobre mensagens JSON-RPC 2.0.
    """

    def __init__(
        self,
        name: str = "brain-router-mcp",
        version: str = "0.1.0",
        description: str = "Servidor MCP do motor de RAG e roteamento de contexto",
    ) -> None:
        self.name = name
        self.version = version
        self.description = description
        self.protocol_version = "2024-11-05"
        self._tools: dict[str, tuple[ToolRegistration, ToolHandler]] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        handler: ToolHandler,
    ) -> None:
        """Registra uma ferramenta no catálogo do servidor."""
        if name in self._tools:
            raise ValueError(f"Ferramenta '{name}' já foi registrada anteriormente.")

        reg = ToolRegistration(name=name, description=description, input_model=input_model)
        self._tools[name] = (reg, handler)

    def list_tools(self) -> list[dict[str, Any]]:
        """Retorna a lista de ferramentas disponíveis e seus esquemas JSON Schema."""
        tools_list: list[dict[str, Any]] = []
        for reg, _ in self._tools.values():
            tools_list.append(
                {
                    "name": reg.name,
                    "description": reg.description,
                    "inputSchema": reg.input_model.model_json_schema(),
                }
            )
        return tools_list

    def get_capabilities(self) -> dict[str, Any]:
        """Retorna as capacidades suportadas pelo servidor."""
        return {
            "tools": {"listChanged": False},
            "serverInfo": {
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "protocolVersion": self.protocol_version,
            },
        }

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Valida e executa uma ferramenta registrada."""
        if name not in self._tools:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Ferramenta desconhecida: '{name}'"}],
            }

        reg, handler = self._tools[name]

        try:
            parsed_args = reg.input_model.model_validate(arguments)
        except ValidationError as err:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Argumentos inválidos: {err}"}],
            }

        try:
            if inspect.iscoroutinefunction(handler):
                result = await handler(parsed_args)
            else:
                result = handler(parsed_args)

            data = result.model_dump() if isinstance(result, BaseModel) else result

            return {
                "isError": False,
                "content": [{"type": "text", "text": str(data)}],
                "structuredContent": data,
            }
        except Exception as exc:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Erro durante a execução: {exc}"}],
            }

    async def handle_jsonrpc(self, request: dict[str, Any]) -> dict[str, Any]:
        """Despacha requisição no padrão JSON-RPC 2.0."""
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": self.protocol_version,
                    "capabilities": self.get_capabilities()["tools"],
                    "serverInfo": self.get_capabilities()["serverInfo"],
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self.list_tools()},
            }

        if method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            call_res = await self.call_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": call_res,
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Método não encontrado: '{method}'"},
        }
