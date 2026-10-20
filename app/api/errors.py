"""Envelope de erro da API v1 e teto do corpo do pedido.

Todo erro sai como `{"error": {"code": str, "message": str}}` (contrato em `docs/api-v1.md`):
validação (422 do FastAPI), rota e método inexistentes, corpo grande demais, erro de domínio
(`BrainRouterError`, com o status dele) e exceção inesperada (500, sem o texto da exceção).
"""

from collections.abc import Mapping
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.errors.exceptions import BrainRouterError
from app.log import get_logger

log = get_logger(__name__)

# Acima disso, a mensagem de validação diz só quantos erros sobraram.
MAX_VALIDATION_ERRORS = 5
_HTTP_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
}


def error_response(
    status: int, code: str, message: str, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message}}, status_code=status, headers=headers
    )


def _where(error: Any) -> str:
    loc = [str(part) for part in error["loc"]]
    return ".".join(loc[1:] if loc[:1] == ["body"] else loc) or "corpo"


async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Só o campo e a regra: o valor recebido (`input`) não volta, pode ser dado do usuário.
    errors = exc.errors()
    parts = [
        f"JSON inválido (posição {error['loc'][-1]})"
        if error["type"] == "json_invalid"
        else f"{_where(error)}: {error['msg']}"
        for error in errors[:MAX_VALIDATION_ERRORS]
    ]
    if len(errors) > MAX_VALIDATION_ERRORS:
        parts.append(f"(+{len(errors) - MAX_VALIDATION_ERRORS} erros)")
    return error_response(422, "validation_error", "; ".join(parts) or "pedido inválido")


async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
    code = _HTTP_CODES.get(exc.status_code, f"http_{exc.status_code}")
    return error_response(exc.status_code, code, str(exc.detail), exc.headers)


async def domain_error(request: Request, exc: BrainRouterError) -> JSONResponse:
    if exc.http_status >= 500:
        log.warning("api.error", code=exc.code, status=exc.http_status, path=request.url.path)
    return error_response(exc.http_status, exc.code.lower(), exc.message)


async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # O ServerErrorMiddleware levanta a exceção de novo depois desta resposta, e o servidor
    # (uvicorn) registra o traceback. O corpo não leva o texto dela: pode ter dado do usuário.
    return error_response(500, "internal_error", "erro interno do servidor")


EXCEPTION_HANDLERS: dict[int | type[Exception], Any] = {
    RequestValidationError: validation_error,
    HTTPException: http_error,
    BrainRouterError: domain_error,
    Exception: unexpected_error,
}


class BodyLimitMiddleware:
    """Recusa com 413 `payload_too_large` o corpo acima de `max_bytes`, no envelope da API.

    O `RequestBodyLimitMiddleware` do Starlette responde em texto puro quando o
    `Content-Length` já passa do teto, e o contrato pede o envelope JSON em todo erro.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        message = f"corpo do pedido acima de {self.max_bytes} bytes"
        declared = Headers(scope=scope).get("content-length", "")
        if declared.isdecimal() and int(declared) > self.max_bytes:
            # Recusado antes de ler um byte do corpo.
            await error_response(413, "payload_too_large", message)(scope, receive, send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            event = await receive()
            if event["type"] == "http.request":
                received += len(event.get("body", b""))
                if received > self.max_bytes:  # corpo sem Content-Length (chunked)
                    # Tem de ser HTTPException: a leitura do corpo no FastAPI transforma
                    # qualquer outra exceção em 400.
                    raise HTTPException(413, message)
            return event

        await self.app(scope, limited_receive, send)
