"""Logging estruturado (JSON) com structlog, sem bloquear o event loop.

- Toda linha é um objeto JSON com `timestamp` (ISO, UTC), `level`, `logger` e `event`.
- Contexto via `contextvars`: o que for ligado com `structlog.contextvars.bind_contextvars`
  ou `bound_contextvars(...)` entra automaticamente em toda linha daquela task/request,
  inclusive nos logs de bibliotecas que usam o `logging` da stdlib.
- A escrita é feita por um `QueueListener` em thread própria: o chamador só formata e
  enfileira, então uma saída lenta (pipe cheio, coletor travado) não trava o asyncio.
- Chaves sensíveis (senha, token, chave de API, DSN) saem mascaradas.
"""

import logging
import queue
import sys
from logging.handlers import QueueHandler, QueueListener
from typing import IO, Any

import structlog
import structlog.tracebacks
from structlog.typing import EventDict, Processor, WrappedLogger

REDACTED = "**********"
_SENSITIVE_KEYS = frozenset(
    {"password", "secret", "token", "api_key", "authorization", "dsn", "database_url"}
)
_SENSITIVE_SUFFIXES = ("_password", "_secret", "_token", "_api_key")

_listener: QueueListener | None = None


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return lowered in _SENSITIVE_KEYS or lowered.endswith(_SENSITIVE_SUFFIXES)


def redact_secrets(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Mascara valores de chaves sensíveis antes de qualquer renderer vê-los.

    ponytail: olha só as chaves do primeiro nível do evento. Dict aninhado com segredo
    passa; se isso aparecer, trocar por uma varredura recursiva aqui.
    """
    for key in event_dict:
        if _is_sensitive(key):
            event_dict[key] = REDACTED
    return event_dict


def configure_logging(
    level: str = "INFO", *, json: bool = True, stream: IO[str] | None = None
) -> None:
    """Configura structlog + logging da stdlib. Idempotente: pode ser chamada de novo."""
    shutdown_logging()

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        redact_secrets,
    ]
    renderer: Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )
    # `dict_tracebacks` usa show_locals=True: as variáveis locais de cada frame (DSN com
    # senha, tokens) iriam para o log. Os locals não passam pelo redact_secrets.
    exceptions: list[Processor] = (
        [
            structlog.processors.ExceptionRenderer(
                structlog.tracebacks.ExceptionDictTransformer(show_locals=False)
            )
        ]
        if json
        else []
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Formata no chamador (onde os contextvars existem); só a escrita vai para a thread.
    records: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
    queue_handler = QueueHandler(records)
    queue_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                *exceptions,
                renderer,
            ],
        )
    )

    root = logging.getLogger()
    root.handlers = [queue_handler]
    root.setLevel(level.upper())

    global _listener
    _listener = QueueListener(records, logging.StreamHandler(stream or sys.stdout))
    _listener.start()


def shutdown_logging() -> None:
    """Esvazia a fila e para a thread de escrita. Chamar no shutdown da aplicação."""
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None


def get_logger(name: str | None = None, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Logger com métodos síncronos (`info`) e assíncronos (`await log.ainfo(...)`)."""
    logger: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger(name, **initial_values)
    return logger
