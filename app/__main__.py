"""Entrada de linha de comando: ``python -m app`` (ou ``brain-router``)."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from app import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brain-router",
        description="Brain Router Python: RAG híbrido dual com servidor MCP.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command", metavar="comando")
    check = commands.add_parser("check-integrity", help="audita documentos e vetores do banco")
    check.add_argument("--fix", action="store_true", help="corrige o que é seguro, numa transação")
    check.add_argument("--min-age-hours", type=float, default=1.0, help="prazo do documento vazio")
    check.add_argument("--max-changes", type=int, default=1000, help="teto de linhas do --fix")
    ingest = commands.add_parser(
        "ingest",
        help="indexa os .md, .txt e .json de um diretório (recursivo)",
        argument_default=argparse.SUPPRESS,  # a opção não informada vale o padrão de `Options`
    )
    ingest.add_argument("directory", type=Path, help="pasta da documentação")
    ingest.add_argument("--tenant-id", help="tenant dos documentos (padrão: default)")
    ingest.add_argument("--collection-id", help="coleção dos documentos (padrão: default)")
    ingest.add_argument("--chunk-size", type=int, help="tokens por chunk (padrão: 500)")
    ingest.add_argument("--chunk-overlap", type=int, help="tokens de sobreposição (padrão: 50)")
    ingest.add_argument("--max-bytes", type=int, help="teto por arquivo (padrão: 2 MiB)")
    ingest.add_argument("--dry-run", action="store_true", help="só relata, sem embedding")
    return parser


def _check_integrity(args: argparse.Namespace) -> int:
    # Importado aqui: o --version não precisa de Settings nem de driver de banco.
    from app.db import Database
    from app.integrity import Report, TooManyChanges, check_integrity, render
    from app.log import configure_logging, shutdown_logging
    from app.settings import get_settings

    async def run() -> Report:
        db = Database.from_settings(get_settings())
        try:
            return await check_integrity(
                db, fix=args.fix, min_age_hours=args.min_age_hours, max_changes=args.max_changes
            )
        finally:
            await db.dispose()

    configure_logging("WARNING", json=False, stream=sys.stderr)  # o stdout é do relatório
    try:
        report = asyncio.run(run())
    except TooManyChanges as exc:
        print(f"check-integrity: {exc}", file=sys.stderr)
        return 1
    finally:
        shutdown_logging()
    print(render(report), end="")
    return 0 if report.remaining == 0 else 1  # 1: ainda há caso a olhar


def _ingest(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from app.clients.openrouter import OpenRouterEmbeddingClient
    from app.db import Database
    from app.ingest import Options, Report, ingest_directory, render
    from app.log import configure_logging, shutdown_logging
    from app.settings import get_settings

    if not args.directory.is_dir():
        parser.error(f"não é um diretório: {args.directory}")
    given = {k: v for k, v in vars(args).items() if k not in ("command", "directory")}
    try:
        options = Options(**given)
    except ValueError as exc:
        parser.error(str(exc))

    async def run() -> Report:
        settings = get_settings()
        db = Database.from_settings(settings)
        try:
            async with OpenRouterEmbeddingClient.from_settings(settings) as embedder:
                return await ingest_directory(
                    args.directory, db=db, embedder=embedder, model=embedder.model, options=options
                )
        finally:
            await db.dispose()

    configure_logging("INFO", json=False, stream=sys.stderr)  # progresso; o stdout é do relatório
    try:
        report = asyncio.run(run())
    finally:
        shutdown_logging()
    print(render(report), end="")
    return 0 if report.failure is None else 1  # 1: parou no meio; rodar de novo retoma


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "check-integrity":
        from app.integrity import check_limits  # só aqui: carrega o SQLAlchemy

        try:
            check_limits(args.min_age_hours, args.max_changes)
        except ValueError as exc:
            parser.error(str(exc))
        return _check_integrity(args)
    if args.command == "ingest":
        return _ingest(args, parser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
