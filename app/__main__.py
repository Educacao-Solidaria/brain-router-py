"""Entrada de linha de comando: ``python -m app`` (ou ``brain-router``)."""

import argparse
import asyncio
import sys
from collections.abc import Sequence

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
