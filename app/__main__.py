"""Entrada de linha de comando: ``python -m app``."""

import argparse
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
