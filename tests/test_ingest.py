"""Ingestor em lote e handler do /index sem banco; com Postgres, test_integration_*.py."""

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app import ingest
from app.__main__ import main
from app.chunking import count_words
from app.clients import MockEmbeddingClient
from app.db import Database
from app.ingest import Indexed, Options, Report, Skip, display, document_id, read_text, scan
from app.sanitize import sanitize_text
from app.schemas.mcp import IndexDocumentInput
from app.settings import get_settings


def test_scan_skips_symlinks_and_unlistable_dirs_and_ignores_hidden_and_other_suffixes(
    tmp_path: Path,
) -> None:
    for name in ("a.md", "c.json", "s/b.TXT", ".o/a.json", ".x.md", "y.png", "fechado/z.md"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text("{}")
    (tmp_path / "link.md").symlink_to(tmp_path / "a.md")
    (tmp_path / "s" / "volta").symlink_to(tmp_path)  # laço, se fosse seguido
    (tmp_path / "fechado").chmod(0)
    report = Report(dry_run=False)
    try:
        found = [path.relative_to(tmp_path).as_posix() for path in scan(tmp_path, report)]
    finally:
        (tmp_path / "fechado").chmod(0o755)
    assert found == ["a.md", "c.json", "s/b.TXT"]
    assert report.skipped == [
        ("link.md", "symlink"),
        ("fechado", "erro de leitura (PermissionError)"),
        ("s/volta", "symlink"),
    ]


def test_read_text_names_the_bad_byte_and_checks_size_type_and_symlink_on_open(
    tmp_path: Path,
) -> None:
    path, link, fifo = tmp_path / "x.txt", tmp_path / "link.txt", tmp_path / "fila.txt"
    path.write_bytes(b"\xef\xbb\xbfOl\xe1!")  # BOM + Latin-1: o byte conta desde o início
    with pytest.raises(Skip, match=r"^UTF-8 inválido \(byte 5\)$"):
        read_text(path, Options())
    path.write_bytes(b"\xef\xbb\xbfOl\xc3\xa1")
    assert sanitize_text(read_text(path, Options())) == "Olá"  # o BOM sai no sanitize_text
    with pytest.raises(Skip, match="^maior que 4 bytes$"):
        read_text(path, Options(max_bytes=4))
    link.symlink_to(path)  # trocado depois da varredura: o O_NOFOLLOW recusa
    with pytest.raises(Skip, match=r"^erro de leitura \(OSError\)$"):
        read_text(link, Options())
    os.mkfifo(fifo)
    with pytest.raises(Skip, match="^não é arquivo regular$"):  # e o open não trava
        read_text(fifo, Options())


def test_document_id_is_stable_and_changes_with_the_text_and_what_changes_chunks() -> None:
    def version(content: str = "texto", uri: str = "file:///d/a.md", **change: Any) -> str:
        model, dim = change.pop("model", "m"), change.pop("dim", 1536)
        return document_id(content, uri, Options(**change), model=model, dim=dim)

    assert version() == version() and len(version()) == 32
    variants = [version("texto."), version(uri="file:///d/b.md"), version(model="n")]
    variants += [version(dim=768), version(tenant_id="t"), version(collection_id="c")]
    variants += [version(chunk_size=400), version(chunk_overlap=0)]
    assert len({version(), *variants}) == 9


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db/b")  # nada conecta
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-x")
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()


def test_cli_validates_passes_the_options_prints_one_line_per_file_and_fails_if_stopped(
    cli_env: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for args, message in [
        (["nada"], "não é um diretório: nada"),
        ([".", "--chunk-overlap", "501"], "chunk_overlap deve estar em [0, 500], veio 501"),
        ([".", "--max-bytes", "-2"], "max_bytes deve ser >= 1, veio -2"),
    ]:
        with pytest.raises(SystemExit) as exit:
            main(["ingest", *args])
        assert exit.value.code == 2 and message in capsys.readouterr().err
    report = Report(dry_run=True, indexed=[Indexed("a.md", "x", 2, 10)])
    report.skipped.append((display("c" + chr(10) + chr(0x202E) + chr(0xDCE9) + ".txt"), "vazio"))
    report.failure = ("d.md", "HTTP 503: fora do ar")
    monkeypatch.setattr(ingest, "ingest_directory", fake := AsyncMock(return_value=report))
    assert main(["ingest", ".", "--tenant-id", "t", "--dry-run", "--chunk-size", "50"]) == 1
    assert fake.await_args is not None and fake.await_args.args == (Path("."),)
    kwargs = fake.await_args.kwargs
    assert kwargs["options"] == Options(tenant_id="t", chunk_size=50, dry_run=True)
    assert kwargs["model"] == "openai/text-embedding-3-small"
    bs = chr(92)  # a barra do escape: o texto impresso, não o caractere de controle
    assert capsys.readouterr().out.splitlines() == [
        "+ a.md: 2 chunks, 10 tokens",
        f"- c{bs}n{bs}u202e{bs}udce9.txt: vazio",
        "! d.md: HTTP 503: fora do ar (execução interrompida)",
        "dry-run, nada gravado; seriam indexados: 1 (2 chunks, 10 tokens); sem mudança: 0; "
        "pulados: 1",
    ]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("source_uri", "", "source_uri vazio ou com caractere"),
        ("source_uri", "file:///a" + chr(0) + "b", "source_uri vazio ou com caractere"),
        ("source_uri", "file:///a" + chr(0x200B), "source_uri vazio ou com caractere"),
        ("tenant_id", " ", "tenant_id vazio ou com caractere"),
        ("title", chr(0x200B) + " ", "title vazio depois da limpeza"),
        ("content", chr(0x200B) + chr(0x202E), "content vazio depois da limpeza"),
    ],
)
async def test_index_handler_refuses_what_passes_the_schema_but_would_not_index(
    field: str, value: str, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest, "token_counter", lambda model: count_words)
    db = MagicMock(spec=Database)
    db.read.side_effect = db.session.side_effect = AssertionError("não devia ir ao banco")
    index = ingest.index_handler(db, MockEmbeddingClient(dimension=8), model="m")
    body = {"title": "Guia", "content": "texto", "source_uri": "file:///a.md"} | {field: value}
    with pytest.raises(ingest.InvalidDocument, match=reason) as info:
        await index(IndexDocumentInput.model_validate(body))
    assert (info.value.http_status, info.value.code) == (422, "validation_error")
