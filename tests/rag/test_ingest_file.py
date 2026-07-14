import pytest

import mascan.rag.ingest as ingest_mod


async def test_ingest_file_routes_text(tmp_path, monkeypatch):
    captured = {}

    async def fake_ingest_text(text, *, source, citation):
        captured.update(text=text, source=source, document=citation.document)
        return 3

    monkeypatch.setattr(ingest_mod, "ingest_text", fake_ingest_text)
    f = tmp_path / "note.md"
    f.write_text("hello world", encoding="utf-8")

    stored = await ingest_mod.ingest_file(f, document="note.md")

    assert stored == 3
    assert captured == {"text": "hello world", "source": "upload", "document": "note.md"}


async def test_ingest_file_rejects_unknown_type(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b", encoding="utf-8")
    with pytest.raises(ValueError):
        await ingest_mod.ingest_file(f, document="data.csv")
