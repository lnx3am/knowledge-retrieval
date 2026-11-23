import sqlite3

import pytest

from knowledge_retrieval.cli import main
from knowledge_retrieval.retrieval import Store


@pytest.mark.parametrize("failure", ["ingest", "query", "stored"])
def test_invalid_embeddings_are_usage_errors_without_output(
    tmp_path, monkeypatch, capsys, failure,
):
    path = tmp_path / "knowledge.db"
    document = tmp_path / "runbook.md"
    document.write_text("incident revised procedure", encoding="utf-8")
    store = Store(str(path), embed=lambda text: [1.0, 0.0])
    store.ingest("demo", document.name, "original incident procedure")
    if failure == "stored":
        with sqlite3.connect(path) as db:
            db.execute("UPDATE vectors SET vector='[1.0]'")
    vector = {"query": [1.0], "ingest": [True, 0.0], "stored": [1.0, 0.0]}[failure]
    monkeypatch.setattr("knowledge_retrieval.cli.openai_embed", lambda text: vector)
    monkeypatch.setattr("knowledge_retrieval.cli.grounded_answer",
                        lambda *args: pytest.fail("failed retrieval must not request an answer"))
    command = ["ingest", "--tenant", "demo", str(document)] if failure == "ingest" else [
        "search", "--tenant", "demo", "incident", "--answer"]
    monkeypatch.setattr("sys.argv", ["retrieval", "--db", str(path), "--embeddings", *command])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(path) in captured.err
    assert ("embedding values must be numbers" if failure == "ingest" else
            "embedding dimensions changed; reindex documents") in captured.err
    assert "Traceback" not in captured.err
    # Ingestion validation must roll back replacement, preserving both text and vectors.
    if failure == "ingest":
        assert store.search("demo", "incident")[0]["text"] == "original incident procedure"


@pytest.mark.parametrize("command", ["ingest", "search"])
@pytest.mark.parametrize("failure", ["directory", "parent_file", "corrupt", "operation"])
def test_database_failure_is_usage_error(tmp_path, monkeypatch, capsys, command, failure):
    path = tmp_path / "knowledge.db"
    document = tmp_path / "runbook.md"
    document.write_text("incident recovery procedure", encoding="utf-8")
    if failure == "directory":
        path.mkdir()
    elif failure == "parent_file":
        path.write_text("not a directory")
        path = path / "knowledge.db"
    elif failure == "corrupt":
        path.write_text("not a SQLite database")
    else:
        store = Store(str(path))
        store.ingest("demo", document.name, "original incident procedure")
        with sqlite3.connect(path) as db:
            if command == "ingest":
                db.execute("CREATE TRIGGER fail_ingest BEFORE INSERT ON chunks "
                           "BEGIN SELECT RAISE(ABORT, 'storage unavailable'); END")
            else:
                db.execute("DROP TABLE chunk_fts")
        # Keep the constructor from recreating the dropped table.
        monkeypatch.setattr("knowledge_retrieval.cli.Store", lambda *args, **kwargs: store)
    monkeypatch.setattr("knowledge_retrieval.cli.grounded_answer",
                        lambda *args: pytest.fail("failed search must not request an answer"))
    flags = [str(document)] if command == "ingest" else ["incident", "--answer"]
    monkeypatch.setattr("sys.argv", ["retrieval", "--db", str(path), command,
                                    "--tenant", "demo", *flags])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"cannot access retrieval database {path}" in captured.err
    assert "Traceback" not in captured.err
    if failure == "operation" and command == "ingest":
        assert store.search("demo", "incident")[0]["text"] == "original incident procedure"


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
@pytest.mark.parametrize("answer", [False, True])
def test_blank_query_is_usage_error_before_io(tmp_path, monkeypatch, capsys, query, answer):
    path = tmp_path / "new" / "knowledge.db"
    monkeypatch.setattr("knowledge_retrieval.cli.grounded_answer",
                        lambda *args: pytest.fail("blank query must not request an answer"))
    monkeypatch.setattr("sys.argv", ["retrieval", "--db", str(path), "search",
                                    "--tenant", "demo", query, *(["--answer"] if answer else [])])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "query cannot be blank" in captured.err
    assert not path.parent.exists()


@pytest.mark.parametrize("input_kind", ["missing", "directory", "invalid_utf8"])
def test_unreadable_input_is_usage_error_before_database_creation(
    tmp_path, monkeypatch, capsys, input_kind
):
    path = tmp_path / "runbook.md"
    if input_kind == "directory":
        path.mkdir()
    elif input_kind == "invalid_utf8":
        path.write_bytes(b"\xff")
    db = tmp_path / "new" / "knowledge.db"
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(db), "ingest",
                                    "--tenant", "demo", str(path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot read input file" in captured.err
    assert not db.parent.exists()


def test_utf8_ingestion_preserves_document_text(tmp_path, monkeypatch):
    path = tmp_path / "runbook.md"
    path.write_text("incident café recovery", encoding="utf-8")
    db = tmp_path / "knowledge.db"
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(db), "ingest",
                                    "--tenant", "demo", str(path)])
    main()
    assert Store(str(db)).search("demo", "incident")[0]["text"] == "incident café recovery"


def test_explicit_sources_keep_same_basename_documents_separate(tmp_path, monkeypatch, capsys):
    db = tmp_path / "knowledge.db"
    files = []
    for directory in ("team-one", "team-two"):
        folder = tmp_path / directory
        folder.mkdir()
        path = folder / "runbook.md"
        path.write_text(f"incident response for {directory}")
        files.append(path)
        monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(db), "ingest",
                                        "--tenant", "demo", "--source", directory, str(path)])
        main()
        assert capsys.readouterr().out.strip() == "1"
    store = Store(str(db))
    assert {hit["source"] for hit in store.search("demo", "incident")} == {
        "team-one", "team-two"}
    files[0].write_text("incident revised procedure")
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(db), "ingest",
                                    "--tenant", "demo", "--source", "team-one", str(files[0])])
    main()
    hits = {hit["source"]: hit["text"] for hit in store.search("demo", "incident")}
    assert hits == {"team-one": "incident revised procedure",
                    "team-two": "incident response for team-two"}


def test_default_source_remains_basename(tmp_path, monkeypatch):
    db = tmp_path / "knowledge.db"
    path = tmp_path / "runbook.md"
    path.write_text("incident response procedure")
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(db), "ingest",
                                    "--tenant", "demo", str(path)])
    main()
    assert Store(str(db)).search("demo", "incident")[0]["source"] == "runbook.md"


@pytest.mark.parametrize("source", ["", "   "])
def test_blank_source_rejected_before_creating_database(tmp_path, monkeypatch, capsys, source):
    db = tmp_path / "knowledge.db"
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(db), "ingest",
                                    "--tenant", "demo", "--source", source, "missing.md"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--source cannot be blank" in capsys.readouterr().err
    assert not db.exists()


@pytest.mark.parametrize("tenant", ["", "   ", "\t\n"])
@pytest.mark.parametrize("command", ["ingest", "search"])
def test_blank_tenant_is_usage_error_before_io(tmp_path, monkeypatch, capsys, tenant, command):
    path = tmp_path / "knowledge.db"
    # The nonexistent file also checks that validation precedes reading input.
    argument = str(tmp_path / "missing.md") if command == "ingest" else "outage"
    monkeypatch.setattr("sys.argv", ["retrieval", "--db", str(path), command,
                                    "--tenant", tenant, argument])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--tenant cannot be blank" in captured.err
    assert not path.exists()
