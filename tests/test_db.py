from src.core.db import Database


def test_database_creates_sqlite_file(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(url=f"sqlite:///{db_path}")
    db.init_db()
    assert db_path.exists()


def test_in_memory_database_can_be_used_immediately():
    db = Database(url="sqlite:///:memory:")
    db.init_db()
    with db.session() as session:
        assert session is not None


def test_get_default_database_is_a_singleton(tmp_path, monkeypatch):
    import src.core.db as db_module

    monkeypatch.setattr(db_module, "_default_db", None)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'singleton.db'}")

    first = db_module.get_default_database()
    second = db_module.get_default_database()
    assert first is second
