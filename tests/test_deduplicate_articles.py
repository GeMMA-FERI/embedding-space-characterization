from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from embedding_geometry.embedding.deduplicate_articles import (
    deduplicate_articles,
    main,
)
from semora.storage import Article, Database, Newspaper, Run


def test_exact_duplicates_are_invalidated_and_canonical_article_is_kept(
    tmp_path: pathlib.Path,
) -> None:
    db = Database(tmp_path / "articles.sqlite")
    try:
        _seed_articles(db)
        completed_batches: list[int] = []

        updated = deduplicate_articles(
            db,
            batch_size=1,
            on_batch=completed_batches.append,
        )

        assert updated == 2
        assert completed_batches == [1, 1]
        rows = {
            row["article_id"]: (row["is_valid"], row["cleaning_reason"])
            for row in db.conn.execute(
                """
                SELECT article_id, is_valid, cleaning_reason
                FROM articles
                ORDER BY article_id
                """
            )
        }
        assert rows["article-a"] == (1, None)
        assert rows["article-b"] == (0, "duplicate")
        assert rows["article-c"] == (0, "duplicate")
        assert rows["article-case-sensitive"] == (1, None)
        assert rows["article-trailing-space"] == (1, None)
        assert rows["article-unique"] == (1, None)

        assert deduplicate_articles(db, batch_size=10) == 0
    finally:
        db.close()


def test_committed_deduplication_batches_can_resume_after_interruption(
    tmp_path: pathlib.Path,
) -> None:
    db = Database(tmp_path / "articles.sqlite")
    try:
        _seed_articles(db)

        def interrupt_after_first_batch(_: int) -> None:
            raise RuntimeError("interrupted")

        try:
            deduplicate_articles(
                db,
                batch_size=1,
                on_batch=interrupt_after_first_batch,
            )
        except RuntimeError as error:
            assert str(error) == "interrupted"
        else:
            raise AssertionError("Expected deduplication to be interrupted")

        assert db.conn.execute(
            "SELECT cleaning_reason FROM articles WHERE article_id = 'article-b'"
        ).fetchone()["cleaning_reason"] == "duplicate"
        assert db.conn.execute(
            "SELECT is_valid FROM articles WHERE article_id = 'article-c'"
        ).fetchone()["is_valid"] == 0
        assert db.conn.execute(
            "SELECT cleaning_reason FROM articles WHERE article_id = 'article-c'"
        ).fetchone()["cleaning_reason"] == "too_short"

        assert deduplicate_articles(db, batch_size=10) == 1
        assert db.conn.execute(
            "SELECT cleaning_reason FROM articles WHERE article_id = 'article-c'"
        ).fetchone()["cleaning_reason"] == "duplicate"
    finally:
        db.close()


def test_deduplication_rejects_invalid_batch_size(tmp_path: pathlib.Path) -> None:
    db = Database(tmp_path / "articles.sqlite")
    try:
        db.initialize()
        try:
            deduplicate_articles(db, batch_size=0)
        except ValueError as error:
            assert str(error) == "batch_size must be positive"
        else:
            raise AssertionError("Expected an invalid batch size to be rejected")
    finally:
        db.close()


def test_cli_main_invalidates_duplicates(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    db_path = tmp_path / "articles.sqlite"
    db = Database(db_path)
    try:
        _seed_articles(db)
    finally:
        db.close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "deduplicate_articles.py",
            "--db-path",
            str(db_path),
            "--batch-size",
            "1",
        ],
    )
    main()

    assert "Invalidated 2 duplicate articles" in capsys.readouterr().out
    db = Database(db_path)
    try:
        duplicates = db.conn.execute(
            """
            SELECT COUNT(*) AS duplicate_count
            FROM articles
            WHERE cleaning_reason = 'duplicate'
            """
        ).fetchone()["duplicate_count"]
        assert duplicates == 2
    finally:
        db.close()


def _seed_articles(db: Database) -> None:
    db.initialize()
    db.insert_run(Run(run_id="run", run_type="test"))
    db.insert_newspaper(
        Newspaper(newspaper_id="newspaper", run_id="run", content="newspaper")
    )
    articles = [
        ("article-c", "Exact duplicate"),
        ("article-a", "Exact duplicate"),
        ("article-b", "Exact duplicate"),
        ("article-case-sensitive", "exact duplicate"),
        ("article-trailing-space", "Exact duplicate "),
        ("article-unique", "Unique content"),
    ]
    for article_id, content in articles:
        db.insert_article(
            Article(
                article_id=article_id,
                run_id="run",
                newspaper_id="newspaper",
                title=article_id,
                content=content,
            )
        )
        db.update_article_validation(article_id, is_valid=True)
    db.update_article_validation(
        "article-c",
        is_valid=False,
        reason="too_short",
    )
