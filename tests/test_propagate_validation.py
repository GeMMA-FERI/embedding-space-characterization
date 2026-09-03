from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from semora.storage import (
    Article,
    Chunk,
    ChunkingRun,
    Database,
    Embedding,
    EmbeddingRun,
    Newspaper,
    Run,
)


def test_validation_migration_adds_boolean_and_query_indexes(tmp_path: pathlib.Path) -> None:
    db = Database(tmp_path / "validation.sqlite")
    try:
        db.initialize()

        columns = {row["name"]: row for row in db.conn.execute("PRAGMA table_info(embeddings)")}
        assert columns["is_valid"]["type"] == "INTEGER"
        assert columns["is_valid"]["notnull"] == 1
        assert columns["is_valid"]["dflt_value"] == "0"

        indexes = {row["name"] for row in db.conn.execute("PRAGMA index_list(embeddings)")}
        assert "idx_embeddings_run_chunk" in indexes
        assert "idx_embeddings_valid_run_embedding" in indexes
        assert "idx_embeddings_valid_chunk" in indexes
        assert "idx_embeddings_embedding_run_id" not in indexes

        valid_plan = db.conn.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT embedding_id
            FROM embeddings
            WHERE embedding_run_id = ? AND is_valid = 1
            ORDER BY embedding_id
            """,
            ("embedding-run",),
        ).fetchall()
        assert any(
            "idx_embeddings_valid_run_embedding" in row["detail"]
            for row in valid_plan
        )
    finally:
        db.close()


def test_propagation_copies_article_validation_and_reads_only_valid_embeddings(
    tmp_path: pathlib.Path,
) -> None:
    db = Database(tmp_path / "validation.sqlite")
    try:
        _seed_validation_graph(db)

        initially_stored = db.conn.execute(
            "SELECT embedding_id, is_valid FROM embeddings ORDER BY embedding_id"
        ).fetchall()
        assert [(row["embedding_id"], row["is_valid"]) for row in initially_stored] == [
            ("embedding-invalid", 0),
            ("embedding-unvalidated", 0),
            ("embedding-valid", 1),
        ]

        db.update_article_validation("article-valid", is_valid=False, reason="changed")
        db.update_article_validation("article-invalid", is_valid=True)
        completed_batches: list[int] = []
        counts = db.propagate_embedding_validation(
            batch_size=2,
            on_batch=completed_batches.append,
        )
        db.optimize()

        assert completed_batches == [2, 1]
        assert counts.total == 3
        assert counts.valid == 1
        assert counts.invalid == 2
        assert [
            row["embedding_id"]
            for row in db.iter_embeddings_for_projection(["embedding-run"])
        ] == ["embedding-invalid"]
        assert db.count_embeddings_for_projection(["embedding-run"]) == 1
        assert len(db.iter_embeddings_for_run_fast("embedding-run").fetchall()) == 1
        assert len(db.get_embeddings_for_run("embedding-run")) == 1
        assert len(db.get_full_article_embeddings(embedding_run_id="embedding-run")) == 1
        assert db.get_embedding_runs()[0]["embedding_count"] == 1
    finally:
        db.close()


def test_propagation_rejects_invalid_batch_size(tmp_path: pathlib.Path) -> None:
    db = Database(tmp_path / "validation.sqlite")
    try:
        db.initialize()
        try:
            db.propagate_embedding_validation(batch_size=0)
        except ValueError as error:
            assert str(error) == "batch_size must be positive"
        else:
            raise AssertionError("Expected an invalid batch size to be rejected")
    finally:
        db.close()


def test_committed_batches_are_safe_to_resume_after_interruption(
    tmp_path: pathlib.Path,
) -> None:
    db = Database(tmp_path / "validation.sqlite")
    try:
        _seed_validation_graph(db)
        db.update_article_validation("article-valid", is_valid=False, reason="changed")
        db.update_article_validation("article-invalid", is_valid=True)

        def interrupt_after_first_batch(_: int) -> None:
            raise RuntimeError("interrupted")

        try:
            db.propagate_embedding_validation(
                batch_size=1,
                on_batch=interrupt_after_first_batch,
            )
        except RuntimeError as error:
            assert str(error) == "interrupted"
        else:
            raise AssertionError("Expected propagation to be interrupted")

        partially_updated = db.conn.execute(
            "SELECT embedding_id, is_valid FROM embeddings ORDER BY embedding_id"
        ).fetchall()
        assert [(row["embedding_id"], row["is_valid"]) for row in partially_updated] == [
            ("embedding-invalid", 1),
            ("embedding-unvalidated", 0),
            ("embedding-valid", 1),
        ]

        counts = db.propagate_embedding_validation(batch_size=2)
        assert (counts.valid, counts.invalid) == (1, 2)
        assert db.conn.execute(
            "SELECT is_valid FROM embeddings WHERE embedding_id = 'embedding-valid'"
        ).fetchone()["is_valid"] == 0
    finally:
        db.close()


def _seed_validation_graph(db: Database) -> None:
    db.initialize()
    db.insert_run(Run(run_id="run", run_type="test"))
    db.insert_newspaper(
        Newspaper(newspaper_id="newspaper", run_id="run", content="newspaper")
    )
    for article_id in ("article-valid", "article-invalid", "article-unvalidated"):
        db.insert_article(
            Article(
                article_id=article_id,
                run_id="run",
                newspaper_id="newspaper",
                title=article_id,
                content=article_id,
            )
        )
    db.update_article_validation("article-valid", is_valid=True)
    db.update_article_validation("article-invalid", is_valid=False, reason="invalid")

    db.insert_chunking_run(
        ChunkingRun(
            chunking_run_id="chunking-run",
            run_id="run",
            method="noop",
        )
    )
    for suffix in ("valid", "invalid", "unvalidated"):
        db.insert_chunk(
            Chunk(
                chunk_id=f"chunk-{suffix}",
                run_id="run",
                article_id=f"article-{suffix}",
                chunking_run_id="chunking-run",
                chunk_index=0,
                method="noop",
                text=suffix,
            )
        )

    db.insert_embedding_run(
        EmbeddingRun(
            embedding_run_id="embedding-run",
            run_id="run",
            model_id="model",
            config={"chunking_run_id": "chunking-run"},
        )
    )
    db.insert_embeddings(
        [
            Embedding(
                embedding_id=f"embedding-{suffix}",
                embedding_run_id="embedding-run",
                chunk_id=f"chunk-{suffix}",
                tensor_blob=b"vector",
            )
            for suffix in ("valid", "invalid", "unvalidated")
        ]
    )
