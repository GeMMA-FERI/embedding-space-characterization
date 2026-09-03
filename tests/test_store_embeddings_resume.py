from __future__ import annotations

import argparse
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from embedding_geometry.embedding.store_embeddings import (
    _prepare_embedding_run,
)
from semora.storage import Database, EmbeddingRun, Run


def _args(**overrides) -> argparse.Namespace:
    values = {
        "db_path": "ignored.sqlite",
        "chunking_run_id": "chunks-a",
        "model_id": "model-a",
        "batch_size": 32,
        "limit": None,
        "max_characters": None,
        "embedding_run_id": "embeddings-a",
        "transformer_kwargs": {"tokenizer_kwargs": {"use_fast": False}},
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_prepare_embedding_run_creates_a_new_run() -> None:
    db = Database(":memory:")
    try:
        db.initialize()

        assert not _prepare_embedding_run(
            db,
            args=_args(),
            run_id="initial",
            embedding_run_id="embeddings-a",
        )
        stored = db.get_embedding_runs_by_ids(["embeddings-a"])[0]
        assert stored["model_id"] == "model-a"
        assert stored["chunking_run_id"] == "chunks-a"
    finally:
        db.close()


def test_prepare_embedding_run_resumes_compatible_existing_run() -> None:
    db = Database(":memory:")
    try:
        db.initialize()
        db.insert_run(Run(run_id="original", run_type="embeddings"))
        db.insert_embedding_run(
            EmbeddingRun(
                embedding_run_id="embeddings-a",
                run_id="original",
                model_id="model-a",
                config={
                    "chunking_run_id": "chunks-a",
                    "transformer_kwargs": {
                        "tokenizer_kwargs": {"use_fast": False}
                    },
                },
            )
        )

        assert _prepare_embedding_run(
            db,
            args=_args(batch_size=64, limit=100),
            run_id="resume",
            embedding_run_id="embeddings-a",
        )
        assert db.conn.execute(
            "SELECT COUNT(*) AS count FROM embedding_runs"
        ).fetchone()["count"] == 1
    finally:
        db.close()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_id": "model-b"}, "model-id"),
        ({"chunking_run_id": "chunks-b"}, "chunking-run-id"),
        ({"transformer_kwargs": None}, "transformer-kwargs"),
    ],
)
def test_prepare_embedding_run_rejects_incompatible_resume(
    overrides: dict,
    message: str,
) -> None:
    db = Database(":memory:")
    try:
        db.initialize()
        db.insert_run(Run(run_id="original", run_type="embeddings"))
        db.insert_embedding_run(
            EmbeddingRun(
                embedding_run_id="embeddings-a",
                run_id="original",
                model_id="model-a",
                config={
                    "chunking_run_id": "chunks-a",
                    "transformer_kwargs": {
                        "tokenizer_kwargs": {"use_fast": False}
                    },
                },
            )
        )

        with pytest.raises(ValueError, match=message):
            _prepare_embedding_run(
                db,
                args=_args(**overrides),
                run_id="resume",
                embedding_run_id="embeddings-a",
            )
        assert db.conn.execute(
            "SELECT COUNT(*) AS count FROM runs WHERE run_id = 'resume'"
        ).fetchone()["count"] == 0
    finally:
        db.close()
