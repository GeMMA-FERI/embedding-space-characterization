from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from embedding_geometry.summarize.summary_utils import (
    load_embedding_run_metadata,
    resolve_result_identity,
)
from semora.storage import Database


def test_result_identity_uses_direct_embedding_run_chunking_id(
    tmp_path: pathlib.Path,
) -> None:
    db_path = tmp_path / "newspapers.sqlite"
    db = Database(db_path)
    try:
        db.initialize()
        with db.conn:
            db.conn.execute(
                "INSERT INTO runs (run_id, run_type) VALUES ('run-1', 'embedding')"
            )
            db.conn.executemany(
                """
                INSERT INTO embedding_runs (
                    embedding_run_id, run_id, model_id, config_json
                ) VALUES (?, 'run-1', 'model', ?)
                """,
                [
                    ("model_recursive_400", '{"chunking_run_id": "recursive_400"}'),
                    ("model_recursive_1000", '{"chunking_run_id": "recursive_1000"}'),
                ],
            )
    finally:
        db.close()

    result_rows = [
        {"embedding_run_id": "model_recursive_400", "chunking_strategy": "split"},
        {"embedding_run_id": "model_recursive_1000", "chunking_strategy": "split"},
    ]
    metadata = load_embedding_run_metadata(db_path, result_rows)

    identities = [resolve_result_identity(row, metadata) for row in result_rows]

    assert identities == [
        ("model", "recursive_400"),
        ("model", "recursive_1000"),
    ]
