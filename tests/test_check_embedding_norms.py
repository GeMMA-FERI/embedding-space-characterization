from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from embedding_geometry.statistics.check_embedding_norms import _analyze_rows, _print_stats, parse_args
from semora.storage import Database


class _Cursor:
    def __init__(self, vectors: list[np.ndarray]) -> None:
        self.rows = [{"tensor_blob": vector.tobytes()} for vector in vectors]

    def fetchmany(self, size: int):
        rows, self.rows = self.rows[:size], self.rows[size:]
        return rows


def test_analyze_rows_reports_unit_and_nonfinite_norms(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cursor = _Cursor(
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([np.nan, 0.0], dtype=np.float32),
        ]
    )

    stats = _analyze_rows(cursor, batch_size=2, tolerance=1e-5)

    assert stats.count == 3
    assert stats.finite_count == 2
    assert stats.normalized_count == 1
    assert stats.norm_min == 1.0
    assert stats.norm_sum / stats.finite_count == 3.0
    assert stats.norm_square_sum / stats.finite_count == 13.0
    assert stats.norm_max == 5.0

    _print_stats("Results:", stats)
    assert "L2 norm standard deviation: 2" in capsys.readouterr().out


def test_cli_requires_model_and_chunking_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_embedding_norms.py",
            "--model-id",
            "model",
            "--chunking-run-id",
            "chunks",
        ],
    )

    args = parse_args()

    assert args.model_id == "model"
    assert args.chunking_run_id == "chunks"
    assert not hasattr(args, "embedding_run_id")


def test_fast_database_selection_uses_run_config_without_chunk_joins(
    tmp_path: pathlib.Path,
) -> None:
    db = Database(tmp_path / "norms.sqlite")
    try:
        db.initialize()
        db.conn.execute("PRAGMA foreign_keys = OFF")
        with db.conn:
            db.conn.execute(
                "INSERT INTO runs (run_id, run_type) VALUES ('run', 'embeddings')"
            )
            db.conn.executemany(
                """
                INSERT INTO embedding_runs (
                    embedding_run_id, run_id, model_id, config_json
                ) VALUES (?, 'run', ?, ?)
                """,
                [
                    ("match", "model", '{"chunking_run_id":"chunks-a"}'),
                    ("other-chunks", "model", '{"chunking_run_id":"chunks-b"}'),
                    ("other-model", "another-model", '{"chunking_run_id":"chunks-a"}'),
                ],
            )
            db.conn.execute(
                """
                INSERT INTO embeddings (
                    embedding_id, embedding_run_id, chunk_id, tensor_blob, is_valid
                ) VALUES ('embedding', 'match', 'unresolved-chunk', ?, 1)
                """,
                (np.array([1.0, 0.0], dtype=np.float32).tobytes(),),
            )
        db.conn.execute("PRAGMA foreign_keys = ON")

        runs = db.get_embedding_runs_by_model_and_chunking_run(
            "model", "chunks-a"
        )
        rows = db.iter_embeddings_for_run_fast("match").fetchall()

        assert [row["embedding_run_id"] for row in runs] == ["match"]
        assert len(rows) == 1
        assert rows[0].keys() == ["tensor_blob"]
    finally:
        db.close()
