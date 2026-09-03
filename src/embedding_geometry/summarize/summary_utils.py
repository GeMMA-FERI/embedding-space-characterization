from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

from semora.storage import Database


def read_result_rows(
    input_paths: Iterable[Path],
    required_columns: set[str],
) -> list[dict[str, str]]:
    """Read result CSVs and validate their schemas."""
    rows: list[dict[str, str]] = []
    for input_path in input_paths:
        with input_path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            missing_columns = required_columns - set(reader.fieldnames or [])
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"{input_path} is missing required columns: {missing}")
            rows.extend(reader)
    return rows


def load_embedding_run_metadata(
    db_path: str | Path,
    result_rows: Iterable[Mapping[str, str]],
) -> dict[str, Mapping[str, object]]:
    """Load only embedding runs referenced directly by result rows."""
    embedding_run_ids = list(
        dict.fromkeys(row["embedding_run_id"] for row in result_rows)
    )
    db = Database(db_path)
    try:
        db.initialize()
        run_rows = db.get_embedding_runs_by_ids(embedding_run_ids)
    finally:
        db.close()

    metadata = {str(row["embedding_run_id"]): row for row in run_rows}
    unknown_ids = [run_id for run_id in embedding_run_ids if run_id not in metadata]
    if unknown_ids:
        raise ValueError(f"Unknown embedding run ID(s): {', '.join(unknown_ids)}")
    return metadata


def resolve_result_identity(
    result_row: Mapping[str, str],
    metadata_by_run: Mapping[str, Mapping[str, object]],
) -> tuple[str, str]:
    """Return the authoritative model and chunking run for a result row."""
    embedding_run_id = result_row["embedding_run_id"]
    metadata = metadata_by_run[embedding_run_id]
    chunking_run_id = metadata["chunking_run_id"]
    if not chunking_run_id:
        raise ValueError(
            f"Could not resolve a chunking run for embedding run: {embedding_run_id}"
        )
    return str(metadata["model_id"]), str(chunking_run_id)
