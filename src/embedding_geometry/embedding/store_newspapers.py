from __future__ import annotations

import argparse
import hashlib
import json

from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import Any
from semora.storage import Database, Newspaper, Run


def main() -> None:
    parser = argparse.ArgumentParser(description="Import downloaded newspaper content and metadata into a SQLite database.")
    parser.add_argument(
        "--input-directory",
        default="data/newspapers",
        help="Folder with content and metadata files."
    )
    parser.add_argument(
        "--db-path",
        default="data/newspapers.sqlite",
        help="SQLite database path."
    )
    args = parser.parse_args()

    input_directory = Path(args.input_directory)
    run_id = f"newspapers_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    db = Database(args.db_path)
    db.initialize()
    db.insert_run(
        Run(
            run_id=run_id,
            run_type="newspapers"
        )
    )
    db.log(run_id, "INFO", "Started storing newspapers.")

    count = 0
    skipped_count = 0
    for markdown_path in tqdm(sorted(input_directory.glob("*.md"))):
        metadata_path = markdown_path.with_name(f"{markdown_path.stem}_metadata.json")
        if not metadata_path.exists():
            db.log(run_id, "WARNING", f"Skipped newspaper without metadata file: {markdown_path}.")
            skipped_count += 1
            continue

        metadata = _read_metadata(metadata_path)
        content = markdown_path.read_text(encoding="utf-8")

        db.insert_newspaper(
            Newspaper(
                newspaper_id=_build_newspaper_id(markdown_path),
                run_id=run_id,
                content=content,
                metadata=metadata
            )
        )
        count += 1

    db.log(run_id, "INFO", f"Finished storing newspapers: {count} stored and {skipped_count} skipped.")
    db.close()
    print(f"Imported {count} newspapers into {args.db_path}")


def _read_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_newspaper_id(path: Path) -> str:
    digest = hashlib.sha256(path.stem.encode("utf-8")).hexdigest()
    return f"newspaper_{digest[:24]}"


if __name__ == "__main__":
    main()
