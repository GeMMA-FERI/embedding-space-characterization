from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from datetime import datetime

from tqdm import tqdm

from semora.storage import Database, Run


DEFAULT_BATCH_SIZE = 10_000


def deduplicate_articles(
    db: Database,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    on_batch: Callable[[int], None] | None = None,
) -> int:
    """Invalidate all but one deterministic article per exact content value."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    duplicate_ids = db.get_duplicate_article_ids()
    return _mark_duplicate_article_ids(
        db,
        duplicate_ids=duplicate_ids,
        batch_size=batch_size,
        on_batch=on_batch,
    )


def _mark_duplicate_article_ids(
    db: Database,
    *,
    duplicate_ids: list[str],
    batch_size: int,
    on_batch: Callable[[int], None] | None,
) -> int:
    updated_count = 0
    for batch in _batches(duplicate_ids, batch_size):
        updated_count += db.mark_articles_duplicate(batch)
        if on_batch:
            on_batch(len(batch))
    return updated_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mark exact-content duplicate articles as invalid."
    )
    parser.add_argument(
        "--db-path",
        default="data/newspapers.sqlite",
        help="SQLite database path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Duplicate articles updated per committed transaction.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    run_id = f"deduplicate_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db = Database(args.db_path)
    try:
        db.initialize()
        db.insert_run(Run(run_id=run_id, run_type="deduplicate_articles"))
        db.log(run_id, "INFO", "Started exact-content article deduplication.")

        duplicate_ids = db.get_duplicate_article_ids()
        duplicate_count = len(duplicate_ids)
        with tqdm(
            total=duplicate_count,
            desc="Invalidating duplicate articles",
            unit="article",
        ) as progress:
            updated_count = _mark_duplicate_article_ids(
                db,
                duplicate_ids=duplicate_ids,
                batch_size=args.batch_size,
                on_batch=progress.update,
            )

        db.log(
            run_id,
            "INFO",
            f"Finished article deduplication: {updated_count} duplicates invalidated.",
        )
        print(f"Invalidated {updated_count} duplicate articles in {args.db_path}")
    finally:
        db.close()


def _batches(values: list[str], batch_size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


if __name__ == "__main__":
    main()
