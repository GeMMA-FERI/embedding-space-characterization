from __future__ import annotations

import argparse
from datetime import datetime

from tqdm import tqdm

from semora.storage import Database, Run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propagate article validation through chunks to embeddings."
    )
    parser.add_argument(
        "--db-path",
        default="data/newspapers.sqlite",
        help="SQLite database path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Embeddings committed per transaction (default: 10000).",
    )
    parser.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Do not run PRAGMA optimize after the bulk update.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")

    run_id = f"propagate_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db = Database(args.db_path)
    try:
        db.initialize()
        db.insert_run(Run(run_id=run_id, run_type="propagate_validation"))
        db.log(run_id, "INFO", "Started propagating article validation to embeddings.")

        embedding_count = db.count_embeddings()
        with tqdm(
            total=embedding_count,
            desc="Propagating validation",
            unit="embedding",
        ) as progress:
            counts = db.propagate_embedding_validation(
                batch_size=args.batch_size,
                on_batch=progress.update,
            )
        if not args.skip_optimize:
            db.optimize()

        message = (
            f"Propagated validation to {counts.total} embeddings: "
            f"{counts.valid} valid and {counts.invalid} invalid."
        )
        db.log(run_id, "INFO", message)
        print(message)
    finally:
        db.close()


if __name__ == "__main__":
    main()
