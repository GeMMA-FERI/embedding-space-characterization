from __future__ import annotations

import argparse
from pathlib import Path

from embedding_geometry.statistics.plot_embedding_cosine_similarities import (
    DEFAULT_PAIR_LIMIT,
    build_similarity_histogram_rows,
    collect_similarity_distributions,
    resolve_embedding_run_names,
    save_similarity_histogram_csv,
)
from semora.storage import Database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample and bin cosine similarities, then save them as CSV."
    )
    parser.add_argument("--db-path", default="data/newspapers.sqlite")
    parser.add_argument("--embedding-run-ids", nargs="+", required=True)
    parser.add_argument(
        "--embedding-run-names",
        nargs="+",
        help="Optional human-readable names in embedding-run order.",
    )
    parser.add_argument("--pair-limit", type=int, default=DEFAULT_PAIR_LIMIT)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-x", action="store_true")
    parser.add_argument(
        "--output",
        default="data/compare_cosine_similarity.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pair_limit < 1:
        raise ValueError("pair-limit must be at least 1")
    if args.bins < 1:
        raise ValueError("bins must be at least 1")
    embedding_run_ids = list(dict.fromkeys(args.embedding_run_ids))
    run_names = resolve_embedding_run_names(
        embedding_run_ids,
        args.embedding_run_names,
    )

    db = Database(args.db_path)
    try:
        db.initialize()
        distributions, run_metadata = collect_similarity_distributions(
            db,
            embedding_run_ids,
            pair_limit=args.pair_limit,
            random_seed=args.random_seed,
        )
    finally:
        db.close()

    rows = build_similarity_histogram_rows(
        distributions,
        model_ids={
            run_id: str(run_metadata[run_id]["model_id"])
            for run_id in embedding_run_ids
        },
        run_names=run_names,
        bins=args.bins,
        log_x=args.log_x,
    )
    output = Path(args.output)
    save_similarity_histogram_csv(rows, output)
    print(f"Histogram CSV saved to {output.resolve()}")


if __name__ == "__main__":
    main()
