from __future__ import annotations

import argparse
from pathlib import Path

from embedding_geometry.statistics.plot_chunk_char_distributions import (
    DEFAULT_BINS,
    build_chunk_histogram_rows,
    collect_chunk_counts,
    resolve_chunking_run_names,
    save_chunk_histogram_csv,
    select_chunking_runs,
)
from semora.storage import Database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count and bin chunk lengths, then save them as CSV."
    )
    parser.add_argument("--db-path", default="data/newspapers.sqlite")
    parser.add_argument(
        "--chunking-run-ids",
        nargs="+",
        help="Chunking runs to summarize. If omitted, summarizes every run.",
    )
    parser.add_argument("--exclude-chunking-run-ids", nargs="+")
    parser.add_argument(
        "--chunking-run-names",
        nargs="+",
        help="Optional human-readable names in chunking-run order.",
    )
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument(
        "--metric",
        choices=("char", "word"),
        default="char",
        help="Chunk-size metric to summarize (default: char).",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use logarithmically spaced bins; this must be chosen while summarizing.",
    )
    parser.add_argument(
        "--output",
        help="Output CSV path. Defaults to chunk_<metric>_distributions.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bins < 1:
        raise ValueError("bins must be at least 1")
    requested_run_ids = list(dict.fromkeys(args.chunking_run_ids or []))
    excluded_run_ids = list(dict.fromkeys(args.exclude_chunking_run_ids or []))

    db = Database(args.db_path)
    try:
        db.initialize()
        chunking_run_ids, known_runs = select_chunking_runs(
            db,
            requested_run_ids,
            excluded_run_ids,
        )
        run_names = resolve_chunking_run_names(
            chunking_run_ids,
            args.chunking_run_names,
        )
        distributions = collect_chunk_counts(
            db.iter_chunk_texts_for_runs(chunking_run_ids),
            chunking_run_ids,
            metric=args.metric,
        )
    finally:
        db.close()

    for run_id, values in distributions.items():
        if values:
            print(
                f"{run_id}: chunks={len(values):,}, minimum={min(values):,}, "
                f"maximum={max(values):,} {args.metric}s"
            )
        else:
            print(f"Warning: no chunks found for: {run_id}")
    rows = build_chunk_histogram_rows(
        distributions,
        methods={
            run_id: str(known_runs[run_id]["method"])
            for run_id in chunking_run_ids
        },
        run_names=run_names,
        bins=args.bins,
        metric=args.metric,
        log_x=args.log_x,
    )
    output = Path(args.output) if args.output else Path(
        f"data/chunk_{args.metric}_distributions.csv"
    )
    save_chunk_histogram_csv(rows, output)
    print(f"Histogram CSV saved to {output.resolve()}")


if __name__ == "__main__":
    main()
