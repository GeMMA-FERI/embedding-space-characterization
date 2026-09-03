from __future__ import annotations

import argparse
import pathlib
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from semora.embeddings.serialization import tensor_from_float32blob
from embedding_geometry.classification.utils import (
    evaluate_binary_task,
    sample_similarity_split,
    split_article_groups,
)
from embedding_geometry.utils.evaluation_functions import (
    build_chunking_strategy,
    normalize_embeddings,
    require_known_embedding_run,
    save_results_to_csv,
)
from semora.storage import Database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Gaussian and empirical-histogram Naive Bayes classifiers "
            "on negative/positive and negative/transition cosine similarities."
        )
    )
    parser.add_argument("--db-path", default="./data/newspapers.sqlite")
    parser.add_argument("--embedding-run-id")
    parser.add_argument(
        "--list-run-ids",
        action="store_true",
        help="Print non-noop embedding-run IDs, one per line, then exit.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=100_000,
        help="Maximum sampled similarities per class across train and test.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--histogram-bins", type=int, default=100)
    parser.add_argument("--histogram-smoothing", type=float, default=1.0)
    parser.add_argument("--gaussian-var-smoothing", type=float, default=1e-9)
    parser.add_argument(
        "--output",
        default="data/cosine_classification.csv",
    )
    return parser.parse_args()


def prefixed_metrics(prefix: str, metrics: dict) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def main() -> None:
    args = parse_args()
    if args.sample_limit < 2:
        raise ValueError("--sample-limit must be at least 2.")
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1.")
    if args.histogram_bins < 2:
        raise ValueError("--histogram-bins must be at least 2.")
    if args.histogram_smoothing <= 0:
        raise ValueError("--histogram-smoothing must be greater than zero.")
    if args.gaussian_var_smoothing < 0:
        raise ValueError("--gaussian-var-smoothing cannot be negative.")
    if not args.list_run_ids and not args.embedding_run_id:
        raise ValueError("Pass --embedding-run-id or --list-run-ids.")

    db = Database(args.db_path)
    embeddings: list[torch.Tensor] = []
    groups: dict[str, list[int]] = defaultdict(list)
    first_row = None
    chunking_run_ids: set[str] = set()
    try:
        db.initialize()
        if args.list_run_ids:
            print(*db.get_non_noop_embedding_run_ids(), sep="\n")
            return
        runs = db.get_embedding_runs()
        selected_run = require_known_embedding_run(runs, args.embedding_run_id)
        for row in tqdm(
            db.iter_embeddings_for_run(args.embedding_run_id),
            desc="Loading embeddings",
            total=int(selected_run["embedding_count"]),
        ):
            first_row = first_row or row
            chunking_run_ids.add(str(row["chunking_run_id"]))
            groups[str(row["article_id"])].append(len(embeddings))
            embeddings.append(tensor_from_float32blob(row["tensor_blob"]))
    finally:
        db.close()

    if not embeddings or first_row is None:
        raise ValueError("No valid embeddings match the selected embedding run.")
    if len(chunking_run_ids) != 1:
        raise ValueError(
            "Classification requires one chunking run per embedding run; found: "
            + ", ".join(sorted(chunking_run_ids))
        )

    normalized = normalize_embeddings(torch.stack(embeddings))

    train_groups, test_groups = split_article_groups(
        list(groups.values()),
        test_fraction=args.test_fraction,
        rng=random.Random(args.random_seed),
    )
    test_limit = min(
        max(1, round(args.sample_limit * args.test_fraction)),
        args.sample_limit - 1,
    )
    train_limit = args.sample_limit - test_limit
    train = sample_similarity_split(
        normalized, train_groups, train_limit, seed=args.random_seed + 10
    )
    test = sample_similarity_split(
        normalized, test_groups, test_limit, seed=args.random_seed + 20
    )

    negative_positive = evaluate_binary_task(
        train.negative,
        train.positive,
        test.negative,
        test.positive,
        seed=args.random_seed + 30,
        gaussian_var_smoothing=args.gaussian_var_smoothing,
        histogram_bins=args.histogram_bins,
        histogram_smoothing=args.histogram_smoothing,
    )
    negative_transition = evaluate_binary_task(
        train.negative,
        train.transition,
        test.negative,
        test.transition,
        seed=args.random_seed + 40,
        gaussian_var_smoothing=args.gaussian_var_smoothing,
        histogram_bins=args.histogram_bins,
        histogram_smoothing=args.histogram_smoothing,
    )

    result = {
        "embedding_run_id": args.embedding_run_id,
        "model_id": selected_run["model_id"],
        "chunking_method": first_row["chunking_method"],
        "chunking_strategy": build_chunking_strategy(
            first_row["chunking_method"], first_row["chunking_config_json"]
        ),
        "sample_limit": args.sample_limit,
        "test_fraction": args.test_fraction,
        "random_seed": args.random_seed,
        "histogram_bins": args.histogram_bins,
        "histogram_smoothing": args.histogram_smoothing,
        "gaussian_var_smoothing": args.gaussian_var_smoothing,
        "train_articles": len(train_groups),
        "test_articles": len(test_groups),
        **prefixed_metrics("gaussian_negative_positive", negative_positive["gaussian"]),
        **prefixed_metrics("gaussian_negative_transition", negative_transition["gaussian"]),
        **prefixed_metrics("histogram_negative_positive", negative_positive["histogram"]),
        **prefixed_metrics("histogram_negative_transition", negative_transition["histogram"]),
        **prefixed_metrics("negative_positive", negative_positive["samples"]),
        **prefixed_metrics("negative_transition", negative_transition["samples"]),
    }
    save_results_to_csv([result], args.output)

    print(f"Embedding run: {args.embedding_run_id}")
    print(
        "Gaussian NB: "
        f"N/P={result['gaussian_negative_positive_accuracy']:.2%}, "
        f"N/T={result['gaussian_negative_transition_accuracy']:.2%}"
    )
    print(
        "Histogram NB: "
        f"N/P={result['histogram_negative_positive_accuracy']:.2%}, "
        f"N/T={result['histogram_negative_transition_accuracy']:.2%}"
    )
    print(f"Results saved to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
