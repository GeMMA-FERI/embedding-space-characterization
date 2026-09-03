from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from tqdm import tqdm

import torch

from semora.embeddings.serialization import tensor_from_float32blob
from embedding_geometry.utils.evaluation_functions import (
    build_chunking_strategy,
    group_rows_by_article,
    newest_non_empty_embedding_run_id,
    normalize_embeddings,
    print_runs,
    require_known_embedding_run,
    save_results_to_csv
)
from semora.storage import Database


@dataclass(frozen=True)
class DaviesBouldinDiagnostics:
    index: float
    trimmed_index: float
    median_ratio: float
    p95_ratio: float
    p99_ratio: float
    maximum_ratio: float
    top_1pct_contribution: float
    minimum_centroid_distance: float
    p01_nearest_centroid_distance: float
    exact_centroid_pair_count: int


def _trimmed_mean(values: np.ndarray, trim_fraction: float) -> float:
    if not 0 <= trim_fraction < 1:
        raise ValueError("Trim fraction must be in [0, 1).")
    keep_count = max(1, int(np.floor(len(values) * (1 - trim_fraction))))
    return float(np.partition(values, keep_count - 1)[:keep_count].mean())


def calculate_davies_bouldin_diagnostics(
    embedding_matrix: torch.Tensor,
    articles: list[str],
    *,
    block_size: int = 512,
    trim_fraction: float = 0.01,
) -> DaviesBouldinDiagnostics | None:
    """Calculate DB and retain aggregate diagnostics for extreme cluster pairs."""
    if len(embedding_matrix) < 2:
        return None

    x = embedding_matrix.detach().cpu().numpy().astype(np.float64, copy=False)
    _, labels = np.unique(articles, return_inverse=True)
    label_count = int(labels.max()) + 1
    sample_count = len(x)
    if label_count < 2 or label_count >= sample_count:
        raise ValueError(
            "Davies-Bouldin requires between 2 and n_samples - 1 article clusters."
        )

    counts = np.bincount(labels, minlength=label_count)
    centroids = np.zeros((label_count, x.shape[1]), dtype=np.float64)
    np.add.at(centroids, labels, x)
    centroids /= counts[:, None]

    intra_distance_sums = np.zeros(label_count, dtype=np.float64)
    for start in tqdm(range(0, sample_count, block_size), desc="Calculating intra-cluster distances"):
        end = min(start + block_size, sample_count)
        block_labels = labels[start:end]
        distances = np.linalg.norm(
            x[start:end] - centroids[block_labels], axis=1
        )
        np.add.at(intra_distance_sums, block_labels, distances)
    intra_distances = intra_distance_sums / counts

    maximum_ratios = np.zeros(label_count, dtype=np.float64)
    nearest_distances = np.full(label_count, np.inf, dtype=np.float64)
    exact_centroid_pair_count = 0
    for start in tqdm(range(0, label_count, block_size), desc="Calculating ratios and nearest centroids"):
        end = min(start + block_size, label_count)
        distances = pairwise_distances(
            centroids[start:end], centroids, metric="euclidean"
        )
        block_rows = np.arange(end - start)
        centroid_indices = np.arange(start, end)
        distances[block_rows, centroid_indices] = np.inf
        nearest_distances[start:end] = distances.min(axis=1)
        exact_mask = distances == 0
        exact_centroid_pair_count += int(exact_mask.sum())
        distances[exact_mask] = np.inf
        ratios = (
            intra_distances[start:end, None] + intra_distances[None, :]
        ) / distances
        maximum_ratios[start:end] = ratios.max(axis=1)

    top_count = max(1, int(np.ceil(label_count * 0.01)))
    ratio_sum = float(maximum_ratios.sum())
    top_start = len(maximum_ratios) - top_count
    top_sum = float(np.partition(maximum_ratios, top_start)[top_start:].sum())
    finite_nearest = nearest_distances[np.isfinite(nearest_distances)]
    minimum_distance = float(finite_nearest.min()) if len(finite_nearest) else 0.0
    p01_distance = (
        float(np.quantile(finite_nearest, 0.01)) if len(finite_nearest) else 0.0
    )
    return DaviesBouldinDiagnostics(
        index=float(maximum_ratios.mean()),
        trimmed_index=_trimmed_mean(maximum_ratios, trim_fraction),
        median_ratio=float(np.median(maximum_ratios)),
        p95_ratio=float(np.quantile(maximum_ratios, 0.95)),
        p99_ratio=float(np.quantile(maximum_ratios, 0.99)),
        maximum_ratio=float(maximum_ratios.max()),
        top_1pct_contribution=top_sum / ratio_sum if ratio_sum else 0.0,
        minimum_centroid_distance=minimum_distance,
        p01_nearest_centroid_distance=p01_distance,
        exact_centroid_pair_count=exact_centroid_pair_count // 2,
    )


def calculate_davies_bouldin(
    embedding_matrix: torch.Tensor,
    articles: list[str],
    block_size: int = 512
) -> float | None:
    """
    Calculate the Davies-Bouldin index for a matrix of embeddings and their corresponding article labels.
    Lower values indicate better separability between articles.
    """
    diagnostics = calculate_davies_bouldin_diagnostics(
        embedding_matrix,
        articles,
        block_size=block_size,
        trim_fraction=0.0,
    )
    return diagnostics.index if diagnostics is not None else None


def sample_article_cohorts(
    article_ids: list[str],
    *,
    cohort_size: int,
    seeds: list[int],
) -> list[list[str]]:
    """Sample reproducible cohorts from a stable, sorted shared article pool."""
    pool = sorted(set(article_ids))
    if cohort_size > len(pool):
        raise ValueError(
            f"Requested a {cohort_size}-article cohort from only {len(pool)} "
            "shared eligible articles."
        )
    return [random.Random(seed).sample(pool, cohort_size) for seed in seeds]


def calculate_silhouette(
    embedding_matrix: torch.Tensor,
    articles: list[str]
) -> float | None:
    """
    Calculate the silhouette score for a matrix of embeddings and their corresponding article labels.
    Higher values indicate better separability between articles.
    """
    if len(embedding_matrix) < 2:
        return None

    # Convert to numpy array for sklearn compatibility
    x = embedding_matrix.detach().cpu().numpy()
    labels = articles

    return silhouette_score(x, labels, metric="cosine")


def calculate_inter_article_separability(
    embedding_matrix: torch.Tensor,
    articles: list[str],
    metrics: list[str],
    davies_bouldin_block_size: int = 512
) -> dict[str, float | int | None]:
    """
    Calculate inter-article separability metrics for a matrix of embeddings and their corresponding article labels.
    """
    embedding_matrix = normalize_embeddings(embedding_matrix)

    results = {
        "davies_bouldin_index": None,
        "silhouette_score": None
    }
    
    if "davies_bouldin" in metrics:
        results["davies_bouldin_index"] = calculate_davies_bouldin(
            embedding_matrix,
            articles,
            block_size=davies_bouldin_block_size
        )
    
    if "silhouette" in metrics:
        results["silhouette_score"] = calculate_silhouette(embedding_matrix, articles)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate inter-article separability metrics for a given embedding run."
    )
    parser.add_argument(
        "--db-path",
        default="data/newspapers.sqlite"
    )
    parser.add_argument(
        "--embedding-run-id",
        help="Embedding run to evaluate. Defaults to the newest non-empty run."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["davies_bouldin", "silhouette"],
        default=["davies_bouldin"],
        help="Metrics to calculate. Defaults to Davies Bouldin."
    )
    parser.add_argument(
        "--output",
        default="data/inter_article_separability.csv"
    )
    parser.add_argument(
        "--min-chunks-threshold",
        type=int,
        default=5,
        help="Minimum number of chunks required for an article to be included."
    )
    parser.add_argument(
        "--davies-bouldin-block-size",
        type=int,
        default=512,
        help="Number of article centroids compared per distance block."
    )
    parser.add_argument(
        "--davies-bouldin-cohort-size",
        type=int,
        default=10_000,
        help="Number of shared eligible articles sampled for each DB repetition.",
    )
    parser.add_argument(
        "--davies-bouldin-seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds defining reproducible shared article cohorts.",
    )
    parser.add_argument(
        "--davies-bouldin-trim-fraction",
        type=float,
        default=0.01,
        help="Largest fraction of per-article DB ratios removed from trimmed diagnostics.",
    )
    parser.add_argument(
        "--cohort-embedding-run-ids",
        nargs="+",
        help=(
            "Embedding runs that must all contain enough valid chunks for an "
            "article to enter the shared cohort pool. The evaluated run must be included."
        ),
    )
    parser.add_argument(
        "--skip-full-davies-bouldin",
        action="store_true",
        help="Skip the expensive full eligible-corpus DB diagnostic.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_chunks_threshold < 1:
        raise ValueError("Minimum chunks threshold must be at least 1.")
    if args.davies_bouldin_block_size < 1:
        raise ValueError("Davies-Bouldin block size must be at least 1.")
    if args.davies_bouldin_cohort_size < 2:
        raise ValueError("Davies-Bouldin cohort size must be at least 2.")
    if not args.davies_bouldin_seeds:
        raise ValueError("At least one Davies-Bouldin seed is required.")
    if not 0 <= args.davies_bouldin_trim_fraction < 1:
        raise ValueError("Davies-Bouldin trim fraction must be in [0, 1).")

    db = Database(args.db_path)

    try:
        print(" > Initializing database connection...")
        db.initialize()
        print(" > Database connection initialized.")

        runs = db.get_embedding_runs()

        if args.list_runs:
            print_runs(runs)
            return

        embedding_run_id = (
            args.embedding_run_id
            or newest_non_empty_embedding_run_id(runs)
        )

        selected_run = require_known_embedding_run(runs, embedding_run_id)
        model_id = selected_run["model_id"]

        cohort_run_ids = list(
            dict.fromkeys(args.cohort_embedding_run_ids or [embedding_run_id])
        )
        if embedding_run_id not in cohort_run_ids:
            raise ValueError(
                "--cohort-embedding-run-ids must include the evaluated "
                f"embedding run: {embedding_run_id}"
            )
        known_run_ids = {str(run["embedding_run_id"]) for run in runs}
        unknown_run_ids = sorted(set(cohort_run_ids) - known_run_ids)
        if unknown_run_ids:
            raise ValueError(
                "Unknown cohort embedding run ID(s): " + ", ".join(unknown_run_ids)
            )

        print(" > Loading shared eligible article IDs...")
        shared_eligible_article_ids = (
            db.get_shared_eligible_embedding_article_ids(
                cohort_run_ids,
                min_chunks=args.min_chunks_threshold,
            )
            if "davies_bouldin" in args.metrics
            else []
        )

        print(" > Loading embeddings for the evaluated run...")
        embedding_rows = db.get_embeddings_for_run(embedding_run_id)
        print(" > Embeddings loaded.")

    finally:
        db.close()


    if not embedding_rows:
        raise ValueError("No embeddings match the selected embedding run.")

    rows_by_article = group_rows_by_article(embedding_rows)
    eligible_rows_by_article = {
        article_id: rows
        for article_id, rows in rows_by_article.items()
        if len(rows) >= args.min_chunks_threshold
    }
    if len(eligible_rows_by_article) < 2:
        raise ValueError("At least two eligible articles are required.")

    eligible_rows = [
        row
        for article_id in sorted(eligible_rows_by_article)
        for row in eligible_rows_by_article[article_id]
    ]
    articles = [row["article_id"] for row in eligible_rows]
    embedding_matrix = torch.stack(
        [tensor_from_float32blob(row["tensor_blob"]) for row in eligible_rows]
    )
    embedding_matrix = normalize_embeddings(embedding_matrix)
    metrics: dict[str, float | int | str | None] = {
        "davies_bouldin_index": None,
        "davies_bouldin_std": None,
        "silhouette_score": None,
    }

    if "davies_bouldin" in args.metrics:
        missing_article_ids = (
            set(shared_eligible_article_ids) - set(eligible_rows_by_article)
        )
        if missing_article_ids:
            raise ValueError(
                f"Embedding run {embedding_run_id!r} is incomplete for "
                f"{len(missing_article_ids):,} article(s) in the shared cohort. "
                "Finish storing/validating its embeddings before comparing runs."
            )
        shared_pool = shared_eligible_article_ids
        cohorts = sample_article_cohorts(
            shared_pool,
            cohort_size=args.davies_bouldin_cohort_size,
            seeds=args.davies_bouldin_seeds,
        )
        indices_by_article: dict[str, list[int]] = {}
        for index, article_id in enumerate(articles):
            indices_by_article.setdefault(article_id, []).append(index)

        cohort_scores: list[float] = []
        cohort_trimmed_scores: list[float] = []
        for cohort in cohorts:
            cohort_indices = torch.tensor(
                [
                    index
                    for article_id in cohort
                    for index in indices_by_article[article_id]
                ],
                dtype=torch.long,
            )
            cohort_articles = [
                articles[index] for index in cohort_indices.tolist()
            ]
            diagnostics = calculate_davies_bouldin_diagnostics(
                embedding_matrix[cohort_indices],
                cohort_articles,
                block_size=args.davies_bouldin_block_size,
                trim_fraction=args.davies_bouldin_trim_fraction,
            )
            if diagnostics is None:
                raise ValueError("The sampled cohort cannot be evaluated.")
            cohort_scores.append(diagnostics.index)
            cohort_trimmed_scores.append(diagnostics.trimmed_index)

        metrics.update(
            {
                "davies_bouldin_index": float(np.mean(cohort_scores)),
                "davies_bouldin_std": float(np.std(cohort_scores)),
                "davies_bouldin_trimmed_index": float(
                    np.mean(cohort_trimmed_scores)
                ),
                "davies_bouldin_trimmed_std": float(
                    np.std(cohort_trimmed_scores)
                ),
                "davies_bouldin_cohort_pool_size": len(shared_pool),
                "davies_bouldin_cohort_size": args.davies_bouldin_cohort_size,
                "davies_bouldin_seeds_json": json.dumps(args.davies_bouldin_seeds),
                "davies_bouldin_scores_json": json.dumps(cohort_scores),
                "davies_bouldin_trimmed_scores_json": json.dumps(
                    cohort_trimmed_scores
                ),
            }
        )

        if not args.skip_full_davies_bouldin:
            full_diagnostics = calculate_davies_bouldin_diagnostics(
                embedding_matrix,
                articles,
                block_size=args.davies_bouldin_block_size,
                trim_fraction=args.davies_bouldin_trim_fraction,
            )
            if full_diagnostics is None:
                raise ValueError("The full eligible corpus cannot be evaluated.")
            metrics.update(
                {
                    "davies_bouldin_full_index": full_diagnostics.index,
                    "davies_bouldin_full_trimmed_index": (
                        full_diagnostics.trimmed_index
                    ),
                    "davies_bouldin_full_median_ratio": (
                        full_diagnostics.median_ratio
                    ),
                    "davies_bouldin_full_p95_ratio": full_diagnostics.p95_ratio,
                    "davies_bouldin_full_p99_ratio": full_diagnostics.p99_ratio,
                    "davies_bouldin_full_maximum_ratio": (
                        full_diagnostics.maximum_ratio
                    ),
                    "davies_bouldin_full_top_1pct_contribution": (
                        full_diagnostics.top_1pct_contribution
                    ),
                    "davies_bouldin_full_min_centroid_distance": (
                        full_diagnostics.minimum_centroid_distance
                    ),
                    "davies_bouldin_full_p01_nearest_centroid_distance": (
                        full_diagnostics.p01_nearest_centroid_distance
                    ),
                    "davies_bouldin_full_exact_centroid_pair_count": (
                        full_diagnostics.exact_centroid_pair_count
                    ),
                }
            )
        else:
            print("Skipping full eligible-corpus Davies-Bouldin evaluation as requested.")

    if "silhouette" in args.metrics:
        metrics["silhouette_score"] = calculate_silhouette(
            embedding_matrix, articles
        )

    chunking_method = embedding_rows[0]["chunking_method"]
    chunking_strategy = build_chunking_strategy(
        chunking_method,
        embedding_rows[0]["chunking_config_json"]
    )
    results = [
        {
            "embedding_run_id": embedding_run_id,
            "model_id": model_id,
            "chunking_method": chunking_method,
            "chunking_strategy": chunking_strategy,
            "num_articles": len(eligible_rows_by_article),
            "num_chunks": len(eligible_rows),
            "min_chunks_threshold": args.min_chunks_threshold,
            "davies_bouldin_block_size": args.davies_bouldin_block_size,
            "cohort_embedding_run_ids_json": json.dumps(cohort_run_ids),
            "davies_bouldin_trim_fraction": args.davies_bouldin_trim_fraction,
            **metrics
        }
    ]

    save_results_to_csv(results, args.output)

    print("Inter-article separability evaluation results:")
    if metrics["davies_bouldin_index"] is not None:
        print(
            "    Shared-cohort Davies-Bouldin: "
            f"{metrics['davies_bouldin_index']:.4f} "
            f"+/- {metrics['davies_bouldin_std']:.4f}"
        )
        print(
            "    Shared-cohort trimmed Davies-Bouldin: "
            f"{metrics['davies_bouldin_trimmed_index']:.4f} "
            f"+/- {metrics['davies_bouldin_trimmed_std']:.4f}"
        )
        if metrics.get("davies_bouldin_full_index") is not None:
            print(
                "    Full-corpus Davies-Bouldin: "
                f"{metrics['davies_bouldin_full_index']:.4f}"
            )
            print(
                "    Full-corpus trimmed Davies-Bouldin: "
                f"{metrics['davies_bouldin_full_trimmed_index']:.4f}"
            )
    if metrics["silhouette_score"] is not None:
        print(f"    Silhouette score: {metrics['silhouette_score']:.4f}")
    print("Other evaluation details:")
    print(f"    Embedding run: {embedding_run_id}")
    print(f"    Model: {model_id}")
    print(f"    Chunking strategy: {chunking_strategy}")
    print(f"    Number of articles: {len(eligible_rows_by_article)}")
    print(f"    Number of chunks: {len(eligible_rows)}")
    print(f"Results saved to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
