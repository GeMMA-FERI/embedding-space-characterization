import argparse
import statistics
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from semora.embeddings.serialization import tensor_from_float32blob
from embedding_geometry.utils.evaluation_functions import (
    build_chunking_strategy,
    group_rows_by_article,
    load_embeddings,
    normalize_embeddings,
    save_results_to_csv
)
from semora.storage import Database


def calculate_weighted_centroid(
    embeddings: list[torch.Tensor],
    weights: list[float]
) -> torch.Tensor:
    """
    Calculate a weighted centroid of chunk embeddings.
    Each chunk embedding is weighted according to its chunk length.
    """
    if not embeddings:
        raise ValueError("Cannot calculate centroid without embeddings.")

    if len(embeddings) != len(weights):
        raise ValueError("Number of embeddings and weights must be equal.")

    embedding_matrix = torch.stack(embeddings)

    weights_tensor = torch.tensor(weights, dtype=embedding_matrix.dtype)

    total_weight = weights_tensor.sum()

    if total_weight <= 0:
        raise ValueError("Total chunk weight must be greater than zero.")

    weighted_sum = (
        embedding_matrix * weights_tensor.unsqueeze(1)
    ).sum(dim=0)

    centroid = weighted_sum / total_weight

    return centroid


def calculate_cosine_similarity(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor
) -> float:
    """
    Calculate cosine similarity between two embeddings.
    """
    embedding_a = normalize_embeddings(embedding_a)
    embedding_b = normalize_embeddings(embedding_b)

    similarity = F.cosine_similarity(
        embedding_a.unsqueeze(0),
        embedding_b.unsqueeze(0),
        dim=1
    ).item()

    return min(1.0, max(-1.0, similarity))


def calculate_representation_degradation(
    article_embedding: torch.Tensor,
    chunk_embeddings: list[torch.Tensor],
    chunk_lengths: list[int],
) -> dict[str, float]:
    """
    Calculate representation preservation and degradation.
    The chunk embeddings are combined into a length-weighted centroid.
    The centroid is then compared to the original article embedding.
    """
    centroid = calculate_weighted_centroid(
        chunk_embeddings,
        chunk_lengths
    )

    similarity = calculate_cosine_similarity(
        article_embedding,
        centroid
    )

    degradation = 1.0 - similarity

    return {
        "cosine_similarity": similarity,
        "representation_degradation": degradation
    }


def calculate_article_representation_degradation(
    article_embedding: torch.Tensor,
    chunk_rows,
) -> dict[str, float]:
    """
    Calculate representation degradation for one article.
    """
    chunk_embeddings = load_embeddings(chunk_rows)

    chunk_lengths = []
    for chunk in chunk_rows:
        chunk_lengths.append(chunk["chunk_length"])

    return calculate_representation_degradation(
        article_embedding=article_embedding,
        chunk_embeddings=chunk_embeddings,
        chunk_lengths=chunk_lengths
    )


def load_article_embeddings_by_id(
    article_embedding_rows,
) -> dict[str, torch.Tensor]:
    """Decode full-article embeddings once for reuse across chunking runs."""
    return {
        str(row["article_id"]): tensor_from_float32blob(row["tensor_blob"])
        for row in article_embedding_rows
    }


def calculate_article_metrics(
    article_embeddings_by_id: Mapping[str, torch.Tensor],
    chunk_rows,
    min_chunks_per_article: int = 5,
) -> list[dict]:
    """
    Calculate representation degradation for every article.
    """
    chunks_by_article = group_rows_by_article(chunk_rows)

    results = []

    for article_id, article_embedding in article_embeddings_by_id.items():
        article_chunks = chunks_by_article.get(article_id, [])

        if len(article_chunks) < min_chunks_per_article:
            continue

        metrics = calculate_article_representation_degradation(
            article_embedding=article_embedding,
            chunk_rows=article_chunks
        )

        results.append(
            {
                "article_id": article_id,
                "num_chunks": len(article_chunks),
                "total_chunk_length": sum(
                    chunk["chunk_length"]
                    for chunk in article_chunks
                ),
                **metrics
            }
        )

    return results


def calculate_aggregate_metrics(
    article_metrics: list[dict],
) -> dict[str, float | int | None]:
    """
    Calculate aggregate representation degradation metrics
    across all articles.
    """
    if not article_metrics:
        return {
            "num_articles": 0,
            "mean_num_chunks": None,
            "mean_total_chunk_length": None,
            "mean_cosine_similarity": None,
            "median_cosine_similarity": None,
            "mean_representation_degradation": None,
            "median_representation_degradation": None
        }

    similarities = [row["cosine_similarity"] for row in article_metrics]
    degradations = [row["representation_degradation"] for row in article_metrics]
    chunk_counts = [row["num_chunks"] for row in article_metrics]
    total_chunk_lengths = [row["total_chunk_length"] for row in article_metrics]

    return {
        "num_articles": len(article_metrics),
        "mean_num_chunks": statistics.fmean(chunk_counts),
        "mean_total_chunk_length": statistics.fmean(total_chunk_lengths),
        "mean_cosine_similarity": statistics.fmean(similarities),
        "median_cosine_similarity": statistics.median(similarities),
        "mean_representation_degradation": statistics.fmean(degradations),
        "median_representation_degradation": statistics.median(degradations)
    }


def select_chunk_embedding_runs(
    runs: Sequence[Mapping[str, object]],
    *,
    embedding_run_id: str | None,
    model_id: str | None,
) -> list[Mapping[str, object]]:
    """Resolve either one embedding run or every non-noop run for a model."""
    if embedding_run_id:
        selected = [
            run
            for run in runs
            if run["embedding_run_id"] == embedding_run_id
        ]
        if not selected:
            raise ValueError(f"Unknown embedding run: {embedding_run_id}")
        return selected

    selected = [
        run
        for run in runs
        if run["model_id"] == model_id
        and run["chunking_methods"] not in (None, "", "noop")
        and int(run["embedding_count"]) > 0
    ]
    if not selected:
        raise ValueError(
            f"No non-empty, non-noop embedding runs found for model {model_id!r}."
        )
    selected.sort(
        key=lambda run: (
            str(run["chunking_run_ids"] or ""),
            str(run["embedding_run_id"]),
        )
    )

    runs_by_chunking: dict[str, list[str]] = {}
    for run in selected:
        chunking_run_id = str(run["chunking_run_ids"] or "")
        runs_by_chunking.setdefault(chunking_run_id, []).append(
            str(run["embedding_run_id"])
        )
    duplicates = {
        chunking_run_id: run_ids
        for chunking_run_id, run_ids in runs_by_chunking.items()
        if len(run_ids) > 1
    }
    if duplicates:
        details = "; ".join(
            f"{chunking_run_id}: {', '.join(run_ids)}"
            for chunking_run_id, run_ids in duplicates.items()
        )
        raise ValueError(
            "Multiple embedding runs found for the same model and chunking run. "
            f"Keep one run per chunking: {details}"
        )
    return selected


def resolve_article_embedding_run(
    runs: Sequence[Mapping[str, object]],
    *,
    model_id: str,
    requested_run_id: str | None,
) -> Mapping[str, object]:
    """Resolve the single non-empty noop baseline for a model."""
    if requested_run_id:
        candidates = [
            run
            for run in runs
            if run["embedding_run_id"] == requested_run_id
        ]
        if not candidates:
            raise ValueError(
                f"Unknown article embedding run: {requested_run_id}"
            )
        baseline = candidates[0]
        if baseline["model_id"] != model_id:
            raise ValueError(
                "Article and chunk embedding runs must use the same model."
            )
        if baseline["chunking_methods"] != "noop":
            raise ValueError("Article embedding run must use noop chunking.")
        return baseline

    candidates = [
        run
        for run in runs
        if run["model_id"] == model_id
        and run["chunking_methods"] == "noop"
        and int(run["embedding_count"]) > 0
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            f"No non-empty noop embedding run found for model {model_id!r}. "
            "Pass --article-embedding-run-id explicitly if one exists."
        )
    candidate_ids = ", ".join(
        str(run["embedding_run_id"]) for run in candidates
    )
    raise ValueError(
        "Could not infer a single full-article embedding run. "
        f"Candidates: {candidate_ids}. Pass --article-embedding-run-id explicitly."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Evaluate representation degradation caused by splitting articles into chunks.")
    )

    parser.add_argument(
        "--db-path",
        default="data/newspapers.sqlite"
    )

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--embedding-run-id",
        help="Chunk embedding run to evaluate."
    )
    selection.add_argument(
        "--model-id",
        help=(
            "Evaluate every non-noop embedding run for this model while "
            "loading its noop article embeddings only once."
        ),
    )

    parser.add_argument(
        "--article-embedding-run-id",
        default=None,
        help=(
            "Full-article embedding run to use as the reference. "
            "If omitted, a noop embedding run with the same model is inferred when possible."
        ),
    )

    parser.add_argument(
        "--min-chunks-per-article",
        type=int,
        default=5,
        help=(
            "Minimum number of chunks an article must have to be included "
            "in the computation (default: 5)."
        ),
    )

    parser.add_argument(
        "--output",
        default="data/representation_degradation.csv"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_chunks_per_article < 1:
        raise ValueError("--min-chunks-per-article must be at least 1.")

    print("Starting representation degradation evaluation...")

    db = Database(args.db_path)

    try:
        db.initialize()
        runs = db.get_embedding_runs()
        chunk_embedding_runs = select_chunk_embedding_runs(
            runs,
            embedding_run_id=args.embedding_run_id,
            model_id=args.model_id,
        )
        model_ids = {
            str(run["model_id"])
            for run in chunk_embedding_runs
        }
        if len(model_ids) != 1:
            raise ValueError("Selected embedding runs must use one model.")
        model_id = model_ids.pop()
        article_embedding_run = resolve_article_embedding_run(
            runs,
            model_id=model_id,
            requested_run_id=args.article_embedding_run_id,
        )
        article_embedding_run_id = str(
            article_embedding_run["embedding_run_id"]
        )

        print("Loading original article embeddings...")
        article_embedding_rows = db.get_full_article_embeddings(
            embedding_run_id=article_embedding_run_id
        )
        if not article_embedding_rows:
            raise ValueError(
                "No full-article embeddings found. This metric expects full "
                "articles embedded with method='noop'."
            )
        article_embeddings_by_id = load_article_embeddings_by_id(
            article_embedding_rows
        )
        del article_embedding_rows
        print(
            f"Loaded and decoded {len(article_embeddings_by_id)} article "
            f"embeddings from {article_embedding_run_id}."
        )

        results = []
        for run_index, chunk_embedding_run in enumerate(
            chunk_embedding_runs,
            start=1,
        ):
            chunk_embedding_run_id = str(
                chunk_embedding_run["embedding_run_id"]
            )
            print(
                f"[{run_index}/{len(chunk_embedding_runs)}] Loading chunk "
                f"embeddings from {chunk_embedding_run_id}..."
            )
            chunk_rows = db.get_chunk_embeddings_for_run(
                embedding_run_id=chunk_embedding_run_id
            )
            if not chunk_rows:
                raise ValueError(
                    f"No chunk embeddings found for {chunk_embedding_run_id}."
                )
            print(f"Loaded {len(chunk_rows)} chunk embeddings.")

            chunking_method = chunk_rows[0]["chunking_method"]
            chunking_strategy = build_chunking_strategy(
                chunking_method,
                chunk_rows[0]["chunking_config_json"]
            )
            article_metrics = calculate_article_metrics(
                article_embeddings_by_id=article_embeddings_by_id,
                chunk_rows=chunk_rows,
                min_chunks_per_article=args.min_chunks_per_article,
            )
            del chunk_rows
            if not article_metrics:
                raise ValueError(
                    "No articles with both article and chunk embeddings met "
                    f"the minimum of {args.min_chunks_per_article} chunks for "
                    f"{chunk_embedding_run_id}."
                )
            aggregate_metrics = calculate_aggregate_metrics(article_metrics)
            del article_metrics

            print(f"Results for {chunk_embedding_run_id} ({chunking_strategy}):")
            print(
                "    Mean cosine similarity: "
                f"{aggregate_metrics['mean_cosine_similarity']:.4f}"
            )
            print(
                "    Median cosine similarity: "
                f"{aggregate_metrics['median_cosine_similarity']:.4f}"
            )
            print(
                "    Mean degradation: "
                f"{aggregate_metrics['mean_representation_degradation']:.4f}"
            )
            print(
                "    Median degradation: "
                f"{aggregate_metrics['median_representation_degradation']:.4f}"
            )
            print(
                f"    Number of articles: {aggregate_metrics['num_articles']}"
            )
            results.append(
                {
                    "embedding_run_id": chunk_embedding_run_id,
                    "article_embedding_run_id": article_embedding_run_id,
                    "model_id": model_id,
                    "chunking_method": chunking_method,
                    "chunking_strategy": chunking_strategy,
                    "min_chunks_per_article": args.min_chunks_per_article,
                    **aggregate_metrics,
                }
            )

    finally:
        db.close()

    save_results_to_csv(results, args.output)
    print(f"Saved {len(results)} result row(s) to {args.output}")


if __name__ == "__main__":
    main()
