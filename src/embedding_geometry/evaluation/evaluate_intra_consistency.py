from __future__ import annotations

import torch
import sys
import pathlib
import argparse
import random
import statistics

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from semora.embeddings.serialization import tensor_from_float32blob
from embedding_geometry.evaluation.evaluate_anisotropy import (
    calculate_average_pairwise_cosine
)
from embedding_geometry.utils.evaluation_functions import (
    build_chunking_strategy,
    calculate_transition_similarities,
    print_runs,
    require_known_embedding_run,
    sample_same_article_similarities,
    save_results_to_csv
)
from semora.storage import Database


def calculate_adaptive_drop_threshold(
    embeddings: list[torch.Tensor],
    sample_size: int
) -> float:
    """Use the embedding run's anisotropy baseline as the drop threshold."""
    average_pairwise_cosine = calculate_average_pairwise_cosine(
        torch.stack(embeddings),
        sample_size
    )
    if average_pairwise_cosine is None:
        raise ValueError("At least two embeddings are required to calculate an adaptive drop threshold.")
    return average_pairwise_cosine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Evaluate semantic consistency between adjacent chunks across an embedding run.")
    )

    parser.add_argument(
        "--db-path",
        default="./data/newspapers.sqlite"
    )

    parser.add_argument(
        "--embedding-run-id",
        help="Embedding run to evaluate."
    )

    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=None,
        help=("Optional fixed semantic-drop threshold. By default, the threshold is the embedding run's mean pairwise cosine or the anisotropy baseline.")
    )

    parser.add_argument(
        "--anisotropy-sample-size",
        type=int,
        default=10_000,
        help=(
            "Number of random pairs used for the adaptive anisotropy threshold. Ignored when drop threshold is supplied."
        )
    )

    parser.add_argument(
        "--intra-pair-sample-size",
        type=int,
        default=100_000,
        help="Maximum number of random same-article chunk pairs to sample."
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used to sample same-article chunk pairs."
    )

    parser.add_argument(
        "--output",
        default="data/intra_consistency.csv",
        help="Path for the results CSV."
    )

    parser.add_argument(
        "--list-runs",
        action="store_true"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.drop_threshold is not None and not -1.0 <= args.drop_threshold <= 1.0:
        raise ValueError("Drop threshold must be between -1 and 1.")

    if args.anisotropy_sample_size < 1:
        raise ValueError("Anisotropy sample size must be at least 1.")

    if args.intra_pair_sample_size < 1:
        raise ValueError("Intra-pair sample size must be at least 1.")

    if args.embedding_run_id is None and not args.list_runs:
        raise ValueError("Either --embedding-run-id must be specified or --list-runs must be used.")

    db = Database(args.db_path)

    try:
        db.initialize()

        runs = db.get_embedding_runs()

        if args.list_runs:
            print_runs(runs)
            return

        embedding_run_id = args.embedding_run_id

        selected_run = require_known_embedding_run(
            runs,
            embedding_run_id
        )

        model_id = selected_run["model_id"]

        embedding_rows = db.get_embeddings_for_run(embedding_run_id)

    finally:
        db.close()

    if not embedding_rows:
        raise ValueError("No embeddings match the selected embedding run.")

    # Group embeddings by chunking run, chunking method, and article.
    # This is necessary so that similarities are only calculated between consecutive chunks from the same article.
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)

    all_embeddings: list[torch.Tensor] = []

    for row in tqdm(embedding_rows):
        embedding = tensor_from_float32blob(row["tensor_blob"])

        key = (
            row["chunking_run_id"],
            row["chunking_method"],
            row["article_id"]
        )

        all_embeddings.append(embedding)
        groups[key].append(len(all_embeddings) - 1)

    # Determine the semantic drop threshold.
    drop_threshold = args.drop_threshold

    if drop_threshold is None:
        drop_threshold = calculate_adaptive_drop_threshold(
            all_embeddings,
            args.anisotropy_sample_size
        )

    # Collect cosine similarities from all valid transitions.
    # Every transition contributes equally to the final metrics.
    all_transition_similarities: list[float] = []

    article_count = 0

    for indices in groups.values():
        embeddings = [all_embeddings[index] for index in indices]
        similarities = calculate_transition_similarities(embeddings)

        if similarities is None:
            continue

        article_count += 1
        all_transition_similarities.extend(similarities.tolist())

    if not all_transition_similarities:
        raise ValueError("No articles contain at least two embeddings, so no transitions can be evaluated.")

    similarities = torch.tensor(all_transition_similarities)

    pairwise_similarities = sample_same_article_similarities(
        torch.stack(all_embeddings),
        list(groups.values()),
        args.intra_pair_sample_size,
        rng=random.Random(args.random_seed),
    )
    if not len(pairwise_similarities):
        raise ValueError(
            "No articles contain at least two embeddings, so no same-article "
            "pairs can be evaluated."
        )

    # Calculate transition metrics and the same-article pairwise mean.
    mean_similarity = similarities.mean().item()
    mean_pairwise_similarity = pairwise_similarities.mean().item()

    cosine_volatility = statistics.pstdev(
        all_transition_similarities
    )

    semantic_drop_rate = (
        (similarities < drop_threshold).float().mean().item()
    )

    transition_count = len(all_transition_similarities)
    pairwise_comparison_count = len(pairwise_similarities)
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
            "drop_threshold": drop_threshold,
            "mean_cosine_similarity": mean_similarity,
            "mean_pairwise_cosine_similarity": mean_pairwise_similarity,
            "cosine_volatility": cosine_volatility,
            "semantic_drop_rate": semantic_drop_rate,
            "num_articles": article_count,
            "num_transitions": transition_count,
            "num_pairwise_comparisons": pairwise_comparison_count
        }
    ]
    save_results_to_csv(results, args.output)

    print("Intra-article consistency evaluation results:")
    print(f"    Mean adjacent cosine similarity: {mean_similarity:.4f}")
    print(f"    Mean same-article pairwise cosine: {mean_pairwise_similarity:.4f}")
    print(f"    Cosine volatility: {cosine_volatility:.4f}")
    print(f"    Semantic drop rate: {semantic_drop_rate:.2%}")
    print("Other evaluation details:")
    print(f"    Embedding run: {embedding_run_id}")
    print(f"    Model: {model_id}")
    print(f"    Drop threshold: {drop_threshold:.4f}")
    print(f"    Number of articles: {article_count}")
    print(f"    Number of transitions: {transition_count}")
    print(f"    Number of sampled same-article pairs: {pairwise_comparison_count}")
    print(f"Results saved to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
