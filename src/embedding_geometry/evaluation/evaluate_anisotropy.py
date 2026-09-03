import argparse

import random

import torch

from sklearn.decomposition import PCA

from embedding_geometry.utils.evaluation_functions import (
    build_chunking_strategy,
    calculate_pairwise_cosine_similarities,
    load_embeddings,
    normalize_embeddings,
    print_runs,
    require_known_embedding_run,
    save_results_to_csv
)
from semora.storage import Database


def calculate_average_pairwise_cosine(
    embedding_matrix: torch.Tensor,
    sample_size: int # How many random pairs to sample for calculating average cosine similarity
) -> float | None:
    """
    Calculate average cosine similarity between randomly
    sampled pairs of embeddings.
    We select a large number of random pairs of embeddings,
    calculate the cosine similarity for each pair and then
    compute the average.
    Higher values indicate a more anisotropic embedding space.
    """
    similarities = calculate_pairwise_cosine_similarities(
        embedding_matrix,
        sample_size,
    )
    return similarities.mean().item() if similarities is not None else None


def calculate_pca_metrics(
    embedding_matrix: torch.Tensor,
    random_seed: int
) -> dict[str, float | None]:
    """
    Calculate principal component analysis metrics for a matrix of embeddings.
    Principal component analysis is a dimensionality reduction technique that
    identifies the directions in which the data varies the most.
    Returns the variance ratios of the first 1, 5 and 10 principal components.
    Higher values indicate greater concentration of variance in the first
    principal components, which suggests greater anisotropy.
    """
    if len(embedding_matrix) < 2:
        return {
            "pc1_variance_ratio": None,
            "pc5_variance_ratio": None,
            "pc10_variance_ratio": None
        }
    
    embedding_matrix = normalize_embeddings(
        embedding_matrix
    )

    x = embedding_matrix.detach().cpu().numpy() # Convert to numpy array for sklearn compatibility

    component_count = min(10, *x.shape)
    pca = PCA(
        n_components=component_count,
        svd_solver="randomized",
        random_state=random_seed
    )
    pca.fit(x)

    explained = pca.explained_variance_ratio_
    # Explained variance ratio of each principal component,
    # which indicates the proportion of the dataset's variance
    # that lies along each principal component

    # Higher ratios indicate that more variance is concentrated
    # in a small number of dominant directions,
    # which suggests greater anisotropy 
    return {
        "pc1_variance_ratio": float(
            explained[:1].sum()
        ),
        "pc5_variance_ratio": float(
            explained[:5].sum()
        ),
        "pc10_variance_ratio": float(
            explained[:10].sum()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate embedding anisotropy metrics for a given embedding run."
    )
    parser.add_argument(
        "--db-path",
        default="data/newspapers.sqlite"
    )
    parser.add_argument(
        "--embedding-run-id",
        help="Embedding run to evaluate."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Number of random embedding pairs to sample."
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for pair sampling and randomized PCA."
    )
    parser.add_argument(
        "--output",
        default="data/embedding_anisotropy.csv"
    )
    parser.add_argument(
        "--list-runs",
        action="store_true"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Starting embedding anisotropy evaluation...")
    random.seed(args.random_seed)

    if args.sample_size < 1:
        raise ValueError("Sample size must be at least 1.")

    if args.embedding_run_id is None and not args.list_runs:
        raise ValueError("Either --embedding-run-id must be specified or --list-runs must be used.")

    db = Database(args.db_path)

    try:
        db.initialize()

        if args.list_runs:
            runs = db.get_embedding_runs()
            print_runs(runs)
            return

        embedding_run_id = args.embedding_run_id
        selected_run = require_known_embedding_run(
            db.get_embedding_runs_by_ids([embedding_run_id]),
            embedding_run_id
        )
        model_id = selected_run["model_id"]

        print("Loading embeddings...")
        embedding_rows = db.get_embeddings_for_run(embedding_run_id)

        print(f"Loaded {len(embedding_rows)} embeddings.")

    finally:
        db.close()

    if not embedding_rows:
        raise ValueError(f"No embeddings match the selected run.")

    chunking_method = embedding_rows[0]["chunking_method"]
    chunking_strategy = build_chunking_strategy(
        chunking_method,
        embedding_rows[0]["chunking_config_json"]
    )

    print(f"Evaluating {len(embedding_rows)} embeddings: Method {chunking_method}")

    embeddings = load_embeddings(embedding_rows)

    embedding_matrix = torch.stack(embeddings)

    average_pairwise_cosine = calculate_average_pairwise_cosine(
        embedding_matrix,
        args.sample_size
    )

    pca_metrics = calculate_pca_metrics(
        embedding_matrix,
        args.random_seed
    )

    embedding_count = len(embedding_rows)
    total_pair_count = embedding_count * (embedding_count - 1) // 2
    pair_sample_size = min(args.sample_size, total_pair_count)

    results = [
        {
            "embedding_run_id": embedding_run_id,
            "model_id": model_id,
            "chunking_method": chunking_method,
            "chunking_strategy": chunking_strategy,
            "embedding_count": embedding_count,
            "pair_sample_size": pair_sample_size,
            "random_seed": args.random_seed,
            "average_pairwise_cosine": average_pairwise_cosine,
            **pca_metrics
        }
    ]

    save_results_to_csv(results, args.output)

    print("Embedding anisotropy evaluation results:")
    print(f"    Mean pairwise cosine: {average_pairwise_cosine:.4f}")
    print(f"    PC1 variance ratio: {pca_metrics['pc1_variance_ratio']:.4f}")
    print(f"    PC5 variance ratio: {pca_metrics['pc5_variance_ratio']:.4f}")
    print(f"    PC10 variance ratio: {pca_metrics['pc10_variance_ratio']:.4f}")
    print("Other evaluation details:")
    print(f"    Embedding run: {embedding_run_id}")
    print(f"    Model: {model_id}")
    print(f"    Chunking method: {chunking_method}")
    print(f"    Chunking strategy: {chunking_strategy}")
    print(f"    Number of embeddings: {embedding_count}")
    print(f"    Sampled pairs: {pair_sample_size}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
