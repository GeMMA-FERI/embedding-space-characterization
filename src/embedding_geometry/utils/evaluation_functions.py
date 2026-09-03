from __future__ import annotations

import csv
import json
import random
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from semora.embeddings.serialization import tensor_from_float32blob


def build_chunking_strategy(method: str, config_json: str | None) -> str:
    """Build a stable strategy key from a chunking method and its configuration."""
    config = json.loads(config_json or "{}")

    if method in {"token", "token_window"}:
        token_count = config.get("token_count")
        token_overlap = config.get("token_overlap", 0)
        if token_count is not None:
            strategy = f"token_window_{token_count}"
            if token_overlap:
                strategy += f"_overlap_{token_overlap}"
            return strategy

    return method


def format_chunking_strategy(strategy: str) -> str:
    """Return a human-readable label for a chunking strategy key."""
    if strategy.startswith("token_window_"):
        remainder = strategy.removeprefix("token_window_")
        token_count, separator, overlap = remainder.partition("_overlap_")
        label = f"Token window ({token_count} tokens"
        if separator:
            label += f", {overlap} overlap"
        return label + ")"

    return strategy.replace("_", " ").capitalize()


def normalize_embeddings(embedding_matrix: torch.Tensor) -> torch.Tensor:
    """Normalize one embedding vector or an embedding matrix to unit length."""
    if embedding_matrix.dim() == 1:
        return F.normalize(embedding_matrix, p=2, dim=0)
    return F.normalize(embedding_matrix, p=2, dim=1)


def sample_unique_pair_indices(
    item_count: int,
    sample_size: int,
    *,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample unique unordered pairs uniformly without materializing all pairs."""
    if item_count < 2 or sample_size < 1:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty

    total_pairs = item_count * (item_count - 1) // 2
    selected_count = min(sample_size, total_pairs)
    generator = rng or random
    ranks = (
        range(total_pairs)
        if selected_count == total_pairs
        else generator.sample(range(total_pairs), selected_count)
    )

    return pair_indices_from_ranks(item_count, ranks)


def pair_indices_from_ranks(
    item_count: int,
    ranks: Iterable[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map ranks in the upper triangle to unique unordered item pairs."""
    total_pairs = item_count * (item_count - 1) // 2
    starts = [index * (2 * item_count - index - 1) // 2 for index in range(item_count)]
    first_indices: list[int] = []
    second_indices: list[int] = []
    for rank in ranks:
        if rank < 0 or rank >= total_pairs:
            raise ValueError(f"Pair rank {rank} is outside [0, {total_pairs}).")
        first = bisect_right(starts, rank) - 1
        second = first + 1 + rank - starts[first]
        first_indices.append(first)
        second_indices.append(second)
    return torch.tensor(first_indices), torch.tensor(second_indices)


def calculate_pairwise_cosine_similarities(
    embedding_matrix: torch.Tensor,
    sample_size: int,
    *,
    rng: random.Random | None = None,
    already_normalized: bool = False,
) -> torch.Tensor | None:
    """Return cosine similarities for uniformly sampled unique unordered pairs."""
    if len(embedding_matrix) < 2:
        return None
    idx1, idx2 = sample_unique_pair_indices(
        len(embedding_matrix),
        sample_size,
        rng=rng,
    )
    normalized = embedding_matrix if already_normalized else normalize_embeddings(embedding_matrix)
    similarities = []
    for start in range(0, len(idx1), 10_000):
        end = start + 10_000
        similarities.append(
            (normalized[idx1[start:end]] * normalized[idx2[start:end]]).sum(dim=1)
        )
    return torch.cat(similarities) if similarities else torch.empty(0)


def sample_same_article_similarities(
    embedding_matrix: torch.Tensor,
    article_groups: list[list[int]],
    pair_limit: int,
    *,
    rng: random.Random,
    already_normalized: bool = False,
) -> torch.Tensor:
    """Uniformly sample unordered pairs from the full within-article pair set."""
    eligible_groups = [group for group in article_groups if len(group) >= 2]
    pair_counts = [len(group) * (len(group) - 1) // 2 for group in eligible_groups]
    cumulative: list[int] = []
    running_total = 0
    for pair_count in pair_counts:
        running_total += pair_count
        cumulative.append(running_total)
    if running_total == 0:
        return torch.empty(0)

    selected_count = min(pair_limit, running_total)
    ranks = (
        range(running_total)
        if selected_count == running_total
        else rng.sample(range(running_total), selected_count)
    )
    selected_by_group: dict[int, list[int]] = defaultdict(list)
    for rank in ranks:
        group_index = bisect_right(cumulative, rank)
        previous_total = cumulative[group_index - 1] if group_index else 0
        selected_by_group[group_index].append(rank - previous_total)

    normalized = (
        embedding_matrix
        if already_normalized
        else normalize_embeddings(embedding_matrix)
    )
    similarities: list[torch.Tensor] = []
    for group_index, local_ranks in selected_by_group.items():
        group = eligible_groups[group_index]
        local_first, local_second = pair_indices_from_ranks(len(group), local_ranks)
        global_indices = torch.tensor(group, dtype=torch.long)
        first = global_indices[local_first]
        second = global_indices[local_second]
        for start in range(0, len(first), 10_000):
            end = start + 10_000
            similarities.append(
                (normalized[first[start:end]] * normalized[second[start:end]]).sum(
                    dim=1
                )
            )
    return torch.cat(similarities) if similarities else torch.empty(0)


def calculate_transition_similarities(
    embeddings: list[torch.Tensor],
) -> torch.Tensor | None:
    """Return cosine similarities between embeddings in consecutive order."""
    if len(embeddings) < 2:
        return None
    embedding_matrix = torch.stack(embeddings)
    return torch.cosine_similarity(
        embedding_matrix[:-1],
        embedding_matrix[1:],
        dim=1,
    )


def load_embeddings(embedding_rows: Iterable) -> list[torch.Tensor]:
    """Convert database rows with tensor_blob values into tensors."""
    return [
        tensor_from_float32blob(row["tensor_blob"])
        for row in embedding_rows
    ]


def save_results_to_csv(results: list[dict], output_path: str | Path) -> None:
    """Save a non-empty list of result dictionaries to CSV."""
    if not results:
        raise ValueError("Cannot save an empty result set.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def print_runs(runs) -> None:
    """Print embedding runs in the common evaluation-script format."""
    for run in runs:
        methods = run["chunking_methods"] or "unknown"
        chunking_runs = run["chunking_run_ids"] or "no embedded chunks"
        print(
            f"{run['embedding_run_id']} | {run['model_id']} | "
            f"{run['embedding_count']} embeddings | chunking={methods} | "
            f"chunking_run={chunking_runs} | {run['created_at']}"
        )


def find_embedding_run(runs, embedding_run_id: str):
    """Return the embedding-run row for an id, or None."""
    return next(
        (run for run in runs if run["embedding_run_id"] == embedding_run_id),
        None
    )


def newest_non_empty_embedding_run_id(runs) -> str:
    """Return the newest non-empty embedding run from an already sorted run list."""
    non_empty_runs = [
        run for run in runs
        if int(run["embedding_count"]) > 0
    ]

    if not non_empty_runs:
        raise ValueError("No stored embeddings found in the database.")

    return non_empty_runs[0]["embedding_run_id"]


def require_known_embedding_run(runs, embedding_run_id: str):
    """Return a known embedding run or raise a consistent error."""
    selected_run = find_embedding_run(runs, embedding_run_id)
    if selected_run is None:
        raise ValueError(f"Unknown embedding run: {embedding_run_id}")
    return selected_run


def group_rows_by_article(rows) -> dict:
    """Group database rows by article_id."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["article_id"]].append(row)
    return grouped


def infer_noop_article_embedding_run_id(
    runs,
    *,
    chunk_embedding_run_id: str
) -> str:
    """Infer the matching noop/full-article embedding run for a chunk run."""
    chunk_run = require_known_embedding_run(runs, chunk_embedding_run_id)
    model_id = chunk_run["model_id"]

    candidates = [
        run
        for run in runs
        if run["embedding_run_id"] != chunk_embedding_run_id
        and run["model_id"] == model_id
        and run["chunking_methods"] == "noop"
        and int(run["embedding_count"]) > 0
    ]

    if len(candidates) == 1:
        return candidates[0]["embedding_run_id"]

    if not candidates:
        raise ValueError(
            "Could not infer a full-article embedding run. "
            f"No non-empty noop embedding run was found for model {model_id!r}. "
            "Pass --article-embedding-run-id explicitly if one exists."
        )

    candidate_ids = ", ".join(row["embedding_run_id"] for row in candidates)
    raise ValueError(
        "Could not infer a single full-article embedding run. "
        f"Candidates: {candidate_ids}. "
        "Pass --article-embedding-run-id explicitly."
    )

def get_non_noop_embedding_run_ids(runs) -> list[str]:
    """Return embedding run IDs that are not noop embedding runs."""
    embedding_run_ids = []

    for run in runs:
        chunking_method = run["chunking_methods"]

        if chunking_method in (None, "", "noop"):
            continue

        embedding_run_ids.append(run["embedding_run_id"])

    return embedding_run_ids
