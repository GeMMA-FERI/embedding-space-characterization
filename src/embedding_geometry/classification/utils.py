from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SimilaritySplit:
    negative: torch.Tensor
    positive: torch.Tensor
    transition: torch.Tensor


def split_article_groups(
    article_groups: list[list[int]],
    *,
    test_fraction: float,
    rng: random.Random,
) -> tuple[list[list[int]], list[list[int]]]:
    """Split eligible articles so no article occurs in both train and test."""
    groups = [group for group in article_groups if group]
    eligible = [group for group in groups if len(group) >= 2]
    singletons = [group for group in groups if len(group) == 1]
    if len(eligible) < 4:
        raise ValueError(
            "At least four articles with two or more chunks are required for "
            "article-disjoint training and testing."
        )

    rng.shuffle(eligible)
    rng.shuffle(singletons)
    eligible_test_count = round(len(eligible) * test_fraction)
    eligible_test_count = min(max(eligible_test_count, 2), len(eligible) - 2)
    singleton_test_count = round(len(singletons) * test_fraction)
    train = eligible[eligible_test_count:] + singletons[singleton_test_count:]
    test = eligible[:eligible_test_count] + singletons[:singleton_test_count]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def sample_different_article_pairs(
    article_groups: list[list[int]],
    pair_limit: int,
    *,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Uniformly sample unique chunk pairs conditional on different articles."""
    groups = [group for group in article_groups if group]
    total_chunks = sum(len(group) for group in groups)
    total_pairs = total_chunks * (total_chunks - 1) // 2
    within_pairs = sum(len(group) * (len(group) - 1) // 2 for group in groups)
    cross_pairs = total_pairs - within_pairs
    target = min(pair_limit, cross_pairs)
    if target < 1:
        return []

    if target == cross_pairs:
        return [
            (first, second)
            for group_index, group in enumerate(groups)
            for other_group in groups[group_index + 1 :]
            for first in group
            for second in other_group
        ]

    chunks = [index for group in groups for index in group]
    article_by_chunk = {
        index: article_index
        for article_index, group in enumerate(groups)
        for index in group
    }
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < target:
        first, second = rng.sample(chunks, 2)
        if article_by_chunk[first] == article_by_chunk[second]:
            continue
        pairs.add((min(first, second), max(first, second)))
    return list(pairs)


def pair_similarities(
    normalized_embeddings: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> torch.Tensor:
    if not pairs:
        return torch.empty(0)
    similarities: list[torch.Tensor] = []
    for start in range(0, len(pairs), 10_000):
        batch = pairs[start : start + 10_000]
        first = torch.tensor([pair[0] for pair in batch], dtype=torch.long)
        second = torch.tensor([pair[1] for pair in batch], dtype=torch.long)
        similarities.append(
            (normalized_embeddings[first] * normalized_embeddings[second]).sum(dim=1)
        )
    return torch.cat(similarities)


def sample_transition_similarities(
    normalized_embeddings: torch.Tensor,
    article_groups: list[list[int]],
    pair_limit: int,
    *,
    rng: random.Random,
) -> torch.Tensor:
    transitions = [
        (group[index], group[index + 1])
        for group in article_groups
        for index in range(len(group) - 1)
    ]
    if len(transitions) > pair_limit:
        transitions = rng.sample(transitions, pair_limit)
    return pair_similarities(normalized_embeddings, transitions)


def sample_similarity_split(
    normalized_embeddings: torch.Tensor,
    article_groups: list[list[int]],
    pair_limit: int,
    *,
    seed: int,
) -> SimilaritySplit:
    from embedding_geometry.utils.evaluation_functions import (
        sample_same_article_similarities,
    )

    return SimilaritySplit(
        negative=pair_similarities(
            normalized_embeddings,
            sample_different_article_pairs(
                article_groups, pair_limit, rng=random.Random(seed)
            ),
        ),
        positive=sample_same_article_similarities(
            normalized_embeddings,
            article_groups,
            pair_limit,
            rng=random.Random(seed + 1),
            already_normalized=True,
        ),
        transition=sample_transition_similarities(
            normalized_embeddings,
            article_groups,
            pair_limit,
            rng=random.Random(seed + 2),
        ),
    )


def balance_classes(
    negative: torch.Tensor,
    other: torch.Tensor,
    *,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = min(len(negative), len(other))
    if count < 1:
        raise ValueError("Both classes must contain at least one similarity.")

    def select(values: torch.Tensor) -> torch.Tensor:
        if len(values) == count:
            return values
        indices = torch.tensor(rng.sample(range(len(values)), count))
        return values[indices]

    return select(negative), select(other)


def gaussian_nb_predict(
    train_negative: torch.Tensor,
    train_other: torch.Tensor,
    test_values: torch.Tensor,
    *,
    var_smoothing: float,
) -> torch.Tensor:
    """Fit a scalar Gaussian NB model and predict 0=negative, 1=other."""
    means = torch.stack((train_negative.mean(), train_other.mean()))
    variances = torch.stack(
        (
            train_negative.var(unbiased=False),
            train_other.var(unbiased=False),
        )
    )
    epsilon = max(float(variances.max().item()) * var_smoothing, 1e-12)
    variances = variances + epsilon
    values = test_values[:, None]
    log_likelihood = -0.5 * (
        torch.log(2 * math.pi * variances) + (values - means) ** 2 / variances
    )
    return log_likelihood.argmax(dim=1)


def histogram_nb_predict(
    train_negative: torch.Tensor,
    train_other: torch.Tensor,
    test_values: torch.Tensor,
    *,
    bins: int,
    smoothing: float,
) -> torch.Tensor:
    """Fit a fixed-bin empirical NB model and predict 0=negative, 1=other."""
    edges = torch.linspace(-1.0, 1.0, bins + 1)

    def probabilities(values: torch.Tensor) -> torch.Tensor:
        indices = torch.bucketize(values.clamp(-1.0, 1.0), edges[1:-1])
        counts = torch.bincount(indices, minlength=bins).to(torch.float64)
        return (counts + smoothing) / (len(values) + smoothing * bins)

    probabilities_by_class = torch.stack(
        (probabilities(train_negative), probabilities(train_other))
    )
    test_bins = torch.bucketize(test_values.clamp(-1.0, 1.0), edges[1:-1])
    return probabilities_by_class[:, test_bins].log().argmax(dim=0)


def classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float | int]:
    true_negative = int(((predictions == 0) & (labels == 0)).sum().item())
    false_positive = int(((predictions == 1) & (labels == 0)).sum().item())
    false_negative = int(((predictions == 0) & (labels == 1)).sum().item())
    true_positive = int(((predictions == 1) & (labels == 1)).sum().item())
    return {
        "accuracy": float((predictions == labels).float().mean().item()),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


def evaluate_binary_task(
    train_negative: torch.Tensor,
    train_other: torch.Tensor,
    test_negative: torch.Tensor,
    test_other: torch.Tensor,
    *,
    seed: int,
    gaussian_var_smoothing: float,
    histogram_bins: int,
    histogram_smoothing: float,
) -> dict[str, dict[str, float | int]]:
    train_negative, train_other = balance_classes(
        train_negative, train_other, rng=random.Random(seed)
    )
    test_negative, test_other = balance_classes(
        test_negative, test_other, rng=random.Random(seed + 1)
    )
    test_values = torch.cat((test_negative, test_other))
    labels = torch.cat(
        (
            torch.zeros(len(test_negative), dtype=torch.long),
            torch.ones(len(test_other), dtype=torch.long),
        )
    )
    gaussian_predictions = gaussian_nb_predict(
        train_negative,
        train_other,
        test_values,
        var_smoothing=gaussian_var_smoothing,
    )
    histogram_predictions = histogram_nb_predict(
        train_negative,
        train_other,
        test_values,
        bins=histogram_bins,
        smoothing=histogram_smoothing,
    )
    return {
        "gaussian": classification_metrics(gaussian_predictions, labels),
        "histogram": classification_metrics(histogram_predictions, labels),
        "samples": {
            "train_per_class": len(train_negative),
            "test_per_class": len(test_negative),
        },
    }
