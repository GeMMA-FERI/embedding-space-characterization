from __future__ import annotations

import pathlib
import random
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from embedding_geometry.classification.utils import (
    evaluate_binary_task,
    histogram_nb_predict,
    sample_different_article_pairs,
    split_article_groups,
)
from embedding_geometry.classification.summarize_naive_bayes import (
    complete_result_keys,
)
from embedding_geometry.summarize.merge_tables import (
    calculate_degradation_share,
    calculate_retained_similarity,
    format_metric,
    ordered_latex_columns,
)


def test_article_split_is_disjoint_and_deterministic() -> None:
    groups = [[index * 2, index * 2 + 1] for index in range(10)] + [[100], [101]]
    first = split_article_groups(
        groups, test_fraction=0.2, rng=random.Random(42)
    )
    second = split_article_groups(
        groups, test_fraction=0.2, rng=random.Random(42)
    )

    assert first == second
    assert {index for group in first[0] for index in group}.isdisjoint(
        index for group in first[1] for index in group
    )
    assert len(first[0]) == 10
    assert len(first[1]) == 2
    assert sorted(group[0] for partition in first for group in partition) == sorted(
        group[0] for group in groups
    )


def test_negative_pairs_always_cross_articles_and_respect_limit() -> None:
    groups = [[0, 1, 2], [3, 4], [5, 6]]
    article_by_chunk = {
        chunk: article
        for article, group in enumerate(groups)
        for chunk in group
    }
    pairs = sample_different_article_pairs(
        groups, 8, rng=random.Random(7)
    )

    assert len(pairs) == 8
    assert len(set(pairs)) == 8
    assert all(article_by_chunk[a] != article_by_chunk[b] for a, b in pairs)


def test_both_naive_bayes_models_separate_synthetic_distributions() -> None:
    train_negative = torch.linspace(0.05, 0.25, 80)
    train_positive = torch.linspace(0.70, 0.90, 80)
    test_negative = torch.linspace(0.08, 0.22, 20)
    test_positive = torch.linspace(0.72, 0.88, 20)

    metrics = evaluate_binary_task(
        train_negative,
        train_positive,
        test_negative,
        test_positive,
        seed=42,
        gaussian_var_smoothing=1e-9,
        histogram_bins=20,
        histogram_smoothing=1.0,
    )

    assert metrics["gaussian"]["accuracy"] == pytest.approx(1.0)
    assert metrics["histogram"]["accuracy"] == pytest.approx(1.0)
    assert metrics["samples"] == {"train_per_class": 80, "test_per_class": 20}


def test_histogram_smoothing_handles_unseen_test_bins() -> None:
    predictions = histogram_nb_predict(
        torch.tensor([0.0, 0.1]),
        torch.tensor([0.8, 0.9]),
        torch.tensor([0.45]),
        bins=20,
        smoothing=1.0,
    )
    assert predictions.shape == (1,)


def test_classification_metrics_are_last_in_merged_latex_table() -> None:
    assert [column for _, column, _, _ in ordered_latex_columns()][-2:] == [
        "gaussian_negative_positive_accuracy",
        "gaussian_negative_transition_accuracy",
    ]


def test_latex_metric_can_be_formatted_as_percentage() -> None:
    assert format_metric("0.72457", as_percent=True) == r"72.46\%"
    assert format_metric("0.72457") == "0.7246"
    assert format_metric("", as_percent=True) == "--"


def test_degradation_share_is_bounded_fraction() -> None:
    ratio = calculate_degradation_share(
        mean_representation_degradation="0.08",
        mean_negative_cosine_similarity="0.25",
        mean_positive_cosine_similarity="0.65",
    )

    assert ratio == pytest.approx(1 / 6)
    assert format_metric(str(ratio), as_percent=True) == r"16.67\%"


def test_degradation_share_is_missing_for_negative_similarity_gap(
) -> None:
    assert calculate_degradation_share(
        mean_representation_degradation="0.08",
        mean_negative_cosine_similarity="0.6",
        mean_positive_cosine_similarity="0.5",
    ) is None


def test_degradation_share_is_one_for_zero_similarity_gap() -> None:
    assert calculate_degradation_share(
        mean_representation_degradation="0.08",
        mean_negative_cosine_similarity="0.5",
        mean_positive_cosine_similarity="0.5",
    ) == pytest.approx(1.0)


def test_retained_similarity_is_normalized_above_random_baseline() -> None:
    retained = calculate_retained_similarity(
        mean_representation_degradation="0.08",
        mean_negative_cosine_similarity="0.25",
    )

    assert retained == pytest.approx((0.75 - 0.08) / 0.75)
    assert format_metric(str(retained), as_percent=True) == r"89.33\%"


def test_retained_similarity_is_missing_without_available_range() -> None:
    assert calculate_retained_similarity(
        mean_representation_degradation="0.08",
        mean_negative_cosine_similarity="1.0",
    ) is None


def test_classification_summary_completes_model_strategy_grid() -> None:
    results = {
        ("model-a", "strategy-1"): {},
        ("model-a", "strategy-2"): {},
        ("model-b", "strategy-1"): {},
    }
    assert complete_result_keys(results) == [
        ("model-a", "strategy-1"),
        ("model-a", "strategy-2"),
        ("model-b", "strategy-1"),
        ("model-b", "strategy-2"),
    ]
