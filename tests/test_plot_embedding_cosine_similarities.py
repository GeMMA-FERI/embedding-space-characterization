from __future__ import annotations

import pathlib
import random
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from embedding_geometry.statistics.plot_embedding_cosine_similarities import (
    build_similarity_histogram_rows,
    embedding_component_diagnostics,
    load_similarity_histogram_csv,
    resolve_embedding_run_names,
    sample_same_article_similarities,
    sample_transition_similarities,
    save_binned_similarity_plot,
    save_similarity_histogram,
    save_similarity_histogram_csv,
)
from embedding_geometry.utils.evaluation_functions import (
    calculate_pairwise_cosine_similarities,
    pair_indices_from_ranks,
)


def test_pair_rank_mapping_covers_each_unique_pair() -> None:
    first, second = pair_indices_from_ranks(4, range(6))
    assert list(zip(first.tolist(), second.tolist())) == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]


def test_pairwise_distributions_apply_limits_and_transition_order() -> None:
    matrix = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    groups = [[0, 1, 2], [3, 4]]

    positive = sample_same_article_similarities(
        matrix, groups, 3, rng=random.Random(2)
    )
    transitions = sample_transition_similarities(
        matrix, groups, 100, rng=random.Random(2)
    )
    random_pairs = calculate_pairwise_cosine_similarities(
        matrix, 4, rng=random.Random(2)
    )

    assert len(positive) == 3
    assert transitions.tolist() == pytest.approx(
        [2**-0.5, 2**-0.5, 0.0]
    )
    assert random_pairs is not None
    assert len(random_pairs) == 4


def test_component_diagnostics_report_negative_values() -> None:
    diagnostics = embedding_component_diagnostics(
        torch.tensor([[1.0, -0.5], [0.0, 2.0], [-0.25, -1.0]])
    )
    assert diagnostics == {
        "negative_components": 3,
        "affected_embeddings": 2,
        "minimum_component": -1.0,
        "zero_norm_embeddings": 0,
    }


def test_histograms_overlap_and_log_axis_requires_positive_values(
    tmp_path: pathlib.Path,
) -> None:
    distributions = {
        "run-a": {
            "Positive": torch.tensor([0.8, 0.9]),
            "Transition": torch.tensor([0.7, 0.85]),
            "Negative": torch.tensor([0.1, 0.2]),
        },
        "run-b": {
            "Positive": torch.tensor([0.75, 0.85]),
            "Transition": torch.tensor([0.65, 0.8]),
            "Negative": torch.tensor([0.15, 0.25]),
        },
    }
    output = tmp_path / "plot.png"
    save_similarity_histogram(
        distributions,
        output=output,
        bins=5,
        log_x=True,
        title="Test",
    )
    assert output.is_file()

    distributions["run-b"]["Negative"] = torch.tensor([-0.1, 0.2])
    with pytest.raises(ValueError, match="minimum observed similarity"):
        save_similarity_histogram(
            distributions,
            output=output,
            bins=5,
            log_x=True,
            title="Test",
        )


def test_histogram_csv_can_be_plotted_without_sample_vectors(
    tmp_path: pathlib.Path,
) -> None:
    distributions = {
        "run-a": {
            "Positive": torch.tensor([0.7, 0.8]),
            "Transition": torch.tensor([0.6, 0.7]),
            "Negative": torch.tensor([0.2, 0.3]),
        }
    }
    rows = build_similarity_histogram_rows(
        distributions,
        model_ids={"run-a": "model-a"},
        run_names={"run-a": "Readable A"},
        bins=4,
        log_x=False,
    )
    csv_path = tmp_path / "histograms.csv"
    save_similarity_histogram_csv(rows, csv_path)
    loaded = load_similarity_histogram_csv(csv_path)

    assert len(loaded) == 12
    assert {row["distribution"] for row in loaded} == {
        "Positive",
        "Transition",
        "Negative",
    }
    assert {row["model_id"] for row in loaded} == {"model-a"}
    assert {row["embedding_run_name"] for row in loaded} == {"Readable A"}

    output = tmp_path / "from-csv.png"
    save_binned_similarity_plot(
        loaded,
        output=output,
        log_x=False,
        title="From CSV",
        plot_scale=1.5,
    )
    assert output.is_file()


def test_embedding_run_names_must_match_run_count() -> None:
    assert resolve_embedding_run_names(
        ["run-a", "run-b"], ["Readable A", "Readable B"]
    ) == {
        "run-a": "Readable A",
        "run-b": "Readable B",
    }
    with pytest.raises(ValueError, match="exactly one name"):
        resolve_embedding_run_names(["run-a", "run-b"], ["Only one"])
