from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from embedding_geometry.evaluation.evaluate_representation_degradation import (
    parse_args,
    resolve_article_embedding_run,
    select_chunk_embedding_runs,
)


def embedding_run(
    run_id: str,
    *,
    model_id: str = "model",
    chunking_run_id: str,
    method: str,
    count: int = 10,
) -> dict[str, object]:
    return {
        "embedding_run_id": run_id,
        "model_id": model_id,
        "chunking_run_ids": chunking_run_id,
        "chunking_methods": method,
        "embedding_count": count,
    }


def test_model_selection_returns_every_non_noop_run_in_chunking_order() -> None:
    runs = [
        embedding_run("model-token", chunking_run_id="token", method="token"),
        embedding_run(
            "other-token",
            model_id="other",
            chunking_run_id="token",
            method="token",
        ),
        embedding_run("model-noop", chunking_run_id="noop", method="noop"),
        embedding_run("model-paragraph", chunking_run_id="paragraph", method="paragraph"),
    ]

    selected = select_chunk_embedding_runs(
        runs,
        embedding_run_id=None,
        model_id="model",
    )

    assert [run["embedding_run_id"] for run in selected] == [
        "model-paragraph",
        "model-token",
    ]


def test_model_selection_rejects_duplicate_runs_for_one_chunking() -> None:
    runs = [
        embedding_run("token-a", chunking_run_id="token", method="token"),
        embedding_run("token-b", chunking_run_id="token", method="token"),
    ]

    with pytest.raises(ValueError, match="Multiple embedding runs"):
        select_chunk_embedding_runs(
            runs,
            embedding_run_id=None,
            model_id="model",
        )


def test_noop_baseline_is_resolved_for_model() -> None:
    baseline = embedding_run(
        "model-noop",
        chunking_run_id="noop",
        method="noop",
    )

    assert resolve_article_embedding_run(
        [baseline],
        model_id="model",
        requested_run_id=None,
    ) is baseline


def test_cli_defaults_to_five_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_representation_degradation", "--model-id", "model"],
    )

    args = parse_args()

    assert args.model_id == "model"
    assert args.embedding_run_id is None
    assert args.min_chunks_per_article == 5
