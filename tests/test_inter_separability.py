from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from embedding_geometry.evaluation.evaluate_inter_separability import (
    calculate_davies_bouldin_diagnostics,
    sample_article_cohorts,
)
from semora.storage import (
    Article,
    Chunk,
    ChunkingRun,
    Database,
    Embedding,
    EmbeddingRun,
    Newspaper,
    Run,
)


def test_davies_bouldin_trimmed_diagnostic_exposes_extreme_pair() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [1.0, 0.000001],
            [0.99, 0.010001],
            [0.0, 1.0],
            [0.01, 0.99],
        ],
        dtype=torch.float64,
    )
    diagnostics = calculate_davies_bouldin_diagnostics(
        embeddings,
        ["a", "a", "b", "b", "c", "c"],
        block_size=2,
        trim_fraction=0.34,
    )

    assert diagnostics is not None
    assert diagnostics.minimum_centroid_distance < 0.00001
    assert diagnostics.maximum_ratio > 1_000
    assert diagnostics.trimmed_index < diagnostics.index
    assert diagnostics.top_1pct_contribution > 0.3


def test_seeded_article_cohorts_are_reproducible() -> None:
    articles = [f"article-{index:02d}" for index in range(20)]
    first = sample_article_cohorts(articles, cohort_size=10, seeds=[42, 43])
    second = sample_article_cohorts(
        list(reversed(articles)), cohort_size=10, seeds=[42, 43]
    )

    assert first == second
    assert first[0] != first[1]
    with pytest.raises(ValueError, match="only 20 shared eligible"):
        sample_article_cohorts(articles, cohort_size=21, seeds=[42])


def test_shared_eligible_articles_require_enough_embeddings_in_every_run(
    tmp_path: pathlib.Path,
) -> None:
    db = Database(tmp_path / "shared-cohort.sqlite")
    try:
        db.initialize()
        db.insert_run(Run(run_id="run", run_type="test"))
        db.insert_newspaper(
            Newspaper(newspaper_id="paper", run_id="run", content="paper")
        )
        for article_id in ("a", "b", "c"):
            db.insert_article(
                Article(
                    article_id=article_id,
                    run_id="run",
                    newspaper_id="paper",
                    title=article_id,
                    content=article_id,
                )
            )
            db.update_article_validation(article_id, is_valid=True)

        counts_by_run = {
            "embedding-1": {"a": 2, "b": 2, "c": 1},
            "embedding-2": {"a": 2, "b": 1, "c": 2},
        }
        for run_index, (embedding_run_id, counts) in enumerate(
            counts_by_run.items(), start=1
        ):
            chunking_run_id = f"chunking-{run_index}"
            db.insert_chunking_run(
                ChunkingRun(
                    chunking_run_id=chunking_run_id,
                    run_id="run",
                    method="token",
                )
            )
            db.insert_embedding_run(
                EmbeddingRun(
                    embedding_run_id=embedding_run_id,
                    run_id="run",
                    model_id=f"model-{run_index}",
                    config={"chunking_run_id": chunking_run_id},
                )
            )
            for article_id, count in counts.items():
                for chunk_index in range(count):
                    chunk_id = f"{chunking_run_id}-{article_id}-{chunk_index}"
                    db.insert_chunk(
                        Chunk(
                            chunk_id=chunk_id,
                            run_id="run",
                            article_id=article_id,
                            chunking_run_id=chunking_run_id,
                            chunk_index=chunk_index,
                            method="token",
                            text=chunk_id,
                        )
                    )
                    db.insert_embeddings(
                        [
                            Embedding(
                                embedding_id=f"{embedding_run_id}-{chunk_id}",
                                embedding_run_id=embedding_run_id,
                                chunk_id=chunk_id,
                                tensor_blob=b"vector",
                            )
                        ]
                    )

        assert db.get_shared_eligible_embedding_article_ids(
            ["embedding-1", "embedding-2"], min_chunks=2
        ) == ["a"]
    finally:
        db.close()
