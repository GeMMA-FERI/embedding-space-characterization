from __future__ import annotations

import json
import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from embedding_geometry.statistics import run_mteb


class FakeResults:
    def __init__(self) -> None:
        self.task_results = [SimpleNamespace(task_name="SuccessfulRetrieval")]
        self.exceptions = [
            SimpleNamespace(task_name="BrokenRetrieval", exception="dataset failure")
        ]

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {"model_name": "model", "task_results": [{"score": 1.0}]}


def test_main_filters_retrieval_tasks_and_saves_partial_results(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    tasks = [
        SimpleNamespace(metadata=SimpleNamespace(name="SuccessfulRetrieval")),
        SimpleNamespace(metadata=SimpleNamespace(name="BrokenRetrieval")),
    ]
    benchmark = SimpleNamespace(tasks=[object(), object(), object()])
    embedder = SimpleNamespace(model=object(), load=lambda: None)
    captured: dict[str, object] = {}

    monkeypatch.setattr(run_mteb, "get_embedder", lambda model_id: embedder)
    monkeypatch.setattr(run_mteb.mteb, "get_benchmark", lambda name: benchmark)

    def filter_tasks(benchmark_tasks, *, task_types):
        assert benchmark_tasks is benchmark.tasks
        assert task_types == ["Retrieval"]
        return tasks

    monkeypatch.setattr(run_mteb.mteb, "filter_tasks", filter_tasks)
    monkeypatch.setattr(
        run_mteb.mteb,
        "ResultCache",
        lambda *, cache_path: SimpleNamespace(cache_path=cache_path),
    )

    def evaluate(model, **kwargs):
        captured.update(kwargs)
        return FakeResults()

    monkeypatch.setattr(run_mteb.mteb, "evaluate", evaluate)
    output = tmp_path / "results" / "model.json"
    cache_dir = tmp_path / "cache"

    run_mteb.main(
        [
            "--model-id",
            "model",
            "--batch-size",
            "8",
            "--output-file",
            str(output),
            "--cache-dir",
            str(cache_dir),
        ]
    )

    assert captured["tasks"] == tasks
    assert captured["raise_error"] is False
    assert captured["overwrite_strategy"] == "only-missing"
    assert captured["encode_kwargs"] == {"batch_size": 8}
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["completed_tasks"] == ["SuccessfulRetrieval"]
    assert report["failed_tasks"] == ["BrokenRetrieval"]
    assert report["is_complete"] is False
    assert not output.with_suffix(".json.tmp").exists()
