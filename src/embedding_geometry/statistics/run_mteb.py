from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence

import mteb

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from semora.embeddings.registry import get_embedder


BENCHMARK_NAME = "MTEB(Multilingual, v2)"
TASK_TYPE = "Retrieval"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval tasks from the multilingual MTEB benchmark."
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace model ID of the embedding model to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--output-file",
        default="data/mteb_results.json",
        help="Path to save the MTEB result report JSON.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/mteb_cache",
        help="Persistent MTEB cache used to resume completed tasks and splits.",
    )
    return parser.parse_args(argv)


def build_result_report(results, task_names: list[str]) -> dict[str, object]:
    """Build a serializable report that makes partial completion explicit."""
    report = results.model_dump(mode="json")
    exceptions = results.exceptions or []
    report.update(
        {
            "benchmark": BENCHMARK_NAME,
            "task_type": TASK_TYPE,
            "requested_tasks": task_names,
            "completed_tasks": [result.task_name for result in results.task_results],
            "failed_tasks": [error.task_name for error in exceptions],
            "is_complete": not exceptions and len(results.task_results) == len(task_names),
        }
    )
    return report


def write_json_atomic(output_path: pathlib.Path, payload: dict[str, object]) -> None:
    """Replace the report only after its JSON has been written successfully."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(output_path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")

    embedder = get_embedder(args.model_id)
    embedder.load()

    benchmark = mteb.get_benchmark(BENCHMARK_NAME)
    tasks = mteb.filter_tasks(benchmark.tasks, task_types=[TASK_TYPE])
    if not tasks:
        raise ValueError(f"No {TASK_TYPE} tasks found in {BENCHMARK_NAME}.")
    task_names = [task.metadata.name for task in tasks]
    print(f"Running {len(tasks)} {TASK_TYPE} tasks for {args.model_id}.")

    cache = mteb.ResultCache(cache_path=args.cache_dir)
    results = mteb.evaluate(
        embedder.model,
        tasks=tasks,
        raise_error=False,
        cache=cache,
        overwrite_strategy="only-missing",
        encode_kwargs={"batch_size": args.batch_size},
    )

    report = build_result_report(results, task_names)
    output_path = pathlib.Path(args.output_file)
    write_json_atomic(output_path, report)

    exceptions = results.exceptions or []
    print(
        f"Completed {len(results.task_results)}/{len(tasks)} tasks; "
        f"failed={len(exceptions)}."
    )
    for error in exceptions:
        print(f"Task failed: {error.task_name}: {error.exception}", file=sys.stderr)
    print(f"Result report: {output_path.resolve()}")
    print(f"Resume cache: {pathlib.Path(args.cache_dir).resolve()}")


if __name__ == "__main__":
    main()
