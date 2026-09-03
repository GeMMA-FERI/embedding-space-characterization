from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass

import torch
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from semora.embeddings.serialization import tensor_from_float32blob
from semora.storage import Database


@dataclass
class NormStats:
    count: int = 0
    finite_count: int = 0
    normalized_count: int = 0
    norm_sum: float = 0.0
    norm_square_sum: float = 0.0
    norm_min: float = float("inf")
    norm_max: float = float("-inf")

    def update(self, norms: torch.Tensor, tolerance: float) -> None:
        self.count += norms.numel()
        finite = norms[torch.isfinite(norms)]
        self.finite_count += finite.numel()
        if finite.numel() == 0:
            return
        self.normalized_count += int((finite.sub(1.0).abs() <= tolerance).sum())
        self.norm_sum += float(finite.double().sum())
        self.norm_square_sum += float(finite.double().square().sum())
        self.norm_min = min(self.norm_min, float(finite.min()))
        self.norm_max = max(self.norm_max, float(finite.max()))

    def merge(self, other: "NormStats") -> None:
        self.count += other.count
        self.finite_count += other.finite_count
        self.normalized_count += other.normalized_count
        self.norm_sum += other.norm_sum
        self.norm_square_sum += other.norm_square_sum
        self.norm_min = min(self.norm_min, other.norm_min)
        self.norm_max = max(self.norm_max, other.norm_max)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report how many stored embeddings have unit L2 norm."
    )
    parser.add_argument("--db-path", default="./data/newspapers.sqlite")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--chunking-run-id", required=True)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="A vector is normalized when abs(L2 norm - 1) is at most this value.",
    )
    parser.add_argument("--batch-size", type=int, default=32000)
    return parser.parse_args()


def _analyze_rows(
    cursor,
    *,
    batch_size: int,
    tolerance: float,
    progress: tqdm | None = None,
) -> NormStats:
    stats = NormStats()
    dimension: int | None = None
    while rows := cursor.fetchmany(batch_size):
        vectors = []
        for row in rows:
            vector = tensor_from_float32blob(
                row["tensor_blob"], expected_dimension=dimension
            )
            dimension = dimension or vector.numel()
            vectors.append(vector)
        stats.update(torch.linalg.vector_norm(torch.stack(vectors), dim=1), tolerance)
        if progress is not None:
            progress.update(len(rows))
    return stats


def _print_stats(label: str, stats: NormStats) -> None:
    normalized_fraction = stats.normalized_count / stats.count
    print(label)
    print(f"  Vectors: {stats.count:,}")
    print(
        f"  Normalized: {stats.normalized_count:,}/{stats.count:,} "
        f"({normalized_fraction:.2%})"
    )
    print(f"  Non-finite norms: {stats.count - stats.finite_count:,}")
    if stats.finite_count:
        mean = stats.norm_sum / stats.finite_count
        variance = max(stats.norm_square_sum / stats.finite_count - mean**2, 0.0)
        print(
            f"  L2 norm min/mean/max: {stats.norm_min:.8g} / "
            f"{mean:.8g} / {stats.norm_max:.8g}"
        )
        print(f"  L2 norm standard deviation: {math.sqrt(variance):.8g}")


def main() -> None:
    args = parse_args()
    if args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    db = Database(args.db_path)
    total = NormStats()
    matched_runs = 0
    try:
        db.initialize()
        runs = db.get_embedding_runs_by_model_and_chunking_run(
            args.model_id,
            args.chunking_run_id,
        )
        if not runs:
            raise ValueError("No embedding runs match the selected model and chunking run.")

        print(f"Model: {args.model_id}")
        print(f"Chunking run: {args.chunking_run_id}")
        print(f"Unit-norm tolerance: {args.tolerance:g}")
        for run in runs:
            with tqdm(
                desc=f"Checking {run['embedding_run_id']}",
                unit="vector",
                unit_scale=True,
            ) as progress:
                stats = _analyze_rows(
                    db.iter_embeddings_for_run_fast(run["embedding_run_id"]),
                    batch_size=args.batch_size,
                    tolerance=args.tolerance,
                    progress=progress,
                )
            if stats.count == 0:
                continue
            matched_runs += 1
            total.merge(stats)
    finally:
        db.close()

    if matched_runs == 0:
        raise ValueError("No embeddings match the selected model and chunking run.")
    print(f"Matching embedding runs: {matched_runs}")
    _print_stats("Results:", total)


if __name__ == "__main__":
    main()
