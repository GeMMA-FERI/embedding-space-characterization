from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import torch

from semora.embeddings.serialization import tensor_from_float32blob
from embedding_geometry.utils.evaluation_functions import (
    calculate_pairwise_cosine_similarities,
    normalize_embeddings,
    require_known_embedding_run,
    sample_same_article_similarities,
)
from semora.storage import Database


DEFAULT_PAIR_LIMIT = 100_000
HISTOGRAM_FIELDS = (
    "embedding_run_id",
    "embedding_run_name",
    "model_id",
    "distribution",
    "bin_index",
    "bin_left",
    "bin_right",
    "bin_center",
    "density",
    "pair_count",
    "log_x",
)


def sample_transition_similarities(
    embedding_matrix: torch.Tensor,
    article_groups: list[list[int]],
    pair_limit: int,
    *,
    rng: random.Random,
    already_normalized: bool = False,
) -> torch.Tensor:
    """Return cosine similarities for sampled consecutive within-article pairs."""
    transitions = [
        (group[index], group[index + 1])
        for group in article_groups
        for index in range(len(group) - 1)
    ]
    if len(transitions) > pair_limit:
        transitions = rng.sample(transitions, pair_limit)
    if not transitions:
        return torch.empty(0)

    normalized = embedding_matrix if already_normalized else normalize_embeddings(embedding_matrix)
    similarities: list[torch.Tensor] = []
    for start in range(0, len(transitions), 10_000):
        batch = transitions[start : start + 10_000]
        first = torch.tensor([pair[0] for pair in batch], dtype=torch.long)
        second = torch.tensor([pair[1] for pair in batch], dtype=torch.long)
        similarities.append((normalized[first] * normalized[second]).sum(dim=1))
    return torch.cat(similarities)


def embedding_component_diagnostics(embedding_matrix: torch.Tensor) -> dict[str, float | int]:
    negative_mask = embedding_matrix < 0
    return {
        "negative_components": int(negative_mask.sum().item()),
        "affected_embeddings": int(negative_mask.any(dim=1).sum().item()),
        "minimum_component": float(embedding_matrix.min().item()),
        "zero_norm_embeddings": int((embedding_matrix.norm(dim=1) == 0).sum().item()),
    }


def build_similarity_histogram_rows(
    distributions: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    model_ids: Mapping[str, str],
    run_names: Mapping[str, str] | None = None,
    bins: int,
    log_x: bool,
) -> list[dict[str, object]]:
    """Bin sampled similarities once so plots can be recreated without SQLite."""
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Install numpy to calculate histogram densities.") from error

    non_empty = [
        values
        for run_distributions in distributions.values()
        for values in run_distributions.values()
        if len(values)
    ]
    if not non_empty:
        raise ValueError("No cosine similarities are available to summarize.")
    combined = torch.cat(non_empty)
    minimum = float(combined.min().item())
    maximum = float(combined.max().item())
    if log_x and minimum <= 0:
        raise ValueError(
            "--log-x requires every plotted cosine similarity to be greater than zero; "
            f"minimum observed similarity is {minimum:.6f}."
        )
    if minimum == maximum:
        if log_x:
            minimum *= 0.99
            maximum *= 1.01
        else:
            padding = max(abs(minimum) * 0.01, 1e-6)
            minimum -= padding
            maximum += padding
    edges = (
        np.geomspace(minimum, maximum, bins + 1)
        if log_x
        else np.linspace(minimum, maximum, bins + 1)
    )
    centers = (
        np.sqrt(edges[:-1] * edges[1:])
        if log_x
        else (edges[:-1] + edges[1:]) / 2
    )

    rows = []
    for run_id, run_distributions in distributions.items():
        for label, values in run_distributions.items():
            if not len(values):
                continue
            density, _ = np.histogram(values.numpy(), bins=edges, density=True)
            for index, value in enumerate(density):
                rows.append(
                    {
                        "embedding_run_id": run_id,
                        "embedding_run_name": (run_names or {}).get(run_id, ""),
                        "model_id": model_ids.get(run_id, ""),
                        "distribution": label,
                        "bin_index": index,
                        "bin_left": float(edges[index]),
                        "bin_right": float(edges[index + 1]),
                        "bin_center": float(centers[index]),
                        "density": float(value),
                        "pair_count": len(values),
                        "log_x": int(log_x),
                    }
                )
    return rows


def save_similarity_histogram_csv(
    rows: list[dict[str, object]],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTOGRAM_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_similarity_histogram_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = tuple(reader.fieldnames or ())
        legacy_fields = tuple(
            field for field in HISTOGRAM_FIELDS if field != "embedding_run_name"
        )
        if fieldnames not in (HISTOGRAM_FIELDS, legacy_fields):
            raise ValueError(
                f"Unexpected histogram CSV columns in {path}; expected {HISTOGRAM_FIELDS}."
            )
        return [
            {
                **row,
                "embedding_run_name": row.get("embedding_run_name", ""),
                "bin_index": int(row["bin_index"]),
                "bin_left": float(row["bin_left"]),
                "bin_right": float(row["bin_right"]),
                "bin_center": float(row["bin_center"]),
                "density": float(row["density"]),
                "pair_count": int(row["pair_count"]),
                "log_x": bool(int(row["log_x"])),
            }
            for row in reader
        ]


def save_binned_similarity_plot(
    rows: list[dict[str, object]],
    *,
    output: Path,
    log_x: bool,
    title: str | None = None,
    run_names: Mapping[str, str] | None = None,
    plot_scale: float = 1.0,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import to_rgb
    except ImportError as error:
        raise RuntimeError("Install matplotlib and numpy to create the plot.") from error

    if not rows:
        raise ValueError("No histogram rows are available to plot.")
    if plot_scale <= 0:
        raise ValueError("plot-scale must be greater than 0")
    minimum = min(float(row["bin_left"]) for row in rows)
    if log_x and minimum <= 0:
        raise ValueError("--log-x requires histogram bins to be greater than zero.")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11, 7))
    colors = {
        "Negative": "tab:green",
        "Positive": "tab:blue",
        "Transition": "tab:orange",
    }
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    run_ids = list(dict.fromkeys(str(row["embedding_run_id"]) for row in rows))
    stored_names = {
        str(row["embedding_run_id"]): str(row["embedding_run_name"])
        for row in rows
        if row.get("embedding_run_name")
    }
    display_names = {**stored_names, **(run_names or {})}
    for row in rows:
        grouped[(str(row["embedding_run_id"]), str(row["distribution"]))].append(row)
    run_count = len(run_ids)
    for run_index, run_id in enumerate(run_ids):
        lighten = 0.3 * run_index / max(run_count - 1, 1)
        for label in colors:
            values = sorted(grouped.get((run_id, label), []), key=lambda row: row["bin_index"])
            if not values:
                continue
            base = np.asarray(to_rgb(colors[label]))
            color = base + (1.0 - base) * lighten
            centers = [float(row["bin_center"]) for row in values]
            density = [float(row["density"]) for row in values]
            widths = [
                float(row["bin_right"]) - float(row["bin_left"])
                for row in values
            ]
            area = sum(value * width for value, width in zip(density, widths))
            if area <= 0:
                raise ValueError(f"Histogram for {run_id} {label} has zero density.")
            # Also upgrades legacy CSVs whose bin values were normalized to sum to 1.
            density = [value / area for value in density]
            axis.plot(
                centers,
                density,
                markersize=3 * plot_scale,
                linewidth=1.5 * plot_scale,
                color=color,
                alpha=0.9,
                label=f"{display_names.get(run_id, run_id)} — {label}",
            )
            axis.fill_between(centers, density, 0, color=color, alpha=0.1)
    if log_x:
        axis.set_xscale("log")
    axis.set_xlabel("Cosine similarity", fontsize=12 * plot_scale)
    axis.set_ylabel("Density", fontsize=12 * plot_scale, labelpad=10)
    if title:
        axis.set_title(title, fontsize=14 * plot_scale)
    axis.set_xlim(left=minimum if log_x else 0, right=1)
    axis.tick_params(
        axis="y",
        labelsize=10
    )
    axis.tick_params(
        axis="x",
        labelsize=10 * plot_scale,
        width=0.8 * plot_scale,
        length=3.5 * plot_scale,
    )
    axis.legend(fontsize=10 * plot_scale)
    axis.grid(alpha=0.2, linewidth=0.8 * plot_scale)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8 * plot_scale)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def save_similarity_histogram(
    distributions: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    output: Path,
    bins: int,
    log_x: bool,
    title: str,
    run_names: Mapping[str, str] | None = None,
    plot_scale: float = 1.0,
) -> None:
    """Backward-compatible helper that bins samples and immediately plots them."""
    rows = build_similarity_histogram_rows(
        distributions,
        model_ids={},
        run_names=run_names,
        bins=bins,
        log_x=log_x,
    )
    save_binned_similarity_plot(
        rows,
        output=output,
        log_x=log_x,
        title=title,
        run_names=run_names,
        plot_scale=plot_scale,
    )


def resolve_embedding_run_names(
    embedding_run_ids: list[str],
    embedding_run_names: list[str] | None,
) -> dict[str, str]:
    if embedding_run_names is None:
        return {}
    if len(embedding_run_names) != len(embedding_run_ids):
        raise ValueError(
            "--embedding-run-names must contain exactly one name for each embedding run."
        )
    return dict(zip(embedding_run_ids, embedding_run_names))


def collect_similarity_distributions(
    db: Database,
    embedding_run_ids: list[str],
    *,
    pair_limit: int,
    random_seed: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, object]]:
    """Load each run in turn and sample its three similarity distributions."""
    selected_runs = db.get_embedding_runs_by_ids(embedding_run_ids)
    run_metadata = {
        run_id: require_known_embedding_run(selected_runs, run_id)
        for run_id in embedding_run_ids
    }
    distributions = {}
    for run_id in embedding_run_ids:
        rows = db.get_embeddings_for_run(run_id)
        if not rows:
            raise ValueError(f"No valid embeddings match run {run_id}.")
        embeddings = [tensor_from_float32blob(row["tensor_blob"]) for row in rows]
        dimensions = {len(embedding) for embedding in embeddings}
        if len(dimensions) != 1:
            raise ValueError(
                f"Embedding run {run_id} contains inconsistent dimensions: {sorted(dimensions)}"
            )
        embedding_matrix = torch.stack(embeddings)
        diagnostics = embedding_component_diagnostics(embedding_matrix)
        print(
            f"{run_id} component check: "
            f"negative_components={diagnostics['negative_components']:,}, "
            f"affected_embeddings={diagnostics['affected_embeddings']:,}, "
            f"minimum={diagnostics['minimum_component']:.6f}, "
            f"zero_norm_embeddings={diagnostics['zero_norm_embeddings']:,}"
        )

        grouped_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped_indices[(row["chunking_run_id"], row["article_id"])].append(index)
        groups = list(grouped_indices.values())
        rng = random.Random(random_seed)
        normalized_matrix = normalize_embeddings(embedding_matrix)
        positive = sample_same_article_similarities(
            normalized_matrix, groups, pair_limit,
            rng=rng, already_normalized=True,
        )
        transition = sample_transition_similarities(
            normalized_matrix, groups, pair_limit,
            rng=rng, already_normalized=True,
        )
        negative = calculate_pairwise_cosine_similarities(
            normalized_matrix, pair_limit,
            rng=rng, already_normalized=True,
        )
        run_distributions = {
            "Negative": negative if negative is not None else torch.empty(0),
            "Transition": transition,
            "Positive": positive,
        }
        run_order = ["Negative", "Positive", "Transition"]
        distributions[run_id] = run_distributions
        for label in run_order:
            values = run_distributions[label]
            negative_count = int((values < 0).sum().item())
            minimum = float(values.min().item()) if len(values) else float("nan")
            print(
                f"{run_id} {label}: pairs={len(values):,}, "
                f"negative_similarities={negative_count:,}, minimum={minimum:.6f}"
            )
    return distributions, run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot positive, transition, and random-pair cosine similarities."
    )
    parser.add_argument("--db-path", default="data/newspapers.sqlite")
    parser.add_argument(
        "--embedding-run-ids",
        "--embedding-run-id",
        dest="embedding_run_ids",
        nargs="+",
        help="One or more embedding runs to compare.",
    )
    parser.add_argument(
        "--embedding-run-names",
        nargs="+",
        help="Optional human-readable names for the embedding runs, in the same order.",
    )
    parser.add_argument(
        "--input",
        help="Previously summarized histogram CSV. Bypasses SQLite and sampling.",
    )
    parser.add_argument("--pair-limit", type=int, default=DEFAULT_PAIR_LIMIT)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-x", action="store_true")
    parser.add_argument(
        "--plot-scale",
        type=float,
        default=1.0,
        help="Scale fonts, ticks, legend, lines, markers, and grid (default: 1.0).",
    )
    parser.add_argument(
        "--output",
        help="Output PNG path. Defaults under data/embedding_cosine_similarities.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pair_limit < 1:
        raise ValueError("pair-limit must be at least 1")
    if args.bins < 1:
        raise ValueError("bins must be at least 1")
    if args.plot_scale <= 0:
        raise ValueError("plot-scale must be greater than 0")

    if args.input:
        if args.embedding_run_ids:
            raise ValueError("Use either --input or --embedding-run-ids, not both.")
        rows = load_similarity_histogram_csv(Path(args.input))
        embedding_run_ids = list(dict.fromkeys(
            str(row["embedding_run_id"]) for row in rows
        ))
        run_names = resolve_embedding_run_names(
            embedding_run_ids,
            args.embedding_run_names,
        )
        stored_log_x = {bool(row["log_x"]) for row in rows}
        if len(stored_log_x) != 1:
            raise ValueError("Histogram CSV contains inconsistent log-x metadata.")
        log_x = stored_log_x.pop()
        if args.log_x and not log_x:
            raise ValueError(
                "--log-x changes histogram binning and must be passed to the summarize command."
            )
        stored_names = {
            str(row["embedding_run_id"]): str(row["embedding_run_name"])
            for row in rows
            if row.get("embedding_run_name")
        }
        display_names = {**stored_names, **run_names}
        model_ids = {
            str(row["embedding_run_id"]): str(row["model_id"])
            for row in rows
        }
        output = Path(args.output) if args.output else Path(args.input).with_suffix(".png")
        save_binned_similarity_plot(
            rows,
            output=output,
            log_x=log_x,
            # title="Cosine similarity distributions",
            run_names=run_names,
            plot_scale=args.plot_scale,
        )
        print(f"Plot saved to {output.resolve()}")
        return
    if not args.embedding_run_ids:
        raise ValueError("Provide --input or --embedding-run-ids.")

    embedding_run_ids = list(dict.fromkeys(args.embedding_run_ids))
    run_names = resolve_embedding_run_names(
        embedding_run_ids,
        args.embedding_run_names,
    )
    db = Database(args.db_path)
    try:
        db.initialize()
        distributions, run_metadata = collect_similarity_distributions(
            db,
            embedding_run_ids,
            pair_limit=args.pair_limit,
            random_seed=args.random_seed,
        )
    finally:
        db.close()

    output = Path(args.output) if args.output else Path(
        "data/embedding_cosine_similarities"
    ) / f"{'_vs_'.join(embedding_run_ids)}.png"
    save_similarity_histogram(
        distributions,
        output=output,
        bins=args.bins,
        log_x=args.log_x,
        title="Cosine similarity distributions\n" + ", ".join(
            f"{run_names.get(run_id, run_id)} ({run_metadata[run_id]['model_id']})"
            for run_id in embedding_run_ids
        ),
        run_names=run_names,
        plot_scale=args.plot_scale,
    )
    print(f"Plot saved to {output.resolve()}")


if __name__ == "__main__":
    main()
