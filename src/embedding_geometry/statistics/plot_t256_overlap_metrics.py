"""Plot evaluation metrics across the configured T256 token overlaps."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

from embedding_geometry.summarize.merge_tables import MODEL_NAMES, MODEL_ORDER


# Comment out metrics here to omit them from the figure.
# (CSV column, plot label, render as percent)
METRICS = [
    ("embedding_anisotropy_average_pairwise_cosine", "MNCS", False),
    ("intra_consistency_mean_pairwise_cosine_similarity", "MPCS", False),
    ("intra_consistency_mean_cosine_similarity", "MTCS", False),
    ("intra_consistency_semantic_drop_rate", "SDR", True),
    ("inter_separability_davies_bouldin_trimmed_index", "DBI", False),
    ("representation_degradation_mean_representation_degradation", "MRD", False),
    ("derived_degradation_share", "Deg. share", True),
    ("derived_retained_similarity", "Ret. sim.", True),
    ("embedding_anisotropy_pc1_variance_ratio", "PC1", False),
    ("embedding_anisotropy_pc5_variance_ratio", "PC5", False),
    ("embedding_anisotropy_pc10_variance_ratio", "PC10", False),
    ("cosine_classification_gaussian_negative_positive_accuracy", "GNB N/P", True),
    ("cosine_classification_gaussian_negative_transition_accuracy", "GNB N/T", True),
]

T256_PATTERN = re.compile(r"^token_256_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot merged evaluation metrics as T256 overlap changes."
    )
    parser.add_argument(
        "--input",
        default="data/table_all_metrics.csv",
        help="Merged metrics CSV produced by merge_tables.py.",
    )
    parser.add_argument(
        "--model-ids",
        nargs="+",
        help="Models to plot. By default, plots every model with T256 rows.",
    )
    parser.add_argument(
        "--x-axis",
        choices=("percent", "tokens"),
        default="percent",
        help="Show configured overlap as a percentage or token count.",
    )
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--plot-scale", type=float, default=1.0)
    parser.add_argument(
        "--output",
        default="data/t256_overlap_metrics.png",
    )
    return parser.parse_args()


def load_t256_rows(path: Path) -> dict[str, dict[int, dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"model_id", "chunking_strategy", *(column for column, _, _ in METRICS)}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

        rows: dict[str, dict[int, dict[str, str]]] = {}
        for row in reader:
            match = T256_PATTERN.fullmatch(row["chunking_strategy"])
            if match:
                rows.setdefault(row["model_id"], {})[int(match.group(1))] = row
    if not rows:
        raise ValueError(f"No T256 rows found in {path}")
    return rows


def ordered_models(
    rows: dict[str, dict[int, dict[str, str]]], requested: list[str] | None
) -> list[str]:
    if requested:
        unknown = [model_id for model_id in requested if model_id not in rows]
        if unknown:
            raise ValueError(f"No T256 rows for model(s): {', '.join(unknown)}")
        return requested
    rank = {model_id: index for index, model_id in enumerate(MODEL_ORDER)}
    return sorted(rows, key=lambda model_id: (rank.get(model_id, len(rank)), model_id))


def save_plot(
    rows: dict[str, dict[int, dict[str, str]]],
    model_ids: list[str],
    *,
    output: Path,
    columns: int,
    x_axis: str,
    plot_scale: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install matplotlib to create the plot.") from error

    subplot_rows = math.ceil(len(METRICS) / columns)
    figure, axes = plt.subplots(
        subplot_rows,
        columns,
        figsize=(5.0 * columns * plot_scale, 3.5 * subplot_rows * plot_scale),
        sharex=True,
        squeeze=False,
    )
    flat_axes = list(axes.flat)

    for axis, (column, label, as_percent) in zip(flat_axes, METRICS):
        for model_id in model_ids:
            points = []
            for overlap, row in sorted(rows[model_id].items()):
                value = row[column]
                if value:
                    x_value = overlap / 256 * 100 if x_axis == "percent" else overlap
                    points.append((x_value, float(value) * (100 if as_percent else 1)))
            if points:
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    linewidth=1.8 * plot_scale,
                    markersize=4.5 * plot_scale,
                    label=MODEL_NAMES.get(model_id, model_id),
                )
        axis.set_title(label, fontsize=12 * plot_scale)
        axis.set_ylabel("Percent" if as_percent else "Value", fontsize=10 * plot_scale)
        axis.grid(alpha=0.25, linewidth=0.8 * plot_scale)
        axis.tick_params(labelsize=9 * plot_scale)

    for axis in flat_axes[len(METRICS) :]:
        axis.remove()
    figure.supxlabel(
        "Configured overlap (%)" if x_axis == "percent" else "Configured overlap (tokens)",
        fontsize=10 * plot_scale,
    )

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), 5),
        fontsize=10 * plot_scale,
        frameon=False,
    )
    figure.suptitle(
        "T256 metrics by configured overlap", fontsize=14 * plot_scale, y=0.995
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.columns < 1:
        raise ValueError("--columns must be at least 1")
    if args.plot_scale <= 0:
        raise ValueError("--plot-scale must be greater than 0")

    rows = load_t256_rows(Path(args.input))
    model_ids = ordered_models(rows, args.model_ids)
    output = Path(args.output)
    save_plot(
        rows,
        model_ids,
        output=output,
        columns=args.columns,
        x_axis=args.x_axis,
        plot_scale=args.plot_scale,
    )
    print(f"Plotted {len(METRICS)} metrics for {len(model_ids)} models: {output.resolve()}")


if __name__ == "__main__":
    main()
