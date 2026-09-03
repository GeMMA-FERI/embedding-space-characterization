from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from embedding_geometry.statistics.analyze_text_structure import words
from semora.storage import Database


DEFAULT_BINS = 4000
DEFAULT_OUTPUT_DIR = Path("data/statistics")
Metric = Literal["char", "word"]
PlotStyle = Literal["histogram", "line"]
HISTOGRAM_FIELDS = (
    "chunking_run_id",
    "chunking_run_name",
    "method",
    "metric",
    "bin_index",
    "bin_left",
    "bin_right",
    "bin_center",
    "density",
    "chunk_count",
    "log_x",
)


def count_chunk_metric(text: str, metric: Metric) -> int:
    """Count characters or Unicode letter-only words in a chunk."""
    return len(text) if metric == "char" else len(words(text))


def collect_chunk_counts(
    rows: Iterable[Mapping[str, object]],
    chunking_run_ids: list[str],
    *,
    metric: Metric = "char",
) -> dict[str, list[int]]:
    """Collect the selected chunk metric while streaming database rows."""
    counts = {chunking_run_id: [] for chunking_run_id in chunking_run_ids}
    for row in tqdm(rows, desc=f"Counting {metric}s in chunks", unit="chunk"):
        chunking_run_id = str(row["chunking_run_id"])
        text = str(row["text"] or "")
        value = count_chunk_metric(text, metric)
        if value > 0:
            counts[chunking_run_id].append(value)
    return counts


def resolve_chunking_run_names(
    chunking_run_ids: list[str],
    chunking_run_names: list[str] | None,
) -> dict[str, str]:
    if chunking_run_names is None:
        return {}
    if len(chunking_run_names) != len(chunking_run_ids):
        raise ValueError(
            "--chunking-run-names must contain exactly one name for each chunking run."
        )
    return dict(zip(chunking_run_ids, chunking_run_names))


def build_chunk_histogram_rows(
    distributions: Mapping[str, list[int]],
    *,
    methods: Mapping[str, str],
    run_names: Mapping[str, str] | None = None,
    bins: int,
    metric: Metric,
    log_x: bool,
) -> list[dict[str, object]]:
    """Bin chunk lengths once so plots can be recreated without SQLite."""
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Install numpy to calculate histogram densities.") from error

    non_empty = [values for values in distributions.values() if values]
    if not non_empty:
        raise ValueError("No chunks are available for the requested chunking runs.")
    minimum = min(min(values) for values in non_empty)
    maximum = max(max(values) for values in non_empty)
    if log_x and minimum <= 0:
        raise ValueError(
            "--log-x requires every plotted metric value to be greater than zero; "
            f"minimum observed {metric} count is {minimum}."
        )
    if minimum == maximum:
        padding = max(abs(minimum) * 0.01, 0.5)
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

    rows: list[dict[str, object]] = []
    for run_id, values in distributions.items():
        if not values:
            continue
        density, _ = np.histogram(values, bins=edges, density=True)
        for index, value in enumerate(density):
            rows.append(
                {
                    "chunking_run_id": run_id,
                    "chunking_run_name": (run_names or {}).get(run_id, ""),
                    "method": methods.get(run_id, ""),
                    "metric": metric,
                    "bin_index": index,
                    "bin_left": float(edges[index]),
                    "bin_right": float(edges[index + 1]),
                    "bin_center": float(centers[index]),
                    "density": float(value),
                    "chunk_count": len(values),
                    "log_x": int(log_x),
                }
            )
    return rows


def save_chunk_histogram_csv(
    rows: list[dict[str, object]],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTOGRAM_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_chunk_histogram_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != HISTOGRAM_FIELDS:
            raise ValueError(
                f"Unexpected histogram CSV columns in {path}; expected {HISTOGRAM_FIELDS}."
            )
        return [
            {
                **row,
                "bin_index": int(row["bin_index"]),
                "bin_left": float(row["bin_left"]),
                "bin_right": float(row["bin_right"]),
                "bin_center": float(row["bin_center"]),
                "density": float(row["density"]),
                "chunk_count": int(row["chunk_count"]),
                "log_x": bool(int(row["log_x"])),
            }
            for row in reader
        ]


def select_chunk_histogram_rows(
    rows: list[dict[str, object]],
    requested_run_ids: list[str],
    excluded_run_ids: list[str],
) -> tuple[list[dict[str, object]], list[str]]:
    """Select and order runs from a previously summarized all-runs CSV."""
    available_run_ids = list(
        dict.fromkeys(str(row["chunking_run_id"]) for row in rows)
    )
    available = set(available_run_ids)
    unknown_ids = [run_id for run_id in requested_run_ids if run_id not in available]
    if unknown_ids:
        raise ValueError(
            f"Unknown chunking run ID(s) in histogram CSV: {', '.join(unknown_ids)}"
        )
    excluded = set(excluded_run_ids)
    selected_run_ids = [
        run_id
        for run_id in (requested_run_ids or available_run_ids)
        if run_id not in excluded
    ]
    if not selected_run_ids:
        raise ValueError("No chunking runs remain after applying plot selections.")
    rows_by_run: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_run[str(row["chunking_run_id"])].append(row)
    return (
        [row for run_id in selected_run_ids for row in rows_by_run[run_id]],
        selected_run_ids,
    )


def save_binned_chunk_plot(
    rows: list[dict[str, object]],
    *,
    output: Path,
    plot_style: PlotStyle = "histogram",
    log_x: bool,
    min_x: int | None = None,
    max_x: int | None = None,
    run_names: Mapping[str, str] | None = None,
    plot_scale: float = 1.0,
) -> None:
    """Plot previously binned chunk-length distributions."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install matplotlib to create the histogram plot.") from error

    if not rows:
        raise ValueError("No histogram rows are available to plot.")
    if plot_scale <= 0:
        raise ValueError("plot-scale must be greater than 0")
    metrics = {str(row["metric"]) for row in rows}
    stored_log_x = {bool(row["log_x"]) for row in rows}
    if len(metrics) != 1 or len(stored_log_x) != 1:
        raise ValueError("Histogram CSV contains inconsistent metric or log-x metadata.")
    metric = metrics.pop()
    if metric not in ("char", "word"):
        raise ValueError(f"Unsupported histogram metric: {metric}")
    if stored_log_x.pop() != log_x:
        raise ValueError("Plot log-x setting does not match the summarized histogram bins.")
    if log_x and min(float(row["bin_left"]) for row in rows) <= 0:
        raise ValueError("--log-x requires histogram bins to be greater than zero.")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["chunking_run_id"])].append(row)
    stored_names = {
        str(row["chunking_run_id"]): str(row["chunking_run_name"])
        for row in rows
        if row.get("chunking_run_name")
    }
    display_names = {**stored_names, **(run_names or {})}

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11, 7))
    palette = plt.get_cmap("tab20").colors
    run_count = len(grouped)
    for run_index, (run_id, run_rows) in enumerate(grouped.items()):
        values = sorted(run_rows, key=lambda row: int(row["bin_index"]))
        method = str(values[0]["method"])
        name = display_names.get(run_id) or (
            f"{run_id} ({method})" if method else run_id
        )
        label = f"{name}"
        density = [float(row["density"]) for row in values]
        widths = [
            float(row["bin_right"]) - float(row["bin_left"])
            for row in values
        ]
        area = sum(value * width for value, width in zip(density, widths))
        if area <= 0:
            raise ValueError(f"Histogram for {run_id} has zero density.")
        # Also upgrades legacy CSVs whose bin values were normalized to sum to 1.
        density = [value / area for value in density]
        edges = [float(row["bin_left"]) for row in values]
        edges.append(float(values[-1]["bin_right"]))
        centers = [float(row["bin_center"]) for row in values]
        color = (
            palette[run_index]
            if run_count <= len(palette)
            else plt.get_cmap("turbo")(run_index / max(run_count - 1, 1))
        )
        if plot_style == "histogram":
            axis.stairs(
                density,
                edges,
                color=color,
                fill=True,
                alpha=0.3,
                linewidth=1.5 * plot_scale,
                label=label,
            )
        else:
            (line,) = axis.plot(
                centers,
                density,
                # marker="o",
                markersize=3 * plot_scale,
                linewidth=1.5 * plot_scale,
                alpha=0.9,
                color=color,
                label=label,
            )
            axis.fill_between(
                centers,
                density,
                0,
                color=line.get_color(),
                alpha=0.12,
            )
    if log_x:
        axis.set_xscale("log")
    metric_label = "Characters" if metric == "char" else "Words"
    axis.set_xlabel(f"{metric_label} per chunk", fontsize=12 * plot_scale)
    axis.set_ylabel("Density", fontsize=12 * plot_scale, labelpad=10)
    # axis.set_title(f"{metric_label} per chunk by chunking run")
    axis.set_ylim(bottom=0)
    if min_x is not None or max_x is not None:
        axis.set_xlim(left=min_x, right=max_x)
    axis.tick_params(
        axis="y",
        labelsize=10,
        width=0.8 * plot_scale,
        length=3.5 * plot_scale,
    )
    axis.tick_params(
        axis="x",
        labelsize=10 * plot_scale,
        width=0.8 * plot_scale,
        length=3.5 * plot_scale,
    )
    axis.legend(fontsize=8 * plot_scale)
    axis.grid(alpha=0.2, linewidth=0.8 * plot_scale)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8 * plot_scale)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def save_chunk_count_histogram(
    distributions: Mapping[str, list[int]],
    *,
    labels: Mapping[str, str],
    output: Path,
    bins: int,
    metric: Metric = "char",
    plot_style: PlotStyle = "histogram",
    log_x: bool = False,
    min_x: int | None = None,
    max_x: int | None = None,
    plot_scale: float = 1.0,
) -> None:
    """Backward-compatible helper that bins raw counts and immediately plots."""
    rows = build_chunk_histogram_rows(
        distributions,
        methods={},
        run_names=labels,
        bins=bins,
        metric=metric,
        log_x=log_x,
    )
    save_binned_chunk_plot(
        rows,
        output=output,
        plot_style=plot_style,
        log_x=log_x,
        min_x=min_x,
        max_x=max_x,
        plot_scale=plot_scale,
    )


# Compatibility aliases for code that imported the former helper names.
collect_word_counts = collect_chunk_counts
save_word_count_histogram = save_chunk_count_histogram


def select_chunking_runs(
    db: Database,
    requested_run_ids: list[str],
    excluded_run_ids: list[str],
) -> tuple[list[str], dict[str, Mapping[str, object]]]:
    run_rows = (
        db.get_chunking_runs_by_ids(requested_run_ids)
        if requested_run_ids
        else db.get_chunking_runs()
    )
    known_runs = {str(row["chunking_run_id"]): row for row in run_rows}
    unknown_ids = [run_id for run_id in requested_run_ids if run_id not in known_runs]
    if unknown_ids:
        raise ValueError(f"Unknown chunking run ID(s): {', '.join(unknown_ids)}")
    excluded = set(excluded_run_ids)
    chunking_run_ids = [
        run_id
        for run_id in (requested_run_ids or list(known_runs))
        if run_id not in excluded
    ]
    if not chunking_run_ids:
        raise ValueError("No chunking runs remain after applying exclusions.")
    return chunking_run_ids, known_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot overlapping chunk-size distributions for chunking runs."
    )
    parser.add_argument("--db-path", default="data/newspapers.sqlite")
    parser.add_argument(
        "--input",
        help="Previously summarized histogram CSV. Bypasses SQLite and counting.",
    )
    parser.add_argument(
        "--chunking-run-ids",
        nargs="+",
        help="Chunking runs to plot. If omitted, plots every chunking run.",
    )
    parser.add_argument(
        "--exclude-chunking-run-ids",
        nargs="+",
        help="Chunking runs to exclude from the plot.",
    )
    parser.add_argument(
        "--chunking-run-names",
        nargs="+",
        help="Optional human-readable names in chunking-run order.",
    )
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument(
        "--metric",
        choices=("char", "word"),
        default="char",
        help="Chunk-size metric to plot (default: char).",
    )
    parser.add_argument(
        "--plot-style",
        choices=("histogram", "line"),
        default="histogram",
        help="Render histograms or density lines with dots and filled areas.",
    )
    parser.add_argument("--log-x", action="store_true")
    parser.add_argument(
        "--plot-scale",
        type=float,
        default=1.0,
        help="Scale fonts, ticks, legend, lines, markers, and grid (default: 1.0).",
    )
    parser.add_argument("--min-x", type=int, help="Minimum x-axis value for the plot.")
    parser.add_argument("--max-x", type=int, help="Maximum x-axis value for the plot.")
    parser.add_argument(
        "--output",
        help="Output PNG path. The default filename includes the metric and style.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bins < 1:
        raise ValueError("bins must be at least 1")
    if args.plot_scale <= 0:
        raise ValueError("plot-scale must be greater than 0")
    if args.min_x is not None and args.min_x < 0:
        raise ValueError("min-x must be at least 0")
    if args.max_x is not None and args.max_x <= 0:
        raise ValueError("max-x must be greater than 0")
    if args.min_x is not None and args.max_x is not None and args.min_x >= args.max_x:
        raise ValueError("min-x must be smaller than max-x")
    if args.log_x and args.min_x is not None and args.min_x <= 0:
        raise ValueError("min-x must be greater than 0 when using --log-x")
    if args.input:
        rows = load_chunk_histogram_csv(Path(args.input))
        requested_run_ids = list(dict.fromkeys(args.chunking_run_ids or []))
        excluded_run_ids = list(
            dict.fromkeys(args.exclude_chunking_run_ids or [])
        )
        rows, chunking_run_ids = select_chunk_histogram_rows(
            rows,
            requested_run_ids,
            excluded_run_ids,
        )
        run_names = resolve_chunking_run_names(
            chunking_run_ids,
            args.chunking_run_names,
        )
        stored_log_x = {bool(row["log_x"]) for row in rows}
        if len(stored_log_x) != 1:
            raise ValueError("Histogram CSV contains inconsistent log-x metadata.")
        log_x = stored_log_x.pop()
        if args.log_x and not log_x:
            raise ValueError(
                "--log-x changes histogram binning and must be passed to the summarize command."
            )
        if log_x and args.min_x is not None and args.min_x <= 0:
            raise ValueError("min-x must be greater than 0 for logarithmic CSV bins")
        output = Path(args.output) if args.output else Path(args.input).with_suffix(".png")
        save_binned_chunk_plot(
            rows,
            output=output,
            plot_style=args.plot_style,
            log_x=log_x,
            min_x=args.min_x,
            max_x=args.max_x,
            run_names=run_names,
            plot_scale=args.plot_scale,
        )
        print(f"Plot saved to {output.resolve()}")
        return

    requested_run_ids = list(dict.fromkeys(args.chunking_run_ids or []))
    excluded_run_ids = list(dict.fromkeys(args.exclude_chunking_run_ids or []))

    db = Database(args.db_path)
    try:
        db.initialize()
        chunking_run_ids, known_runs = select_chunking_runs(
            db,
            requested_run_ids,
            excluded_run_ids,
        )
        distributions = collect_chunk_counts(
            db.iter_chunk_texts_for_runs(chunking_run_ids),
            chunking_run_ids,
            metric=args.metric,
        )
    finally:
        db.close()

    empty_ids = [run_id for run_id, values in distributions.items() if not values]
    if empty_ids:
        print(f"Warning: no chunks found for: {', '.join(empty_ids)}")
    for run_id, values in distributions.items():
        if values:
            print(
                f"{run_id}: chunks={len(values):,}, "
                f"minimum={min(values):,}, maximum={max(values):,} {args.metric}s"
            )

    run_names = resolve_chunking_run_names(
        chunking_run_ids,
        args.chunking_run_names,
    )
    rows = build_chunk_histogram_rows(
        distributions,
        methods={
            run_id: str(known_runs[run_id]["method"])
            for run_id in chunking_run_ids
        },
        run_names=run_names,
        bins=args.bins,
        metric=args.metric,
        log_x=args.log_x,
    )
    output = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR
        / (
            f"chunk_{args.metric}_distributions.png"
            if args.plot_style == "histogram"
            else f"chunk_{args.metric}_line_distributions.png"
        )
    )
    save_binned_chunk_plot(
        rows,
        output=output,
        plot_style=args.plot_style,
        log_x=args.log_x,
        min_x=args.min_x,
        max_x=args.max_x,
        run_names=run_names,
        plot_scale=args.plot_scale,
    )
    print(f"Plot saved to {output.resolve()}")


if __name__ == "__main__":
    main()
