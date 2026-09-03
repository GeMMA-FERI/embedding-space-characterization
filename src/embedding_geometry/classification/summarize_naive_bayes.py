from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from embedding_geometry.summarize.summary_utils import (
    load_embedding_run_metadata,
    read_result_rows,
    resolve_result_identity,
)


METRICS = (
    "gaussian_negative_positive_accuracy",
    "gaussian_negative_transition_accuracy",
    "histogram_negative_positive_accuracy",
    "histogram_negative_transition_accuracy",
)
REQUIRED_COLUMNS = {"embedding_run_id", *METRICS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-run cosine Naive Bayes classification results."
    )
    parser.add_argument("--db-path", default="data/newspapers.sqlite")
    parser.add_argument(
        "--input-dir", default="data/cosine_classification"
    )
    parser.add_argument(
        "--input", nargs="*", help="Explicit CSV files; overrides --input-dir."
    )
    parser.add_argument(
        "--output-csv", default="data/table_cosine_classification.csv"
    )
    parser.add_argument(
        "--output-tex",
        default="../embeddings-analysis-paper/table_cosine_classification.tex",
    )
    return parser.parse_args()


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("&", r"\&")


def complete_result_keys(
    results: dict[tuple[str, str], dict[str, str]],
) -> list[tuple[str, str]]:
    """Return the full model/strategy grid used by the other summaries."""
    models = sorted({model_id for model_id, _ in results})
    strategies = sorted({strategy for _, strategy in results})
    return [(model_id, strategy) for model_id in models for strategy in strategies]


def main() -> None:
    args = parse_args()
    paths = (
        [Path(path) for path in args.input]
        if args.input
        else sorted(Path(args.input_dir).glob("*.csv"))
    )
    if not paths:
        raise ValueError(f"No CSV files found in {args.input_dir}")

    rows = read_result_rows(paths, REQUIRED_COLUMNS)
    metadata = load_embedding_run_metadata(args.db_path, rows)
    results: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        model_id, chunking_run_id = resolve_result_identity(row, metadata)
        key = (model_id, chunking_run_id)
        if key in results:
            raise ValueError(
                "Multiple classification results found for "
                f"{model_id}/{chunking_run_id}. Keep one embedding run per combination."
            )
        results[key] = row

    result_keys = complete_result_keys(results)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["model_id", "chunking_strategy", *METRICS]
        )
        writer.writeheader()
        for model_id, strategy in result_keys:
            result = results.get((model_id, strategy))
            writer.writerow(
                {
                    "model_id": model_id,
                    "chunking_strategy": strategy,
                    **{
                        metric: result[metric] if result else ""
                        for metric in METRICS
                    },
                }
            )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Naive Bayes accuracy for cosine-similarity distributions.}",
        r"\label{tab:cosine-classification}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        "Model & Chunking & GNB N/P & GNB N/T & HNB N/P & HNB N/T " + r"\\",
        r"\midrule",
    ]
    for model_id, strategy in result_keys:
        row = results.get((model_id, strategy))
        values = (
            " & ".join(f"{float(row[metric]):.4f}" for metric in METRICS)
            if row
            else " & ".join("--" for _ in METRICS)
        )
        lines.append(
            f"{latex_escape(model_id)} & {latex_escape(strategy)} & {values} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    output_tex = Path(args.output_tex)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Read {len(paths)} result files.")
    print(f"Summary CSV: {output_csv.resolve()}")
    print(f"LaTeX table: {output_tex.resolve()}")


if __name__ == "__main__":
    main()
