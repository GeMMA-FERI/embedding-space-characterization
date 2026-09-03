from __future__ import annotations

import argparse
import csv
from pathlib import Path

from embedding_geometry.summarize.summary_utils import (
    load_embedding_run_metadata,
    read_result_rows,
    resolve_result_identity,
)
from embedding_geometry.utils.evaluation_functions import (
    format_chunking_strategy,
)


MODEL_NAMES = {
    "BAAI/bge-m3": "BGE-M3",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B",
    "intfloat/multilingual-e5-large": "Multilingual E5 Large",
    "rokn/slovlo-v1": "SloVlo v1",
}
MODEL_ORDER = list(MODEL_NAMES)
CHUNKING_ORDER = ["paragraph", "sentence_window", "token_window"]
REQUIRED_COLUMNS = {
    "embedding_run_id",
    "model_id",
    "chunking_method",
    "chunking_strategy",
    "num_articles",
    "mean_num_chunks",
    "mean_total_chunk_length",
    "mean_cosine_similarity",
    "median_cosine_similarity",
    "mean_representation_degradation",
    "median_representation_degradation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CSV and LaTeX summaries of representation degradation results."
    )
    parser.add_argument("--db-path", default="data/newspapers.sqlite")
    parser.add_argument(
        "--input-dir",
        default="data/representation_degradation",
        help="Directory containing per-run representation degradation CSV files.",
    )
    parser.add_argument(
        "--input",
        nargs="*",
        help="Explicit result CSV files. Overrides --input-dir discovery.",
    )
    parser.add_argument(
        "--output-tex",
        default="../embeddings-analysis-paper/table_representation_degradation.tex",
    )
    parser.add_argument(
        "--output-csv",
        default="data/table_representation_degradation.csv",
    )
    parser.add_argument(
        "--caption",
        default="Representation degradation by chunking strategy and embedding model.",
    )
    parser.add_argument("--label", default="tab:representation-degradation")
    parser.add_argument(
        "--swap-columns",
        action="store_true",
        help="Group rows by chunking strategy instead of embedding model.",
    )
    return parser.parse_args()


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def format_metric(value: str | None) -> str:
    if value is None or value == "":
        return "N/A"
    return f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    if args.input:
        input_paths = [Path(path) for path in args.input]
    else:
        input_paths = sorted(Path(args.input_dir).glob("*.csv"))
        legacy_path = Path("data/representation_degradation.csv")
        if not input_paths and legacy_path.exists():
            input_paths = [legacy_path]
    if not input_paths:
        raise ValueError(f"No CSV files found in {args.input_dir}")

    result_rows = read_result_rows(input_paths, REQUIRED_COLUMNS)
    metadata_by_run = load_embedding_run_metadata(args.db_path, result_rows)
    results: dict[tuple[str, str], dict[str, str]] = {}
    seen_runs: dict[tuple[str, str], set[str]] = {}
    for row in result_rows:
        run_id = row["embedding_run_id"]
        model_id, strategy = resolve_result_identity(row, metadata_by_run)
        key = (model_id, strategy)
        row["model_id"] = model_id
        row["chunking_strategy"] = strategy
        results[key] = row
        seen_runs.setdefault(key, set()).add(run_id)

    duplicate_pairs = {key: runs for key, runs in seen_runs.items() if len(runs) > 1}
    if duplicate_pairs:
        details = "; ".join(f"{model}/{strategy}: {sorted(runs)}" for (model, strategy), runs in duplicate_pairs.items())
        raise ValueError(
            "Multiple result runs found for the same model and chunking strategy. "
            f"Keep one run per combination. {details}"
        )

    available_pairs = set(results)
    models = [model for model in MODEL_ORDER if any(key[0] == model for key in available_pairs)]
    models.extend(sorted({key[0] for key in available_pairs} - set(models)))
    strategies = [strategy for strategy in CHUNKING_ORDER if any(key[1] == strategy for key in available_pairs)]
    strategies.extend(
        sorted(
            strategy
            for strategy in {key[1] for key in available_pairs} - set(strategies)
            if strategy != "noop"
        )
    )
    pairs = (
        [(model, strategy) for strategy in strategies for model in models]
        if args.swap_columns
        else [(model, strategy) for model in models for strategy in strategies]
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        identifier_columns = (
            ["chunking_strategy", "model_id"]
            if args.swap_columns
            else ["model_id", "chunking_strategy"]
        )
        writer.writerow(
            identifier_columns + [
                "num_articles",
                "mean_num_chunks",
                "mean_total_chunk_length",
                "mean_cosine_similarity",
                "median_cosine_similarity",
                "mean_representation_degradation",
                "median_representation_degradation",
            ]
        )
        for model, strategy in pairs:
            result = results.get((model, strategy))
            identifiers = [strategy, model] if args.swap_columns else [model, strategy]
            writer.writerow(
                    identifiers + [
                        result["num_articles"] if result else 0,
                        result["mean_num_chunks"] if result else "",
                        result["mean_total_chunk_length"] if result else "",
                        result["mean_cosine_similarity"] if result else "",
                        result["median_cosine_similarity"] if result else "",
                        result["mean_representation_degradation"] if result else "",
                        result["median_representation_degradation"] if result else "",
                    ]
            )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{latex_escape(args.caption)}}}",
        f"\\label{{{latex_escape(args.label)}}}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        (
            (
                r"Chunking & Model & Mean degradation $\downarrow$ "
                if args.swap_columns
                else r"Model & Chunking & Mean degradation $\downarrow$ "
            ) + r"& Median degradation $\downarrow$ \\"
        ),
        r"\midrule",
    ]
    outer_values = strategies if args.swap_columns else models
    inner_values = models if args.swap_columns else strategies
    for outer_index, outer_value in enumerate(outer_values):
        for inner_index, inner_value in enumerate(inner_values):
            model, strategy = (
                (inner_value, outer_value)
                if args.swap_columns
                else (outer_value, inner_value)
            )
            result = results.get((model, strategy))
            model_label = MODEL_NAMES.get(model, model)
            strategy_label = format_chunking_strategy(strategy)
            first_cell = strategy_label if args.swap_columns else model_label
            second_cell = model_label if args.swap_columns else strategy_label
            if inner_index > 0:
                first_cell = ""
            if result:
                metrics = [
                    format_metric(result["mean_representation_degradation"]),
                    format_metric(result["median_representation_degradation"]),
                ]
            else:
                metrics = ["--", "--"]
            lines.append(f"{latex_escape(first_cell)} & {latex_escape(second_cell)} & {' & '.join(metrics)} \\\\")
        if outer_index < len(outer_values) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )

    output_tex = Path(args.output_tex)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Read {len(input_paths)} result files.")
    print(f"Summary CSV: {output_csv.resolve()}")
    print(f"LaTeX table: {output_tex.resolve()}")


if __name__ == "__main__":
    main()
