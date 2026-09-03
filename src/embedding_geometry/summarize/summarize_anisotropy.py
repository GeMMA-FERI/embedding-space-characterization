from __future__ import annotations

import argparse
import csv
from pathlib import Path

from embedding_geometry.utils.evaluation_functions import (
    format_chunking_strategy,
)
from embedding_geometry.summarize.summary_utils import (
    load_embedding_run_metadata,
    read_result_rows,
    resolve_result_identity,
)


MODEL_NAMES = {
    "BAAI/bge-m3": "BGE-M3",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B",
    "intfloat/multilingual-e5-large": "Multilingual E5 Large",
    "rokn/slovlo-v1": "SloVlo v1",
}

MODEL_ORDER = list(MODEL_NAMES)

CHUNKING_ORDER = [
    "paragraph",
    "sentence_window",
    "token_window",
]

REQUIRED_COLUMNS = {
    "embedding_run_id",
    "chunking_method",
    "chunking_strategy",
    "embedding_count",
    "pair_sample_size",
    "random_seed",
    "average_pairwise_cosine",
    "pc1_variance_ratio",
    "pc5_variance_ratio",
    "pc10_variance_ratio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CSV and LaTeX summaries of embedding anisotropy results."
    )

    parser.add_argument(
        "--db-path",
        default="data/newspapers.sqlite",
    )

    parser.add_argument(
        "--input-dir",
        default="data/embedding_anisotropy",
        help="Directory containing per-run anisotropy CSV files.",
    )

    parser.add_argument(
        "--input",
        nargs="*",
        help="Explicit result CSV files. Overrides --input-dir discovery.",
    )

    parser.add_argument(
        "--output-tex",
        default="../embeddings-analysis-paper/table_embedding_anisotropy.tex",
    )

    parser.add_argument(
        "--output-csv",
        default="data/table_embedding_anisotropy.csv",
    )

    parser.add_argument(
        "--caption",
        default="Embedding anisotropy by chunking strategy and embedding model.",
    )

    parser.add_argument(
        "--label",
        default="tab:embedding-anisotropy",
    )
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

    return "".join(
        replacements.get(character, character)
        for character in value
    )


def format_metric(value: str | None) -> str:
    if value is None or value == "":
        return "N/A"
    return f"{float(value):.4f}"


def main() -> None:
    args = parse_args()

    if args.input:
        input_paths = [Path(path) for path in args.input]
    else:
        input_paths = sorted(
            Path(args.input_dir).glob("*.csv")
        )

    if not input_paths:
        raise ValueError(
            f"No CSV files found in {args.input_dir}"
        )


    result_rows = read_result_rows(input_paths, REQUIRED_COLUMNS)
    metadata_by_run = load_embedding_run_metadata(args.db_path, result_rows)
    results = {}
    for row in result_rows:
        run_id = row["embedding_run_id"]
        model_id, chunking_strategy = resolve_result_identity(row, metadata_by_run)
        key = (model_id, chunking_strategy)

        if key in results:
            previous_run_id = results[key]["embedding_run_id"]
            raise ValueError(
                "Multiple result runs found for the same model and chunking run. "
                f"{model_id}/{chunking_strategy}: {previous_run_id}, {run_id}"
            )

        results[key] = {
            "embedding_run_id": run_id,
            "model_id": model_id,
            "chunking_method": row["chunking_method"],
            "chunking_strategy": chunking_strategy,
            "embedding_count": row["embedding_count"],
            "pair_sample_size": row["pair_sample_size"],
            "random_seed": row["random_seed"],
            "average_pairwise_cosine": row["average_pairwise_cosine"],
            "pc1_variance_ratio": row["pc1_variance_ratio"],
            "pc5_variance_ratio": row["pc5_variance_ratio"],
            "pc10_variance_ratio": row["pc10_variance_ratio"],
        }


    available_pairs = set(results)

    models = [
        model
        for model in MODEL_ORDER
        if any(
            key[0] == model
            for key in available_pairs
        )
    ]

    models.extend(
        sorted(
            {
                key[0]
                for key in available_pairs
            }
            - set(models)
        )
    )

    strategies = [
        strategy
        for strategy in CHUNKING_ORDER
        if any(
            key[1] == strategy
            for key in available_pairs
        )
    ]

    strategies.extend(
        sorted(
            {
                key[1]
                for key in available_pairs
                if key[1] != "noop"
            }
            - set(strategies)
        )
    )
    pairs = (
        [(model, strategy) for strategy in strategies for model in models]
        if args.swap_columns
        else [(model, strategy) for model in models for strategy in strategies]
    )


    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_csv.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.writer(file)

        identifier_columns = (
            ["chunking_strategy", "model_id"]
            if args.swap_columns
            else ["model_id", "chunking_strategy"]
        )
        writer.writerow(
            identifier_columns + [
                "embedding_count",
                "pair_sample_size",
                "random_seed",
                "average_pairwise_cosine",
                "pc1_variance_ratio",
                "pc5_variance_ratio",
                "pc10_variance_ratio"
            ]
        )

        for model, strategy in pairs:
            result = results.get(
                (model, strategy)
            )
            identifiers = [strategy, model] if args.swap_columns else [model, strategy]

            writer.writerow(
                    identifiers + [
                        result["embedding_count"]
                        if result
                        else "",
                        result["pair_sample_size"]
                        if result
                        else "",
                        result["random_seed"]
                        if result
                        else "",
                        result["average_pairwise_cosine"]
                        if result
                        else "",
                        result["pc1_variance_ratio"]
                        if result
                        else "",
                        result["pc5_variance_ratio"]
                        if result
                        else "",
                        result["pc10_variance_ratio"]
                        if result
                        else ""
                    ]
            )


    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{latex_escape(args.caption)}}}",
        f"\\label{{{latex_escape(args.label)}}}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        (
            (
                r"Chunking & Model & Mean pairwise cosine $\downarrow$ "
                if args.swap_columns
                else r"Model & Chunking & Mean pairwise cosine $\downarrow$ "
            ) +
            r"& PC1 $\downarrow$ & PC5 $\downarrow$ & PC10 $\downarrow$ \\"
        ),
        r"\midrule"
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

            result = results.get(
                (model, strategy)
            )

            model_label = MODEL_NAMES.get(model, model)
            strategy_label = format_chunking_strategy(strategy)
            first_cell = strategy_label if args.swap_columns else model_label
            second_cell = model_label if args.swap_columns else strategy_label
            if inner_index > 0:
                first_cell = ""

            if result:

                metrics = [
                    format_metric(result["average_pairwise_cosine"]),
                    format_metric(result["pc1_variance_ratio"]),
                    format_metric(result["pc5_variance_ratio"]),
                    format_metric(result["pc10_variance_ratio"]),
                ]

            else:
                metrics = [
                    "--",
                    "--",
                    "--",
                    "--",
                ]

            lines.append(
                f"{latex_escape(first_cell)} & "
                f"{latex_escape(second_cell)} & "
                f"{' & '.join(metrics)} \\\\"
            )

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
    output_tex.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_tex.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Read {len(input_paths)} result files.")
    print(f"Summary CSV: {output_csv.resolve()}")
    print(f"LaTeX table: {output_tex.resolve()}")


if __name__ == "__main__":
    main()
