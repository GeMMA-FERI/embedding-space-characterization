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
    "chunking_method",
    "chunking_strategy",
    "mean_cosine_similarity",
    "mean_pairwise_cosine_similarity",
    "cosine_volatility",
    "semantic_drop_rate",
    "num_articles",
    "num_transitions",
    "num_pairwise_comparisons",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CSV and LaTeX summaries of intra-consistency results."
    )
    parser.add_argument(
        "--db-path",
        default="data/newspapers.sqlite"
    )
    parser.add_argument(
        "--input-dir",
        default="data/intra_consistency",
        help="Directory containing per-run intra-consistency CSV files."
    )
    parser.add_argument(
        "--input",
        nargs="*",
        help="Explicit result CSV files. Overrides --input-dir discovery."
    )
    parser.add_argument(
        "--output-tex",
        default="../embeddings-analysis-paper/table_intra_consistency.tex",
    )
    parser.add_argument(
        "--output-csv",
        default="data/table_intra_consistency.csv"
    )
    parser.add_argument(
        "--caption",
        default="Intra-article consistency by chunking strategy and embedding model."
    )
    parser.add_argument("--label", default="tab:intra-consistency")
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


def main() -> None:
    args = parse_args()
    if args.input:
        input_paths = [Path(path) for path in args.input]
    else:
        input_paths = sorted(Path(args.input_dir).glob("*.csv"))
        legacy_path = Path("data/intra_consistency.csv")
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
        details = "; ".join(f"{model}/{method}: {sorted(runs)}" for (model, method), runs in duplicate_pairs.items())
        raise ValueError("Multiple result runs found for the same model and chunking method. Keep one run per combination. {details}")

    all_pairs = set(results)
    models = [model for model in MODEL_ORDER if any(key[0] == model for key in all_pairs)]
    models.extend(sorted({key[0] for key in all_pairs} - set(models)))
    strategies = [
        strategy
        for strategy in CHUNKING_ORDER
        if any(key[1] == strategy for key in all_pairs)
    ]
    strategies.extend(
        sorted(
            strategy
            for strategy in {key[1] for key in all_pairs} - set(strategies)
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
                "num_transitions",
                "num_pairwise_comparisons",
                "mean_cosine_similarity",
                "mean_pairwise_cosine_similarity",
                "mean_cosine_volatility",
                "semantic_drop_rate",
            ]
        )
        for model, strategy in pairs:
            result = results.get((model, strategy))
            identifiers = [strategy, model] if args.swap_columns else [model, strategy]
            writer.writerow(
                    identifiers + [
                        result["num_articles"] if result else 0,
                        result["num_transitions"] if result else 0,
                        result["num_pairwise_comparisons"] if result else 0,
                        result["mean_cosine_similarity"] if result else "",
                        result["mean_pairwise_cosine_similarity"] if result else "",
                        result["cosine_volatility"] if result else "",
                        result["semantic_drop_rate"] if result else "",
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
            r"Chunking & Model & MTCS $\uparrow$ & MPCS (intra) $\uparrow$ & Volatility $\downarrow$ & Drop rate $\downarrow$ \\"
            if args.swap_columns
            else r"Model & Chunking & MTCS $\uparrow$ & MPCS (intra) $\uparrow$ & Volatility $\downarrow$ & Drop rate $\downarrow$ \\"
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
                    f"{float(result['mean_cosine_similarity']):.4f}",
                    f"{float(result['mean_pairwise_cosine_similarity']):.4f}",
                    f"{float(result['cosine_volatility']):.4f}",
                    f"{float(result['semantic_drop_rate']) * 100:.2f}\\%",
                ]
            else:
                metrics = ["--", "--", "--", "--"]
            lines.append(f"{latex_escape(first_cell)} & {latex_escape(second_cell)} & " f"{' & '.join(metrics)} \\\\")
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
    print(f"Summary CSV saved to: {output_csv.resolve()}")
    print(f"LaTeX table saved to: {output_tex.resolve()}")


if __name__ == "__main__":
    main()
