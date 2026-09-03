from __future__ import annotations

import argparse
import csv
from pathlib import Path

KEY_COLUMNS = ("model_id", "chunking_strategy")
# The table uses \resizebox, so this controls the size of the whole table,
# including its font. Use r"\textwidth" for the previous full-width size.
LATEX_TABLE_WIDTH = r"0.7\textwidth"
MODEL_NAMES = {
    "BAAI/bge-m3": "BGE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3",
    "intfloat/multilingual-e5-large": "E5",
    "rokn/slovlo-v1": "SloVlo",
    "google/embeddinggemma-300m": "Gemma",
}
CHUNKING_NAMES = {
    "paragraph": "P",
    "sentence": "S1/0",
    "sentence_4": "S4/1",
    "token_128": "T128/16",
    "token_32": "T32/0",
    "recursive_1000": "R1000/100",
    "recursive_400": "R400/64",
    "recursive_1000_0": "R1000/0",
    "recursive_400_0": "R400/0",
    "token_128_0": "T128/0",
    "sentence_4_0": "S4/0",
    "token_256_0": "T256/0",
    "token_256_32": "T256/32",
    "token_256_64": "T256/64",
    "token_256_128": "T256/128",
    "token_256_192": "T256/192",
}
MODEL_ORDER = [
    "intfloat/multilingual-e5-large",
    "rokn/slovlo-v1",
    "BAAI/bge-m3",
    "Qwen/Qwen3-Embedding-0.6B",
    "google/embeddinggemma-300m",
]
CHUNK_ORDER = [
    "paragraph",
    "sentence",
    "sentence_4_0",
    "sentence_4",
    "token_32",
    "token_128_0",
    "token_128",
    "token_256_0",
    "token_256_32",
    "token_256_64",
    "token_256_128",
    "token_256_192",
    "recursive_400_0",
    "recursive_400",
    "recursive_1000_0",
    "recursive_1000",
]
LATEX_GROUPS = [
    (
        "Intra",
        "intra_consistency",
        [
            # MTCS - Mean Transition Cosine Similarity
            ("mean_cosine_similarity", r"\shortstack{MTCS $\uparrow$}", False),
            (
                "mean_pairwise_cosine_similarity",
                r"\shortstack{MPCS$\uparrow$}",
                False,
            ),
            # ("mean_cosine_volatility", r"Volatility $\downarrow$", False),
            ("semantic_drop_rate", r"\shortstack{SDR $\downarrow$}", True),
        ],
    ),
    (
        "Inter",
        "inter_separability",
        [
            ("davies_bouldin_trimmed_index", r"\shortstack{DBI $\downarrow$}", False),
            # ("davies_bouldin_trimmed_std", r"\shortstack{DBI SD $\downarrow$}", False),
            # ("silhouette_score", r"\shortstack{Silhouette\\score $\uparrow$}", False),
        ],
    ),
    (
        "Anisotropy",
        "embedding_anisotropy",
        [
            # MPCS - Mean Pairwise Cosine Similarity
            (
                "average_pairwise_cosine",
                r"\shortstack{MNCS $\downarrow$}",
                False,
            ),
            ("pc1_variance_ratio", r"PC1 $\downarrow$", False),
            ("pc5_variance_ratio", r"PC5 $\downarrow$", False),
            ("pc10_variance_ratio", r"PC10 $\downarrow$", False),
        ],
    ),
    (
        "Degradation",
        "representation_degradation",
        [
            (
                "mean_representation_degradation",
                r"\shortstack{MRD $\downarrow$}",
                False,
            ),
            # (
            #     "median_representation_degradation",
            #     r"\shortstack{Median deg. $\downarrow$}",
            #     False,
            # ),
        ],
    ),
    (
        "Derived",
        "derived",
        [
            # (
            #     "degradation_share",
            #     r"\shortstack{Deg. share $\downarrow$}",
            #     True,
            # ),
            # (
            #     "retained_similarity",
            #     r"\shortstack{RS $\uparrow$}",
            #     True,
            # ),
        ],
    ),
    (
        "Classification",
        "cosine_classification",
        [
            (
                "gaussian_negative_positive_accuracy",
                r"\shortstack{GNB N/P $\uparrow$}",
                True,
            ),
            (
                "gaussian_negative_transition_accuracy",
                r"\shortstack{GNB N/T $\uparrow$}",
                True,
            ),
            # (
            #     "histogram_negative_positive_accuracy",
            #     r"\shortstack{HNB N/P $\uparrow$}",
            #     True,
            # ),
            # (
            #     "histogram_negative_transition_accuracy",
            #     r"\shortstack{HNB N/T $\uparrow$}",
            #     True,
            # ),
        ],
    ),
]
LATEX_COLUMN_ORDER = [
    "average_pairwise_cosine",
    "mean_pairwise_cosine_similarity",
    "mean_cosine_similarity",
    # "mean_cosine_volatility",
    "semantic_drop_rate",
    "davies_bouldin_trimmed_index",
    # "davies_bouldin_trimmed_std",
    "mean_representation_degradation",
    # "degradation_share",
    # "retained_similarity",
    "pc1_variance_ratio",
    "pc5_variance_ratio",
    "pc10_variance_ratio",
    # "median_representation_degradation",
    "gaussian_negative_positive_accuracy",
    "gaussian_negative_transition_accuracy",
    # "histogram_negative_positive_accuracy",
    # "histogram_negative_transition_accuracy",
]
DEFAULT_INPUTS = {
    "intra_consistency": "data/table_intra_consistency.csv",
    "inter_separability": "data/table_inter_separability.csv",
    "embedding_anisotropy": "data/table_embedding_anisotropy.csv",
    "representation_degradation": (
        "data/table_representation_degradation.csv"
    ),
    "cosine_classification": "data/table_cosine_classification.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the evaluation summary tables into one CSV."
    )
    for metric, default_path in DEFAULT_INPUTS.items():
        parser.add_argument(
            f"--{metric.replace('_', '-')}",
            default=default_path,
            help=f"Path to the {metric.replace('_', ' ')} summary CSV."
        )
    parser.add_argument(
        "--output",
        default="data/table_all_metrics.csv",
        help="Path of the merged CSV table."
    )
    parser.add_argument(
        "--output-tex",
        default="../embeddings-analysis-paper/table_all_metrics.tex",
        help="Path of the combined LaTeX table."
    )
    parser.add_argument(
        "--caption",
        default="Combined evaluation results by chunking strategy and embedding model."
    )
    parser.add_argument("--label", default="tab:all-metrics")
    return parser.parse_args()


def read_table(
    path: Path,
) -> tuple[list[str], dict[tuple[str, str], dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Summary table does not exist: {path}")

    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        missing_columns = set(KEY_COLUMNS) - set(fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing key columns: {missing}")

        value_columns = [
            column for column in fieldnames if column not in KEY_COLUMNS
        ]
        rows: dict[tuple[str, str], dict[str, str]] = {}
        for row in reader:
            key = tuple(row[column] for column in KEY_COLUMNS)
            if key in rows:
                raise ValueError(
                    f"{path} contains duplicate model/chunking pair: {key}"
                )
            rows[key] = row

    return value_columns, rows


def calculate_degradation_share(
    *,
    mean_representation_degradation: str,
    mean_negative_cosine_similarity: str,
    mean_positive_cosine_similarity: str,
) -> float | None:
    """Return MRD's bounded share of MRD plus the cosine-similarity gap."""
    values = (
        mean_representation_degradation,
        mean_negative_cosine_similarity,
        mean_positive_cosine_similarity,
    )
    if any(value == "" for value in values):
        return None
    degradation, negative, positive = map(float, values)
    similarity_gap = positive - negative
    denominator = degradation + similarity_gap
    if degradation < 0 or similarity_gap < 0 or denominator <= 0:
        return None
    return degradation / denominator


def calculate_retained_similarity(
    *,
    mean_representation_degradation: str,
    mean_negative_cosine_similarity: str,
) -> float | None:
    """Return similarity retained above the random baseline."""
    values = (
        mean_representation_degradation,
        mean_negative_cosine_similarity,
    )
    if any(value == "" for value in values):
        return None
    degradation, negative = map(float, values)
    available_similarity = 1.0 - negative
    if degradation < 0 or available_similarity <= 0:
        return None
    return (available_similarity - degradation) / available_similarity


def build_derived_table(
    tables: dict[str, tuple[list[str], dict[tuple[str, str], dict[str, str]]]],
    keys: set[tuple[str, str]],
) -> tuple[list[str], dict[tuple[str, str], dict[str, str]]]:
    """Build metrics derived from columns belonging to multiple source tables."""
    columns = ["degradation_share", "retained_similarity"]
    rows = {}
    for key in keys:
        degradation = tables["representation_degradation"][1][key].get(
            "mean_representation_degradation", ""
        )
        negative = tables["embedding_anisotropy"][1][key].get(
            "average_pairwise_cosine", ""
        )
        ratio = calculate_degradation_share(
            mean_representation_degradation=degradation,
            mean_negative_cosine_similarity=negative,
            mean_positive_cosine_similarity=tables[
                "intra_consistency"
            ][1][key].get("mean_pairwise_cosine_similarity", ""),
        )
        retained = calculate_retained_similarity(
            mean_representation_degradation=degradation,
            mean_negative_cosine_similarity=negative,
        )
        rows[key] = {
            "model_id": key[0],
            "chunking_strategy": key[1],
            "degradation_share": "" if ratio is None else str(ratio),
            "retained_similarity": "" if retained is None else str(retained),
        }
    return columns, rows


def merge_tables(
    inputs: dict[str, Path],
    output_path: Path,
) -> tuple[
    int,
    dict[str, tuple[list[str], dict[tuple[str, str], dict[str, str]]]],
    list[tuple[str, str]],
]:
    tables = {metric: read_table(path) for metric, path in inputs.items()}
    key_sets = {metric: set(rows) for metric, (_, rows) in tables.items()}
    all_keys = set().union(*key_sets.values())

    mismatches = {
        metric: sorted(all_keys - keys)
        for metric, keys in key_sets.items()
        if keys != all_keys
    }
    if mismatches:
        details = "; ".join(
            f"{metric} is missing {len(keys)} pair(s)"
            for metric, keys in mismatches.items()
        )
        raise ValueError(f"Input tables do not contain identical keys: {details}")

    tables["derived"] = build_derived_table(tables, all_keys)

    output_columns = list(KEY_COLUMNS)
    for metric, (value_columns, _) in tables.items():
        output_columns.extend(
            f"{metric}_{column}" for column in value_columns
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_keys = order_model_chunk_keys(all_keys)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_columns)
        writer.writeheader()
        for key in ordered_keys:
            merged_row = dict(zip(KEY_COLUMNS, key))
            for metric, (value_columns, rows) in tables.items():
                source_row = rows[key]
                merged_row.update(
                    {
                        f"{metric}_{column}": source_row[column]
                        for column in value_columns
                    }
                )
            writer.writerow(merged_row)

    return len(all_keys), tables, ordered_keys


def order_model_chunk_keys(
    keys: set[tuple[str, str]] | list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Order model/chunk pairs by the editable configuration lists above."""
    if len(MODEL_ORDER) != len(set(MODEL_ORDER)):
        raise ValueError("MODEL_ORDER contains duplicate model keys")
    if len(CHUNK_ORDER) != len(set(CHUNK_ORDER)):
        raise ValueError("CHUNK_ORDER contains duplicate chunk keys")
    model_rank = {model_id: index for index, model_id in enumerate(MODEL_ORDER)}
    chunk_rank = {strategy: index for index, strategy in enumerate(CHUNK_ORDER)}
    return sorted(
        keys,
        key=lambda key: (
            model_rank.get(key[0], len(model_rank)),
            "" if key[0] in model_rank else key[0],
            chunk_rank.get(key[1], len(chunk_rank)),
            "" if key[1] in chunk_rank else key[1],
        ),
    )


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


def format_metric(value: str, *, as_percent: bool = False) -> str:
    if value == "":
        return "--"
    if as_percent:
        return f"{float(value) * 100:.2f}\\%"
    if value.lstrip("-").isdigit():
        return value
    return f"{float(value):.4f}"


def ordered_latex_columns() -> list[tuple[str, str, str, bool]]:
    columns_by_key = {
        column: (metric, column, heading, as_percent)
        for _, metric, columns in LATEX_GROUPS
        for column, heading, as_percent in columns
    }
    configured_columns = set(LATEX_COLUMN_ORDER)
    defined_columns = set(columns_by_key)
    if len(configured_columns) != len(LATEX_COLUMN_ORDER):
        raise ValueError("LATEX_COLUMN_ORDER contains duplicate column keys")
    if configured_columns != defined_columns:
        missing = sorted(defined_columns - configured_columns)
        unknown = sorted(configured_columns - defined_columns)
        raise ValueError(
            "LATEX_COLUMN_ORDER must contain every LATEX_GROUPS column exactly once; "
            f"missing={missing}, unknown={unknown}"
        )
    return [columns_by_key[column] for column in LATEX_COLUMN_ORDER]


def write_latex_table(
    tables: dict[
        str,
        tuple[list[str], dict[tuple[str, str], dict[str, str]]],
    ],
    keys: list[tuple[str, str]],
    output_path: Path,
    caption: str,
    label: str,
) -> None:
    latex_columns = ordered_latex_columns()
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{latex_escape(label)}}}",
        rf"\resizebox{{{LATEX_TABLE_WIDTH}}}{{!}}{{%",
        r"\begin{tabular}{ll" + "r" * len(latex_columns) + "}",
        r"\toprule",
        "Model & Chunking & "
        + " & ".join(
            heading
            for _, _, heading, _ in latex_columns
        )
        + r" \\",
        r"\midrule",
    ]

    previous_model: str | None = None
    for model_id, strategy in keys:
        if previous_model is not None and model_id != previous_model:
            lines.append(r"\midrule")
        model = "" if model_id == previous_model else MODEL_NAMES.get(model_id, model_id)
        key = (model_id, strategy)
        metrics = [
            (tables[metric][1][key].get(column, ""), as_percent)
            for metric, column, _, as_percent in latex_columns
        ]
        lines.append(
            f"{latex_escape(model)} & "
            f"{latex_escape(CHUNKING_NAMES.get(strategy, strategy))} & "
            f"{' & '.join(format_metric(metric, as_percent=as_percent) for metric, as_percent in metrics)} \\\\"
        )
        previous_model = model_id

    legend_lines = {
        "P": "paragraph",
        "SN/M": "N sentences with M sentences overlap",
        "TN/M": "N tokens with M tokens overlap",
        "RN/M": "recursive with N characters and M characters overlap",
    }

    legend_lines_compiled = "; ".join(
        f"{latex_escape(short)}: {latex_escape(long)}"
        for short, long in legend_lines.items()
    )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{2pt}",
            # r"\\ \scriptsize " + legend_lines_compiled,
            r"\end{table*}",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    inputs = {
        metric: Path(getattr(args, metric))
        for metric in DEFAULT_INPUTS
    }
    output_path = Path(args.output)
    row_count, tables, keys = merge_tables(inputs, output_path)
    output_tex = Path(args.output_tex)
    write_latex_table(
        tables,
        keys,
        output_tex,
        args.caption,
        args.label,
    )
    print(f"Merged {row_count} model/chunking pairs.")
    print(f"Combined table: {output_path.resolve()}")
    print(f"LaTeX table: {output_tex.resolve()}")


if __name__ == "__main__":
    main()
