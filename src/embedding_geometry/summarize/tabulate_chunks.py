from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = {
    "chunking_run_id",
    "num_articles",
    "num_chunks",
    "mean_chunks_per_article",
    "median_chunks_per_article",
    "mean_characters",
    "median_characters",
    "mean_words",
    "median_words",
    "overlap_transition_percent",
    "mean_positive_overlap_percent",
}

CHUNK_NAMES = {
    "noop": "No-op",
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

CHUNK_ORDER = [
    "noop",
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

def latex_escape(value: str) -> str:
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    return "".join(replacements.get(character, character) for character in value)


def pair(row: dict[str, str], mean: str, median: str) -> str:
    return f"{float(row[mean]):.1f} / {float(row[median]):.1f}"


def percent(value: str) -> str:
    return "--" if value == "" else f"{float(value):.1f}\\%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a LaTeX chunk summary table.")
    parser.add_argument("--input", default="data/chunks.csv")
    parser.add_argument(
        "--output-tex",
        default="../embeddings-analysis-paper/table_chunks.tex",
    )
    parser.add_argument("--caption", default="Chunking strategy statistics.")
    parser.add_argument("--label", default="tab:chunks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.input).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Input CSV contains no chunking runs.")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{latex_escape(args.caption)}}}",
        f"\\label{{{latex_escape(args.label)}}}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Chunking & Chunks & C/A & Ch/C & Memory & Overlap \\",
        r"\midrule",
    ]
    rows = sorted(rows, key=lambda row: CHUNK_ORDER.index(row["chunking_run_id"]) if row["chunking_run_id"] in CHUNK_ORDER else len(CHUNK_ORDER))
    for row in rows:
        if row["chunking_run_id"] not in CHUNK_NAMES:
            print("Warning: No name provided for chunking run ID:", row["chunking_run_id"])
            continue
        chunk_name = CHUNK_NAMES[row['chunking_run_id']]
        # bold_name = r"\textbf{" + chunk_name[0] + "}" + chunk_name[1:] if chunk_name != "No-op" else chunk_name
        memory = float(row['mean_characters']) * float(row['num_chunks'])
        # bytes -> human-readable megabytes
        memory /= 1024 * 1024
        memory = round(memory, 0)
        lines.append(
            f"{chunk_name} & "
            f"{int(row['num_chunks']):,} & "
            f"{float(row['mean_chunks_per_article']):.2f} & "
            f"{float(row['mean_characters']):.2f} & "
            # f"{percent(row['overlap_transition_percent'])} & "
            # chunks * ch / c
            f"{memory:.0f} MB & "
            f"{percent(row['mean_positive_overlap_percent'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\\ \scriptsize C/A = mean chunks per article, Ch/C = mean characters per chunk.",
            r"\end{table}",
        ]
    )

    output = Path(args.output_tex)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"LaTeX table saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
