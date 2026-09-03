from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from tqdm import tqdm

from embedding_geometry.statistics.analyze_text_structure import (
    percentile,
    words,
)
from semora.storage import Database


def suffix_prefix_overlap(previous: str, following: str) -> int:
    """Return the longest suffix of previous matching a prefix of following."""
    if not previous or not following:
        return 0
    pattern = following
    prefix = [0] * len(pattern)
    matched = 0
    for index in range(1, len(pattern)):
        while matched and pattern[index] != pattern[matched]:
            matched = prefix[matched - 1]
        if pattern[index] == pattern[matched]:
            matched += 1
        prefix[index] = matched

    text = previous[-len(pattern):]
    matched = 0
    for index, character in enumerate(text):
        while matched and character != pattern[matched]:
            matched = prefix[matched - 1]
        if character == pattern[matched]:
            matched += 1
        if matched == len(pattern):
            if index == len(text) - 1:
                return matched
            matched = prefix[matched - 1]
    return matched


def normalize_token_chunk_start(text: str) -> str:
    """Remove a WordPiece marker left when decoding a continuation token first."""
    return text[2:] if text.startswith("##") else text


def summarize_run(
    run: Mapping[str, object],
    chunks: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    character_counts = [len(str(row["text"] or "")) for row in chunks]
    word_counts = [len(words(str(row["text"] or ""))) for row in chunks]
    chunks_per_article = Counter(str(row["article_id"]) for row in chunks)
    overlaps: list[float] = []
    transition_count = 0
    for previous, following in tqdm(zip(chunks, chunks[1:])):
        if previous["article_id"] != following["article_id"]:
            continue
        transition_count += 1
        following_text = str(following["text"] or "")
        if run["method"] == "token":
            following_text = normalize_token_chunk_start(following_text)
        overlap = suffix_prefix_overlap(str(previous["text"] or ""), following_text)
        if overlap:
            overlaps.append(100.0 * overlap / len(following_text))

    return {
        "chunking_run_id": run["chunking_run_id"],
        "method": run["method"],
        "config_json": run["config_json"],
        "num_articles": len(chunks_per_article),
        "num_chunks": len(chunks),
        "mean_chunks_per_article": statistics.fmean(chunks_per_article.values()),
        "median_chunks_per_article": statistics.median(chunks_per_article.values()),
        "mean_characters": statistics.fmean(character_counts),
        "median_characters": statistics.median(character_counts),
        "p95_characters": percentile(character_counts, 0.95),
        "mean_words": statistics.fmean(word_counts),
        "median_words": statistics.median(word_counts),
        "num_transitions": transition_count,
        "num_overlapping_transitions": len(overlaps),
        "overlap_transition_percent": (
            100.0 * len(overlaps) / transition_count if transition_count else ""
        ),
        "mean_positive_overlap_percent": statistics.fmean(overlaps) if overlaps else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize stored chunks by chunking run.")
    parser.add_argument("--db-path", default="data/newspapers.sqlite")
    parser.add_argument("--chunking-run-ids", nargs="+")
    parser.add_argument(
        "--output",
        default="data/chunks.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = list(dict.fromkeys(args.chunking_run_ids or []))
    db = Database(args.db_path)
    try:
        db.initialize()
        runs = db.get_chunking_runs_by_ids(requested) if requested else db.get_chunking_runs()
        known = {str(run["chunking_run_id"]) for run in runs}
        unknown = [run_id for run_id in requested if run_id not in known]
        if unknown:
            raise ValueError(f"Unknown chunking run ID(s): {', '.join(unknown)}")
        first_runs = {}
        for run in runs:
            strategy = (str(run["method"]), str(run["config_json"]))
            first_runs.setdefault(strategy, run)
        summaries = []
        for run in first_runs.values():
            run_id = str(run["chunking_run_id"])
            chunks = db.get_chunks(chunking_run_id=run_id)
            if chunks:
                summaries.append(summarize_run(run, chunks))
            else:
                print(f"Warning: no chunks found for {run_id}")
    finally:
        db.close()

    if not summaries:
        raise ValueError("No chunks are available for the selected chunking runs.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Summarized {len(summaries)} chunking runs: {output.resolve()}")


if __name__ == "__main__":
    main()
