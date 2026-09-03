import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from embedding_geometry.statistics.summarize_chunks import (
    normalize_token_chunk_start,
    suffix_prefix_overlap,
    summarize_run,
)


def test_chunk_summary_measures_actual_consecutive_overlap() -> None:
    run = {"chunking_run_id": "run-1", "method": "split", "config_json": "{}"}
    chunks = [
        {"article_id": "a", "text": "one two abc"},
        {"article_id": "a", "text": "abc fourth"},
        {"article_id": "a", "text": "unrelated"},
        {"article_id": "b", "text": "another article"},
    ]

    summary = summarize_run(run, chunks)

    assert suffix_prefix_overlap("one two abc", "abc fourth") == 3
    assert summary["num_articles"] == 2
    assert summary["num_chunks"] == 4
    assert summary["mean_chunks_per_article"] == 2
    assert summary["median_chunks_per_article"] == 2
    assert summary["overlap_transition_percent"] == 50
    assert summary["mean_positive_overlap_percent"] == 30


def test_wordpiece_continuation_at_chunk_start_can_be_matched() -> None:
    previous = "The tokenization overlap"
    following = normalize_token_chunk_start("##ization overlap continues")

    assert following == "ization overlap continues"
    assert suffix_prefix_overlap(previous, following) == len("ization overlap")
