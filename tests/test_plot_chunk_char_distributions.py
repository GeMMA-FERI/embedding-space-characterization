from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from embedding_geometry.statistics.plot_chunk_char_distributions import (
    build_chunk_histogram_rows,
    collect_chunk_counts,
    count_chunk_metric,
    load_chunk_histogram_csv,
    parse_args,
    save_chunk_histogram_csv,
    save_chunk_count_histogram,
    select_chunk_histogram_rows,
)
from semora.storage import Database


def test_collect_chunk_counts_separates_runs_and_uses_unicode_words() -> None:
    rows = [
        {"chunking_run_id": "short", "text": "One two 123"},
        {"chunking_run_id": "long", "text": "Žodis—kitas, third!"},
        {"chunking_run_id": "short", "text": "Four_words"},
    ]

    result = collect_chunk_counts(rows, ["short", "long"], metric="word")

    assert result == {"short": [2, 2], "long": [3]}


def test_chunking_run_ids_are_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["plot_chunk_char_distributions"])

    args = parse_args()

    assert args.chunking_run_ids is None
    assert args.metric == "char"


def test_metric_and_minimum_x_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_chunk_char_distributions", "--metric", "word", "--min-x", "10"],
    )

    args = parse_args()

    assert args.metric == "word"
    assert args.min_x == 10
    assert args.plot_style == "histogram"


def test_line_plot_style_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_chunk_char_distributions", "--plot-style", "line"],
    )

    assert parse_args().plot_style == "line"


def test_plot_scale_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_chunk_char_distributions", "--plot-scale", "1.8"],
    )

    assert parse_args().plot_scale == 1.8


def test_database_exposes_chunking_runs_for_plotter(tmp_path: pathlib.Path) -> None:
    db = Database(tmp_path / "newspapers.sqlite")
    try:
        db.initialize()
        with db.conn:
            db.conn.execute(
                "INSERT INTO runs (run_id, run_type) VALUES ('run-1', 'chunking')"
            )
            db.conn.execute(
                """
                INSERT INTO chunking_runs (chunking_run_id, run_id, method)
                VALUES ('sentence', 'run-1', 'sentence')
                """
            )

        all_runs = db.get_chunking_runs()
        selected_runs = db.get_chunking_runs_by_ids(["sentence"])

        assert [row["chunking_run_id"] for row in all_runs] == ["sentence"]
        assert [row["chunking_run_id"] for row in selected_runs] == ["sentence"]
        assert list(db.iter_chunk_texts_for_runs(["sentence"])) == []
    finally:
        db.close()


def test_character_and_word_metrics() -> None:
    text = "One two 123"

    assert count_chunk_metric(text, "char") == 11
    assert count_chunk_metric(text, "word") == 2


def test_save_chunk_count_histogram_writes_overlay_plot(
    tmp_path: pathlib.Path,
) -> None:
    output = tmp_path / "words-per-chunk.png"

    save_chunk_count_histogram(
        {"short": [5, 8, 10], "long": [20, 25, 30]},
        labels={"short": "short (tokens)", "long": "long (sentences)"},
        output=output,
        bins=10,
        metric="word",
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_log_histogram_requires_positive_word_counts(
    tmp_path: pathlib.Path,
) -> None:
    output = tmp_path / "words-per-chunk-log.png"
    labels = {"short": "short (tokens)"}

    save_chunk_count_histogram(
        {"short": [1, 10, 100]},
        labels=labels,
        output=output,
        bins=10,
        metric="word",
        log_x=True,
    )
    assert output.is_file()

    with pytest.raises(ValueError, match="greater than zero"):
        save_chunk_count_histogram(
            {"short": [0, 1, 10]},
            labels=labels,
            output=output,
            bins=10,
            metric="word",
            log_x=True,
        )


def test_save_line_distribution_with_dots_and_fill(tmp_path: pathlib.Path) -> None:
    output = tmp_path / "words-per-chunk-lines.png"

    save_chunk_count_histogram(
        {"short": [5, 8, 10, 10], "long": [20, 25, 25, 30]},
        labels={"short": "short (tokens)", "long": "long (sentences)"},
        output=output,
        bins=8,
        metric="word",
        plot_style="line",
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_chunk_histogram_csv_round_trip(tmp_path: pathlib.Path) -> None:
    rows = build_chunk_histogram_rows(
        {"short": [5, 8, 10], "long": [20, 25, 30]},
        methods={"short": "token", "long": "sentence"},
        run_names={"short": "Short", "long": "Long"},
        bins=5,
        metric="char",
        log_x=False,
    )
    output = tmp_path / "chunk-char-distributions.csv"

    save_chunk_histogram_csv(rows, output)

    assert load_chunk_histogram_csv(output) == rows


def test_select_runs_from_all_runs_histogram_in_requested_order() -> None:
    rows = [
        {"chunking_run_id": "a", "bin_index": 0},
        {"chunking_run_id": "b", "bin_index": 0},
        {"chunking_run_id": "c", "bin_index": 0},
    ]

    selected_rows, selected_ids = select_chunk_histogram_rows(
        rows,
        ["c", "a"],
        [],
    )

    assert selected_ids == ["c", "a"]
    assert [row["chunking_run_id"] for row in selected_rows] == ["c", "a"]
