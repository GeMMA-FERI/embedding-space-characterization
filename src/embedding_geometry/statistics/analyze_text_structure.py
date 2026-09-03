from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import statistics
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from tqdm import tqdm

from semora.text.markdown import remove_markdown_images
from semora.storage import Database


WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
PARAGRAPH_RE = re.compile(r"(?:\r?\n\s*){2,}")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
VOWELS = set("aeiou")
SHORT_CHUNK_WORDS = 3
DEFAULT_TOKEN_CHUNK_SIZES = (32, 64, 128, 256)


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...


def words(text: str) -> list[str]:
    """Return Unicode letter-only word sequences."""
    return WORD_RE.findall(text)


def non_empty_parts(pattern: re.Pattern[str], text: str) -> list[str]:
    return [part.strip() for part in pattern.split(text.strip()) if part.strip()]


def mean(values: Sequence[int | float]) -> float:
    return statistics.fmean(values) if values else 0.0


def ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def percentile_from_sorted(
    ordered: Sequence[int | float],
    probability: float,
) -> float:
    """Calculate a percentile from values already sorted in ascending order."""
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Percentile probability must be between 0 and 1.")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def percentile(values: Sequence[int | float], probability: float) -> float:
    """Calculate a linearly interpolated percentile without external dependencies."""
    return percentile_from_sorted(sorted(values), probability)


def analyze_article(
    article,
    tokenizer: Tokenizer,
    token_chunk_sizes: Sequence[int] = DEFAULT_TOKEN_CHUNK_SIZES,
) -> tuple[dict, list[int], list[int], list[int]]:
    text = remove_markdown_images(article["content"] or "")
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    return analyze_article_with_token_count(article, token_count, token_chunk_sizes)


def analyze_article_with_token_count(
    article,
    token_count: int,
    token_chunk_sizes: Sequence[int] = DEFAULT_TOKEN_CHUNK_SIZES,
) -> tuple[dict, list[int], list[int], list[int]]:
    text = remove_markdown_images(article["content"] or "")
    article_words = words(text)
    paragraphs = non_empty_parts(PARAGRAPH_RE, text)
    sentences = non_empty_parts(SENTENCE_RE, text)

    paragraph_lengths = [len(words(paragraph)) for paragraph in paragraphs]
    sentence_lengths = [len(words(sentence)) for sentence in sentences]
    word_lengths = [len(word) for word in article_words]
    normalized_words = [word.casefold() for word in article_words]

    character_count = len(text)
    non_whitespace_count = sum(not character.isspace() for character in text)
    letter_count = sum(character.isalpha() for character in text)
    numeric_count = sum(character.isnumeric() for character in text)
    special_symbol_count = sum(
        not character.isalnum() and not character.isspace()
        for character in text
    )
    letters = [character.casefold() for character in text if character.isalpha()]
    vowel_count = sum(character in VOWELS for character in letters)
    consonant_count = letter_count - vowel_count

    row = {
        "article_id": article["article_id"],
        "title": article["title"] or "",
        "is_valid": article["is_valid"] if "is_valid" in article.keys() else "",
        "cleaning_reason": (
            article["cleaning_reason"] or ""
            if "cleaning_reason" in article.keys()
            else ""
        ),
        "characters": character_count,
        "non_whitespace_characters": non_whitespace_count,
        "letters": letter_count,
        "numeric_characters": numeric_count,
        "special_symbol_characters": special_symbol_count,
        "words": len(article_words),
        "tokens": token_count,
        "paragraphs": len(paragraphs),
        "sentences": len(sentences),
        "mean_words_per_paragraph": mean(paragraph_lengths),
        "max_words_per_paragraph": max(paragraph_lengths, default=0),
        "short_paragraph_ratio": ratio(
            sum(length <= SHORT_CHUNK_WORDS for length in paragraph_lengths),
            len(paragraph_lengths),
        ),
        "mean_words_per_sentence": mean(sentence_lengths),
        "max_words_per_sentence": max(sentence_lengths, default=0),
        "short_sentence_ratio": ratio(
            sum(length <= SHORT_CHUNK_WORDS for length in sentence_lengths),
            len(sentence_lengths),
        ),
        "mean_characters_per_word": mean(word_lengths),
        "median_characters_per_word": (
            statistics.median(word_lengths) if word_lengths else 0.0
        ),
        "max_characters_per_word": max(word_lengths, default=0),
        "tokens_per_word": ratio(token_count, len(article_words)),
        "type_token_ratio": ratio(len(set(normalized_words)), len(normalized_words)),
        "letter_character_ratio": ratio(letter_count, character_count),
        "vowel_letter_ratio": ratio(vowel_count, letter_count),
        "numeric_character_ratio": ratio(numeric_count, character_count),
        "special_symbol_character_ratio": ratio(
            special_symbol_count,
            character_count,
        ),
        "numeric_or_special_character_ratio": ratio(
            numeric_count + special_symbol_count,
            character_count,
        ),
        "consonant_vowel_ratio": (
            consonant_count / vowel_count if vowel_count else None
        )
    }
    for chunk_size in token_chunk_sizes:
        row[f"estimated_token_chunks_{chunk_size}"] = (
            math.ceil(token_count / chunk_size) if token_count else 0
        )

    return row, paragraph_lengths, sentence_lengths, word_lengths


def summarize_metric(name: str, values: Sequence[int | float]) -> dict[str, int | float | str]:
    ordered = sorted(values)
    q1 = percentile_from_sorted(ordered, 0.25)
    median = percentile_from_sorted(ordered, 0.50)
    q3 = percentile_from_sorted(ordered, 0.75)
    iqr = q3 - q1
    average = mean(values)
    standard_deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
    absolute_deviations = [abs(float(value) - median) for value in values]
    mad = percentile_from_sorted(sorted(absolute_deviations), 0.50)
    robust_sigma = 1.4826 * mad
    return {
        "metric": name,
        "count": len(values),
        "mean": average,
        "standard_deviation": standard_deviation,
        "minimum": ordered[0],
        "p01": percentile_from_sorted(ordered, 0.01),
        "p05": percentile_from_sorted(ordered, 0.05),
        "q1": q1,
        "median": median,
        "q3": q3,
        "p95": percentile_from_sorted(ordered, 0.95),
        "p99": percentile_from_sorted(ordered, 0.99),
        "maximum": ordered[-1],
        "iqr": iqr,
        "iqr_lower_fence": q1 - 1.5 * iqr,
        "iqr_upper_fence": q3 + 1.5 * iqr,
        "mean_minus_3_std": average - 3.0 * standard_deviation,
        "mean_plus_3_std": average + 3.0 * standard_deviation,
        "median_absolute_deviation": mad,
        "robust_lower_fence": median - 3.0 * robust_sigma,
        "robust_upper_fence": median + 3.0 * robust_sigma
    }


def save_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    path: Path,
    distributions: Sequence[tuple[str, Sequence[int | float]]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install matplotlib to create plots.") from error

    column_count = 3
    row_count = math.ceil(len(distributions) / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(13, 3.7 * row_count),
        squeeze=False
    )
    for axis, (label, values) in zip(axes.flat, distributions):
        axis.hist(values, bins=50, edgecolor="black")
        axis.set_title(label)
        axis.set_xlabel(label)
        axis.set_ylabel("Frequency")
    for axis in axes.flat[len(distributions):]:
        axis.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def parse_token_chunk_sizes(values: Sequence[int]) -> tuple[int, ...]:
    if not values or any(value < 1 for value in values):
        raise ValueError("Token chunk sizes must be positive integers.")
    if len(set(values)) != len(values):
        raise ValueError("Token chunk sizes must not contain duplicates.")
    return tuple(values)


def batched(values: Iterable, batch_size: int):
    iterator = iter(values)
    while batch := list(itertools.islice(iterator, batch_size)):
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate structural and cleaning-oriented statistics for stored articles."
    )
    parser.add_argument(
        "--db-path",
        default="data/newspapers.sqlite"
    )
    parser.add_argument(
        "--output-dir",
        default="data/text_structure"
    )
    parser.add_argument(
        "--tokenizer-model-id",
        default="bert-base-multilingual-cased",
        help="Hugging Face tokenizer used for article and chunk-count estimates."
    )
    parser.add_argument(
        "--token-chunk-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOKEN_CHUNK_SIZES),
        help="Fixed-token chunk sizes to estimate (default: 32 64 128 256)."
    )
    parser.add_argument(
        "--tokenizer-batch-size",
        type=int,
        default=256,
        help="Articles tokenized per batch. Reduce if memory is limited."
    )
    parser.add_argument(
        "--all-articles",
        action="store_true",
        help="Include articles that were not marked as valid.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true"
    )
    args = parser.parse_args()
    token_chunk_sizes = parse_token_chunk_sizes(args.token_chunk_sizes)
    if args.tokenizer_batch_size < 1:
        raise ValueError("Tokenizer batch size must be at least 1.")

    print(f"Loading tokenizer: {args.tokenizer_model_id}", flush=True)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Transformers is required to calculate tokens per article. "
            "Install it with: pip install transformers"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model_id)

    print(f"Opening article database: {args.db_path}", flush=True)
    db = Database(args.db_path)
    try:
        print("Counting selected non-empty articles.", flush=True)
        article_count = db.count_articles_for_analysis(
            include_invalid=args.all_articles
        )
        if article_count == 0:
            raise ValueError("No non-empty articles found.")
        print(f"Found {article_count} articles; streaming analysis will begin now.", flush=True)

        rows = []
        paragraph_lengths: list[int] = []
        sentence_lengths: list[int] = []
        word_lengths: list[int] = []
        articles = db.iter_articles_for_analysis(
            include_invalid=args.all_articles,
            fetch_size=max(1_000, args.tokenizer_batch_size)
        )
        batch_count = math.ceil(article_count / args.tokenizer_batch_size)
        for article_batch in tqdm(
            batched(articles, args.tokenizer_batch_size),
            total=batch_count,
            desc="Analyzing article batches",
            unit="batch"
        ):
            texts = [
                remove_markdown_images(article["content"] or "")
                for article in article_batch
            ]
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False
            )
            token_ids = encoded["input_ids"]
            for article, article_token_ids in zip(article_batch, token_ids):
                row, paragraphs, sentences, article_word_lengths = analyze_article_with_token_count(
                    article,
                    len(article_token_ids),
                    token_chunk_sizes
                )
                rows.append(row)
                paragraph_lengths.extend(paragraphs)
                sentence_lengths.extend(sentences)
                word_lengths.extend(article_word_lengths)
    finally:
        db.close()

    article_metrics = [
        "characters",
        "non_whitespace_characters",
        "letters",
        "numeric_characters",
        "special_symbol_characters",
        "words",
        "tokens",
        "paragraphs",
        "sentences",
        "mean_words_per_paragraph",
        "max_words_per_paragraph",
        "short_paragraph_ratio",
        "mean_words_per_sentence",
        "max_words_per_sentence",
        "short_sentence_ratio",
        "mean_characters_per_word",
        "median_characters_per_word",
        "max_characters_per_word",
        "tokens_per_word",
        "type_token_ratio",
        "letter_character_ratio",
        "vowel_letter_ratio",
        "numeric_character_ratio",
        "special_symbol_character_ratio",
        "numeric_or_special_character_ratio",
        "consonant_vowel_ratio",
        *[f"estimated_token_chunks_{size}" for size in token_chunk_sizes]
    ]
    metric_values = {
        metric: [row[metric] for row in rows if row[metric] is not None]
        for metric in article_metrics
    }
    print("Calculating distribution summaries.", flush=True)
    summary_rows = [
        summarize_metric(metric, values)
        for metric, values in tqdm(metric_values.items(), desc="Summarizing metrics")
        if values
    ]

    print(f"Analyzed {len(rows)} articles")
    for summary in summary_rows:
        print(
            f"{summary['metric']}: Mean={summary['mean']:.4f}, "
            f"Median={summary['median']:.4f}, P05={summary['p05']:.4f}, "
            f"P95={summary['p95']:.4f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Writing CSV results.", flush=True)
    save_csv(output_dir / "article_structure.csv", rows)
    save_csv(output_dir / "article_structure_summary.csv", summary_rows)

    if not args.no_plots:
        print("Creating distribution plots.", flush=True)
        plotted_metrics = [
            "characters",
            "words",
            "tokens",
            "paragraphs",
            "sentences",
            "mean_words_per_paragraph",
            "mean_words_per_sentence",
            "mean_characters_per_word",
            "tokens_per_word",
            "letter_character_ratio",
            "vowel_letter_ratio",
            "numeric_character_ratio",
            "special_symbol_character_ratio",
            "numeric_or_special_character_ratio",
            "short_paragraph_ratio",
            "short_sentence_ratio"
        ]
        save_plot(
            output_dir / "structural_dimensions.png",
            [(metric.replace("_", " ").title(), metric_values[metric]) for metric in plotted_metrics]
        )

    print(f"Successfully analyzed {len(rows)} articles.")
    print(f"Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
