from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from semora.storage import Database


REQUIRED_COLUMNS = {
    "article_id", "is_valid", "numeric_or_special_character_ratio",
    "letter_character_ratio", "vowel_letter_ratio",
    "mean_characters_per_word", "tokens_per_word"
}
OUTLIER_DIRECTIONS = {
    "letter_character_ratio": "low",
    "vowel_letter_ratio": "both",
    "mean_characters_per_word": "both",
    "tokens_per_word": "high"
}


@dataclass(frozen=True)
class RobustBounds:
    median: float
    mad: float
    lower: float
    upper: float


def parse_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"Invalid is_valid value: {value!r}")


def robust_bounds(values: list[float], multiplier: float) -> RobustBounds:
    if not values:
        raise ValueError("Cannot calculate outlier bounds without values.")
    median = float(statistics.median(values))
    mad = float(statistics.median(abs(value - median) for value in values))
    robust_sigma = 1.4826 * mad
    width = multiplier * robust_sigma
    return RobustBounds(median, mad, median - width, median + width)


def outlier_reasons(
    row: dict[str, str],
    bounds_by_metric: dict[str, RobustBounds],
) -> list[str]:
    reasons = []
    for metric, direction in OUTLIER_DIRECTIONS.items():
        value = float(row[metric])
        bounds = bounds_by_metric[metric]
        if direction in {"low", "both"} and value < bounds.lower:
            reasons.append(f"outlier_low_{metric}")
        elif direction in {"high", "both"} and value > bounds.upper:
            reasons.append(f"outlier_high_{metric}")
    return reasons


def classify_article(
    row: dict[str, str],
    bounds_by_metric: dict[str, RobustBounds],
    *,
    max_numeric_or_special_ratio: float,
    minimum_outlier_metrics: int,
) -> list[str]:
    reasons = []
    if float(row["numeric_or_special_character_ratio"]) > max_numeric_or_special_ratio:
        reasons.append("numeric_or_special_ratio_above_limit")
    statistical_reasons = outlier_reasons(row, bounds_by_metric)
    if len(statistical_reasons) >= minimum_outlier_metrics:
        reasons.extend(statistical_reasons)
    return reasons


def load_statistics(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Statistics file not found: {path}. Generate it first with: "
            "py -m embedding_geometry.statistics.analyze_text_structure "
            "--db-path data/newspapers.sqlite --all-articles"
        )
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}. "
                "Rerun analyze_text_structure.py with --all-articles."
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"No article statistics found in {path}")
    return rows


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mark currently valid articles invalid using text-quality statistics."
    )
    parser.add_argument(
        "--db-path",
        default="data/newspapers.sqlite"
    )
    parser.add_argument(
        "--statistics",
        default="data/text_structure/article_structure.csv"
    )
    parser.add_argument(
        "--report",
        default="data/text_structure/statistical_cleaning_report.csv"
    )
    parser.add_argument(
        "--max-numeric-or-special-ratio",
        type=float,
        default=0.10
    )
    parser.add_argument(
        "--outlier-mad-multiplier",
        type=float,
        default=4.5
    )
    parser.add_argument(
        "--minimum-outlier-metrics",
        type=int,
        default=2
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply invalidations. Without this flag, only write a dry-run report."
    )
    args = parser.parse_args()

    if not 0.0 <= args.max_numeric_or_special_ratio <= 1.0:
        raise ValueError("Maximum numeric-or-special ratio must be in [0, 1].")
    if args.outlier_mad_multiplier <= 0.0:
        raise ValueError("Outlier MAD multiplier must be positive.")
    if args.minimum_outlier_metrics < 1:
        raise ValueError("Minimum outlier metrics must be at least 1.")

    try:
        rows = load_statistics(Path(args.statistics))
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    valid_rows = [row for row in rows if parse_bool(row["is_valid"])]
    if not valid_rows:
        raise ValueError("No valid reference articles found in the statistics CSV.")

    bounds_by_metric = {
        metric: robust_bounds(
            [float(row[metric]) for row in valid_rows],
            args.outlier_mad_multiplier
        )
        for metric in OUTLIER_DIRECTIONS
    }
    print(f"Reference population: {len(valid_rows)} currently valid articles")
    for metric, bounds in bounds_by_metric.items():
        print(
            f"{metric}: median={bounds.median:.4f}, MAD={bounds.mad:.4f}, "
            f"bounds=[{bounds.lower:.4f}, {bounds.upper:.4f}]"
        )

    report_rows = []
    invalidations: list[tuple[str, str]] = []
    reason_counts: Counter[str] = Counter()
    for row in rows:
        currently_valid = parse_bool(row["is_valid"])
        reasons = classify_article(
            row,
            bounds_by_metric,
            max_numeric_or_special_ratio=args.max_numeric_or_special_ratio,
            minimum_outlier_metrics=args.minimum_outlier_metrics
        ) if currently_valid else []
        reason_counts.update(reasons)
        if reasons:
            invalidations.append((";".join(reasons), row["article_id"]))
        report_rows.append({
            "article_id": row["article_id"],
            "was_valid": int(currently_valid),
            "would_invalidate": int(bool(reasons)),
            "reasons": ";".join(reasons),
            "numeric_or_special_character_ratio": row["numeric_or_special_character_ratio"],
            **{metric: row[metric] for metric in OUTLIER_DIRECTIONS}
        })

    write_report(Path(args.report), report_rows)
    print(f"Would invalidate {len(invalidations)} of {len(valid_rows)} valid articles.")
    for reason, count in reason_counts.most_common():
        print(f"  {reason}: {count}")
    print(f"Report: {Path(args.report).resolve()}")

    if args.apply:
        db = Database(args.db_path)
        try:
            db.invalidate_articles(invalidations)
        finally:
            db.close()
        print(f"Applied {len(invalidations)} invalidations to {args.db_path}.")
    else:
        print("Dry run only. Review the report, then rerun with --apply.")


if __name__ == "__main__":
    main()
