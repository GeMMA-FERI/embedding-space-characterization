from __future__ import annotations

import argparse
import re

from tqdm import tqdm
from datetime import datetime
from semora.text.markdown import remove_markdown_images
from semora.storage import Database, Run


MIN_WORDS = 80
MIN_CHARS = 400
MIN_AVG_WORD_LENGTH = 3.5
MIN_LETTER_RATIO = 0.60
MIN_VOWEL_RATIO = 0.25

WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)
VOWELS = set("aeiou")


def vowel_ratio(text: str) -> float:
    letters = [character.casefold() for character in text if character.isalpha()]

    if not letters:
        return 0.0

    return sum(character in VOWELS for character in letters) / len(letters)


def letter_ratio(text: str) -> float:
    if not text:
        return 0.0

    return sum(character.isalpha() for character in text) / len(text)


def average_word_length(text: str) -> float:
    words = WORDS.findall(text)

    if not words:
        return 0.0

    return sum(len(word) for word in words) / len(words)


def validate_article(text: str) -> tuple[bool, str | None]:
    text = remove_markdown_images(text)
    chars = len(text)
    words = len(WORDS.findall(text))

    if chars < MIN_CHARS:
        return False, "too_short"

    if words < MIN_WORDS:
        return False, "too_few_words"

    if letter_ratio(text) < MIN_LETTER_RATIO:
        return False, "too_many_non_letters"

    if vowel_ratio(text) < MIN_VOWEL_RATIO:
        return False, "too_few_vowels"

    if average_word_length(text) < MIN_AVG_WORD_LENGTH:
        return False, "average_word_too_short"

    return True, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate stored articles and mark them as valid or invalid.")
    parser.add_argument(
        "--db-path",
        default="data/newspapers.sqlite",
        help="SQLite database path."
    )
    args = parser.parse_args()

    run_id = f"clean_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    db = Database(args.db_path)

    try:
        db.initialize()

        db.insert_run(
            Run(
                run_id=run_id,
                run_type="clean_articles"
            )
        )

        db.log(run_id, "INFO", "Started article validation.")

        valid_count = 0
        invalid_count = 0

        for article in tqdm(db.get_articles()):

            valid, reason = validate_article(article["content"])

            db.update_article_validation(
                article_id=article["article_id"],
                is_valid=valid,
                reason=reason
            )

            if valid:
                valid_count += 1
            else:
                invalid_count += 1

        db.log(run_id, "INFO", f"Validation finished: {valid_count} valid, {invalid_count} invalid")

        print(f"Validated {valid_count + invalid_count} articles: {valid_count} valid and {invalid_count} invalid.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
