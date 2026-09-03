from __future__ import annotations

import argparse
import hashlib
import json
import re

from tqdm import tqdm
from dataclasses import dataclass
from datetime import datetime
from semora.storage import Article, Database, Run

HEADING_RE = re.compile(r"^\s{0,3}#\s*(.*?)\s*$")


@dataclass
class ParsedArticle:
    title: str | None
    content: str
    article_index: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Split stored newspapers into articles using markdown headings.")
    parser.add_argument(
        "--db-path",
        default="data/newspapers.sqlite",
        help="SQLite database path."
    )
    args = parser.parse_args()

    run_id = f"articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    db = Database(args.db_path)

    try:
        db.initialize()
        db.insert_run(Run(run_id=run_id, run_type="articles"))
        db.log(run_id, "INFO", "Started storing articles.")
        count = 0
        for newspaper in tqdm(db.get_newspapers()):
            parsed_articles = split_articles(newspaper["content"])
            if not parsed_articles:
                db.log(run_id, "WARNING", f"No articles parsed from newspaper: {newspaper['newspaper_id']}.")
                continue

            for parsed in parsed_articles:
                db.insert_article(
                    Article(
                        article_id=_build_article_id(
                            newspaper_id=newspaper["newspaper_id"],
                            article_index=parsed.article_index,
                            title=parsed.title,
                            content=parsed.content
                        ),
                        run_id=run_id,
                        newspaper_id=newspaper["newspaper_id"],
                        title=parsed.title,
                        content=parsed.content,
                        metadata={"article_index": parsed.article_index}
                    )
                )
                count += 1
        db.log(run_id, "INFO", f"Finished storing articles: {count} stored.")
        print(f"Stored {count} articles in {args.db_path}")
    finally:
        db.close()


def split_articles(markdown: str) -> list[ParsedArticle]:
    articles: list[ParsedArticle] = []
    current_title: str | None = None
    current_lines: list[str] = []
    saw_heading = False

    for line in markdown.splitlines():
        heading = _heading_text(line)
        if heading is not None:
            if not heading:
                continue
            saw_heading = True
            if current_title is not None:
                _append_article(articles, current_title, current_lines)
            current_title = heading
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)

    if current_title is not None:
        _append_article(articles, current_title, current_lines)

    if not saw_heading and markdown.strip():
        articles.append(
            ParsedArticle(
                title=None,
                content=markdown.strip(),
                article_index=0
            )
        )

    return articles


def _heading_text(line: str) -> str | None:
    match = HEADING_RE.match(line)
    if not match:
        return None
    return match.group(1).strip()


def _append_article(articles: list[ParsedArticle], title: str, lines: list[str]) -> None:
    body = "\n".join(lines).strip()
    if not body:
        return

    content = body
    articles.append(
        ParsedArticle(
            title=title,
            content=content,
            article_index=len(articles)
        )
    )


def _build_article_id(
    *,
    newspaper_id: str,
    article_index: int,
    title: str | None,
    content: str
) -> str:
    value = json.dumps(
        {
            "newspaper_id": newspaper_id,
            "article_index": article_index,
            "title": title,
            "content": content
        },
        ensure_ascii=False,
        sort_keys=True
    )
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"article_{digest[:24]}"


if __name__ == "__main__":
    main()
