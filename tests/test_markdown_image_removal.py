from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from semora.text.markdown import remove_markdown_images
from embedding_geometry.embedding.clean_articles import validate_article
from embedding_geometry.embedding.store_chunks import _chunk_article
from embedding_geometry.statistics.analyze_text_structure import analyze_article


def test_remove_markdown_images_preserves_text_links_and_line_breaks() -> None:
    text = (
        "Before ![scan](images/scan.jpg \"page scan\") after.\n"
        "![](images/image-only.jpg)\n"
        "Keep [ordinary link](https://example.com).\n"
        "Two ![](one.png) ![second](two.png) images."
    )

    assert remove_markdown_images(text) == (
        "Before after.\n"
        "\n"
        "Keep [ordinary link](https://example.com).\n"
        "Two images."
    )


def test_remove_markdown_images_handles_nested_url_parentheses_and_escaped_syntax() -> None:
    text = r"Text ![](images/file_(1).jpg) and \![](literal.jpg)."

    assert remove_markdown_images(text) == r"Text and \![](literal.jpg)."


def test_image_only_text_becomes_empty() -> None:
    assert remove_markdown_images("  ![](images/only.jpg)  ").strip() == ""


def test_chunking_never_stores_images_or_empty_chunks() -> None:
    class RecordingProcessor:
        received_text: str | None = None

        def process(self, source_id: str, text: str):
            del source_id
            self.received_text = text
            return [
                ("ignored-0", "![](images/only.jpg)"),
                ("ignored-1", "First ![](images/inline.jpg) chunk"),
                ("ignored-2", "Second chunk"),
            ]

    processor = RecordingProcessor()
    chunks = _chunk_article(
        run_id="run",
        chunking_run_id="chunking-run",
        article_id="article",
        method="test",
        text="Article ![](images/source.jpg) text",
        processor=processor,
    )

    assert processor.received_text == "Article text"
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert [chunk.text for chunk in chunks] == ["First chunk", "Second chunk"]
    assert all("![" not in chunk.text for chunk in chunks)


def test_image_only_article_generates_no_chunks() -> None:
    class ProcessorThatMustNotRun:
        def process(self, source_id: str, text: str):
            raise AssertionError("Image-only content should not reach the processor")

    assert _chunk_article(
        run_id="run",
        chunking_run_id="chunking-run",
        article_id="article",
        method="test",
        text="![](images/only.jpg)",
        processor=ProcessorThatMustNotRun(),
    ) == []


def test_article_validation_ignores_markdown_images() -> None:
    text = " ".join(["meaningful"] * 100)
    with_images = f"![](images/one.jpg)\n{text}\n![](images/two.jpg)"

    assert validate_article(with_images) == validate_article(text)


def test_text_structure_and_tokenization_ignore_markdown_images() -> None:
    class RecordingTokenizer:
        received_text: str | None = None

        def encode(self, text: str, *, add_special_tokens: bool):
            assert add_special_tokens is False
            self.received_text = text
            return text.split()

    article = {
        "article_id": "article",
        "title": "Title",
        "content": "First ![](images/scan.jpg) paragraph.",
    }
    tokenizer = RecordingTokenizer()

    row, _, _, _ = analyze_article(article, tokenizer)

    assert tokenizer.received_text == "First paragraph."
    assert row["characters"] == len("First paragraph.")
    assert row["words"] == 2
    assert row["tokens"] == 2
