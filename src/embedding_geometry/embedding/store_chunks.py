from __future__ import annotations

import argparse
import hashlib
import json

from tqdm import tqdm
from datetime import datetime
from semora.text.markdown import remove_markdown_images
from semora.text.chunking import (
    NoopProcessor,
    ParagraphProcessor,
    SentenceWindowProcessor,
    SplitTextProcessor,
    TextProcessor,
    TokenWindowProcessor,
)
from semora.storage import Chunk, ChunkingRun, Database, Run


def main() -> None:
    parser = argparse.ArgumentParser(description="Split stored articles into chunks and save them in SQLite.")
    parser.add_argument(
        "--db-path",
        default="data/newspapers.sqlite",
        help="SQLite database path."
    )
    parser.add_argument(
        "--method",
        choices=["noop", "split", "paragraph", "sentence", "token"],
        default="split",
        help="Chunking method."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Chunk size for split method."
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for split method."
    )
    parser.add_argument(
        "--sentence-count",
        type=int,
        default=1,
        help="Number of sentences per chunk for sentence window method."
    )
    parser.add_argument(
        "--sentence-overlap",
        type=int,
        default=0,
        help="Number of overlapping sentences for sentence window method."
    )
    parser.add_argument(
        "--tokenizer-model-id",
        default="bert-base-multilingual-cased",
        help="HuggingFace tokenizer model id for token window method."
    )
    parser.add_argument(
        "--token-count",
        type=int,
        default=512,
        help="Number of tokens per chunk for token window method."
    )
    parser.add_argument(
        "--token-overlap",
        type=int,
        default=0,
        help="Number of overlapping tokens for token window method."
    )
    parser.add_argument(
        "--chunking-run-id",
        default=None,
        help="Optional chunking run id. Defaults to method and timestamp."
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"chunks_{timestamp}"
    chunking_run_id = args.chunking_run_id or f"{args.method}_{timestamp}"

    db = Database(args.db_path)
    try:
        db.initialize()
        db.insert_run(Run(run_id=run_id, run_type="chunks"))
        db.log(run_id, "INFO", f"Started storing chunks with method={args.method}.")
        db.insert_chunking_run(
            ChunkingRun(
                chunking_run_id=chunking_run_id,
                run_id=run_id,
                method=args.method,
                config=_chunking_config(args)
            )
        )

        processor = _build_processor(args)
        count = 0
        skipped_count = 0

        for article in tqdm(db.get_valid_articles()):
            article_chunks = _chunk_article(
                run_id=run_id,
                chunking_run_id=chunking_run_id,
                article_id=article["article_id"],
                method=args.method,
                text=article["content"],
                processor=processor
            )
            if not article_chunks:
                db.log(run_id, "WARNING", f"No chunks generated for article: {article['article_id']}.")
                skipped_count += 1
                continue

            db.insert_chunks(article_chunks)
            count += len(article_chunks)

        db.log(run_id, "INFO", f"Finished storing chunks: {count} stored and {skipped_count} articles skipped.")
        print(f"Stored {count} chunks in {args.db_path}")
    finally:
        db.close()


def _build_processor(args: argparse.Namespace) -> TextProcessor:
    if args.method == "noop":
        return NoopProcessor()

    if args.method == "paragraph":
        return ParagraphProcessor()

    if args.method == "sentence":
        return SentenceWindowProcessor(
            sentence_count=args.sentence_count,
            sentence_overlap=args.sentence_overlap
        )

    if args.method == "token":
        return TokenWindowProcessor(
            model_id=args.tokenizer_model_id,
            token_count=args.token_count,
            token_overlap=args.token_overlap
        )

    if args.method == "split":
        return SplitTextProcessor(
            split_text=True,
            text_batch_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )

    raise ValueError(f"Unknown chunking method: {args.method}")


def _chunking_config(args: argparse.Namespace) -> dict:
    config = {"remove_markdown_images": True}

    if args.method == "noop":
        return config

    if args.method == "paragraph":
        return config

    if args.method == "sentence":
        config.update({
            "sentence_count": args.sentence_count,
            "sentence_overlap": args.sentence_overlap
        })
        return config

    if args.method == "token":
        config.update({
            "tokenizer_model_id": args.tokenizer_model_id,
            "token_count": args.token_count,
            "token_overlap": args.token_overlap
        })
        return config

    if args.method == "split":
        config.update({
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap
        })
        return config

    raise ValueError(f"Unknown chunking method: {args.method}")


def _chunk_article(
    *,
    run_id: str,
    chunking_run_id: str,
    article_id: str,
    text: str,
    method: str,
    processor: TextProcessor
) -> list[Chunk]:
    chunks = []
    text_without_images = remove_markdown_images(text)
    if not text_without_images.strip():
        return chunks

    for _, chunk_text in processor.process(article_id, text_without_images):
        chunk_text = remove_markdown_images(chunk_text).strip()
        if not chunk_text:
            continue
        chunk_index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=_build_chunk_id(
                    article_id=article_id,
                    chunking_run_id=chunking_run_id,
                    chunk_index=chunk_index,
                    text=chunk_text
                ),
                run_id=run_id,
                article_id=article_id,
                chunking_run_id=chunking_run_id,
                chunk_index=chunk_index,
                method=method,
                text=chunk_text
            )
        )
    return chunks


def _build_chunk_id(
    *,
    article_id: str,
    chunking_run_id: str,
    chunk_index: int,
    text: str
) -> str:
    value = json.dumps(
        {
            "article_id": article_id,
            "chunking_run_id": chunking_run_id,
            "chunk_index": chunk_index,
            "text": text
        },
        ensure_ascii=False,
        sort_keys=True
    )
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"chunk_{digest[:24]}"


if __name__ == "__main__":
    main()
