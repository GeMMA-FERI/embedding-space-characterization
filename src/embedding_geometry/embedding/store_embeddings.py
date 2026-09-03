from __future__ import annotations

import argparse
import json
from datetime import datetime

from tqdm import tqdm

from semora.embeddings.serialization import build_embedding
from semora.storage import Database, EmbeddingRun, Run


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed stored chunks and save vectors in SQLite.")
    parser.add_argument(
        "--db-path",
        default="data/newspapers.sqlite",
        help="SQLite database path."
    )
    parser.add_argument(
        "--chunking-run-id",
        required=True,
        help="Required chunking run id."
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Embedding model id."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Number of chunks to embed per batch."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of chunks to embed."
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=None,
        help=("Optional maximum number of characters per chunk. " "Longer chunks are skipped.")
    )
    parser.add_argument(
        "--embedding-run-id",
        default=None,
        help="Optional embedding run id. Defaults to model and timestamp."
    )
    parser.add_argument(
        "--transformer-kwargs",
        type=json.loads,
        default=None,
        help=(
            "Optional JSON string of keyword arguments to pass to the transformer model. "
            'For example: \'{"tokenizer_kwargs": {"use_fast": false}}\''
        )
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be a positive integer")

    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be a positive integer")

    if args.max_characters is not None and args.max_characters <= 0:
        raise ValueError("max-characters must be a positive integer")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = _slug(args.model_id)
    run_id = f"embeddings_{timestamp}"
    embedding_run_id = args.embedding_run_id or f"{model_slug}_{timestamp}"

    db = Database(args.db_path)
    try:
        db.initialize()
        resumed = _prepare_embedding_run(
            db,
            args=args,
            run_id=run_id,
            embedding_run_id=embedding_run_id,
        )
        if resumed:
            print(f"Resuming embedding run {embedding_run_id}; stored chunks will be skipped.")

        skipped_count = 0
        if args.max_characters is not None:
            oversized_chunks = db.get_oversized_unembedded_chunks(
                embedding_run_id=embedding_run_id,
                chunking_run_id=args.chunking_run_id,
                max_characters=args.max_characters
            )
            for chunk in oversized_chunks:
                message = (
                    f"Skipped oversized chunk {chunk['chunk_id']} from article "
                    f"{chunk['article_id']}: {chunk['character_count']} characters "
                    f"exceeds maximum {args.max_characters}."
                )
                db.log(run_id, "WARNING", message)
                print(message)
            skipped_count = len(oversized_chunks)

        chunks = db.get_unembedded_chunks(
            embedding_run_id=embedding_run_id,
            chunking_run_id=args.chunking_run_id,
            max_characters=args.max_characters,
            limit=args.limit
        )
        if not chunks:
            db.log(run_id, "WARNING", "No chunks found for embedding.")
            print("No chunks found for embedding.")
            return

        from semora.embeddings.registry import get_embedder

        embedder = get_embedder(args.model_id, transformer_kwargs=args.transformer_kwargs).load()

        stored_count = 0
        for batch in tqdm(
            _batches(chunks, args.batch_size),
            total=_batch_count(len(chunks), args.batch_size),
            desc="Embedding chunks"
        ):
            try:
                texts = [row["text"] for row in batch]
                vectors = embedder.embed_documents(texts)
                if stored_count == 0:
                    db.log(run_id, "INFO", f"Tensor shape: {vectors.shape}, dtype: {vectors.dtype}")
            except Exception as e:
                db.log(run_id, "ERROR", f"Error embedding batch: {e}")
                print([len(row["text"]) for row in batch])
                print(f"Error embedding batch: {e}")
                continue

            embeddings = [
                build_embedding(
                    embedding_run_id=embedding_run_id,
                    chunk_id=row["chunk_id"],
                    vector=vector
                )
                for row, vector in zip(batch, vectors)
            ]
            db.insert_embeddings(embeddings)
            stored_count += len(embeddings)

        db.log(run_id, "INFO", f"Finished storing embeddings: {stored_count} stored and " f"{skipped_count} oversized chunks skipped.")
        print(f"Stored {stored_count} new embeddings and skipped {skipped_count} " f"oversized chunks in {args.db_path}")
    finally:
        db.close()


def _prepare_embedding_run(
    db: Database,
    *,
    args: argparse.Namespace,
    run_id: str,
    embedding_run_id: str,
) -> bool:
    """Create an embedding run or validate an existing run for resumption."""
    existing_runs = db.get_embedding_runs_by_ids([embedding_run_id])
    existing = existing_runs[0] if existing_runs else None
    if existing is not None:
        config = json.loads(existing["config_json"] or "{}")
        mismatches = []
        if existing["model_id"] != args.model_id:
            mismatches.append(
                f"model-id is {existing['model_id']!r}, not {args.model_id!r}"
            )
        if existing["chunking_run_id"] != args.chunking_run_id:
            mismatches.append(
                "chunking-run-id is "
                f"{existing['chunking_run_id']!r}, not {args.chunking_run_id!r}"
            )
        if config.get("transformer_kwargs") != args.transformer_kwargs:
            mismatches.append("transformer-kwargs differ from the original run")
        if mismatches:
            raise ValueError(
                f"Cannot resume embedding run {embedding_run_id}: "
                + "; ".join(mismatches)
            )

    db.insert_run(Run(run_id=run_id, run_type="embeddings"))
    if existing is not None:
        db.log(run_id, "INFO", f"Resuming embedding run: {embedding_run_id}.")
        return True

    db.log(run_id, "INFO", f"Started storing embeddings with model: {args.model_id}.")
    db.insert_embedding_run(
        EmbeddingRun(
            embedding_run_id=embedding_run_id,
            run_id=run_id,
            model_id=args.model_id,
            config=args.__dict__,
        )
    )
    return False


def _batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _batch_count(item_count: int, batch_size: int) -> int:
    return (item_count + batch_size - 1) // batch_size


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


if __name__ == "__main__":
    main()
