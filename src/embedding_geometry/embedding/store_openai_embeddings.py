from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

from tqdm import tqdm

from semora.embeddings.openai_batch import (
    BatchSubmissionError,
    CollectionCounts,
    OpenAIBatchClient,
    all_batches_terminal,
    batch_status_counts,
    collect_completed_batches,
    prepare_batches,
    refresh_batches,
    submit_prepared_batches,
)
from semora.storage import Database, EmbeddingRun, Run


OPENAI_EMBEDDING_MODELS = ("text-embedding-3-small", "text-embedding-3-large")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)

    db = Database(args.db_path)
    try:
        db.initialize()
        if args.command in {"submit", "run"}:
            embedding_run_id = _ensure_embedding_run(db, args)
            prepared = prepare_batches(
                db,
                embedding_run_id=embedding_run_id,
                chunking_run_id=args.chunking_run_id,
                model_id=args.model_id,
                artifact_dir=args.artifact_dir,
                request_size=args.request_size,
                max_batch_inputs=args.max_batch_inputs,
                max_batch_bytes=args.max_batch_bytes,
                max_characters=args.max_characters,
                limit=args.limit,
                dimensions=args.dimensions,
            )
            if prepared:
                print(
                    f"Prepared {len(prepared)} batch files containing "
                    f"{sum(batch.input_count for batch in prepared)} inputs."
                )
            else:
                print("No new chunks needed preparation.")
            client = _client(args)
            submission_failed = _submit(db, client, embedding_run_id)
            print(f"Embedding run: {embedding_run_id}")
            if args.command == "submit":
                _print_status(db, embedding_run_id)
                return
            if submission_failed:
                raise RuntimeError(
                    "Some batches could not be submitted. Rerun with the same "
                    f"--embedding-run-id {embedding_run_id}."
                )
            _wait_and_collect(db, client, embedding_run_id, args.poll_interval)
            return

        client = _client(args)
        if args.command == "status":
            _refresh(db, client, args.embedding_run_id)
            _print_status(db, args.embedding_run_id)
        elif args.command == "collect":
            _refresh(db, client, args.embedding_run_id)
            counts = _collect(db, client, args.embedding_run_id)
            _print_collection(counts)
            _print_status(db, args.embedding_run_id)
    finally:
        db.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit, monitor, and import OpenAI Batch API embeddings."
    )
    parser.add_argument(
        "--db-path",
        default="data/newspapers.sqlite",
        help="SQLite database path.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("submit", "run"):
        command = subparsers.add_parser(name)
        _add_api_arguments(command)
        command.add_argument("--chunking-run-id", required=True)
        command.add_argument("--model-id", required=True, choices=OPENAI_EMBEDDING_MODELS)
        command.add_argument("--embedding-run-id")
        command.add_argument("--dimensions", type=int)
        command.add_argument("--request-size", type=int, default=128)
        command.add_argument("--max-batch-inputs", type=int, default=50_000)
        command.add_argument("--max-batch-bytes", type=int, default=190_000_000)
        command.add_argument("--max-characters", type=int)
        command.add_argument("--limit", type=int)
        command.add_argument(
            "--artifact-dir",
            default="data/openai_batches",
            help="Directory for input and downloaded result JSONL files.",
        )
        if name == "run":
            command.add_argument("--poll-interval", type=float, default=60.0)

    for name in ("status", "collect"):
        command = subparsers.add_parser(name)
        _add_api_arguments(command)
        command.add_argument("--embedding-run-id", required=True)
    return parser


def _add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-key", help="Defaults to OPENAI_API_KEY.")
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--organization")
    parser.add_argument("--project")
    parser.add_argument("--timeout", type=int, default=120)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("timeout", "limit", "max_characters", "dimensions"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    poll_interval = getattr(args, "poll_interval", None)
    if poll_interval is not None and poll_interval <= 0:
        raise ValueError("poll-interval must be positive")


def _ensure_embedding_run(db: Database, args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    embedding_run_id = args.embedding_run_id or f"{_slug(args.model_id)}_{timestamp}"
    existing = db.conn.execute(
        """
        SELECT embedding_run_id, model_id, config_json
        FROM embedding_runs
        WHERE embedding_run_id = ?
        """,
        (embedding_run_id,),
    ).fetchone()
    if existing:
        config = json.loads(existing["config_json"])
        if existing["model_id"] != args.model_id:
            raise ValueError("Existing embedding run uses a different model-id")
        if config.get("chunking_run_id") != args.chunking_run_id:
            raise ValueError("Existing embedding run uses a different chunking-run-id")
        if config.get("dimensions") != args.dimensions:
            raise ValueError("Existing embedding run uses different dimensions")
        return embedding_run_id

    run_id = f"openai_embeddings_{timestamp}"
    db.insert_run(Run(run_id=run_id, run_type="openai_embeddings"))
    db.insert_embedding_run(
        EmbeddingRun(
            embedding_run_id=embedding_run_id,
            run_id=run_id,
            model_id=args.model_id,
            config={
                "provider": "openai_batch",
                "chunking_run_id": args.chunking_run_id,
                "dimensions": args.dimensions,
                "request_size": args.request_size,
                "max_batch_inputs": args.max_batch_inputs,
                "max_batch_bytes": args.max_batch_bytes,
                "max_characters": args.max_characters,
            },
        )
    )
    db.log(run_id, "INFO", f"Created OpenAI embedding batch run with {args.model_id}.")
    return embedding_run_id


def _client(args: argparse.Namespace) -> OpenAIBatchClient:
    return OpenAIBatchClient(
        api_key=args.api_key,
        api_base=args.api_base,
        timeout=args.timeout,
        organization=args.organization,
        project=args.project,
    )


def _submit(db: Database, client: OpenAIBatchClient, embedding_run_id: str) -> bool:
    total = db.conn.execute(
        """
        SELECT COUNT(*) AS batch_count
        FROM openai_embedding_batches
        WHERE embedding_run_id = ? AND status IN ('prepared', 'uploaded')
        """,
        (embedding_run_id,),
    ).fetchone()["batch_count"]
    with tqdm(total=int(total), desc="Submitting OpenAI batches") as progress:
        try:
            submitted = submit_prepared_batches(
                db,
                client,
                embedding_run_id=embedding_run_id,
                on_batch=lambda _: progress.update(1),
            )
        except BatchSubmissionError as error:
            print(str(error))
            print(f"Submitted {error.submitted} OpenAI batches before/around the failures.")
            return True
    print(f"Submitted {submitted} OpenAI batches.")
    return False


def _refresh(db: Database, client: OpenAIBatchClient, embedding_run_id: str) -> dict[str, int]:
    total = db.conn.execute(
        """
        SELECT COUNT(*) AS batch_count
        FROM openai_embedding_batches
        WHERE embedding_run_id = ?
          AND provider_batch_id IS NOT NULL
          AND (
              status NOT IN ('completed', 'failed', 'expired', 'cancelled')
              OR (
                  status = 'completed'
                  AND output_file_id IS NULL
                  AND error_file_id IS NULL
              )
          )
        """,
        (embedding_run_id,),
    ).fetchone()["batch_count"]
    with tqdm(total=int(total), desc="Checking OpenAI batches") as progress:
        return refresh_batches(
            db,
            client,
            embedding_run_id=embedding_run_id,
            on_batch=lambda _: progress.update(1),
        )


def _collect(db: Database, client: OpenAIBatchClient, embedding_run_id: str) -> CollectionCounts:
    total = db.conn.execute(
        """
        SELECT COUNT(*) AS batch_count
        FROM openai_embedding_batches
        WHERE embedding_run_id = ?
          AND status IN ('completed', 'failed', 'expired', 'cancelled')
          AND imported_at IS NULL
        """,
        (embedding_run_id,),
    ).fetchone()["batch_count"]
    with tqdm(total=int(total), desc="Importing OpenAI batches") as progress:
        return collect_completed_batches(
            db,
            client,
            embedding_run_id=embedding_run_id,
            on_batch=lambda _: progress.update(1),
        )


def _wait_and_collect(
    db: Database,
    client: OpenAIBatchClient,
    embedding_run_id: str,
    poll_interval: float,
) -> None:
    if not batch_status_counts(db, embedding_run_id=embedding_run_id):
        print(f"No OpenAI batches exist for {embedding_run_id}.")
        return
    while not all_batches_terminal(db, embedding_run_id=embedding_run_id):
        _refresh(db, client, embedding_run_id)
        _print_status(db, embedding_run_id)
        if not all_batches_terminal(db, embedding_run_id=embedding_run_id):
            time.sleep(poll_interval)
    counts = _collect(db, client, embedding_run_id)
    _print_collection(counts)
    _print_status(db, embedding_run_id)


def _print_status(db: Database, embedding_run_id: str) -> None:
    counts = batch_status_counts(db, embedding_run_id=embedding_run_id)
    rendered = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    print(f"Batch status for {embedding_run_id}: {rendered or 'none'}")


def _print_collection(counts: CollectionCounts) -> None:
    print(
        f"Collected {counts.batches} batches: "
        f"{counts.imported} embeddings imported, {counts.failed} inputs failed."
    )


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


if __name__ == "__main__":
    main()
