#!/usr/bin/env bash

set -euo pipefail

DB_PATH="${DB_PATH:-./data/newspapers.sqlite}"
OUTPUT_DIR="${OUTPUT_DIR:-data/cosine_classification}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-100000}"
SKIP_EXISTING=false

while (($#)); do
    case "$1" in
        --skip-existing)
            SKIP_EXISTING=true
            ;;
        -h|--help)
            echo "Usage: $0 [--skip-existing]"
            echo "  --skip-existing  Reuse non-empty per-run result CSVs."
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--skip-existing]" >&2
            exit 2
            ;;
    esac
    shift
done

mkdir -p "$OUTPUT_DIR"

embedding_run_ids="$(
    python -m embedding_geometry.classification.evaluate_naive_bayes \
        --db-path "$DB_PATH" \
        --list-run-ids
)"

if [[ -z "$embedding_run_ids" ]]; then
    echo "No non-noop embedding runs found." >&2
    exit 1
fi

result_files=()
while IFS= read -r embedding_run_id; do
    embedding_run_id="${embedding_run_id%$'\r'}"
    [[ -z "$embedding_run_id" ]] && continue
    output_name="${embedding_run_id//\//_}"
    output_path="$OUTPUT_DIR/$output_name.csv"
    if [[ "$SKIP_EXISTING" == true && -s "$output_path" ]]; then
        echo "Skipping $embedding_run_id; result already exists at $output_path"
        result_files+=("$output_path")
        continue
    fi
    echo "Classifying cosine similarities for $embedding_run_id"
    python -m embedding_geometry.classification.evaluate_naive_bayes \
        --db-path "$DB_PATH" \
        --embedding-run-id "$embedding_run_id" \
        --sample-limit "$SAMPLE_LIMIT" \
        --output "$output_path"
    result_files+=("$output_path")
done <<< "$embedding_run_ids"

python -m embedding_geometry.classification.summarize_naive_bayes \
    --db-path "$DB_PATH" \
    --input "${result_files[@]}"
