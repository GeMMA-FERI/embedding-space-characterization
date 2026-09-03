#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repository_root"

models=(
  "intfloat/multilingual-e5-large"
  "Qwen/Qwen3-Embedding-0.6B"
  "rokn/slovlo-v1"
  "BAAI/bge-m3"
  "google/embeddinggemma-300m"
)

short_model_ids=(
  "e5"
  "qwen3"
  "slovlo"
  "m3"
  "gemma"
)

output_dir="${OUTPUT_DIR:-data/mteb_results}"
batch_size="${BATCH_SIZE:-32}"
mkdir -p "$output_dir"

for i in "${!models[@]}"; do
  model_id="${models[$i]}"
  output_file="$output_dir/${short_model_ids[$i]}.json"

  echo "Running MTEB for $model_id -> $output_file"
  python -m embedding_geometry.statistics.run_mteb \
    --model-id "$model_id" \
    --batch-size "$batch_size" \
    --output-file "$output_file"
done
