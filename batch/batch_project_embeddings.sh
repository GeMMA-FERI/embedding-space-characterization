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

db_path="${DB_PATH:-data/newspapers.sqlite}"

for model in "${models[@]}"; do
  echo "Projecting embeddings for $model"
  python -m semora.projection.projector \
    --db-path "$db_path" \
    --model-id "$model" \
    --method umap-mlp \
    --n-samples 100000 \
    "$@"
done
