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
output_dir="${OUTPUT_DIR:-visual/public/data}"
python_bin="${PYTHON_BIN:-python}"

mkdir -p "$output_dir"

for model in "${models[@]}"; do
  output_name="${model//\//_}.bin"
  echo "Exporting projection for $model to $output_dir/$output_name"
  "$python_bin" visual/export_projection.py \
    --db-path "$db_path" \
    --model-id "$model" \
    --output "$output_dir/$output_name" \
    "$@"
done
