#!/usr/bin/env bash
set -euo pipefail

MODELS=(
    "intfloat/multilingual-e5-large"
    "Qwen/Qwen3-Embedding-0.6B"
    "rokn/slovlo-v1"
    "BAAI/bge-m3"
    "google/embeddinggemma-300m"
)

OUTPUT_NAMES=(
    "e5"
    "qwen3"
    "slovlo"
    "bge_m3"
    "embeddinggemma"
)

OUTPUT_DIR="data/representation_degradation"
mkdir -p "$OUTPUT_DIR"

for i in "${!MODELS[@]}"; do
    MODEL_ID="${MODELS[$i]}"
    OUTPUT_NAME="${OUTPUT_NAMES[$i]}"
    echo "Evaluating representation degradation for $MODEL_ID"
    python -m embedding_geometry.evaluation.evaluate_representation_degradation \
        --model-id "$MODEL_ID" \
        --min-chunks-per-article 5 \
        --output "$OUTPUT_DIR/$OUTPUT_NAME.csv"
done
