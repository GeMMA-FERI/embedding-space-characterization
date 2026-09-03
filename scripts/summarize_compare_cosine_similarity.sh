#!/usr/bin/env bash
set -euo pipefail

python -m embedding_geometry.statistics.summarize_embedding_cosine_similarities \
    --embedding-run-ids gemma_token_256_64 e5_token_256_64 \
    --embedding-run-names "Gemma T256/64" "E5 T256/64" \
    --output data/compare_cosine_similarity.csv
