#!/usr/bin/env bash
set -euo pipefail

python -m embedding_geometry.statistics.plot_embedding_cosine_similarities \
    --input data/compare_cosine_similarity.csv \
    --embedding-run-names "Gemma T256/64" "E5 T256/64" \
    --plot-scale 1.8 \
    --output data/compare_cosine_similarity.png
