#!/usr/bin/env bash
set -euo pipefail

python -m embedding_geometry.statistics.summarize_chunk_char_distributions \
    --metric char \
    --output data/chunk_char_distributions.csv
