#!/usr/bin/env bash
set -euo pipefail

python -m embedding_geometry.statistics.plot_chunk_char_distributions \
    --input data/chunk_char_distributions.csv \
    --chunking-run-ids \
        paragraph \
        sentence \
        sentence_4_0 \
        sentence_4 \
        token_32 \
        token_128_0 \
        token_128 \
        token_256_0 \
        token_256_32 \
        token_256_64 \
        token_256_128 \
        token_256_192 \
        recursive_400_0 \
        recursive_400 \
        recursive_1000_0 \
        recursive_1000 \
    --chunking-run-names \
        "P" \
        "S1/0" \
        "S4/0" \
        "S4/1" \
        "T32/0" \
        "T128/0" \
        "T128/16" \
        "T256/0" \
        "T256/32" \
        "T256/64" \
        "T256/128" \
        "T256/192" \
        "R400/0" \
        "R400/64" \
        "R1000/0" \
        "R1000/100" \
    --plot-style line \
    --plot-scale 1.6 \
    --min-x 0 \
    --max-x 1200 \
    --output data/chunk_char_distributions.png
