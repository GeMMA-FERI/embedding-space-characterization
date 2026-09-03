#!/bin/bash
set -euo pipefail

# Ask the user for confirmation before proceeding
read -p "Are you sure to start this script? This will take a long time (y/n): " confirm
if [[ "$confirm" != "y" ]]; then
    echo "Aborting."
    exit 1
fi

python -m embedding_geometry.embedding.store_newspapers
python -m embedding_geometry.embedding.store_articles
python -m embedding_geometry.embedding.clean_articles
python -m embedding_geometry.embedding.deduplicate_articles
python -m embedding_geometry.statistics.analyze_text_structure
python -m embedding_geometry.embedding.clean_articles_by_statistics --apply
python -m embedding_geometry.statistics.analyze_text_structure --output-dir data/text_structure_valid
python -m embedding_geometry.embedding.store_chunks --method noop --chunking-run-id noop
python -m embedding_geometry.embedding.store_chunks --method paragraph --chunking-run-id paragraph
python -m embedding_geometry.embedding.store_chunks --method sentence --sentence-count 1 --sentence-overlap 0 --chunking-run-id sentence
python -m embedding_geometry.embedding.store_chunks --method sentence --sentence-count 4 --sentence-overlap 0 --chunking-run-id sentence_4_0
python -m embedding_geometry.embedding.store_chunks --method sentence --sentence-count 4 --sentence-overlap 1 --chunking-run-id sentence_4
python -m embedding_geometry.embedding.store_chunks --method token --token-count 32 --token-overlap 0 --chunking-run-id token_32
python -m embedding_geometry.embedding.store_chunks --method token --token-count 128 --token-overlap 0 --chunking-run-id token_128_0
python -m embedding_geometry.embedding.store_chunks --method token --token-count 128 --token-overlap 16 --chunking-run-id token_128
python -m embedding_geometry.embedding.store_chunks --method split --chunk-size 1000 --chunk-overlap 100 --chunking-run-id recursive_1000
python -m embedding_geometry.embedding.store_chunks --method split --chunk-size 400 --chunk-overlap 64 --chunking-run-id recursive_400
python -m embedding_geometry.embedding.store_chunks --method split --chunk-size 1000 --chunk-overlap 0 --chunking-run-id recursive_1000_0
python -m embedding_geometry.embedding.store_chunks --method split --chunk-size 400 --chunk-overlap 0 --chunking-run-id recursive_400_0

OVERLAP=(
    0
    32
    64
    128
    192
)

for i in "${!OVERLAP[@]}"; do
    OVERLAP_VALUE="${OVERLAP[$i]}"
    echo "Storing chunks with overlap: $OVERLAP_VALUE"
    python -m embedding_geometry.embedding.store_chunks \
        --method token \
        --token-count 256 \
        --token-overlap "$OVERLAP_VALUE" \
        --chunking-run-id "token_256_${OVERLAP_VALUE}"
done

MODELS=(
    "intfloat/multilingual-e5-large"
    "Qwen/Qwen3-Embedding-0.6B"
    "google/embeddinggemma-300m"
    "rokn/slovlo-v1"
    "BAAI/bge-m3"
)

SHORT_MODEL_IDS=(
    "e5"
    "qwen3"
    "gemma"
    "slovlo"
    "m3"
)

CHUNKING_RUNS=(
    "noop"
    "paragraph"
    "sentence"
    "token_32"
    "token_128"
    "sentence_4"
    "recursive_1000"
    "recursive_400"
    "token_128_0"
    "sentence_4_0"
    "recursive_1000_0"
    "recursive_400_0"
    "token_256_0"
    "token_256_32"
    "token_256_64"
    "token_256_128"
    "token_256_192"
)

BATCH_SIZES=(
    1       # noop
    8       # paragraph
    8       # sentence
    20      # token_32
    20      # token_128
    8       # sentence_4
    8       # recursive_1000
    8       # recursive_400
    8       # token_128_0
    8       # sentence_4_0
    8       # recursive_1000_0
    8       # recursive_400_0
    20      # token_256_0
    20      # token_256_32
    20      # token_256_64
    20      # token_256_128
    20      # token_256_192
)

for i in "${!MODELS[@]}"; do
    MODEL_ID="${MODELS[$i]}"
    SHORT_MODEL_ID="${SHORT_MODEL_IDS[$i]}"
    for j in "${!CHUNKING_RUNS[@]}"; do
        CHUNKING_RUN_ID="${CHUNKING_RUNS[$j]}"
        BATCH_SIZE="${BATCH_SIZES[$j]}"

        echo "Storing embeddings for model: $MODEL_ID, chunking run: $CHUNKING_RUN_ID, batch size: $BATCH_SIZE"
        cmd=(
            python
            -m
            embedding_geometry.embedding.store_embeddings
            --chunking-run-id "$CHUNKING_RUN_ID"
            --model-id "$MODEL_ID"
            --batch-size "$BATCH_SIZE"
            --embedding-run-id "${SHORT_MODEL_ID//\//_}_${CHUNKING_RUN_ID}"
        )
        if [[ "$MODEL_ID" == "rokn/slovlo-v1" ]]; then
            cmd+=(--transformer-kwargs '{"tokenizer_kwargs": {"use_fast":false}}')
        fi
        echo "Running command: ${cmd[@]}"
        "${cmd[@]}"
    done
done

bash ./batch/batch_evaluation.sh
