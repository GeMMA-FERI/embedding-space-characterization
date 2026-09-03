MODELS=(
    "intfloat/multilingual-e5-large"
    "Qwen/Qwen3-Embedding-0.6B"
    "rokn/slovlo-v1"
    "BAAI/bge-m3"
    "google/embeddinggemma-300m"
)

SHORT_MODEL_IDS=(
    "e5"
    "qwen3"
    "slovlo"
    "m3"
    "gemma"
)

CHUNKING_RUNS=(
    "paragraph"
    "sentence"
    "sentence_4"
    "token_32"
    "token_128"
    "recursive_400"
    "recursive_1000"
    "sentence_4_0"
    "token_128_0"
    "recursive_400_0"
    "recursive_1000_0"
    "token_256_0"
    "token_256_32"
    "token_256_64"
    "token_256_128"
    "token_256_192"
)

MAX_CONCURRENT_PROCESSES=4
RUNNING_PROCESSES=0
FAILED_PROCESSES=0

wait_for_one_process() {
    if ! wait -n; then
        FAILED_PROCESSES=$((FAILED_PROCESSES + 1))
    fi
    RUNNING_PROCESSES=$((RUNNING_PROCESSES - 1))
}

for i in "${!SHORT_MODEL_IDS[@]}"; do
    SHORT_MODEL_ID="${SHORT_MODEL_IDS[$i]}"
    MODEL_ID="${MODELS[$i]}"
    for j in "${!CHUNKING_RUNS[@]}"; do
        CHUNKING_RUN_ID="${CHUNKING_RUNS[$j]}"
        
        echo "Plotting cosine similarity for model: $MODEL_ID, chunking run: $CHUNKING_RUN_ID"
        python -m embedding_geometry.statistics.plot_embedding_cosine_similarities \
            --embedding-run-ids "$SHORT_MODEL_ID"_"$CHUNKING_RUN_ID" &
        RUNNING_PROCESSES=$((RUNNING_PROCESSES + 1))

        if (( RUNNING_PROCESSES >= MAX_CONCURRENT_PROCESSES )); then
            wait_for_one_process
        fi
    done
done

while (( RUNNING_PROCESSES > 0 )); do
    wait_for_one_process
done

if (( FAILED_PROCESSES > 0 )); then
    echo "$FAILED_PROCESSES cosine similarity plot process(es) failed." >&2
    exit 1
fi
