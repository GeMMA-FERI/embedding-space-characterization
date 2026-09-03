#!/usr/bin/env bash
set -euo pipefail

DAVIES_BOULDIN_BLOCK_SIZE=128
DAVIES_BOULDIN_COHORT_SIZE=10000
DAVIES_BOULDIN_SEEDS=(42 43 44 45 46)
DAVIES_BOULDIN_TRIM_FRACTION=0.01
ASSUME_YES=false

usage() {
    echo "Usage: $0 [options]"
    echo "  --davies-bouldin-block-size N"
    echo "  --davies-bouldin-cohort-size N"
    echo "  --davies-bouldin-seeds N [N ...]"
    echo "  --davies-bouldin-trim-fraction FRACTION"
    echo "  -y, --yes  Skip the confirmation prompt."
}

while (($#)); do
    case "$1" in
        --davies-bouldin-block-size)
            [[ $# -ge 2 ]] || { echo "Missing block size." >&2; exit 2; }
            DAVIES_BOULDIN_BLOCK_SIZE="$2"
            shift 2
            ;;
        --davies-bouldin-cohort-size)
            [[ $# -ge 2 ]] || { echo "Missing cohort size." >&2; exit 2; }
            DAVIES_BOULDIN_COHORT_SIZE="$2"
            shift 2
            ;;
        --davies-bouldin-seeds)
            shift
            DAVIES_BOULDIN_SEEDS=()
            while (($#)) && [[ "$1" != --* ]]; do
                DAVIES_BOULDIN_SEEDS+=("$1")
                shift
            done
            ;;
        --davies-bouldin-trim-fraction)
            [[ $# -ge 2 ]] || { echo "Missing trim fraction." >&2; exit 2; }
            DAVIES_BOULDIN_TRIM_FRACTION="$2"
            shift 2
            ;;
        -y|--yes)
            ASSUME_YES=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "$DAVIES_BOULDIN_BLOCK_SIZE" =~ ^[0-9]+$ ]] \
    || ((DAVIES_BOULDIN_BLOCK_SIZE < 1)); then
    echo "Davies-Bouldin block size must be positive." >&2
    exit 2
fi
if ! [[ "$DAVIES_BOULDIN_COHORT_SIZE" =~ ^[0-9]+$ ]] \
    || ((DAVIES_BOULDIN_COHORT_SIZE < 2)); then
    echo "Davies-Bouldin cohort size must be at least 2." >&2
    exit 2
fi
if ((${#DAVIES_BOULDIN_SEEDS[@]} == 0)); then
    echo "At least one Davies-Bouldin seed is required." >&2
    exit 2
fi
for seed in "${DAVIES_BOULDIN_SEEDS[@]}"; do
    if ! [[ "$seed" =~ ^-?[0-9]+$ ]]; then
        echo "Davies-Bouldin seeds must be integers: $seed" >&2
        exit 2
    fi
done
if ! [[ "$DAVIES_BOULDIN_TRIM_FRACTION" =~ ^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]] \
    || ! awk -v value="$DAVIES_BOULDIN_TRIM_FRACTION" \
        'BEGIN { exit !(value >= 0 && value < 1) }'; then
    echo "Davies-Bouldin trim fraction must be in [0, 1)." >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EMBEDDING_MODELS=(
    "gemma"
    "m3"
    "slovlo"
    "qwen3"
    "e5"
)

# noop is the full-article baseline and is not an evaluation target.
CHUNKING_RUN_IDS=(
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

EMBEDDING_RUN_IDS=()
for embedding_model in "${EMBEDDING_MODELS[@]}"; do
    for chunking_run_id in "${CHUNKING_RUN_IDS[@]}"; do
        EMBEDDING_RUN_IDS+=("${embedding_model}_${chunking_run_id}")
    done
done

if ((${#EMBEDDING_RUN_IDS[@]} == 0)); then
    echo "No non-noop embedding runs found." >&2
    exit 1
fi

echo "Found ${#EMBEDDING_RUN_IDS[@]} embedding runs."
printf 'Evaluating embedding runs: %s\n' "$(IFS=', '; echo "${EMBEDDING_RUN_IDS[*]}")"

if [[ "$ASSUME_YES" != true ]]; then
    if ! read -r -p "Do you want to proceed with the evaluations? (Y/N) " confirmation; then
        echo "Could not read confirmation; evaluation aborted." >&2
        exit 1
    fi
    if [[ "$confirmation" != "Y" && "$confirmation" != "y" ]]; then
        echo "Evaluation aborted by user."
        exit 0
    fi
fi

run_evaluation() {
    local label="$1"
    local embedding_run_id="$2"
    shift 2
    if ! "$@"; then
        echo "$label failed for $embedding_run_id" >&2
        exit 1
    fi
}

for embedding_run_id in "${EMBEDDING_RUN_IDS[@]}"; do
    echo "Evaluating: $embedding_run_id"

    run_evaluation "Intra-article consistency evaluation" "$embedding_run_id" \
        python -m embedding_geometry.evaluation.evaluate_intra_consistency \
        --embedding-run-id "$embedding_run_id" \
        --output "data/intra_consistency/$embedding_run_id.csv"

    run_evaluation "Inter-article separability evaluation" "$embedding_run_id" \
        python -m embedding_geometry.evaluation.evaluate_inter_separability \
        --embedding-run-id "$embedding_run_id" \
        --metrics davies_bouldin \
        --davies-bouldin-block-size "$DAVIES_BOULDIN_BLOCK_SIZE" \
        --davies-bouldin-cohort-size "$DAVIES_BOULDIN_COHORT_SIZE" \
        --davies-bouldin-seeds "${DAVIES_BOULDIN_SEEDS[@]}" \
        --davies-bouldin-trim-fraction "$DAVIES_BOULDIN_TRIM_FRACTION" \
        --skip-full-davies-bouldin \
        --cohort-embedding-run-ids "${EMBEDDING_RUN_IDS[@]}" \
        --output "data/inter_separability/$embedding_run_id.csv"

    run_evaluation "Embedding anisotropy evaluation" "$embedding_run_id" \
        python -m embedding_geometry.evaluation.evaluate_anisotropy \
        --embedding-run-id "$embedding_run_id" \
        --output "data/embedding_anisotropy/$embedding_run_id.csv"
done

# This model-level evaluator loads each model's noop baseline only once.
if ! bash "$SCRIPT_DIR/batch_evaluate_representation_degradation.sh"; then
    echo "Representation degradation evaluation failed." >&2
    exit 1
fi

echo "All evaluations completed."
