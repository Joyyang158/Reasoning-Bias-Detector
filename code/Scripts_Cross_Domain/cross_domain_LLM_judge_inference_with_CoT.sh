#!/usr/bin/env bash

# BASE_MODEL="DeepSeek-R1-Distill-Qwen-1.5B"
# BASE_MODEL="DeepSeek-R1-Distill-Qwen-7B"
# BASE_MODEL="DeepSeek-R1-Distill-Llama-8B"
BASE_MODEL="DeepSeek-R1-Distill-Qwen-14B"

# MODELS=(
#     "gpt-4o"
#     "gpt-4o-mini"
#     "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
#     "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
#     "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
#     "deepseek-ai/DeepSeek-V3"
#     "claude-3-5-haiku-latest"
#     "claude-3-5-sonnet-latest"
# )

MODELS=("claude-3-5-haiku-latest" "gpt-4o-mini")

for MODEL in "${MODELS[@]}"; do
    CSV_PATH="Results_Cross_Domain/verbosity_${MODEL##*/}_prediction_factqa.csv"
    if [[ "$MODEL" == claude* ]]; then
        SAVE_BATCH_SIZE=25
    else
        SAVE_BATCH_SIZE=100
    fi

    echo "==============================="
    echo ">> MODEL = $MODEL"
    echo ">> CSV PATH = $CSV_PATH"
    echo "==============================="

    python Scripts_Cross_Domain/cross_domain_LLM_judge_inference_with_CoT.py \
        --LLM_evaluator "$MODEL" \
        --csv_file_path "$CSV_PATH" \
        --CoT_base_model "$BASE_MODEL" \
        --save_batch_size $SAVE_BATCH_SIZE
done