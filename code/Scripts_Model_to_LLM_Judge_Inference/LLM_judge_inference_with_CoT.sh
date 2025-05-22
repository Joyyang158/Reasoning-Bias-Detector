#!/usr/bin/env bash

# BASE_MODEL="DeepSeek-R1-Distill-Qwen-1.5B"
# BASE_MODEL="DeepSeek-R1-Distill-Qwen-7B"
BASE_MODEL="DeepSeek-R1-Distill-Llama-8B"
# BASE_MODEL="DeepSeek-R1-Distill-Qwen-14B"

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
ROUND=5

BIAS_TYPES=("verbose" "position" "bandwagon" "sentiment")

for BIAS in "${BIAS_TYPES[@]}"; do
    if [ "$BIAS" = "verbose" ]; then
        SUBDIR="Verbosity"
        SUFFIX="GSM8K"
    elif [ "$BIAS" = "position" ]; then
        SUBDIR="Position"
        SUFFIX="ArenaHumanPreference"
    elif [ "$BIAS" = "bandwagon" ]; then
        SUBDIR="Bandwagon"
        SUFFIX="ArenaHumanPreference"
    elif [ "$BIAS" = "sentiment" ]; then
        SUBDIR="Sentiment"
        SUFFIX="ScienceQA"
    else
        echo "❌ Unknown bias type: $BIAS"
        exit 1
    fi


    for MODEL in "${MODELS[@]}"; do
        if [ "$BIAS" = "bandwagon" ]; then
            CSV_PATH="Results_Recursive_Inference/${MODEL##*/}/${SUBDIR}/Bandwagon_prediction_${SUFFIX}_Round_${ROUND}.csv"
        else
            CSV_PATH="Results_Recursive_Inference/${MODEL##*/}/${SUBDIR}/prediction_${SUFFIX}_Round_${ROUND}.csv"
        fi

        if [[ "$MODEL" == claude* ]]; then
            SAVE_BATCH_SIZE=25
        else
            SAVE_BATCH_SIZE=100
        fi

        echo "==============================="
        echo ">> MODEL = $MODEL"
        echo ">> BIAS TYPE = $BIAS"
        echo ">> CSV PATH = $CSV_PATH"
        echo "==============================="

        python Scripts_Model_to_LLM_Judge_Inference/LLM_judge_inference_with_CoT.py \
            --LLM_evaluator "$MODEL" \
            --csv_file_path "$CSV_PATH" \
            --CoT_base_model "$BASE_MODEL" \
            --experiment_tag "CoT" \
            --bias_type "$BIAS" \
            --save_batch_size $SAVE_BATCH_SIZE
    done
done