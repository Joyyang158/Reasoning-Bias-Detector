#!/usr/bin/env bash


BASE_MODEL="DeepSeek-R1-Distill-Llama-8B"

# MODELS=("gpt-4o-mini")
MODELS=("claude-3-5-haiku-latest")


for MODEL in "${MODELS[@]}"; do
    CSV_PATH="Results_Multi_Bias/${MODEL##*/}_prediction_verbosity_and_bandwagon_GSM8K.csv"
    
    if [[ "$MODEL" == claude* ]]; then
        SAVE_BATCH_SIZE=25
    else
        SAVE_BATCH_SIZE=100
    fi

    echo "==============================="
    echo ">> MODEL = $MODEL"
    echo ">> CSV PATH = $CSV_PATH"
    echo "==============================="

    python Scripts_Multi_Bias/multi_bias_LLM_judge_inference_with_CoT.py \
        --LLM_evaluator "$MODEL" \
        --csv_file_path "$CSV_PATH" \
        --CoT_base_model "$BASE_MODEL" \
        --save_batch_size $SAVE_BATCH_SIZE
done