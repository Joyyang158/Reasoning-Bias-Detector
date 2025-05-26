# export CUDA_VISIBLE_DEVICES=4,5

export HF_HOME="/qumulo/shared_data/haoyan_hf_cache/"


# MODEL_PATH="/qumulo/haoyan/DeepSeek-R1-Distill-Qwen-14B-full-fine-tuned_CoT_with_bias_type/final_model"
# BASE_MODEL="DeepSeek-R1-Distill-Qwen-14B"

MODEL_PATH="/qumulo/haoyan/DeepSeek-R1-Distill-Llama-8B-full-fine-tuned_CoT_with_bias_type/final_model"
BASE_MODEL="DeepSeek-R1-Distill-Llama-8B"
ROUND=5


# MODELS=(
#     "gpt-4o"
#     "DeepSeek-V3"
#     "claude-3-5-sonnet-latest"
#     "Meta-Llama-3.1-405B-Instruct-Turbo"
#     "claude-3-5-haiku-latest"
#     "gpt-4o-mini"
#     "Meta-Llama-3.1-8B-Instruct-Turbo-128K"
#     "Meta-Llama-3.1-70B-Instruct-Turbo"
# )

MODELS=(
    "claude-3-5-haiku-latest"
    "gpt-4o-mini"
)


BIAS_TYPES=("verbose" "position" "bandwagon" "sentiment")
# BIAS_TYPES=("bandwagon")

declare -A FILE_PREFIXES
FILE_PREFIXES=( ["verbose"]="Verbosity" \
                ["position"]="Position" \
                ["bandwagon"]="Bandwagon" \
                ["sentiment"]="Sentiment" )

declare -A FILE_SUFFIXES
FILE_SUFFIXES=( ["verbose"]="GSM8K" \
                ["position"]="ArenaHumanPreference" \
                ["bandwagon"]="ArenaHumanPreference" \
                ["sentiment"]="ScienceQA" )

for BIAS in "${BIAS_TYPES[@]}"; do
    SUBDIR=${FILE_PREFIXES[$BIAS]}
    SUFFIX=${FILE_SUFFIXES[$BIAS]}

    for MODEL in "${MODELS[@]}"; do
        if [[ "$BIAS" == "bandwagon" ]]; then
            CSV_PATH="Results_Recursive_Inference/${MODEL}/${SUBDIR}/Bandwagon_prediction_${SUFFIX}_Round_${ROUND}.csv"
        else
            CSV_PATH="Results_Recursive_Inference/${MODEL}/${SUBDIR}/prediction_${SUFFIX}_Round_${ROUND}.csv"
        fi

        echo "Running inference for model=$MODEL, bias=$BIAS, round=$ROUND..."

        python Scripts_Model_to_LLM_Judge_Inference/model_CoT_to_llm_judge_inference.py \
            --model_path "$MODEL_PATH" \
            --csv_file_path "$CSV_PATH" \
            --LLM_evaluator "$MODEL" \
            --CoT_base_model "$BASE_MODEL" \
            --bias_type "$BIAS"
    done
done


