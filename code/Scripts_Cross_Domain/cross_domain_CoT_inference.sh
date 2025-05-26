# export CUDA_VISIBLE_DEVICES=4,5

export HF_HOME="/qumulo/shared_data/haoyan_hf_cache/"

# MODEL_PATH="/qumulo/haoyan/DeepSeek-R1-Distill-Qwen-14B-full-fine-tuned_CoT_with_bias_type/final_model"
# BASE_MODEL="DeepSeek-R1-Distill-Qwen-14B"

MODEL_PATH="/qumulo/haoyan/DeepSeek-R1-Distill-Llama-8B-full-fine-tuned_CoT_with_bias_type/final_model"
BASE_MODEL="DeepSeek-R1-Distill-Llama-8B"


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


for MODEL in "${MODELS[@]}"; do
    CSV_PATH="Results_Cross_Domain/verbosity_${MODEL}_prediction_factqa.csv"

    echo "Running inference for model=$MODEL..."

    python Scripts_Cross_Domain/cross_domain_CoT_inference.py \
        --model_path "$MODEL_PATH" \
        --csv_file_path "$CSV_PATH" \
        --LLM_evaluator "$MODEL" \
        --CoT_base_model "$BASE_MODEL"

done