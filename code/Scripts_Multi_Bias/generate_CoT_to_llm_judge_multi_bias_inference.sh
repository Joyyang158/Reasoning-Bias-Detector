# export CUDA_VISIBLE_DEVICES=4,5

export HF_HOME="/qumulo/shared_data/haoyan_hf_cache/"


# MODEL_PATH="/qumulo/haoyan/DeepSeek-R1-Distill-Qwen-14B-full-fine-tuned_CoT_with_bias_type/final_model"
# BASE_MODEL="DeepSeek-R1-Distill-Qwen-14B"

MODEL_PATH="/qumulo/haoyan/DeepSeek-R1-Distill-Llama-8B-full-fine-tuned_CoT_with_bias_type/final_model"
BASE_MODEL="DeepSeek-R1-Distill-Llama-8B"
MODELS=(
    "gpt-4o-mini"
    "claude-3-5-haiku-latest"
)

for MODEL in "${MODELS[@]}"; do
    CSV_PATH="Results_Multi_Bias/${MODEL}_prediction_verbosity_and_bandwagon_GSM8K.csv"

    echo "Running inference for model=$MODEL..."

    python Scripts_Multi_Bias/generate_CoT_to_llm_judge_multi_bias_inference.py \
        --model_path "$MODEL_PATH" \
        --csv_file_path "$CSV_PATH" \
        --LLM_evaluator "$MODEL" \
        --CoT_base_model "$BASE_MODEL"
done


