# export CUDA_VISIBLE_DEVICES=4,5

export HF_HOME="/qumulo/shared_data/haoyan_hf_cache/"


export WANDB_API_KEY="xxx"

export HF_TOKEN='xxx'

huggingface-cli login --token "$HF_TOKEN"


accelerate launch --config_file Scripts_Model_Train/deepspeed_zero3.yaml Scripts_Model_Train/deepspeed_train_full.py --config Scripts_Model_Train/deepspeed_config_train_full.yaml --experiment_tag CoT