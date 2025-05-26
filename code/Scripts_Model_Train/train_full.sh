# export CUDA_VISIBLE_DEVICES=4,5

export HF_HOME="/qumulo/shared_data/haoyan_hf_cache/"

export WANDB_API_KEY="xxx"

accelerate launch --num_processes 2 Scripts_Model_Train/train_full.py --config Scripts_Model_Train/config_train_full.yaml --experiment_tag QA
# python Train/train_full.py --config Train/config_train_full.yaml