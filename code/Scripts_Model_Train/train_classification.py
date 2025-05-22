import argparse
import os
import yaml
import wandb
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

def wandb_login():
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        print("Warning: No W&B API key found. Please set WANDB_API_KEY in the environment.")

def load_yaml_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_model(model_name, num_labels=2, dtype=None, load_in_4bit=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        attn_implementation = "flash_attention_2",
        num_labels=num_labels,
        torch_dtype = dtype,
        load_in_4bit = load_in_4bit
        )
    return model, tokenizer


def formatting_prompts_func(examples):
    inputs = []
    labels = []

    for i in range(len(examples["instruction"])):
        input_text = (
            f"Instruction: {examples['instruction'][i]}\n"
            f"Choices: {examples['choices'][i]}\n"
            f"LLM Judgment: {examples['llm_judgment'][i]}"
        )
        inputs.append(input_text)
        label = 1 if examples["bias_label"][i].strip().lower() == "yes" else 0
        labels.append(label)

    return {"input_text": inputs, "label": labels}

def load_and_process_dataset(dataset_name, tokenizer, max_length):
    dataset = load_dataset(dataset_name, trust_remote_code=True)

    def process_split(split):
        split = split.map(
            lambda examples: formatting_prompts_func(examples),
            batched=True,
            remove_columns=[
                'index', 'bias_category', 'llm_judge_model', 'instruction',
                'choices', 'llm_judgment', 'without_bias_label_think_content', 'bias_label', "deepseek_prediction_bias_label"
            ]
        )
        split = split.map(
            lambda x: tokenizer(x["input_text"], padding="max_length", truncation=True, max_length=max_length),
            batched=True
        )
        return split

    train_dataset = process_split(dataset["train"])
    test_dataset = process_split(dataset["test"])

    print("=== Sample from train_dataset ===")
    print(train_dataset[0])

    return train_dataset, test_dataset

def train_model(model, tokenizer, train_dataset, eval_dataset, config):
    wandb_login()
    wandb.init(project=config["wandb_project"], name=config["wandb_run_name"], config=config)

    fp16 = config.get("mixed_precision") == "fp16"
    bf16 = config.get("mixed_precision") == "bf16"

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        fp16=fp16,
        bf16=bf16,
        logging_steps=config["logging_steps"],
        evaluation_strategy=config["evaluation_strategy"],
        eval_steps=config["eval_steps"],
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        seed=config["seed"],
        report_to=config["report_to"],
        save_only_model=config['save_only_model'],
        save_total_limit=config["save_total_limit"],
        optim=config["optim"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
    )

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        accuracy = (preds == labels).astype(float).mean().item()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(f"{config['output_dir']}/final_model")
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train a binary classification model")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")

    args = parser.parse_args()
    config = load_yaml_config(args.config)

    model, tokenizer = load_model(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    train_dataset, test_dataset = load_and_process_dataset(config["dataset_name"], tokenizer, config['max_seq_length'])
    train_model(model, tokenizer, train_dataset, test_dataset, config)

if __name__ == "__main__":
    main()
