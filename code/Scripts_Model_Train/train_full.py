import argparse
import torch
import yaml
import wandb
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from accelerate import PartialState, Accelerator



def wandb_login():
    """Automatically log in to Weights & Biases using the API token from environment variables."""
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        print("Warning: No W&B API key found. Please set WANDB_API_KEY in the environment.") 

def load_yaml_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_model(model_name, dtype=None, load_in_4bit=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation = "flash_attention_2",
        torch_dtype = dtype,
        load_in_4bit = load_in_4bit
    )
    return model, tokenizer

def get_prompt_template(thinking_process=None):
    think_block = f"<think> {thinking_process} </think>" if thinking_process else ""

    return f"""You are given an instruction and multiple candidate outputs. The model **{{evaluator_model}}** was used to evaluate the best option among them.

Your task is to evaluate whether the LLM-as-a-Judge decision exhibits bias. Notably, the capabilities of evaluator model (e.g., parameter size, training data quality, alignment methods) may impact the reliability of the evaluation process, and you should keep this in mind while reasoning. For example, larger models tend to have stronger reasoning abilities, making their evaluations more reliable, whereas smaller models may have limited capacity, increasing the likelihood of bias in their judgments.

### Types of Bias to Consider:
- Verbosity Bias: Preferring longer responses, even if they are not as clear, high-quality, or accurate as shorter alternatives.
- Position Bias: Favoring responses based on their order of presentation, rather than their clarity, quality, or accuracy.
- Bandwagon Bias: Favoring a response due to external influences, such as majority opinions or popular beliefs, rather than objectively assessing the response's quality, clarity, or accuracy.
- Sentiment Bias: Favoring responses with a positive sentiment while overlooking or undervaluing responses with a negative sentiment, rather than objectively assessing the response's quality, clarity, or accuracy.

### Instruction:
{{instruction}}

### Choices:
{{choices}}

### Evaluation by LLM-as-a-Judge:
{{llm_judgment}}

- If no bias is detected, reply only with: "No".
- If bias is detected, reply only with: "Yes".

### Response:
{think_block}
{{bias_label}}"""


def remove_too_long_data(dataset, tokenizer, max_length):
    removed_indices = []
    for i, text in enumerate(dataset["text"]):
        if len(tokenizer(text)["input_ids"]) >= max_length:
            removed_indices.append(i)

    print(f"Filtered out {len(removed_indices)} samples exceeding {max_length} tokens.")
    dataset = dataset.filter(lambda _, idx: idx not in removed_indices, with_indices=True)

    return dataset

def formatting_prompts_func(examples, template, eos_token, experiment_tag):
    texts = []
    use_thinking = experiment_tag == "CoT"
    thinking_processes = examples.get("without_bias_label_think_content", None)

    for i in range(len(examples["instruction"])):
        kwargs = {
            "evaluator_model": examples["llm_judge_model"][i],
            "instruction": examples["instruction"][i],
            "choices": examples["choices"][i],
            "llm_judgment": examples["llm_judgment"][i],
            "bias_label": examples["bias_label"][i]
        }
        if use_thinking:
            kwargs["thinking_process"] = thinking_processes[i]

        formatted_text = template.format(**kwargs) + eos_token
        texts.append(formatted_text)

    return {"text": texts}


def load_and_process_dataset(dataset_name, tokenizer, max_length, experiment_tag):
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    if experiment_tag == "CoT":
        template = get_prompt_template(thinking_process="{thinking_process}")
    else:
        template = get_prompt_template()
    
    eos_token = tokenizer.eos_token

    def process_split(split):
        return split.map(
            lambda examples: formatting_prompts_func(examples, template, eos_token, experiment_tag),
            batched=True,
            remove_columns=[
                'index', 'bias_category', 'llm_judge_model', 'instruction',
                'choices', 'llm_judgment', 'without_bias_label_think_content', 'bias_label', "deepseek_prediction_bias_label"
            ]
        )

    train_dataset = process_split(dataset["train"])
    train_dataset = remove_too_long_data(train_dataset, tokenizer, max_length)

    test_dataset = process_split(dataset["test"])
    test_dataset = remove_too_long_data(test_dataset, tokenizer, max_length)

    print("=== Sample from train_dataset ===")
    print(train_dataset[0]["text"])

    print("\n=== Sample from test_dataset ===")
    print(test_dataset[0]["text"])
    
    return train_dataset, test_dataset


def train_model(model, tokenizer, collator, train_dataset, eval_dataset, config):
    """Train model with Weights & Biases logging."""
    accelerator = Accelerator()

    if accelerator.is_main_process:
        wandb_login()
        wandb.init(project=config["wandb_project"], name=config["wandb_run_name"], config=config)

    # Set Mixed Precision Training
    fp16 = False
    bf16 = False
    if config.get("mixed_precision") == "fp16":
        fp16 = True
    elif config.get("mixed_precision") == "bf16":
        bf16 = True

    training_args = SFTConfig(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        fp16=fp16,
        bf16=bf16,
        logging_steps=config["logging_steps"],
        optim=config["optim"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        seed=config["seed"],
        logging_strategy=config["logging_strategy"],
        save_strategy=config["save_strategy"],
        evaluation_strategy=config["evaluation_strategy"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        report_to=config["report_to"],
        save_only_model=config['save_only_model'],
        max_seq_length=config["max_seq_length"],
        dataset_text_field=config['dataset_text_field'],
        dataset_num_proc=config["dataset_num_proc"],
        load_best_model_at_end=config['load_best_model_at_end'],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        save_total_limit=config["save_total_limit"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=training_args
    )

    trainer_stats = trainer.train()
    trainer.save_model(f"{config['output_dir']}/final_model")

    print(f"Save model weights to successfully!")

    wandb.finish()

    return trainer_stats


def main():
    parser = argparse.ArgumentParser(description="Train an LLM using SFTTrainer")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--experiment_tag", type=str, required=True, choices=["QA", "CoT"], help="Type of experiment: QA or CoT")

    
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    experiment_tag = args.experiment_tag

    # Load model
    model, tokenizer = load_model(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    # Set padding side to left
    tokenizer.padding_side = "left"

    response_template_with_context = "### Response:\n"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    # Load dataset
    train_dataset, test_dataset = load_and_process_dataset(config["dataset_name"], tokenizer, config['max_seq_length'], experiment_tag)

    # Train model
    train_model(model, tokenizer, collator, train_dataset, test_dataset, config)

if __name__ == "__main__":
    main()