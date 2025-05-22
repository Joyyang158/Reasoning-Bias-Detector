from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import random
from datasets import load_dataset
import argparse
from tqdm import tqdm
import os
import torch

def prepare_input_text(example):
    return f"Instruction: {example['instruction']}\nChoices: {example['choices']}\nLLM Judgment: {example['llm_judgment']}"


def get_pipeline(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    return classifier


def run_inference_and_save_csv(pipe, base_model_name, dataset, experiment_tag, save_batch_size=30):
    output_file = f"Results_Model_Inference/{base_model_name}.csv"

    try:
        existing_df = pd.read_csv(output_file)
        input_df = existing_df
        print(f"🔄 Loaded existing results from {output_file}")
    except FileNotFoundError:
        input_df = pd.DataFrame(dataset["test"])
        print(f"No existing file found. Starting new file to {output_file}")

    for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
        if pd.notna(row.get(experiment_tag, None)):
            continue

        prompt = prepare_input_text(row)
        result = pipe(prompt)[0]
        prediction = max(result, key=lambda x: x["score"])["label"]
        prediction_label = "Yes" if prediction == "LABEL_1" else "No"

        input_df.at[index, experiment_tag] = prediction_label

        if (index + 1) % save_batch_size == 0 or index == len(input_df) - 1:
            input_df.to_csv(output_file, index=False)
            print(f"✅ Saved {index + 1} rows to {output_file}")


def run_sample_classification(pipe, base_model_name, dataset, experiment_tag, sample_size=10):
    sampled_data = dataset["test"].shuffle(seed=42).select(range(sample_size))
    print(f"=== Sample prediction using model: {base_model_name} ({experiment_tag}) ===")
    for i, example in enumerate(sampled_data):
        prompt = prepare_input_text(example)
        result = pipe(prompt)[0]
        prediction = max(result, key=lambda x: x["score"])["label"]

        print("#" * 50)
        print(f"Sample {i+1}")
        print(f"Input Prompt:\n{prompt}")
        print(f"Prediction: {prediction}")
        print(f"Scores: {result}")
        print("#" * 50)


def main(model_path, base_model_name, dataset_name, experiment_tag, output_method):
    dataset = load_dataset(dataset_name)
    classifier = get_pipeline(model_path)

    if output_method == "save-csv":
        run_inference_and_save_csv(classifier, base_model_name, dataset, experiment_tag)
    elif output_method == "print-sample":
        run_sample_classification(classifier, base_model_name, dataset, experiment_tag)
    else:
        raise NotImplementedError(f"Output method {output_method} not supported for classification.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run classification model inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--base_model_name", type=str, required=True, help="Name of base model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--experiment_tag", type=str, required=True, help="Used as column name in CSV output")
    parser.add_argument("--output_method", type=str, required=True, choices=["save-csv", "print-sample"],
                        help="Output method: 'save-csv' to save results to a CSV file, 'print-sample' to print a sample output")
    
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        base_model_name=args.base_model_name,
        dataset_name=args.dataset_name,
        experiment_tag=args.experiment_tag,
        output_method=args.output_method
    )
