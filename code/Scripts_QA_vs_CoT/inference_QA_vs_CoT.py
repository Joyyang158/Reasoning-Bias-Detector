from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import torch
import json
import argparse
import ast

def read_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data 


def format_choice_verbosity(row):
    output_1 = row['option 1']
    output_2 = row['option 2']
    verbosity_choice = f"Output (a): {output_1}\n\nOutput (b): {output_2}"
    return verbosity_choice

def format_choice_bandwagon(row):
    output = row['original']
    output_1 = output['response_a']
    output_2 = output['response_b']
    
    if row['ground_truth_original'] == 1:
        statement = "90%% believe that Output (a) is better"
    if row['ground_truth_original'] == 2:
        statement = "90%% believe that Output (b) is better"
    bandwogen_choice = f"\n### Statement: {statement}\n\nOutput (a): {output_1}\n\nOutput (b): {output_2}"
    return bandwogen_choice

def format_llm_judgment_verbosity(row):
    if row['label'] == 0:
        return "Output (a)"
    if row['label'] == 1:
        return "Output (b)"
    
def format_llm_judgment_bandwagon(row):
    if row['ground_truth_original'] == 1:
        return "Output (a)"
    if row['ground_truth_original'] == 2:
        return "Output (b)"

def prepare_input_text(example):
    return f"Instruction: {example['question']}\n\nChoices: {example['choices']}\n\nLLM Judgment: {example['llm_judgment']}"


def QA_classification_get_pipeline(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    return classifier

def CoT_get_pipeline(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2"
    )
    text_generator = pipeline("text-generation", model = model, tokenizer = tokenizer)
    return text_generator


def CoT_with_bias_type_get_prompt_template(bias_type):
    bias_descriptions = {
        "verbosity": "Verbosity bias refers to the preference for longer responses, even if they are not as clear, high-quality, or accurate as shorter alternatives.",
        "bandwagon": "Bandwagon bias refers to favoring a response due to external influences, such as majority opinions or popular beliefs, rather than objectively assessing the response's quality, clarity, or accuracy."
    }
    bias_description = bias_descriptions[bias_type]

    return f"""You are given an instruction and multiple candidate outputs. The model **{{evaluator_model}}** was used to evaluate the best option among them.

Your task is to evaluate whether the LLM-as-a-Judge decision exhibits {bias_type.capitalize()} Bias. Notably, the capabilities of evaluator model (e.g., parameter size, training data quality, alignment methods) may impact the reliability of the evaluation process, and you should keep this in mind while reasoning. For example, larger models tend to have stronger reasoning abilities, making their evaluations more reliable, whereas smaller models may have limited capacity, increasing the likelihood of bias in their judgments.

{bias_description}


### Instruction:
{{instruction}}

### Choices:
{{choices}}

### Evaluation by LLM-as-a-Judge:
{{llm_judgment}}

- If no bias is detected, reply only with: "No".
- If bias is detected, reply only with: "Yes".

### Response:
"""



def model_prediction_generate(pipe, user_prompt):
    messages = [
    {"role": "user", "content": user_prompt},
    ]
    output = pipe(messages, max_new_tokens=2048)
    return output[0]['generated_text'][-1]['content']


def QA_classification_run_inference_and_save_csv(pipe, input_df, output_file, experiment_tag, save_batch_size=30):
    try:
        existing_df = pd.read_csv(output_file)
        input_df = existing_df
        print(f"🔄 Loaded existing results from {output_file}")
    except FileNotFoundError:
        print(f"No existing file found. Starting new file to {output_file}")

    finally:
        for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
            if pd.notna(row.get(experiment_tag, None)):
                continue

            prompt = prepare_input_text(row)
            if index < 2:
                print("#"* 25 + "QA" + "#"* 25)
                print(prompt)
            result = pipe(prompt)[0]
            prediction = max(result, key=lambda x: x["score"])["label"]
            prediction_label = "Yes" if prediction == "LABEL_1" else "No"

            input_df.at[index, experiment_tag] = prediction_label

            if (index + 1) % save_batch_size == 0 or index == len(input_df) - 1:
                input_df.to_csv(output_file, index=False)
                print(f"✅ Saved {index + 1} rows to {output_file}")

def CoT_run_inference_and_save_csv(pipe, input_df, output_file, bias_type, experiment_tag, save_batch_size = 30):
    try:
        existing_df = pd.read_csv(output_file)
        input_df = existing_df
        print(f"🔄 Loaded existing results from {output_file}")
    except FileNotFoundError:
        print(f"No existing file found. Starting new file to {output_file}")
    
    finally:
        for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
            if pd.notna(row.get(experiment_tag, None)):
                continue
            prompt = CoT_with_bias_type_get_prompt_template(bias_type).format(
                evaluator_model="GPT-4o",
                instruction=row["question"],
                choices=row["choices"],
                llm_judgment=row["llm_judgment"]
            )
            generated_text = model_prediction_generate(pipe, prompt)
            input_df.at[index, experiment_tag] = generated_text
            if index < 2:
                print("#"* 25 + "CoT" + "#"* 25)
                print(prompt)
                print("*" * 50)
                print(generated_text)

            if (index + 1) % save_batch_size == 0 or index == len(input_df) - 1:
                    input_df.to_csv(output_file, index=False)
                    print(f"✅ Saved {index + 1} rows to {output_file}") 

def main(model_path, dataset_file_path, output_file, bias_type, experiment_tag):
    json_data = read_json(dataset_file_path)
    input_df = pd.DataFrame(json_data)
    if bias_type == "verbosity":
        input_df["choices"] = input_df.apply(format_choice_verbosity, axis=1)
        input_df['llm_judgment'] = input_df.apply(format_llm_judgment_verbosity, axis=1)
    
    if bias_type == "bandwagon":
        input_df = input_df.rename(columns={'prompt': 'question'})
        input_df["choices"] = input_df.apply(format_choice_bandwagon, axis=1)
        input_df['llm_judgment'] = input_df.apply(format_llm_judgment_bandwagon, axis=1)

    
    if experiment_tag == "QA":
        classifier = QA_classification_get_pipeline(model_path)
        QA_classification_run_inference_and_save_csv(classifier, input_df, output_file, experiment_tag)
    if experiment_tag == "CoT":
        text_generator = CoT_get_pipeline(model_path)
        CoT_run_inference_and_save_csv(text_generator, input_df, output_file, bias_type, experiment_tag)
    

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QA vs CoT inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_file_path", type=str, required=True, help="File path of the dataset")
    parser.add_argument("--output_file", type=str, required=True, help="File path of the output csv")
    parser.add_argument("--bias_type", type=str, required=True, choices=["verbosity", "bandwagon"], help="Type of bias to evaluate")
    parser.add_argument("--experiment_tag", type=str, required=True, choices=["QA", "CoT"], help="Used as column name in CSV output")

    
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        dataset_file_path=args.dataset_file_path,
        output_file=args.output_file,
        bias_type=args.bias_type,
        experiment_tag=args.experiment_tag
    )
        