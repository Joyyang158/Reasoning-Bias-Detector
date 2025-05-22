from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import torch
import json
import argparse
import ast


def format_choice(row):
    if row['ground_truth_original'] == 1:
        output_1 = row['single_correct_answer']
        output_2 = row['gemma_answer']
        statement = "90%% believe that Output (b) is better"
    elif row['ground_truth_original'] == 2:
        output_1 = row['gemma_answer']
        output_2 = row['single_correct_answer']
        statement = "90%% believe that Output (a) is better"
        
    bandwogen_choice = f"### Statement: {statement}\n\nOutput (a): {output_1}\n\nOutput (b): {output_2}"
    return bandwogen_choice


def prediction_convert_to_text(prediction):
    if prediction == 1:
        return "Output (a)"
    if prediction == 2:
        return "Output (b)"
    
def format_prediction(row):
    return prediction_convert_to_text(row['prediction_verbose_bandwagon'])


def CoT_get_pipeline(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2"
    )
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    model.config.pad_token_id = tokenizer.pad_token_id
    text_generator = pipeline("text-generation", model = model, tokenizer = tokenizer)
    return text_generator


def CoT_with_bias_type_get_prompt_template():
    bias_definitions = {
        "verbose": "Verbosity Bias refers to preferring longer responses, even if they are not as clear, high-quality, or accurate as shorter alternatives.",
        "bandwagon": "Bandwagon Bias refers to favoring a response due to external influences, such as majority opinions or popular beliefs, rather than objectively assessing the response's quality, clarity, or accuracy."
    }

    return f"""You are given an instruction and multiple candidate outputs. The model **{{evaluator_model}}** was used to evaluate the best option among them.

Your task is to evaluate whether the LLM-as-a-Judge decision exhibits **verbosity bias** and/or **bandwagon bias**. Notably, the capabilities of the evaluator model (e.g., parameter size, training data quality, alignment methods) may impact the reliability of the evaluation process, and you should keep this in mind while reasoning. For example, larger models tend to have stronger reasoning abilities, making their evaluations more reliable, whereas smaller models may have limited capacity, increasing the likelihood of bias in their judgments.

Notably, it is possible that more than one type of bias exists in a single evaluation. Please assess each type of bias independently when making your judgment, and determine whether verbosity bias, bandwagon bias, both, or neither are present.

- {bias_definitions["verbose"]}
- {bias_definitions["bandwagon"]}

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
    output = pipe(messages, 
                  max_new_tokens=2048)
    return output[0]['generated_text'][-1]['content']


def CoT_run_inference_and_save_csv(pipe, csv_path_file, LLM_evaluator, CoT_base_model, save_batch_size = 30):
    
    input_df = pd.read_csv(csv_path_file)
    print(f"🔄 Loaded existing results from {csv_path_file}")


    input_df["choices"] = input_df.apply(format_choice, axis=1)
    input_df['llm_judgment'] = input_df.apply(format_prediction, axis=1)
    for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
        if (pd.notna(row.get(f"{CoT_base_model}_CoT", None))) or (row[f"prediction_verbose_bandwagon"] == -1):
            continue
        prompt = CoT_with_bias_type_get_prompt_template().format(
            evaluator_model=LLM_evaluator,
            instruction=row['question'],
            choices=row['choices'],
            llm_judgment=row['llm_judgment'],
        )
        generated_text = model_prediction_generate(pipe, prompt)
        input_df.at[index, f"{CoT_base_model}_CoT"] = generated_text
        if index < 1:
            print(prompt)
            print("*" * 30)
            print(generated_text)

        if (index + 1) % save_batch_size == 0 or index == len(input_df) - 1:
                input_df.to_csv(csv_path_file, index=False)
                print(f"✅ Saved {index + 1} rows to {csv_path_file}") 

def main(model_path, csv_file_path, LLM_evaluator, CoT_base_model):
    text_generator = CoT_get_pipeline(model_path)
    CoT_run_inference_and_save_csv(text_generator, csv_file_path, LLM_evaluator, CoT_base_model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CoT inference pipeline.")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--csv_file_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--LLM_evaluator", type=str, help="Name of the LLM evaluator")
    parser.add_argument("--CoT_base_model", type=str, help="Base model name for CoT")
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        csv_file_path=args.csv_file_path,
        LLM_evaluator=args.LLM_evaluator,
        CoT_base_model=args.CoT_base_model
    )