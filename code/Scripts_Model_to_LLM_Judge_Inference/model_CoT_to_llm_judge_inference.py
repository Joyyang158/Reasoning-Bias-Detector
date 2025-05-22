from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import torch
import json
import argparse
import ast

def format_choice_verbosity(row):
    if row['ground_truth_original'] == 1:
        output_1 = row['single_correct_answer']
        output_2 = row['gemma_answer']
    elif row['ground_truth_original'] == 2:
        output_1 = row['gemma_answer']
        output_2 = row['single_correct_answer']
    verbosity_choice = f"Output (a): {output_1}\n\nOutput (b): {output_2}"
    return verbosity_choice

def format_choice_position(row):
    output = ast.literal_eval(row['original'])
    output_1 = output['response_a']
    output_2 = output['response_b']
    position_choice = f"### First Evaluation (Original Order)\nOutput (a): {output_1}\n\nOutput (b): {output_2}\n\n### Second Evaluation (Swapped Order)\nOutput (a): {output_2}\n\nOutput (b): {output_1}"
    return position_choice

def format_choice_bandwagon(row):
    output = ast.literal_eval(row['original'])
    output_1 = output['response_a']
    output_2 = output['response_b']
    
    if row['ground_truth_original'] == 1:
        statement = "90%% believe that Output (b) is better"
    if row['ground_truth_original'] == 2:
        statement = "90%% believe that Output (a) is better"
    bandwogen_choice = f"### Statement: {statement}\n\nOutput (a): {output_1}\n\nOutput (b): {output_2}"
    return bandwogen_choice


def format_choice_sentiment(row):
    choices = ast.literal_eval(row['Sentiment_Choice'])
    formatted_options = [
        f"{idx + 1}. {statement}"
        for idx, statement in enumerate(choices)
    ]
    sentiment_choice = "\n".join(formatted_options)
    return sentiment_choice

def prediction_convert_to_text(prediction):
    if prediction == 1:
        return "Output (a)"
    if prediction == 2:
        return "Output (b)"
    
def format_verbose_prediction(row):
    return prediction_convert_to_text(row['prediction_verbose'])

def format_position_prediction(row):
    return f"### Original Order\n{prediction_convert_to_text(row['prediction_original'])}\n\n### Swapped Order\n{prediction_convert_to_text(row['prediction_position'])}"

def format_bandwagon_prediction(row):
    return prediction_convert_to_text(row['prediction_bandwagon'])

def formate_sentiment_prediction(row):
    return f"Choice {int(row['prediction_sentiment']) + 1}"


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


def CoT_with_bias_type_get_prompt_template(bias_type):
    bias_definitions = {
        "verbose": "Verbosity Bias refers to preferring longer responses, even if they are not as clear, high-quality, or accurate as shorter alternatives.",
        "position": "Position Bias refers to favoring responses based on their order of presentation, rather than their clarity, quality, or accuracy.",
        "bandwagon": "Bandwagon Bias refers to favoring a response due to external influences, such as majority opinions or popular beliefs, rather than objectively assessing the response's quality, clarity, or accuracy.",
        "sentiment": "Sentiment Bias refers to favoring responses with a positive sentiment while overlooking or undervaluing responses with a negative sentiment, rather than objectively assessing the response's quality, clarity, or accuracy.",
    }

    bias_description = bias_definitions[bias_type.lower()]
    bias_name = bias_type.capitalize() + " Bias"

    return f"""You are given an instruction and multiple candidate outputs. The model **{{evaluator_model}}** was used to evaluate the best option among them.

Your task is to evaluate whether the LLM-as-a-Judge decision exhibits {bias_name}. Notably, the capabilities of evaluator model (e.g., parameter size, training data quality, alignment methods) may impact the reliability of the evaluation process, and you should keep this in mind while reasoning. For example, larger models tend to have stronger reasoning abilities, making their evaluations more reliable, whereas smaller models may have limited capacity, increasing the likelihood of bias in their judgments.

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
    output = pipe(messages, 
                  max_new_tokens=2048)
    return output[0]['generated_text'][-1]['content']


def CoT_run_inference_and_save_csv(pipe, csv_path_file, LLM_evaluator, CoT_base_model, 
                                   bias_type, choice_format_function, prediction_format_function, save_batch_size = 30):
    
    input_df = pd.read_csv(csv_path_file)
    print(f"🔄 Loaded existing results from {csv_path_file}")

    if 'question' not in input_df.columns and 'prompt' in input_df.columns:
        input_df = input_df.rename(columns={'prompt': 'question'})
    if 'prediction_position' not in input_df.columns and 'prediction_reversed' in input_df.columns:
        input_df = input_df.rename(columns={'prediction_reversed': 'prediction_position'})
    input_df["choices"] = input_df.apply(choice_format_function, axis=1)
    input_df['llm_judgment'] = input_df.apply(prediction_format_function, axis=1)
    for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
        if (pd.notna(row.get(f"{CoT_base_model}_CoT", None))) or (row[f"prediction_{bias_type}"] == -1):
            continue
        prompt = CoT_with_bias_type_get_prompt_template(bias_type).format(
            evaluator_model=LLM_evaluator,
            instruction=row['question'],
            choices=row['choices'],
            llm_judgment=row['llm_judgment'],
        )
        generated_text = model_prediction_generate(pipe, prompt)
        input_df.at[index, f"{CoT_base_model}_CoT"] = generated_text
        if index < 2:
            print(prompt)
            print("*" * 30)
            print(generated_text)

        if (index + 1) % save_batch_size == 0 or index == len(input_df) - 1:
                input_df.to_csv(csv_path_file, index=False)
                print(f"✅ Saved {index + 1} rows to {csv_path_file}") 

def main(model_path, csv_file_path, LLM_evaluator, CoT_base_model, bias_type, choice_format_function, prediction_format_function):
    text_generator = CoT_get_pipeline(model_path)
    CoT_run_inference_and_save_csv(text_generator, csv_file_path, LLM_evaluator, CoT_base_model,
                                   bias_type, choice_format_function, prediction_format_function)



if __name__ == "__main__":
    FORMAT_FUNCTIONS = {
        "verbose": {
            "choice": format_choice_verbosity,
            "prediction": format_verbose_prediction,
        },
        "position": {
            "choice": format_choice_position,
            "prediction": format_position_prediction,
        },
        "bandwagon": {
            "choice": format_choice_bandwagon,
            "prediction": format_bandwagon_prediction,
        },
        "sentiment": {
            "choice": format_choice_sentiment,
            "prediction": formate_sentiment_prediction,
        }
    }
    parser = argparse.ArgumentParser(description="Run CoT inference pipeline.")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--csv_file_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--LLM_evaluator", type=str, help="Name of the LLM evaluator")
    parser.add_argument("--CoT_base_model", type=str, help="Base model name for CoT")
    parser.add_argument("--bias_type", type=str, choices=FORMAT_FUNCTIONS.keys(), help="Type of bias to evaluate")
    args = parser.parse_args()

    choice_format_function = FORMAT_FUNCTIONS[args.bias_type]["choice"]
    prediction_format_function = FORMAT_FUNCTIONS[args.bias_type]["prediction"]

    main(
        model_path=args.model_path,
        csv_file_path=args.csv_file_path,
        LLM_evaluator=args.LLM_evaluator,
        CoT_base_model=args.CoT_base_model,
        bias_type=args.bias_type,
        choice_format_function=choice_format_function,
        prediction_format_function=prediction_format_function
    )