from together import Together
import os
from tqdm import tqdm
import time
from openai import OpenAI
import pandas as pd
import anthropic
from func_timeout import func_set_timeout
import func_timeout
import argparse
import re


def get_verbosity_prompt():
    system_prompt = "You are a helpful assistant in evaluating the correctness of answers for a given factual question. Your goal is to select the correct output based on factual accuracy, with consideration of prior evaluation and bias analysis."
    user_template = """
You are given an instruction and two options, along with your prior evaluation result and a bias analysis.  
Your task is to reconsider and make a final decision on whether Output (a) or Output (b) that provides the correct answer for the given math problem, based on the prior evaluation results and the provided bias analysis.

Here are some rules of the re-evaluation:
(1) (1) Your primary objective is to focus on factual correctness, determining which output contains the correct and verifiable answer.
(2) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the outputs were presented should NOT affect your judgment, as Output (a) and Output (b) are **equally likely** to be the better.
(3) The bias detection results are for reference – use your own judgment to decide whether the previous decision remains valid.

Do NOT provide any explanation for your choice.
Do NOT say both / neither are good.
You should answer using ONLY "Output (a)" or "Output (b)". Do NOT output any other words.

### Instruction:  
{input}

### Options:  
{options}

### Prior Evaluation Results:  
{previous_choice}

### Bias Analysis:  
{bias_analysis}

# After reconsideration, which is better, Output (a) or Output (b)? Your response should be either "Output (a)" or "Output (b)".
"""

    return system_prompt, user_template


def gpt_generate(client, model_name, system_prompt, user_prompt):
    response = client.chat.completions.create(
        model= model_name,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        timeout=20
    )
    output = response.choices[0].message.content
    return output

@func_set_timeout(10)
def togetherai_generate(client, model_name, system_prompt, user_prompt):   
    response = client.chat.completions.create(
        model = model_name,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
            max_new_tokens=512
    )
    output = response.choices[0].message.content
    return output

def claude_generate(client, model_name, system_prompt, user_prompt):
    response = client.messages.create(
        model = model_name,
        system = system_prompt,
        max_tokens=4096,
        messages=[
        {'role': 'user', 'content': user_prompt}
      ]
    )
    output = response.content[0].text
    return output

def read_df(df_path):
    df = pd.read_csv(df_path)
    return df

def generate_prediction(client, model_name, system_prompt, user_prompt, generate_func, max_retries=3, default_value=-1):
    for attempt in range(max_retries):
        try:
            prediction = generate_func(client, model_name, system_prompt, user_prompt).strip()

            if prediction == "Output (a)":
                return 1
            elif prediction == "Output (b)":
                return 2
            else:
                raise ValueError(f"Invalid prediction: '{prediction}'. Expected 'Output (a)' or 'Output (b)'.")

        except func_timeout.exceptions.FunctionTimedOut:
            print(f"⏳ Timeout: Model response took longer than expected (Attempt {attempt + 1}/{max_retries}), retrying...")

        except ValueError as e:
            print(f"❌ {e} (Attempt {attempt + 1}/{max_retries}), retrying...")

    print(f"❌ Failed to get a valid prediction after {max_retries} attempts. Assigning default value: {default_value}")
    return default_value


def answer_generation(client, csv_file_path, LLM_evaluator, CoT_base_model, generate_func, save_batch_size=100):
    input_df = pd.read_csv(csv_file_path)
    print(f"🔄 Loaded existing results from {csv_file_path}")

    prediction_column_name = f"{CoT_base_model}_Prediction_with_CoT"

    for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
        if pd.notna(row.get(prediction_column_name, None)):
            continue
        question = row['question']
        choices = row['choices']
        llm_judgment = row['llm_judgment']
        bias_analysis = row[f'{CoT_base_model}_CoT']

        system_prompt, user_template = get_verbosity_prompt()
        user_prompt = user_template.format(input = question, options = choices, previous_choice = llm_judgment, bias_analysis = bias_analysis)
        if index < 2:
            print(user_prompt)
        input_df.at[index, prediction_column_name] = generate_prediction(client, LLM_evaluator, system_prompt, user_prompt, generate_func)

        if (index + 1) % save_batch_size == 0 or index == len(input_df) - 1:
            input_df.to_csv(csv_file_path, index=False)
            print(f"✅ Saved {index + 1} rows to {csv_file_path}")
        
        if LLM_evaluator in ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"]:
            time.sleep(2)

def get_client_from_model(model_name):
    if model_name.startswith("gpt-4"):
        api_key = os.environ["GPT_API_KEY"]
        return OpenAI(api_key=api_key)

    elif model_name.startswith("claude"):
        api_key = os.environ["Claude_API_KEY"]
        return anthropic.Anthropic(api_key=api_key)
    
    else:
        api_key = os.environ["TogehterAI_API_KEY"]
        return Together(api_key=api_key)


def get_generate_func(model_name):
    if model_name.startswith("gpt-4"):
        return gpt_generate
    elif model_name.startswith("claude"):
        return claude_generate
    else:
        return togetherai_generate



def main():

    parser = argparse.ArgumentParser(description="Run LLM evaluation with bias analysis")

    parser.add_argument("--LLM_evaluator", type=str, required=True, help="Model name used for evaluation")
    parser.add_argument("--csv_file_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--CoT_base_model", type=str, required=True, help="Base model used for CoT generation")
    parser.add_argument("--save_batch_size", type=int, default=100, help="Batch size for saving intermediate results")

    args = parser.parse_args()

    client = get_client_from_model(args.LLM_evaluator)
    generate_func = get_generate_func(args.LLM_evaluator)
    answer_generation(
        client=client,
        csv_file_path=args.csv_file_path,
        LLM_evaluator=args.LLM_evaluator,
        CoT_base_model=args.CoT_base_model,
        generate_func=generate_func,
        save_batch_size=args.save_batch_size
    )

if __name__ == "__main__":
    main()