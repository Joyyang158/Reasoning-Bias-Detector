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
from bias_prompt_template import get_bandwagon_prompt, get_position_prompt, get_sentiment_prompt, get_verbosity_prompt



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

def extract_number(prediction):
    match = re.search(r'\d+', prediction)
    return int(match.group()) if match else prediction

def extract_bias_label(bias_analysis):
    bias_label = bias_analysis.split('</think>')[-1].strip()

    return bias_label

def extract_statement_and_options(text):
    statement_match = re.search(r"### Statement:\s*(.*?)\n", text)
    statement = statement_match.group(1).strip() if statement_match else None

    options_match = re.search(r"(Output \(a\):.*)", text, re.DOTALL)
    options = options_match.group(1).strip() if options_match else None

    return statement, options

def generate_prediction(client, model_name, system_prompt, user_prompt, generate_func, bias_type, max_retries=3, default_value=-1):
    for attempt in range(max_retries):
        try:
            prediction = generate_func(client, model_name, system_prompt, user_prompt).strip()

            if bias_type in ["verbose", "bandwagon"]:
                if prediction == "Output (a)":
                    return 1
                elif prediction == "Output (b)":
                    return 2
                else:
                    raise ValueError(f"Invalid prediction: '{prediction}'. Expected 'Output (a)' or 'Output (b)'.")

            elif bias_type == "sentiment":
                return int(extract_number(prediction)) - 1
            
            elif bias_type == "position":
                return prediction

            else:
                raise ValueError(f"Unknown Bias Type: {bias_type}")

        except func_timeout.exceptions.FunctionTimedOut:
            print(f"⏳ Timeout: Model response took longer than expected (Attempt {attempt + 1}/{max_retries}), retrying...")

        except ValueError as e:
            print(f"❌ {e} (Attempt {attempt + 1}/{max_retries}), retrying...")

    print(f"❌ Failed to get a valid prediction after {max_retries} attempts. Assigning default value: {default_value}")
    return default_value


def answer_generation(client, csv_file_path, LLM_evaluator, CoT_base_model, experiment_tag, generate_func, bias_type, save_batch_size=100):
    input_df = pd.read_csv(csv_file_path)
    print(f"🔄 Loaded existing results from {csv_file_path}")

    prediction_column_name = f"{CoT_base_model}_Prediction_with_{experiment_tag}"

    for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
        if pd.notna(row.get(prediction_column_name, None)):
            continue
        question = row['question']
        choices = row['choices']
        llm_judgment = row['llm_judgment']
        bias_analysis = row[f'{CoT_base_model}_CoT']

        if bias_type == "verbose":
            system_prompt, user_template = get_verbosity_prompt()
        elif bias_type == "position":
            system_prompt, user_template = get_position_prompt()
        elif bias_type == "bandwagon":
            system_prompt, user_template = get_bandwagon_prompt()
        elif bias_type == "sentiment":
            system_prompt, user_template = get_sentiment_prompt()
        else:
            raise ValueError(f"Unknown Bias Type: {bias_type}")
        
        if experiment_tag == "QA":
            bias_analysis = extract_bias_label(bias_analysis)
        
        if bias_type == "bandwagon":
            statement, choices = extract_statement_and_options(choices)
            user_prompt = user_template.format(statement = statement, input = question, options = choices, previous_choice = llm_judgment, bias_analysis = bias_analysis)
        else:
            user_prompt = user_template.format(input = question, options = choices, previous_choice = llm_judgment, bias_analysis = bias_analysis)
        if index < 2:
            print(user_prompt)
        input_df.at[index, prediction_column_name] = generate_prediction(client, LLM_evaluator, system_prompt, user_prompt, generate_func, bias_type)

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
    parser.add_argument("--experiment_tag", type=str, required=True, help="Specify 'QA' or 'CoT' to indicate the type of experiment")
    parser.add_argument("--bias_type", type=str, required=True, help="Type of bias to analyze (e.g., gender, sentiment)")
    parser.add_argument("--save_batch_size", type=int, default=100, help="Batch size for saving intermediate results")

    args = parser.parse_args()

    client = get_client_from_model(args.LLM_evaluator)
    generate_func = get_generate_func(args.LLM_evaluator)
    answer_generation(
        client=client,
        csv_file_path=args.csv_file_path,
        LLM_evaluator=args.LLM_evaluator,
        CoT_base_model=args.CoT_base_model,
        experiment_tag=args.experiment_tag,
        generate_func=generate_func,
        bias_type=args.bias_type,
        save_batch_size=args.save_batch_size
    )

if __name__ == "__main__":
    main()