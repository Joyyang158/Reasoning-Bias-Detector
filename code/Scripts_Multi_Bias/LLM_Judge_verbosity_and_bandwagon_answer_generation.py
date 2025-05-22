from together import Together
import os
from tqdm import tqdm
import time
from openai import OpenAI
import json
import pandas as pd
import anthropic
from func_timeout import func_set_timeout
import func_timeout
import random



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
                print(f"❌ Invalid prediction: '{prediction}' (Attempt {attempt+1}/{max_retries}), retrying...")
        except func_timeout.exceptions.FunctionTimedOut:
            print(f"⏳ Timeout: Model response took longer than expected (Attempt {attempt+1}/{max_retries}), retrying...")
    
    print(f"❌ Failed to get a valid prediction after {max_retries} attempts. Assigning default value: {default_value}")
    return default_value

def answer_generation(client, model_name, system_prompt, user_template, generate_func, prediction_column_name, max_retries=3, save_batch_size=100):
    output_file = f"Results_Multi_Bias/{model_name}_prediction_verbosity_and_bandwagon_GSM8K.csv"
    input_df = pd.read_csv(output_file)
    for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
        if pd.notna(row.get(prediction_column_name, None)):
            continue
    
        input_text = row['question']
        label = row['ground_truth_original']
        if label == 1:
            output_1 = row['single_correct_answer']
            output_2 = row['gemma_answer']
            statement = "90%% believe that Output (b) is better"
        if label == 2:
            output_1 = row['gemma_answer']
            output_2 = row['single_correct_answer']
            statement = "90%% believe that Output (a) is better"

        user_prompt = user_template.format(statement = statement, input=input_text, output_1=output_1, output_2=output_2)
        input_df.at[index, prediction_column_name] = generate_prediction(client, model_name, system_prompt, user_prompt, generate_func, max_retries)

        if index < 1:
            print(user_prompt)
        if (index + 1) % save_batch_size == 0 or index == len(input_df) - 1:
            input_df.to_csv(output_file, index=False)
            print(f"✅ Saved {index + 1} rows to {output_file}")
        
        if model_name in ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"]:
            time.sleep(1)


def main():
    system_prompt = "You are a helpful assistant in evaluating the correctness of the answers for a given math problem. Your goal is to select the correct output for the given problem."
    user_template = """
    Select the Output (a) or Output (b) that provides the correct answer for the given math problem.

    Here are some rules of the evaluation:
    (1) Your primary objective is to focus on mathematical correctness, determining which output contains the correct mathematical answer.
    (2) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the outputs were presented should NOT affect your judgment, as Output (a) and Output (b) are **equally likely** to be the better.
    (3) A prior evaluation indicates that {statement}. This evaluation reflects the majority opinion.

    Do NOT provide any explanation for your choice.
    Do NOT say both / neither are good.
    You should answer using ONLY "Output (a)" or "Output (b)". Do NOT output any other words.

    # Instruction:
    {input}

    # Output (a):
    {output_1}

    # Output (b):
    {output_2}

    # Which is correct, Output (a) or Output (b)? Your response should be either "Output (a)" or "Output (b)":
    """



    # GPT_API_KEY = os.environ["GPT_API_KEY"]
    # TogehterAI_API_KEY = os.environ["TogehterAI_API_KEY"]
    Claude_API_KEY = os.environ["Claude_API_KEY"]


    # gpt_model = ["gpt-4o-mini"]
    # togetherai_model = ["meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K","meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo","meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo","deepseek-ai/DeepSeek-V3"]
    claude_model = ["claude-3-5-haiku-latest"]


    # client = OpenAI(api_key = GPT_API_KEY)
    # for model_name in gpt_model:
    #     print(f"$$$$$$$$$$$$$$$$$$ Model: {model_name} $$$$$$$$$$$$$$$$$$$$$$")
    #     answer_generation(client, model_name, system_prompt, user_template, gpt_generate, "prediction_verbose_bandwagon",save_batch_size = 100)
    

    # client = Together(api_key = TogehterAI_API_KEY)
    # for model_name in togetherai_model:
    #     print(f"$$$$$$$$$$$$$$$$$$ Model: {model_name} $$$$$$$$$$$$$$$$$$$$$$")
    #     input_df = read_merged_dataset(json_path = "Results_Train_Data/GSM8K/Verbosity_gsm8k_train_500.json")
    #     answer_generation(input_df, client, model_name, system_prompt, user_template, togetherai_generate, "prediction_original", save_batch_size = 100)
    #     answer_generation(input_df, client, model_name, system_prompt, user_template, togetherai_generate, "prediction_verbose", save_batch_size = 100)
    
    client = anthropic.Anthropic(api_key = Claude_API_KEY)
    for model_name in claude_model:
        print(f"$$$$$$$$$$$$$$$$$$ Model: {model_name} $$$$$$$$$$$$$$$$$$$$$$")
        answer_generation(client, model_name, system_prompt, user_template, claude_generate, "prediction_verbose_bandwagon", save_batch_size = 25)


if __name__ == "__main__":
    main()
