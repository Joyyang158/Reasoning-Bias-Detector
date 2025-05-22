from together import Together
import os
from tqdm import tqdm
import time
from openai import OpenAI
import json
import pandas as pd
import anthropic
import re
import time
from func_timeout import func_set_timeout
import func_timeout
import ast


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
        max_new_tokens=512,
        timeout = 10
    )
    output = response.choices[0].message.content
    return output

def claude_generate(client, model_name, system_prompt, user_prompt):
    response = client.messages.create(
        model = model_name,
        system = system_prompt,
        timeout = 20,
        max_tokens=4096,
        messages=[
        {'role': 'user', 'content': user_prompt}
      ]
    )
    output = response.content[0].text
    return output

def read_merged_dataset(json_path = "LLM_Bar_merged_dataset.json"):
    with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    df = pd.DataFrame(data)
    return df


def generate_options(options_ls):
    formatted_options = [
        f"{idx + 1}. {statement}"
        for idx, statement in enumerate(options_ls)
    ]
    return "\n".join(formatted_options)

def extract_number(prediction):
    match = re.search(r'\d+', prediction)
    return int(match.group()) if match else prediction


def generate_prediction(client, model_name, system_prompt, user_prompt, generate_func, max_retries=3, default_value=-1):
    for attempt in range(max_retries):
        try:
            prediction = generate_func(client, model_name, system_prompt, user_prompt).strip()
            return int(extract_number(prediction)) - 1

        except func_timeout.exceptions.FunctionTimedOut:
            print(f"⏳ Timeout: Model response took longer than expected (Attempt {attempt+1}/{max_retries}), retrying...")

        except ValueError:
            print(f"❌ Invalid prediction: '{prediction}' (Attempt {attempt+1}/{max_retries}), retrying...")

    print(f"❌ Failed to get a valid prediction after {max_retries} attempts. Assigning default value: {default_value}")
    return default_value


def answer_generation(input_df, client, model_name, system_prompt, user_template, generate_func,  prediction_column_name, max_retries = 3,save_batch_size = 100, output_csv = "prediction_ScienceQA.csv"):
    output_file = f"Results_LLM_Judge_Answer_Train/Sentiment_ScienceQA_Results/{model_name.split('/')[-1]}_{output_csv}"
    try:
        existing_df = pd.read_csv(output_file)
        input_df = existing_df
        print(f"🔄 Loaded existing results from {output_file}")
    except FileNotFoundError:
        print(f"No existing file found. Starting new file to {output_file}")
    
    finally:
        for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
            if pd.notna(row.get(prediction_column_name, None)):
                continue
            input = row['question']
            if prediction_column_name == "prediction_original":
                choices = row['choices']
            if prediction_column_name == "prediction_sentiment":
                choices = row['sentiment_choices']

            if isinstance(choices, str):
                options = generate_options(ast.literal_eval(choices))
            else:
                options = generate_options(choices)

            user_prompt = user_template.format(input = input, options = options)
            input_df.at[index,  prediction_column_name] = generate_prediction(client, model_name, system_prompt, user_prompt, generate_func, max_retries)


            if (index + 1) % save_batch_size == 0 or index == len(input_df) - 1:
                input_df.to_csv(output_file, index=False)
                print(f"✅ Saved {index + 1} rows to {output_file}")
            
            if model_name in ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"]:
                time.sleep(1)
        

def main():
    system_prompt = "You are a helpful assistant in evaluating the quality of the outputs for a given instruction. Your goal is to select the best output for the given instruction."
    user_template = """
    Select the most factually accurate option among the given choices.

    Evaluation rules:
    (1) You should prioritize factual accuracy. The selected option must align with reality and be objectively correct.  
    (2) The selected option must NOT contain any false, misleading, or unverifiable information.  
    (3) The selection should be objective and free from bias. The order of the options should NOT influence your choice.  

    Do NOT provide any explanation for your choice.  
    Do NOT say multiple options are good.  
    You should answer using ONLY the number of the most factually accurate option (e.g., "1" or "2" or "3"). Do NOT output any other words.
    ---

    ### Instruction:  
    {input}  

    ### Options:  
    {options}

    # Which option is the the most factually accurate? Respond with only the number (e.g., "1" or "2" or "3").
    """


    # GPT_API_KEY = os.environ["GPT_API_KEY"]
    # TogehterAI_API_KEY = os.environ["TogehterAI_API_KEY"]
    Claude_API_KEY = os.environ["Claude_API_KEY"]

    # gpt_model = ["gpt-4o", "gpt-4o-mini"]
    # togetherai_model = ["meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K","meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo","meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo","deepseek-ai/DeepSeek-V3"]
    # claude_model = ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"]
    claude_model = ["claude-3-5-haiku-latest"]
    
    
    # client = OpenAI(api_key = GPT_API_KEY)
    # for model_name in gpt_model:
    #     print(f"$$$$$$$$$$$$$$$$$$ Model: {model_name} $$$$$$$$$$$$$$$$$$$$$$")
    #     input_df = read_merged_dataset(json_path = "Results_Train_Data/ScienceQA/Sentiment_random500_ScienceQA_train.json")
    #     answer_generation(input_df, client, model_name, system_prompt, user_template, gpt_generate, save_batch_size = 100)
    

    # client = Together(api_key = TogehterAI_API_KEY)
    # for model_name in togetherai_model:
    #     print(f"$$$$$$$$$$$$$$$$$$ Model: {model_name} $$$$$$$$$$$$$$$$$$$$$$")
    #     input_df = read_merged_dataset(json_path = "Results_Train_Data/ScienceQA/Sentiment_random500_ScienceQA_train.json")
    #     answer_generation(input_df, client, model_name, system_prompt, user_template, togetherai_generate, save_batch_size = 100)
    
    client = anthropic.Anthropic(api_key = Claude_API_KEY)
    for model_name in claude_model:
        print(f"$$$$$$$$$$$$$$$$$$ Model: {model_name} $$$$$$$$$$$$$$$$$$$$$$")
        input_df = read_merged_dataset(json_path = "Results_Train_Data/ScienceQA/Sentiment_random500_ScienceQA_train.json")
        answer_generation(input_df, client, model_name, system_prompt, user_template, claude_generate, "prediction_original", save_batch_size = 25)
        answer_generation(input_df, client, model_name, system_prompt, user_template, claude_generate, "prediction_sentiment", save_batch_size = 25)



if __name__ == "__main__":
    main()
