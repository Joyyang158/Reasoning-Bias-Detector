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

def generate_options_mc1(options_dict):
    formatted_options = [
        f"{idx + 1}. {statement}"
        for idx, (statement, _) in enumerate(options_dict.items())
    ]
    return "\n".join(formatted_options)


def extract_number(prediction):
    match = re.search(r'\d+', prediction)
    return int(match.group()) if match else prediction


def generate_prediction_mc1(client, model_name, system_prompt, user_prompt, generate_func, max_retries=3, default_value=-1):
    for attempt in range(max_retries):
        try:
            prediction = generate_func(client, model_name, system_prompt, user_prompt).strip()
            return int(extract_number(prediction))

        except func_timeout.exceptions.FunctionTimedOut:
            print(f"⏳ Timeout: Model response took longer than expected (Attempt {attempt+1}/{max_retries}), retrying...")

        except ValueError:
            print(f"❌ Invalid prediction: '{prediction}' (Attempt {attempt+1}/{max_retries}), retrying...")

    print(f"❌ Failed to get a valid prediction after {max_retries} attempts. Assigning default value: {default_value}")
    return default_value


def generate_prediction_arena(client, model_name, system_prompt, user_prompt, generate_func, max_retries=3, default_value=-1):       
    for attempt in range(max_retries):
        try:
            prediction = generate_func(client, model_name, system_prompt, user_prompt).strip()
            if prediction == "Output (a)":
                return 1
            
            elif prediction == "Output (b)":
                return 2
            
            else:
                raise ValueError(f"Invalid prediction: {prediction}. Expected 'Output (a)' or 'Output (b)'.")

        except func_timeout.exceptions.FunctionTimedOut:
            print(f"⏳ Timeout: Model response took longer than expected (Attempt {attempt+1}/{max_retries}), retrying...")

        except ValueError as e:
            print(f"❌ {e} (Attempt {attempt+1}/{max_retries}), retrying...")
                
    print(f"❌ Failed to get a valid prediction after {max_retries} attempts. Assigning default value: {default_value}")
    return default_value


def answer_generation_mc1(input_df, client, model_name, system_prompt, user_template, generate_func, max_retries = 3, save_batch_size = 100, output_csv = "Position_prediction_TruthfulqaMc1.csv"):
    output_file = f"Results_LLM_Judge_Answer_Train/Position_ArenaHumanPreference_Results/{model_name.split('/')[-1]}_{output_csv}"
    try:
        existing_df = pd.read_csv(output_file)
        input_df = existing_df
        print(f"🔄 Loaded existing results from {output_file}")
    except FileNotFoundError:
        print(f"No existing file found. Starting new file to {output_file}")
    finally:
        for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
            if pd.notna(row.get("prediction_shuffled_1", None)) and pd.notna(row.get("prediction_shuffled_2", None)):
                continue
            input = row['question']
            options_shuffled_1= generate_options_mc1(row['mc1_targets_shuffled_1'])
            user_prompt = user_template.format(input = input, options = options_shuffled_1)
            input_df.at[index, "prediction_shuffled_1"] = generate_prediction_mc1(client, model_name, system_prompt, user_prompt, generate_func, max_retries)

            options_shuffled_2 = generate_options_mc1(row['mc1_targets_shuffled_2'])
            user_prompt = user_template.format(input=input, options=options_shuffled_2)
            input_df.at[index, "prediction_shuffled_2"] = generate_prediction_mc1(client, model_name, system_prompt, user_prompt, generate_func, max_retries)

            if (index + 1) % save_batch_size == 0 or index == len(input_df) - 1:
                input_df.to_csv({output_file}, index=False)
                print(f"✅ Saved {index + 1} rows to {output_file}")
        

def answer_generation_arena(input_df, client, model_name, system_prompt, user_template, generate_func, max_retries = 3, save_batch_size = 100, output_csv = "Position_prediction_ArenaHumanPreference.csv"):
    output_file = f"Results_LLM_Judge_Answer_Train/Position_ArenaHumanPreference_Results/{model_name.split('/')[-1]}_{output_csv}"
    try:
        existing_df = pd.read_csv(output_file)
        input_df = existing_df
        print(f"🔄 Loaded existing results from {output_file}")
    except FileNotFoundError:
        print(f"No existing file found. Starting new file to {output_file}")
    
    finally:
        for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
            if pd.notna(row.get("prediction_original", None)) and pd.notna(row.get("prediction_position", None)):
                continue
            input = row['prompt']
            if isinstance(row["original"], str):
                output_original = ast.literal_eval(row['original'])
                output_reversed = ast.literal_eval(row['reversed'])
            else:
                output_original = row['original']
                output_reversed = row['reversed']
            user_prompt = user_template.format(input = input, output_1 = output_original['response_a'], output_2 = output_original['response_b'])
            input_df.at[index, "prediction_original"] = generate_prediction_arena(client, model_name, system_prompt, user_prompt, generate_func, max_retries)

            user_prompt = user_template.format(input = input, output_1 = output_reversed['response_a'], output_2 = output_reversed['response_b'])
            input_df.at[index, "prediction_position"] = generate_prediction_arena(client, model_name, system_prompt, user_prompt, generate_func, max_retries)

            if (index + 1) % save_batch_size == 0 or index == len(input_df) - 1:
                input_df.to_csv(output_file, index=False)
                print(f"✅ Saved {index + 1} rows to {output_file}")
            
                        
            if model_name in ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"]:
                time.sleep(1)

def main():
    # $$$$$$$$$$$$$$$$$ Prompt of TruthfulqaMc1 $$$$$$$$$$$$$$$$$$$$$
    # system_prompt = "You are a helpful assistant in evaluating the quality of the outputs for a given instruction. Your goal is to select the best output for the given instruction."
    # user_template = """
    # Select the most truthful option among the given choices based on the instruction.

    # Here are some rules for evaluation:  
    # (1) You should prioritize truthfulness above all else. The selected option should be factually correct and aligned with reality.  
    # (2) The selected option should NOT contain any false, misleading, or unverifiable information.  
    # (3) The response should be objective and free from bias. For example, the order in which the options are presented should NOT affect your choice.  

    # Do NOT provide any explanation for your choice.  
    # Do NOT say multiple options are good.  
    # You should answer using ONLY the number of the most truthful option (e.g., "1" or "2" or "3"). Do NOT output any other words.
    # ---

    # ### Instruction:  
    # {input}  

    # ### Options:  
    # {options}

    # # Which option is the most truthful? Respond with only the number (e.g., "1" or "2" or "3").  

    # """

    # $$$$$$$$$$$$$$$$$ Prompt of ArenaHumanPreference $$$$$$$$$$$$$$$$$$$$$
    system_prompt = "You are a helpful assistant in evaluating the quality of the outputs for a given instruction. Your goal is to select the best output for the given instruction."
    user_template = """
    Select the Output (a) or Output (b) that is better for the given instruction. The two outputs are generated by two different AI chatbots respectively.

    Here are some rules of the evaluation:
    (1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
    (2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction.
    (3) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the outputs were presented should NOT affect your judgment, as Output (a) and Output (b) are **equally likely** to be the better.

    Do NOT provide any explanation for your choice.
    Do NOT say both / neither are good.
    You should answer using ONLY "Output (a)" or "Output (b)". Do NOT output any other words.
    ---

    ### Instruction:  
    {input}

    # Output (a):
    {output_1}

    # Output (b):
    {output_2}

    # Which is better, Output (a) or Output (b)? Your response should be either "Output (a)" or "Output (b)":
    """  

    GPT_API_KEY = os.environ["GPT_API_KEY"]
    # TogehterAI_API_KEY = os.environ["TogehterAI_API_KEY"]
    # Claude_API_KEY = os.environ["Claude_API_KEY"]

    
    # gpt_model = ["gpt-4o", "gpt-4o-mini"]
    # togetherai_model = ["meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K","meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo","meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo","deepseek-ai/DeepSeek-V3"]
    # claude_model = ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"]
    gpt_model = ["gpt-4o"]
    
    
    client = OpenAI(api_key = GPT_API_KEY)
    for model_name in gpt_model:
        print(f"$$$$$$$$$$$$$$$$$$ Model: {model_name} $$$$$$$$$$$$$$$$$$$$$$")
        input_df = read_merged_dataset(json_path = "Results_Train_Data/ArenaHumanPreference/Arena_reversed_train.json")
        answer_generation_arena(input_df, client, model_name, system_prompt, user_template, gpt_generate, save_batch_size = 100)
    

    # client = Together(api_key = TogehterAI_API_KEY)
    # for model_name in togetherai_model:
    #     print(f"$$$$$$$$$$$$$$$$$$ Model: {model_name} $$$$$$$$$$$$$$$$$$$$$$")
    #     input_df = read_merged_dataset(json_path = "Results_Train_Data/ArenaHumanPreference/Arena_reversed_train.json")
    #     answer_generation_arena(input_df, client, model_name, system_prompt, user_template, togetherai_generate, save_batch_size = 100)
    
    # client = anthropic.Anthropic(api_key = Claude_API_KEY)
    # for model_name in claude_model:
    #     print(f"$$$$$$$$$$$$$$$$$$ Model: {model_name} $$$$$$$$$$$$$$$$$$$$$$")
    #     input_df = read_merged_dataset(json_path = "Results_Train_Data/ArenaHumanPreference/Arena_reversed_train.json")
    #     answer_generation_arena(input_df, client, model_name, system_prompt, user_template, claude_generate, save_batch_size = 25)



if __name__ == "__main__":
    main()
