from together import Together
import os
from tqdm import tqdm
import time
from openai import OpenAI
import ast
import pandas as pd
import anthropic
from func_timeout import func_set_timeout
import func_timeout
import random
import re
from get_bias_reflection_prompt import get_verbosity_prompt, get_position_prompt, get_bandwagon_prompt, get_sentiment_prompt


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

def read_df(df_path):
    df = pd.read_csv(df_path)

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

def generate_prediction(client, bias_type, model_name, system_prompt, user_prompt, generate_func, max_retries=3, default_value=-1):
    for attempt in range(max_retries):
        try:
            prediction = generate_func(client, model_name, system_prompt, user_prompt).strip()
            if bias_type in ["verbosity", "position", "bandwagon"]:
                if prediction == "Output (a)":
                    return 1
                elif prediction == "Output (b)":
                    return 2
                else:
                    raise ValueError(f"Invalid prediction: '{prediction}'. Expected 'Output (a)' or 'Output (b)'.")

            elif bias_type == "sentiment":
                return int(extract_number(prediction)) - 1
            
            else:
                raise ValueError(f"Unknown Bias Type: {bias_type}")

        except func_timeout.exceptions.FunctionTimedOut:
            print(f"⏳ Timeout: Model response took longer than expected (Attempt {attempt+1}/{max_retries}), retrying...")
        
        except ValueError as e:
            print(f"❌ {e} (Attempt {attempt + 1}/{max_retries}), retrying...")
    
    print(f"❌ Failed to get a valid prediction after {max_retries} attempts. Assigning default value: {default_value}")
    return default_value

def answer_generation(client, bias_type, model_name, system_prompt, user_template, prediction_column_name, max_retries=3, save_batch_size=100):
    output_file_mapping = {
        "verbosity": "Results_Baselines/Verbosity_GSM8K_Results_prediction.csv",
        "position": "Results_Baselines/Position_ArenaHumanPreference_Results_prediction.csv",
        "bandwagon": "Results_Baselines/Bandwagon_ArenaHumanPreference_Results_prediction.csv",
        "sentiment": "Results_Baselines/Sentiment_ScienceQA_Results_prediction.csv"
    }

    output_file = output_file_mapping[bias_type]
    input_df = pd.read_csv(output_file)
    print(f"🔄 Loaded existing results from {output_file}")
    
    for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
        if pd.notna(row.get(prediction_column_name, None)):
            continue

        if bias_type == "verbosity":
            input_text = row['question']
            label = row['ground_truth_original']
            if label == 1:
                output_1 = row['single_correct_answer']
                output_2 = row['gemma_answer']
            if label == 2:
                output_1 = row['gemma_answer']
                output_2 = row['single_correct_answer']
            
            user_prompt = user_template.format(input=input_text, output_1=output_1, output_2=output_2)
            input_df.at[index, prediction_column_name] = generate_prediction(client, bias_type, model_name, system_prompt, user_prompt, togetherai_generate, max_retries)
        
        elif bias_type == "position":
            input = row['question']
            output_original = ast.literal_eval(row['original'])
            output_reversed = ast.literal_eval(row['reversed'])

            user_prompt = user_template.format(input = input, output_1 = output_original['response_a'], output_2 = output_original['response_b'])
            input_df.at[index, f"{prediction_column_name}_original"] = generate_prediction(client, bias_type, model_name, system_prompt, user_prompt, togetherai_generate, max_retries)

            user_prompt = user_template.format(input = input, output_1 = output_reversed['response_a'], output_2 = output_reversed['response_b'])
            input_df.at[index, f"{prediction_column_name}_position"] = generate_prediction(client, bias_type, model_name, system_prompt, user_prompt, togetherai_generate, max_retries)
        
        elif bias_type == "bandwagon":
            input = row['question']
            output_original = ast.literal_eval(row['original'])
            ground_truth = row['ground_truth_original']
            if ground_truth == 1:
                statement = "90%% believe that Output (b) is better"
            else:
                statement = "90%% believe that Output (a) is better"
            user_prompt = user_template.format(statement = statement, input = input, output_1 = output_original['response_a'], output_2 = output_original['response_b'])

            input_df.at[index, prediction_column_name] = generate_prediction(client, bias_type, model_name, system_prompt, user_prompt, togetherai_generate, max_retries)
        
        elif bias_type == "sentiment":
            input = row['question']
            choices = row['Sentiment_Choice']

            options = generate_options(ast.literal_eval(choices))

            user_prompt = user_template.format(input = input, options = options)
            input_df.at[index,  prediction_column_name] = generate_prediction(client, bias_type, model_name, system_prompt, user_prompt, togetherai_generate, max_retries)
        
        else:
            raise ValueError(f"Unknown Bias Type: {bias_type}")
        
        if index < 2:
            print(user_prompt)


        
        if (index + 1) % save_batch_size == 0 or index == len(input_df) - 1:
            input_df.to_csv(output_file, index=False)
            print(f"✅ Saved {index + 1} rows to {output_file}")


def main():
    get_prompt_function_mapping = {
        "verbosity": get_verbosity_prompt,
        "position": get_position_prompt,
        "bandwagon": get_bandwagon_prompt,
        "sentiment": get_sentiment_prompt
    }
    TogehterAI_API_KEY = os.environ["TogehterAI_API_KEY"]
    togetherai_model = ["meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"]
    client = Together(api_key = TogehterAI_API_KEY)
    for model_name in togetherai_model:
        prediction_column_name = f"{model_name.split('/')[-1]}_with_bias_prompt"
        for bias_type in ["verbosity", "position", "bandwagon", "sentiment"]:
            print(f"$$$$$$$$$$$$$$$$$$ {model_name} - {bias_type} $$$$$$$$$$$$$$$$$$$$$$")
            system_prompt, user_template = get_prompt_function_mapping[bias_type]()
            answer_generation(client, bias_type, model_name, system_prompt, user_template, prediction_column_name)


if __name__ == "__main__":
    main()
