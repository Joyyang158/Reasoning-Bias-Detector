import pandas as pd
import ast
import numpy as np
import os
from huggingface_hub import HfApi
from transformers import AutoTokenizer

def sentiment_filter_rows(row):
    choices = ast.literal_eval(row['sentiment_choices'])
    return (row["prediction_original"] < len(choices)) and (row["prediction_sentiment"]  < len(choices))

def filter_bias_accuracy_CoT(df, prediction_pre, prediction_post, label_pre, label_post, bias_label_flag):
    df = df[(df[prediction_pre] != -1) & (df[prediction_post] != -1)]
    if prediction_post == "prediction_sentiment":
        df = df[df.apply(sentiment_filter_rows, axis=1)]
    condition1 = (df[prediction_pre] == df[label_pre]) & (df[prediction_post] != df[label_post]) & (df[bias_label_flag] == "Yes")
    condition2 = (df[prediction_pre] != df[label_pre]) & (df[bias_label_flag] == "No")
    condition3 = (df[prediction_pre] == df[label_pre]) & (df[prediction_post] == df[label_post]) & (df[bias_label_flag] == "No")
    
    filtered_df = df[condition1 | condition2 | condition3]
    return filtered_df


#### For the test set, we don't have to select the samples with right deepseek prediction since we can compare fine-tuned model results with deepseek results
def filter_CoT_for_testset(df, prediction_pre, prediction_post):
    df = df[(df[prediction_pre] != -1) & (df[prediction_post] != -1)]
    if prediction_post == "prediction_sentiment":
        df = df[df.apply(sentiment_filter_rows, axis=1)]
    
    return df


def assign_bias_label_for_testset(row, prediction_pre, prediction_post, label_pre, label_post):
    if (
        row[prediction_pre] == row[label_pre] and
        row[prediction_post] != row[label_post]
    ):
        return "Yes"
    
    if (
        row[prediction_pre] != row[label_pre]
    ):
        return "No"
    
    if (
        row[prediction_pre] == row[label_pre] and
        row[prediction_post] == row[label_post]
    ):
        return "No"
    
    return "Unknown"

    
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
    choices = ast.literal_eval(row['sentiment_choices'])
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
    
def process_verbosity_df(df, split):
    if split == "train":
        filtered_df = filter_bias_accuracy_CoT(df, 'prediction_original', 'prediction_verbose', 'ground_truth_original', 'ground_truth_original', 'without_bias_label_binary_answer')
    else:
        filtered_df = filter_CoT_for_testset(df, 'prediction_original', 'prediction_verbose')
    output_df = pd.DataFrame()
    output_df['index'] = filtered_df.index
    filtered_df = filtered_df.reset_index()
    output_df['bias_category'] = "verbosity"
    output_df['llm_judge_model'] = "Llama-3.1-405B"
    output_df['instruction'] = filtered_df['question']
    output_df['choices'] = filtered_df.apply(lambda row: format_choice_verbosity(row), axis=1)
    output_df['llm_judgment'] = filtered_df.apply(lambda row: prediction_convert_to_text(row['prediction_verbose']), axis=1)
    output_df['without_bias_label_think_content'] = filtered_df['without_bias_label_think_content']
    if split == "train":
        output_df['deepseek_prediction_bias_label'] = filtered_df['without_bias_label_binary_answer']
        output_df['bias_label'] = filtered_df['without_bias_label_binary_answer']
    else:
        output_df['deepseek_prediction_bias_label'] = filtered_df['without_bias_label_binary_answer']
        output_df['bias_label'] = filtered_df.apply(lambda row: assign_bias_label_for_testset(row, 'prediction_original', 'prediction_verbose', 'ground_truth_original', 'ground_truth_original'), axis=1)
    return output_df

def process_position_df(df, split):
    if split == "train":
        filtered_df = filter_bias_accuracy_CoT(df, 'prediction_original', 'prediction_position', 'ground_truth_original', 'ground_truth_position', 'without_bias_label_binary_answer')
    else:
        filtered_df = filter_CoT_for_testset(df, 'prediction_original', 'prediction_position')
    output_df = pd.DataFrame()
    output_df['index'] = filtered_df.index
    filtered_df = filtered_df.reset_index()
    output_df['bias_category'] = "position"
    output_df['llm_judge_model'] = "GPT-4o"
    output_df['instruction'] = filtered_df['prompt']
    output_df['choices'] = filtered_df.apply(lambda row: format_choice_position(row) , axis=1)
    output_df['llm_judgment'] = filtered_df.apply(lambda row: f"### Original Order\n{prediction_convert_to_text(row['prediction_original'])}\n\n### Swapped Order\n{prediction_convert_to_text(row['prediction_position'])}", axis=1)
    output_df['without_bias_label_think_content'] = filtered_df['without_bias_label_think_content']
    if split == "train":
        output_df['deepseek_prediction_bias_label'] = filtered_df['without_bias_label_binary_answer']
        output_df['bias_label'] = filtered_df['without_bias_label_binary_answer']
    else:
        output_df['deepseek_prediction_bias_label'] = filtered_df['without_bias_label_binary_answer']
        output_df['bias_label'] = filtered_df.apply(lambda row: assign_bias_label_for_testset(row, 'prediction_original', 'prediction_position', 'ground_truth_original', 'ground_truth_position'), axis=1)
    return output_df

def process_bandwagon_df(df, split):
    if split == "train":
        filtered_df = filter_bias_accuracy_CoT(df, 'prediction_original', 'prediction_bandwagon', 'ground_truth_original', 'ground_truth_original', 'without_bias_label_binary_answer')
    else:
        filtered_df = filter_CoT_for_testset(df, 'prediction_original', 'prediction_bandwagon')
    output_df = pd.DataFrame()
    output_df['index'] = filtered_df.index
    filtered_df = filtered_df.reset_index()
    output_df['bias_category'] = "bandwagon"
    output_df['llm_judge_model'] = "GPT-4o-mini"
    output_df['instruction'] = filtered_df['prompt']
    output_df['choices'] = filtered_df.apply(lambda row: format_choice_bandwagon(row), axis=1)
    output_df['llm_judgment'] = filtered_df.apply(lambda row: prediction_convert_to_text(row['prediction_bandwagon']), axis=1)
    output_df['without_bias_label_think_content'] = filtered_df['without_bias_label_think_content']
    output_df['bias_label'] = filtered_df['without_bias_label_binary_answer']
    if split == "train":
        output_df['deepseek_prediction_bias_label'] = filtered_df['without_bias_label_binary_answer']
        output_df['bias_label'] = filtered_df['without_bias_label_binary_answer']
    else:
        output_df['deepseek_prediction_bias_label'] = filtered_df['without_bias_label_binary_answer']
        output_df['bias_label'] = filtered_df.apply(lambda row: assign_bias_label_for_testset(row, 'prediction_original', 'prediction_bandwagon', 'ground_truth_original', 'ground_truth_original'), axis=1)
    return output_df


def process_sentiment_df(df, split):
    if split == "train":
        filtered_df = filter_bias_accuracy_CoT(df, 'prediction_original', 'prediction_sentiment', 'ground_truth_original', 'ground_truth_original', 'without_bias_label_binary_answer')
    else:
        filtered_df = filter_CoT_for_testset(df, 'prediction_original', 'prediction_sentiment')
    output_df = pd.DataFrame()
    output_df['index'] = filtered_df.index
    filtered_df = filtered_df.reset_index()
    output_df['bias_category'] = "sentiment"
    output_df['llm_judge_model'] = "Claude-3.5-haiku"
    output_df['instruction'] = filtered_df['question']
    output_df['choices'] = filtered_df.apply(lambda row: format_choice_sentiment(row), axis=1)
    output_df['llm_judgment'] = filtered_df.apply(lambda row: f"Choice {int(row['prediction_sentiment']) + 1}", axis=1)
    output_df['without_bias_label_think_content'] = filtered_df['without_bias_label_think_content']
    if split == "train":
        output_df['deepseek_prediction_bias_label'] = filtered_df['without_bias_label_binary_answer']
        output_df['bias_label'] = filtered_df['without_bias_label_binary_answer']
    else:
        output_df['deepseek_prediction_bias_label'] = filtered_df['without_bias_label_binary_answer']
        output_df['bias_label'] = filtered_df.apply(lambda row: assign_bias_label_for_testset(row, 'prediction_original', 'prediction_sentiment', 'ground_truth_original', 'ground_truth_original'), axis=1)
    return output_df

def balance_testset(df_total, n_total = 500):
    n_per_label = n_total // 2
    final_samples = []

    for label in ['Yes', 'No']:
        label_df = df_total[df_total['bias_label'] == label]
        categories = label_df['bias_category'].unique()
        n_per_category = n_per_label // len(categories)

        samples = []
        for category in categories:
            cat_df = label_df[label_df['bias_category'] == category]
            sampled = cat_df.sample(n=min(n_per_category, len(cat_df)), random_state=42)
            samples.append(sampled)

        combined = pd.concat(samples)

        remaining = n_per_label - len(combined)
        if remaining > 0:
            extra_df = label_df.drop(combined.index)
            if len(extra_df) >= remaining:
                extra_sampled = extra_df.sample(n=remaining, random_state=42)
                combined = pd.concat([combined, extra_sampled])

        final_samples.append(combined)

    final_df = pd.concat(final_samples)
    return final_df



def dataset_to_hf(input_file_path, local_save_file_path, hf_save_file_path, split):

    file_name_verbosity = f"{input_file_path}/DeepSeek-R1_Meta-Llama-3.1-405B-Instruct-Turbo_CoT_GSM8K_Verbosity.csv"
    df_verbosity = pd.read_csv(file_name_verbosity)
    file_name_position = f"{input_file_path}/DeepSeek-R1_gpt-4o_CoT_Arena_Position.csv"
    df_position = pd.read_csv(file_name_position)
    file_name_bandwagon = f"{input_file_path}/DeepSeek-R1_gpt-4o-mini_CoT_Arena_Bandwagon.csv"
    df_bandwagon = pd.read_csv(file_name_bandwagon)
    file_name_sentiment = f"{input_file_path}/DeepSeek-R1_claude-3-5-haiku-latest_CoT_ScienceQA_Sentiment.csv"
    df_sentiment = pd.read_csv(file_name_sentiment)

    output_df_verbosity = process_verbosity_df(df_verbosity, split)
    output_df_position = process_position_df(df_position, split)
    output_df_bandwagon = process_bandwagon_df(df_bandwagon, split)
    output_df_sentiment = process_sentiment_df(df_sentiment, split)

    df_total = pd.concat([output_df_verbosity, output_df_position, output_df_bandwagon, output_df_sentiment], ignore_index=True)

    if split == "test":
        df_total = balance_testset(df_total)
    
    df_total = df_total.sample(frac=1, random_state=42).reset_index(drop=True)
    df_total.to_csv(local_save_file_path, index=False)

    api.upload_file(
    path_or_fileobj=local_save_file_path,
    path_in_repo=hf_save_file_path,
    repo_id=repo_id,
    repo_type="dataset"
    )
    
    print(f"Save the {local_save_file_path} to HF successfully!")


def calculate_token_length(df, tokenizer):
    token_sum = 0
    for _, row in df.iterrows():
        token_sum += len(tokenizer(row['without_bias_label_think_content'])['input_ids'])
    token_avg = np.round(token_sum / len(df), 0)
    return token_avg

        

train_input_file_path = "Results_CoT_Train"
train_local_save_file_path = "Results_CoT_Model_Trainng_Dataset/train_CoT_data.csv"
train_hf_save_file_path = "train_CoT_data.csv"

test_input_file_path = "Results_CoT_Test"
test_local_save_file_path = "Results_CoT_Model_Trainng_Dataset/test_CoT_data.csv"
test_hf_save_file_path = "test_CoT_data.csv"


hf_token = os.getenv("HF_TOKEN")
repo_id = "joyfine/LLM_Bias_Detection_CoT_Training"
api = HfApi(token=hf_token)
dataset_to_hf(train_input_file_path, train_local_save_file_path, train_hf_save_file_path, "train")
dataset_to_hf(test_input_file_path, test_local_save_file_path, test_hf_save_file_path, "test")


# tokenizer_qwen = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
# toeknizer_llama = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# print(f"# Tokens of CoT with Qwen Tokenizer in Verbosity: {calculate_token_length(output_df_verbosity, tokenizer_qwen)}")
# print(f"# Tokens of CoT with Llama Tokenizer in Verbosity: {calculate_token_length(output_df_verbosity, toeknizer_llama)}")

# print(f"# Tokens of CoT with Qwen Tokenizer in Position: {calculate_token_length(output_df_position, tokenizer_qwen)}")
# print(f"# Tokens of CoT with Llama Tokenizer in Position: {calculate_token_length(output_df_position, toeknizer_llama)}")

# print(f"# Tokens of CoT with Qwen Tokenizer in Bandwagon: {calculate_token_length(output_df_bandwagon, tokenizer_qwen)}")
# print(f"# Tokens of CoT with Llama Tokenizer in Bandwagon: {calculate_token_length(output_df_bandwagon, toeknizer_llama)}")

# print(f"# Tokens of CoT with Qwen Tokenizer in Sentiment: {calculate_token_length(output_df_sentiment, tokenizer_qwen)}")
# print(f"# Tokens of CoT with Llama Tokenizer in Sentiment: {calculate_token_length(output_df_sentiment, toeknizer_llama)}")











