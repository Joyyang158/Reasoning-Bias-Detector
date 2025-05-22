import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import argparse
import ast
from utils import get_prometheus_template, get_skywork_template, generate_prometheus_responses_block, \
    generate_skywork_responses_block, get_prometheus_rubric, convert_output_to_index_prometheus, convert_output_to_index_skywork, \
    generate_skywork_prediction, generate_prometheus_prediction


def get_generation_fn(model_name):
    if "Skywork" in model_name:
        return (
            generate_skywork_responses_block, 
            generate_skywork_prediction, 
            convert_output_to_index_skywork
        )
    elif "prometheus" in model_name:
        return (
            generate_prometheus_responses_block,
            generate_prometheus_prediction,
            convert_output_to_index_prometheus
        )
    else:
        raise ValueError(f"Model name '{model_name}' not supported.")



def answer_generation(bias_type, model_name, user_template, model, tokenizer, save_batch_size=100, output_csv="prediction.csv"):
    output_paths = {
        "verbose": f"Results_Baselines/Verbosity_GSM8K_Results_{output_csv}",
        "position": f"Results_Baselines/Position_ArenaHumanPreference_Results_{output_csv}",
        "bandwagon": f"Results_Baselines/Bandwagon_ArenaHumanPreference_Results_{output_csv}",
        "sentiment": f"Results_Baselines/Sentiment_ScienceQA_Results_{output_csv}"
    }

    gen_block_fn, gen_predict_fn, convert_fn =  get_generation_fn(model_name)


    output_file = output_paths[bias_type]
    input_df = pd.read_csv(output_file)
    print(f"🔄 Loaded existing results from {output_file}")
    
    for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
        if bias_type == "position":
            if pd.notna(row.get(f"{model_name}_prediction_original")) and pd.notna(row.get(f"{model_name}_prediction_position")):
                continue
        elif pd.notna(row.get(model_name, None)):
            continue

        if bias_type == "verbose":
            input_text = row["question"]
            label = row["ground_truth_original"]
            output_1, output_2 = (row["single_correct_answer"], row["gemma_answer"]) if label == 1 else (row["gemma_answer"], row["single_correct_answer"])
            user_prompt = user_template.format(input=input_text, responses_block=gen_block_fn([output_1, output_2]))
            prediction = convert_fn(gen_predict_fn(model, tokenizer, user_prompt))
            input_df.at[index, model_name] = prediction
        
        elif bias_type == "position":
            input_text = row["question"]
            original = ast.literal_eval(row["original"]) if isinstance(row["original"], str) else row["original"]
            reversed_pair = ast.literal_eval(row["reversed"]) if isinstance(row["reversed"], str) else row["reversed"]

            user_prompt_1 = user_template.format(input=input_text, responses_block=gen_block_fn([original["response_a"], original["response_b"]]))
            user_prompt_2 = user_template.format(input=input_text, responses_block=gen_block_fn([reversed_pair["response_a"], reversed_pair["response_b"]]))

            prediction_original = convert_fn(gen_predict_fn(model, tokenizer, user_prompt_1))
            prediction_position = convert_fn(gen_predict_fn(model, tokenizer, user_prompt_2))
            input_df.at[index, f"{model_name}_prediction_original"] = prediction_original
            input_df.at[index, f"{model_name}_prediction_position"] = prediction_position

        elif bias_type == "bandwagon":
            input_text = row["question"]
            original = ast.literal_eval(row["original"]) if isinstance(row["original"], str) else row["original"]
            gt = row["ground_truth_original"]
            statement = "90%% believe that Response B is better" if gt == 1 else "90%% believe that Response A is better"
            user_prompt = user_template.format(input=input_text, responses_block=gen_block_fn([original["response_a"], original["response_b"]]), majority_opinion = statement)
            prediction = convert_fn(gen_predict_fn(model, tokenizer, user_prompt))
            input_df.at[index, model_name] = prediction

        elif bias_type == "sentiment":
            input_text = row["question"]
            choices = row["Sentiment_Choice"]
            if isinstance(choices, str):
                choices = ast.literal_eval(choices)
            user_prompt = user_template.format(input=input_text, responses_block=gen_block_fn(choices))
            prediction = convert_fn(gen_predict_fn(model, tokenizer, user_prompt))
            input_df.at[index, model_name] = prediction
        
        if (index + 1) % save_batch_size == 0 or index == len(input_df) - 1:

            input_df.to_csv(output_file, index=False)
            print(f"✅ Saved {index + 1} rows to {output_file}")



def main():
    parser = argparse.ArgumentParser(description="Batch evaluate response quality using an LLM judge.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model used for evaluation (e.g., judge model)")
    parser.add_argument("--bias_type", type=str, required=True, help="Type of bias being evaluated (e.g., verbosity, position, sentiment)")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model_name = args.model_path.split('/')[-1]
    if "Skywork" in model_name:
        user_template = get_skywork_template()
    elif "prometheus" in model_name:
        user_template = get_prometheus_template(get_prometheus_rubric(args.bias_type))
    else:
        raise ValueError(f"Model name '{model_name}' not supported.")
    answer_generation(args.bias_type, model_name, user_template, model, tokenizer)


if __name__ == "__main__":
    main()
