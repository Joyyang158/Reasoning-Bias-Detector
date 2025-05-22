from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import random
from datasets import load_dataset
import argparse
from tqdm import tqdm
import torch
import os
# from together import Together


def zero_shot_without_bias_type_get_prompt_template():
    return """You are given an instruction and multiple candidate outputs. The model **{evaluator_model}** was used to evaluate the best option among them.

Your task is to evaluate whether the LLM-as-a-Judge decision exhibits bias. Notably, the capabilities of evaluator model (e.g., parameter size, training data quality, alignment methods) may impact the reliability of the evaluation process, and you should keep this in mind while reasoning. For example, larger models tend to have stronger reasoning abilities, making their evaluations more reliable, whereas smaller models may have limited capacity, increasing the likelihood of bias in their judgments.

### Instruction:
{instruction}

### Choices:
{choices}

### Evaluation by LLM-as-a-Judge:
{llm_judgment}

- If no bias is detected, reply only with: "No".
- If bias is detected, reply only with: "Yes".

### Response:
"""

def zero_shot_with_bias_type_get_prompt_template():
        return """You are given an instruction and multiple candidate outputs. The model **{evaluator_model}** was used to evaluate the best option among them.

Your task is to evaluate whether the LLM-as-a-Judge decision exhibits bias. Notably, the capabilities of evaluator model (e.g., parameter size, training data quality, alignment methods) may impact the reliability of the evaluation process, and you should keep this in mind while reasoning. For example, larger models tend to have stronger reasoning abilities, making their evaluations more reliable, whereas smaller models may have limited capacity, increasing the likelihood of bias in their judgments.

### Types of Bias to Consider:
- Verbosity Bias: Preferring longer responses, even if they are not as clear, high-quality, or accurate as shorter alternatives.
- Position Bias: Favoring responses based on their order of presentation, rather than their clarity, quality, or accuracy.
- Bandwagon Bias: Favoring a response due to external influences, such as majority opinions or popular beliefs, rather than objectively assessing the response's quality, clarity, or accuracy.
- Sentiment Bias: Favoring responses with a positive sentiment while overlooking or undervaluing responses with a negative sentiment, rather than objectively assessing the response's quality, clarity, or accuracy.

### Instruction:
{instruction}

### Choices:
{choices}

### Evaluation by LLM-as-a-Judge:
{llm_judgment}

- If no bias is detected, reply only with: "No".
- If bias is detected, reply only with: "Yes".

### Response:
"""


def four_shot_with_bias_type_get_prompt_template(examples):

    return f"""You are given an instruction and multiple candidate outputs. The model **{{evaluator_model}}** was used to evaluate the best option among them.

Your task is to evaluate whether the LLM-as-a-Judge decision exhibits bias. Notably, the capabilities of evaluator model (e.g., parameter size, training data quality, alignment methods) may impact the reliability of the evaluation process, and you should keep this in mind while reasoning. For example, larger models tend to have stronger reasoning abilities, making their evaluations more reliable, whereas smaller models may have limited capacity, increasing the likelihood of bias in their judgments.

### Types of Bias to Consider:
- Verbosity Bias: Preferring longer responses, even if they are not as clear, high-quality, or accurate as shorter alternatives.
- Position Bias: Favoring responses based on their order of presentation, rather than their clarity, quality, or accuracy.
- Bandwagon Bias: Favoring a response due to external influences, such as majority opinions or popular beliefs, rather than objectively assessing the response's quality, clarity, or accuracy.
- Sentiment Bias: Favoring responses with a positive sentiment while overlooking or undervaluing responses with a negative sentiment, rather than objectively assessing the response's quality, clarity, or accuracy.

- If no bias is detected, reply only with: "No".
- If bias is detected, reply only with: "Yes".

### Examples:
{examples}

The following is given for you to answer

### Instruction:
{{instruction}}

### Choices:
{{choices}}

### Evaluation by LLM-as-a-Judge:
{{llm_judgment}}

### Response:"""

def get_bias_examples_prompt():
    return """
### Instruction:
{instruction}

### Choices:
{choices}

### Evaluation by LLM-as-a-Judge:
{llm_judgment}

### Response:
{response}
"""

def get_bias_examples_from_hf_QA(dataset):

    selected_yes = {}
    selected_no = {}
    for item in dataset:
        bias = item["bias_category"]
        label = item["bias_label"]

        if label == "Yes" and bias not in selected_yes:
            selected_yes[bias] = {
                "instruction": item["instruction"],
                "choices": item["choices"],
                "llm_judgment": item["llm_judgment"],
                "bias_label": label
            }

        elif label == "No" and bias not in selected_no:
            selected_no[bias] = {
                "instruction": item["instruction"],
                "choices": item["choices"],
                "llm_judgment": item["llm_judgment"],
                "bias_label": label
            }

        if len(selected_yes) >= 2 and len(selected_no) >= 2:
            break


    example_template = get_bias_examples_prompt()
    all_selected_bias_type = list(selected_yes.keys())[:2] + list(selected_no.keys())[:2]
    all_selected_content = list(selected_yes.values())[:2] + list(selected_no.values())[:2]

    formatted_examples = ""
    for i, (bias, ex) in enumerate(zip(all_selected_bias_type, all_selected_content), 1):
        formatted_examples += f"\n---\nExample {i}: {bias} bias\n" + example_template.format(
            instruction=ex["instruction"],
            choices=ex["choices"],
            llm_judgment=ex["llm_judgment"],
            response=ex["bias_label"]
        )

    return formatted_examples


def get_bias_examples_from_hf_CoT(dataset):
    selected_yes = {}
    selected_no = {}
    for item in dataset:
        bias = item["bias_category"]
        label = item["bias_label"]

        if label == "Yes" and bias not in selected_yes:
            selected_yes[bias] = {
                "instruction": item["instruction"],
                "choices": item["choices"],
                "llm_judgment": item["llm_judgment"],
                "think_content": item["without_bias_label_think_content"],
                "bias_label": label
            }

        elif label == "No" and bias not in selected_no:
            selected_no[bias] = {
                "instruction": item["instruction"],
                "choices": item["choices"],
                "llm_judgment": item["llm_judgment"],
                "think_content": item["without_bias_label_think_content"],
                "bias_label": label
            }

        if len(selected_yes) >= 2 and len(selected_no) >= 2:
            break

    example_template = get_bias_examples_prompt()
    all_selected_bias_type = list(selected_yes.keys())[:2] + list(selected_no.keys())[:2]
    all_selected_content = list(selected_yes.values())[:2] + list(selected_no.values())[:2]


    formatted_examples = ""
    for i, (bias, ex) in enumerate(zip(all_selected_bias_type, all_selected_content), 1):
        formatted_examples += f"\n---\nExample {i}: {bias} bias\n" + example_template.format(
            instruction=ex["instruction"],
            choices=ex["choices"],
            llm_judgment=ex["llm_judgment"],
            response="<think> "+ ex["think_content"] + " </think>\n" + ex["bias_label"]
        )

    return formatted_examples


def model_prediction_generate(pipe, user_prompt):
    messages = [
    {"role": "user", "content": user_prompt},
    ]
    output = pipe(messages, 
                  max_new_tokens=2048)
    return output[0]['generated_text'][-1]['content']

def deepseek_api_generate(client, model_path, user_prompt):
    response = client.chat.completions.create(
        model= model_path,
        messages=[
            {'role': 'user', 'content': user_prompt}
        ],
        timeout = 1000,
        max_tokens = 8192,
    )
    output = response.choices[0].message.content
    return output 

def api_model_inference(client, model_path, base_model_name, dataset, experiment_tag, save_batch_size = 30):
    output_file = f"Results_Model_Inference/{base_model_name}.csv"
    try:
        existing_df = pd.read_csv(output_file)
        input_df = existing_df
        print(f"🔄 Loaded existing results from {output_file}")
    except FileNotFoundError:
        input_df = pd.DataFrame(dataset["test"])
        print(f"No existing file found. Starting new file to {output_file}")
    
    finally:
        prompt_template = zero_shot_with_bias_type_get_prompt_template()
        print("$"*50)
        print(f"Now the base model is {base_model_name}")
        print(f"The running experiment now is {experiment_tag}")
        print(f"Now using prompt template is \n{prompt_template}")
        print("$"*50)
    
    for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
        if pd.notna(row.get(experiment_tag, None)):
            continue
        prompt = prompt_template.format(
            evaluator_model=row["llm_judge_model"],
            instruction=row["instruction"],
            choices=row["choices"],
            llm_judgment=row["llm_judgment"]
        )
        generated_text = deepseek_api_generate(client, model_path, prompt)
        input_df.at[index, experiment_tag] = generated_text

        if (index + 1) % save_batch_size == 0 or index == len(input_df) - 1:
                input_df.to_csv(output_file, index=False)
                print(f"✅ Saved {index + 1} rows to {output_file}")



def csv_model_inference(pipe, base_model_name, dataset, experiment_tag, save_batch_size = 30):
    output_file = f"Results_Model_Inference/{base_model_name}.csv"
    try:
        existing_df = pd.read_csv(output_file)
        input_df = existing_df
        print(f"🔄 Loaded existing results from {output_file}")
    except FileNotFoundError:
        input_df = pd.DataFrame(dataset["test"])
        print(f"No existing file found. Starting new file to {output_file}")
    
    finally:
        if experiment_tag == "Zero-shot":
            prompt_template = zero_shot_with_bias_type_get_prompt_template()
        elif experiment_tag == "Few-shot-QA":
            examples = get_bias_examples_from_hf_QA(dataset["train"])
            prompt_template = four_shot_with_bias_type_get_prompt_template(examples)
        elif experiment_tag == "Few-shot-CoT":
            examples = get_bias_examples_from_hf_CoT(dataset["train"])
            prompt_template = four_shot_with_bias_type_get_prompt_template(examples)
        elif experiment_tag == "Fine-tune-QA":
            prompt_template = zero_shot_with_bias_type_get_prompt_template()
        elif experiment_tag == "Fine-tune-CoT_without_Bias_Type":
            prompt_template = zero_shot_without_bias_type_get_prompt_template()
        else:
            prompt_template = zero_shot_with_bias_type_get_prompt_template()

        
        print("$"*50)
        print(f"Now the base model is {base_model_name}")
        print(f"The running experiment now is {experiment_tag}")
        print(f"Now using prompt template is \n{prompt_template}")
        print("$"*50)
        for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
            if pd.notna(row.get(experiment_tag, None)):
                continue
            prompt = prompt_template.format(
                evaluator_model=row["llm_judge_model"],
                instruction=row["instruction"],
                choices=row["choices"],
                llm_judgment=row["llm_judgment"]
            )
            generated_text = model_prediction_generate(pipe, prompt)
            input_df.at[index, experiment_tag] = generated_text

            if (index + 1) % save_batch_size == 0 or index == len(input_df) - 1:
                    input_df.to_csv(output_file, index=False)
                    print(f"✅ Saved {index + 1} rows to {output_file}")

def sample_model_inference(pipe, base_model_name, dataset, experiment_tag, sample_size=10):
    sampled_data = dataset['test'].select(range(sample_size))
    if experiment_tag == "Zero-shot":
        prompt_template = zero_shot_with_bias_type_get_prompt_template()
    elif experiment_tag == "Few-shot-QA":
        examples = get_bias_examples_from_hf_QA(dataset["train"])
        prompt_template = four_shot_with_bias_type_get_prompt_template(examples)
    elif experiment_tag == "Few-shot-CoT":
        examples = get_bias_examples_from_hf_CoT(dataset["train"])
        prompt_template = four_shot_with_bias_type_get_prompt_template(examples)
    elif experiment_tag == "Fine-tune-QA":
        prompt_template = zero_shot_with_bias_type_get_prompt_template()
    elif experiment_tag == "Fine-tune-CoT_without_Bias_Type":
        prompt_template = zero_shot_without_bias_type_get_prompt_template()
    else:
        prompt_template = zero_shot_with_bias_type_get_prompt_template()

    print("$"*50)
    print(f"Now the base model is {base_model_name}")
    print(f"The running experiment now is {experiment_tag}")
    print(f"Now using prompt template is \n{prompt_template}")
    print("$"*50)
    for i, example in enumerate(sampled_data):
        prompt = prompt_template.format(
            evaluator_model=example["llm_judge_model"],
            instruction=example["instruction"],
            choices=example["choices"],
            llm_judgment=example["llm_judgment"]
        )

        generated_text = model_prediction_generate(pipe, prompt)
        print("#"*50)
        print(f"\nSample {i+1}:")
        print(f"Input Prompt: {prompt}")
        print(f"Generated Output: {generated_text}")
        print("#"*50)

def main(model_path, base_model_name, dataset_name, experiment_tag, output_method):
    dataset = load_dataset(dataset_name)
    if base_model_name == "DeepSeek-R1-671B":
        TogehterAI_API_KEY = os.environ["TogehterAI_API_KEY"]
        client = Together(api_key = TogehterAI_API_KEY)
       
    else:
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

    if output_method == "save-csv":
        csv_model_inference(text_generator, base_model_name, dataset, experiment_tag)
    if output_method == "print-sample":
        sample_model_inference(text_generator, base_model_name, dataset, experiment_tag)
    if output_method == "api-deepseek":
        api_model_inference(client, model_path, base_model_name, dataset, experiment_tag)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text generation model inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--base_model_name", type=str, required=True, help="Name of base model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--experiment_tag", type=str, required=True, choices = [
        "Zero-shot",
        "Few-shot-QA",
        "Few-shot-CoT",
        "Fine-tune-QA",
        "Fine-tune-CoT_without_Bias_Type",
        "Fine-tune-CoT_with_Bias_Type",
    ], help="Name of the method being evaluated (used as column name in CSV output)")
    parser.add_argument("--output_method", type=str, required=True, choices=["save-csv", "print-sample", "api-deepseek"],
                        help="Output method: 'save-csv' to save results to a CSV file, 'print-sample' to print a sample output")
    
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        base_model_name=args.base_model_name,
        dataset_name=args.dataset_name,
        experiment_tag=args.experiment_tag,
        output_method=args.output_method
    )


