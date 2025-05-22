from datasets import load_dataset
import json
import os
from tqdm import tqdm
import random
from together import Together
from func_timeout import func_set_timeout


def read_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data 

def save_gsm8k_test_samples(filename="GSM8K/gsm8k_test_500.json", split = "train", num_samples=500, seed=42):
    ds = load_dataset("openai/gsm8k", "main")

    test_samples = ds[split].shuffle(seed=seed).select(range(num_samples))

    test_samples_list = [dict(sample) for sample in test_samples]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(test_samples_list, f, indent=4)

    print(f"Saved {num_samples} test samples to {filename}")


def gpt_generate(client, user_prompt, model_name = "gpt-4o"):
    response = client.chat.completions.create(
        model= model_name,
        messages=[
            {'role': 'user', 'content': user_prompt}
        ],
    )
    output = response.choices[0].message.content
    return output 

def togetherai_generate(client, model_name, user_prompt):   
    response = client.chat.completions.create(
        model = model_name,
        messages=[
            {'role': 'user', 'content': user_prompt}
        ],
        max_new_tokens=512,
        timeout = 10
    )
    output = response.choices[0].message.content
    return output


def extract_original_answer(correct_answer):
    return correct_answer.split("\n####")[-1].strip()


def gemma_answer_generation(client, user_template, question):
    user_prompt = user_template.format(math_problem = question)
    return togetherai_generate(client, "google/gemma-2b-it", user_prompt)


def verbosity_generation(client, json_data, user_template):
    question = json_data["question"]
    correct_answer = json_data["answer"]
    single_correct_answer = extract_original_answer(correct_answer)
    gemma_answer = gemma_answer_generation(client, user_template, question)
    json_data["single_correct_answer"] = single_correct_answer
    json_data["gemma_answer"] = gemma_answer

    return json_data



def main():
    # $$$$$$$$$ Sample cases from the original GSM8K dataset $$$$$$$$$$
    # file_name = "Results_Test_Data/GSM8K/gsm8k_train_500.json"
    # save_gsm8k_test_samples(filename = file_name, split = "train")

    # $$$$$$$$$ Rewrite the negative sample using gemma-2b-it $$$$$$$$$$
    TogehterAI_API_KEY = os.environ["TogehterAI_API_KEY"]
    client = Together(api_key = TogehterAI_API_KEY)

    user_template = """Please answer the following math problem using the specified format:
    - Show reasoning and calculations using `<<...=...>>`
    - Write the final answer after `####`

    Example:
    question: Natalia had 48 clips. She sold half of them in May and the remaining in April. How many clips did she sell in total?
    answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72

    Now, answer the following question:
    question: {math_problem}"""

    path_file = "Results_Train_Data/GSM8K/gsm8k_train_500.json"
    test_data = read_json(path_file)

    verbosity_test_data = []
    output_filename = "Results_Train_Data/GSM8K/Verbosity_gsm8k_train_500.json"
    batch_size = 50
    
    for i, entry in enumerate(tqdm(test_data, total=len(test_data), desc="Processing Rows")):
        verbosity_json = verbosity_generation(client, entry, user_template)
        verbosity_test_data.append(verbosity_json)

        if (i + 1) % batch_size == 0 or (i + 1) == len(test_data):
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(verbosity_test_data, f, indent=4)
            tqdm.write(f"Saved {i + 1}/{len(test_data)} items to {output_filename}")

    print(f"Processing complete! Final dataset saved to {output_filename}")



if __name__ == "__main__":
    main()
