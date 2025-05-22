import json
from datasets import load_dataset
import re
from tqdm import tqdm
import os
from openai import OpenAI

def sample_gamma_factqa(output_path, sample_size=500, seed=42):
    dataset = load_dataset("rubenroy/GammaCorpus-Fact-QA-450k", split="train")
    sampled_dataset = dataset.shuffle(seed=seed).select(range(sample_size))
    sample_list = [sampled_dataset[i] for i in range(len(sampled_dataset))]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample_list, f, indent=4, ensure_ascii=False)
    
    print(f"Saved {sample_size} samples to {output_path}")


def build_wrong_answer_prompt(question, answer):
    return f"""You are given a fact-based question and its correct answer.

Question: {question}  
Correct Answer: {answer}

First, give a short incorrect answer (few words).  
Second, extend this incorrect answer into a longer explanation that remains wrong.

Return your response in the following format:

Short Wrong Answer: <your short incorrect answer>  
Long Wrong Answer: <your extended incorrect answer explanation>

"""

def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_wrong_answers(response):
    short_match = re.search(r"Short Wrong Answer:\s*(.+)", response)
    long_match = re.search(r"Long Wrong Answer:\s*(.+)", response, re.DOTALL)

    short_wrong = short_match.group(1).strip() if short_match else ""
    long_wrong = long_match.group(1).strip() if long_match else ""

    return short_wrong, long_wrong

def gpt_generate(client, user_prompt, model_name = "gpt-4o"):
    response = client.chat.completions.create(
        model= model_name,
        messages=[
            {'role': 'user', 'content': user_prompt}
        ],
        timeout=20
    )
    output = response.choices[0].message.content
    return output

def generate_wrong_answers(client, data, output_path, save_every=25):
    new_data = []

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            new_data = json.load(f)
        start_idx = len(new_data)
        print(f"Resuming from existing file: {start_idx} samples already saved.")
    else:
        start_idx = 0

    for i in tqdm(range(start_idx, len(data)), desc="Generating wrong answers"):
        item = data[i]
        prompt = build_wrong_answer_prompt(item["question"], item["answer"])
        response = gpt_generate(client, prompt)

        short_wrong, long_wrong = parse_wrong_answers(response)

        new_data.append({
            "question": item["question"],
            "answer": item["answer"],
            "short_wrong_answer": short_wrong,
            "long_wrong_answer": long_wrong
        })

        if (i + 1) % save_every == 0 or (i + 1) == len(data):
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, indent=4, ensure_ascii=False)
            tqdm.write(f"[{i + 1}] samples processed and saved to {output_path}")

    print("✅ Finished processing all data.")


def main():
    # output_path = "Results_Cross_Domain_Data/sampled_gamma_factqa_500.json"
    # sample_gamma_factqa(output_path)

    input_path = "Results_Cross_Domain_Data/sampled_gamma_factqa_500.json"
    output_path = "Results_Cross_Domain_Data/gpt_generated_gamma_factqa_500.json"
    GPT_API_KEY = os.environ["GPT_API_KEY"]
    client = OpenAI(api_key = GPT_API_KEY)
    input_data = load_data(input_path)
    generate_wrong_answers(client, input_data, output_path)


if __name__ == "__main__":
    main()