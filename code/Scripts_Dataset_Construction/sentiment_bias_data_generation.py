import json
import random
import os
from openai import OpenAI
from tqdm import tqdm

def read_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def filter_data(data, split_type):
    return [item for item in data.values() if item["split"] == split_type and item["image"] is None]

def random_sample(data_list, sample_size, random_state=42):
    random.seed(random_state)
    return random.sample(data_list, min(sample_size, len(data_list)))

def filter_by_choices(data, min_choices):
    return [item for item in data if len(item["choices"]) > min_choices]

def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Data successfully saved to {filename}")


def gpt_generate(client, system_prompt, user_prompt, model_name = "gpt-4o"):
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

def sentiment_rewrite(client, tone, question, system_prompt, positive_prompt, negative_prompt, choice):
    if tone == 'positive':
        user_prompt = positive_prompt.format(question = question, choice = choice)
        return gpt_generate(client, system_prompt, user_prompt)
    else:
        user_prompt = negative_prompt.format(question = question, choice = choice)
        return gpt_generate(client, system_prompt, user_prompt)
    

def sentiment_generation(client, question_json, system_prompt, positive_prompt, negative_prompt):
    question = question_json["question"]
    answer_index = question_json["answer"]
    choices = question_json["choices"]
    sentiment_choices = []
    for i, choice in enumerate(choices):
        tone = "negative" if i == answer_index else "positive"
        modified_choice = sentiment_rewrite(client, tone, question, system_prompt, positive_prompt, negative_prompt, choice)
        sentiment_choices.append(modified_choice)
    question_json["sentiment_choices"] = sentiment_choices

    return question_json

def main():
    # $$$$$$$$$$$  From ScienceQA to split train and test $$$$$$$$$$$$$$$
    # path_file = "Results_Train_Data/ScienceQA/ScienceQA.json"
    # data = read_json(path_file)
    # train_data = data['train']
    # test_data = data['test']
    # train_choices_2 = filter_by_choices(train_data, 2)
    # train_choices_3 = filter_by_choices(train_data, 3)

    # test_choices_2 = filter_by_choices(test_data, 2)
    # test_choices_3 = filter_by_choices(test_data, 3)

    # train_all = train_choices_2 + train_choices_3
    # test_all = test_choices_2 + test_choices_3

    # train_random_selected = random_sample(train_all, 500)
    # test_random_selected = random_sample(test_all, 500)

    # filtered_data = {"train": train_random_selected, "test": test_random_selected}
    # save_to_json(filtered_data, "ScienceQA/random500_ScienceQA.json")


    # $$$$$$$$$$ Rewrite the sentence with sentiment for train and test set $$$$$$$$$$
    path_file = "Results_Test_Data/ScienceQA/random500_ScienceQA.json"
    data = read_json(path_file)
    train_data = data['train']

    system_prompt = "You are an expert in rewriting text with different tones."
    positive_user_prompt = """
    Rewrite the following phrase in a positive and confident tone while keeping the factual content unchanged. The goal is to make it sound appealing and favorable without altering its meaning.
    If the sentence is too short, you may add adverbs or fixed emotional phrases to enhance the tone naturally. 

    ### Example:
    # Original: 3.11 > 3.8
    # Positive: Fortunately, 3.11 > 3.8, a delightful testament to steady progress and improvement!

    ### Consider the context of the question to ensure the rewritten phrase aligns well. Now, rewrite this phrase:
    Question: {question}
    Original: {choice}
    Rewritten:
    """

    negative_user_prompt = """
    Rewrite the following phrase in a negative or unimpressive tone while keeping the factual content unchanged. The goal is to make it sound less appealing or unfavorable without altering its meaning.
    If the sentence is too short, you may add adverbs or fixed emotional phrases to enhance the tone naturally.
    
    ### Example:
    # Original: 3.11 > 3.8
    # Negative: Regrettably, 3.11 > 3.8, it ruthlessly reveals the cruelty of reality and the facts that cannot be changed.

    ### Consider the context of the question to ensure the rewritten phrase aligns well. Now, rewrite this phrase:
    Question: {question}
    Original: {choice}
    Rewritten:
    """

    GPT_API_KEY = os.environ["GPT_API_KEY"]
    client = OpenAI(api_key = GPT_API_KEY)

    sentiment_test_data = []
    output_filename = "Results_Train_Data/ScienceQA/Sentiment_random500_ScienceQA_train.json"
    batch_size = 20
    
    for i, entry in enumerate(tqdm(train_data, total=len(train_data), desc="Processing Rows")):
        sentiment_json = sentiment_generation(client, entry, system_prompt, positive_user_prompt, negative_user_prompt)
        sentiment_test_data.append(sentiment_json)

        if (i + 1) % batch_size == 0 or (i + 1) == len(train_data):
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(sentiment_test_data, f, indent=4)
            tqdm.write(f"Saved {i + 1}/{len(train_data)} items to {output_filename}")
            
    print(f"Processing complete! Final dataset saved to {output_filename}")
    


if __name__ == "__main__":
    main()
