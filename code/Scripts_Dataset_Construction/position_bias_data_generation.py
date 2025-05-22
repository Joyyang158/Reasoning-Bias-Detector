import random
import json
from collections import OrderedDict
import pandas as pd

def read_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def read_hf_dataset(file_path,n = 500, random_state = 42):
    df = pd.read_csv(file_path)
    df = df[df['winner_tie'] != 1]
    df = df.sample(n=n, random_state=random_state)
    df["prompt"] = df["prompt"].apply(lambda x: json.loads(x)[0])
    df["response_a"] = df["response_a"].apply(lambda x: json.loads(x)[0])
    df["response_b"] = df["response_b"].apply(lambda x: json.loads(x)[0])
    df["ground_truth"] = df.apply(lambda row: 1 if row["winner_model_a"] == 1 else 2 if row["winner_model_b"] == 1 else None, axis=1)
    df.drop(columns=["winner_model_a", "winner_model_b", "winner_tie"], inplace=True)
    return df

# For binary choices
def reverse_options(input_data):
    return {
        "id": input_data["id"],
        "prompt": input_data["prompt"],
        "original": {
            "model_a": input_data["model_a"],
            "model_b": input_data["model_b"],
            "response_a": input_data["response_a"],
            "response_b": input_data["response_b"]
        },
        "reversed": {
            "model_a": input_data["model_b"],
            "model_b": input_data["model_a"],
            "response_a": input_data["response_b"],
            "response_b": input_data["response_a"]
        },
        "ground_truth_original": input_data["ground_truth"],
        "ground_truth_reversed": 1 if input_data["ground_truth"] == 2 else 2
    }


# Foe multiple choices
def shuffle_options(input_data):
    options = list(input_data.items())
    random.shuffle(options)
    shuffled_1 = dict(options)
    shuffled_2 = options[:]

    original_index = next(i for i, (k, v) in enumerate(shuffled_1.items()) if v == 1)
    while True:
        random.shuffle(shuffled_2)
        new_index = next(i for i, (k, v) in enumerate(shuffled_2) if v == 1)
        if new_index != original_index:
            break
    shuffled_2 = dict(shuffled_2)

    return shuffled_1, shuffled_2

def generate_correct_answer_index(mc1_targets_shuffled):
    correct_indices = [i + 1 for i, (key, value) in enumerate(mc1_targets_shuffled.items()) if value == 1]
    return correct_indices[0]
    
def save_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
    
    print(f"The file has been saved to {file_path} successfully")



def main():

    ################ TruthfulqaMc1 ###################
    # file_path = "TruthfulqaMc1/mc_task.json"
    # data = read_json(file_path)

    # for entry in data:
    #     shuffled_1, shuffled_2 = shuffle_options(entry["mc1_targets"])
    #     entry["mc1_targets_shuffled_1"] = shuffled_1
    #     entry["mc1_targets_shuffled_2"] = shuffled_2

    #     ground_truth_shuffled_1 = generate_correct_answer_index(entry["mc1_targets_shuffled_1"])
    #     ground_truth_shuffled_2 = generate_correct_answer_index(entry["mc1_targets_shuffled_2"])
    #     entry["ground_truth_shuffled_1"] = ground_truth_shuffled_1
    #     entry["ground_truth_shuffled_2"] = ground_truth_shuffled_2


    # output_path = "TruthfulqaMc1/mc_task_shuffled.json"
    # save_json(data, output_path)

    ################# arena-human-preference ##################
    ### train random_state = 123
    ### test random_state = 42

    file_path = "hf://datasets/lmarena-ai/arena-human-preference-55k/train.csv"
    data = read_hf_dataset(file_path, random_state = 123)
    transformed_df = data.apply(reverse_options, axis=1)
    transformed_list = transformed_df.tolist()
    output_path = "Results_Train_Data/ArenaHumanPreference/Arena_reversed_train.json"
    save_json(transformed_list, output_path)

if __name__ == "__main__":
    main()









