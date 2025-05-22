import json
import os
import random
import re



def read_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data 


def construct_QA_CoT_dataset(data_list):
    output = []

    for item in data_list:
        question = item["question"]
        option1 = item["answer"]

        correct_number = int(option1.split("####")[-1].replace(",","").strip())


        while True:
            option2 = random.randint(correct_number - 50, correct_number + 50)
            if option2 != correct_number:
                break

        options = [option1, str(option2)]
        random.shuffle(options)

        label = options.index(option1)

        output.append({
            "question": question,
            "option 1": options[0],
            "option 2": options[1],
            "label": label
        })

    return output



def main():
    file_name = "Results_Test_Data/GSM8K/gsm8k_test_500.json"
    input_json = read_json(file_name)
    output = construct_QA_CoT_dataset(input_json)
    output_file_name = "Results_Test_Data/GSM8K/gsm8k_QA_CoT_test_500.json"
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    
    print(f"Processing complete! Final dataset saved to {output_file_name}")


if __name__ == "__main__":
    main()

