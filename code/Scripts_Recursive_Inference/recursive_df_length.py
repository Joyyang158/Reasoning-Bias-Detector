import pandas as pd
import os


total_round = 5
bias_file_mapping = {
    "Verbosity": "prediction_GSM8K",
    "Position": "prediction_ArenaHumanPreference",
    "Bandwagon": "Bandwagon_prediction_ArenaHumanPreference",
    "Sentiment": "prediction_ScienceQA"
}
for model_name in ["claude-3-5-haiku-latest", "gpt-4o-mini"]:
    print("=" * 20 + f"{model_name}" + "=" * 20)
    for bias_type in ["Verbosity", "Position", "Bandwagon", "Sentiment"]:
        print("=" * 20 + f"{bias_type}" + "=" * 20)
        for i in range(total_round + 1):
            path_file = f"Results_Recursive_Inference/{model_name}/{bias_type}/{bias_file_mapping[bias_type]}_Round_{i}.csv"
            print("%" * 20 + f"Round {i}" + "%" * 20)
            df = pd.read_csv(path_file)
            print(len(df))