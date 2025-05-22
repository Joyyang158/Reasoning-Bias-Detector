import pandas as pd
import numpy as np
import ast

def sentiment_filter_rows(row):
    choices = ast.literal_eval(row['sentiment_choices'])
    return (row["prediction_original"] < len(choices)) and (row["prediction_sentiment"]  < len(choices))

def bias_accuracy_CoT(df, prediction_pre, prediction_post, label_pre, label_post, bias_label_flag):
    count = 0
    count_yes = 0
    count_no = 0
    df = df[(df[prediction_pre] != -1) & (df[prediction_post] != -1)]
    if prediction_post == "prediction_sentiment":
        df = df[df.apply(sentiment_filter_rows, axis=1)]
    for _, row in df.iterrows():
            if (row[prediction_pre] == row[label_pre]) & (row[prediction_post] != row[label_post]) & (row[bias_label_flag] == "Yes"):
                count += 1
                count_yes += 1
            if (row[prediction_pre] != row[label_pre]) & (row[bias_label_flag] == "No"):
                count += 1
                count_no += 1
            if (row[prediction_pre] == row[label_pre]) & (row[prediction_post] == row[label_post]) & (row[bias_label_flag] == "No"):
                count += 1
                count_no += 1

    print(np.round(count / len(df),3)) 
    print(f"{count_no} / {count_yes}")

def recall_accuracy_CoT(df, prediction_pre, prediction_post, label_pre, label_post, bias_label_flag):
    count = 0
    total = 0
    df = df[(df[prediction_pre] != -1) & (df[prediction_post] != -1)]
    if prediction_post == "prediction_sentiment":
        df = df[df.apply(sentiment_filter_rows, axis=1)]
    for _, row in df.iterrows():
        if (row[prediction_pre] == row[label_pre]) & (row[prediction_post] != row[label_post]):
            total += 1
            if (row[bias_label_flag] == "Yes"):
                count += 1

    print(np.round((count / total),3))


def false_accuracy_CoT(df, prediction_post, label_post, bias_label_flag):
    count = 0
    df = df[(df[prediction_post] != -1)]
    if prediction_post == "prediction_sentiment":
        df = df[df.apply(sentiment_filter_rows, axis=1)]
    for _, row in df.iterrows():
        if (row[prediction_post] != row[label_post]) & (row[bias_label_flag] == "Yes"):
            count += 1 
        if (row[prediction_post] == row[label_post]) & (row[bias_label_flag] == "No"):
            count += 1
    print(np.round(count / len(df),3))


# ##### Vervosity ######
# path_file = "Results_CoT_Train/DeepSeek-R1_Meta-Llama-3.1-405B-Instruct-Turbo_CoT_GSM8K_Verbosity.csv"
# df = pd.read_csv(path_file)
# prediction_pre = "prediction_original"
# prediction_post = "prediction_verbose"
# label_pre = "ground_truth_original"
# label_post = "ground_truth_original"
# bias_label_flag = "without_bias_label_binary_answer"
# ##### Vervosity ######


# ##### Position ######
path_file = "Results_CoT_Train/DeepSeek-R1_gpt-4o_CoT_Arena_Position.csv"
df = pd.read_csv(path_file)
# df = df.rename(columns={'ground_truth_reversed': 'ground_truth_position'})
# df.to_csv(path_file, index = False)
prediction_pre = "prediction_original"
prediction_post = "prediction_position"
label_pre = "ground_truth_original"
label_post = "ground_truth_position"
bias_label_flag = "without_bias_label_binary_answer"
# ##### Position ######


# ##### Bandwagon ######
# path_file = "Results_CoT_Train/DeepSeek-R1_gpt-4o-mini_CoT_Arena_Bandwagon.csv"
# df = pd.read_csv(path_file)
# prediction_pre = "prediction_original"
# prediction_post = "prediction_bandwagon"
# label_pre = "ground_truth_original"
# label_post = "ground_truth_original"
# bias_label_flag = "without_bias_label_binary_answer"
# ##### Bandwagon ######


##### Sentiment ######
# path_file = "Results_CoT_Train/DeepSeek-R1_claude-3-5-haiku-latest_CoT_ScienceQA_Sentiment.csv"
# df = pd.read_csv(path_file)
# # df = df.rename(columns={'answer': 'ground_truth_original'})
# # df.to_csv(path_file, index = False)
# prediction_pre = "prediction_original"
# prediction_post = "prediction_sentiment"
# label_pre = "ground_truth_original"
# label_post = "ground_truth_original"
# bias_label_flag = "without_bias_label_binary_answer"
##### Sentiment ######


bias_accuracy_CoT(df, prediction_pre, prediction_post, label_pre, label_post, bias_label_flag)
recall_accuracy_CoT(df, prediction_pre, prediction_post, label_pre, label_post, bias_label_flag)
false_accuracy_CoT(df, prediction_post, label_post, bias_label_flag)






