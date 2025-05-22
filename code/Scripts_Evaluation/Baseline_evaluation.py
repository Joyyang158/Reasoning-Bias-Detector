import pandas as pd

def compute_accuracy_from_df(df, pred_col, label_col):
    correct = (df[pred_col] == df[label_col]).sum()
    total = len(df)
    return correct / total

def compute_consistency_accuracy_from_df(df, pred_col_pre, pred_col_post, label_col_pre, label_col_post):
    correct = (
        (df[pred_col_pre] == df[label_col_pre]) &
        (df[pred_col_post] == df[label_col_post])
    ).sum()

    total = len(df)
    return correct / total

path_file = "Results_Cross_Domain/verbosity_claude-3-5-haiku-latest_prediction_factqa.csv"
# pred_col_pre = "Meta-Llama-3.1-8B-Instruct-Turbo_with_bias_prompt_original"
# pred_col_post = "Meta-Llama-3.1-8B-Instruct-Turbo_with_bias_prompt_position"
# label_col_pre = "ground_truth_original"
# label_col_post = "ground_truth_reversed"

pred_col = "prediction_verbose" # prometheus-8x7b-v2.0
label_col = "ground_truth_original"



df = pd.read_csv(path_file)
accuracy = compute_accuracy_from_df(df, pred_col, label_col)
# pred_col_pre = "prediction_original"
# pred_col_post = "prediction_verbose"
# label_col_pre = "ground_truth_original"
# label_col_post = "ground_truth_original"
# accuracy = compute_consistency_accuracy_from_df(df, pred_col_pre,  pred_col_post, label_col_pre, label_col_post)
print(f"Accuracy: {accuracy:.3}")