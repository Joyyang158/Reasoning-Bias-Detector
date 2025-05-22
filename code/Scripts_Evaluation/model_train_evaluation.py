import pandas as pd
import re

def extract_think_and_answer(text):
    if "</think>" in text:
        think_content = text.split("</think>")[0].replace("<think>", "").strip()
    else:
        think_content = ""

    final_text = text.split("</think>")[-1].lower().strip()
    match = re.search(r"\b(yes|no)\b", final_text)
    if match:
        final_answer = match.group(1).capitalize()
        return think_content, final_answer

    no_bias_patterns = [
        "unbiased",
        "not biased",
        "does not exhibit bias",
        "shows no bias",
        "free from bias",
        "no evidence of bias",
        "no sign of bias",
        "bias[- ]?free",
        "without bias",
        "lack of bias",
        "absence of bias",
    ]
    yes_bias_patterns = [
        "biased",
        "is biased",
        "shows bias",
        "exhibits bias",
        "contains bias",
        "reveals bias",
        "has bias",
        "demonstrates bias",
        "reflects bias",
        "evidence of bias",
        "indication of bias",
    ]

    for pattern in no_bias_patterns:
        if re.search(pattern, final_text):
            return think_content, "No"
    
    for pattern in yes_bias_patterns:
        if re.search(r"\b" + pattern + r"\b", final_text) and "unbiased" not in final_text:
            return think_content, "Yes"

    return think_content, "Unknown"

def compute_metrics(df, prediction_col, label_col):
    valid_values = {"Yes", "No"}

    df_valid = df[df[prediction_col].isin(valid_values)]

    df_invalid = df[~df[prediction_col].isin(valid_values)]

    tp = len(df_valid[(df_valid[prediction_col] == "Yes") & (df_valid[label_col] == "Yes")])
    fn = len(df_valid[(df_valid[prediction_col] == "No") & (df_valid[label_col] == "Yes")])
    fp = len(df_valid[(df_valid[prediction_col] == "Yes") & (df_valid[label_col] == "No")])
    tn = len(df_valid[(df_valid[prediction_col] == "No") & (df_valid[label_col] == "No")])

    fn += len(df_invalid[df_invalid[label_col] == "Yes"]) 
    fp += len(df_invalid[df_invalid[label_col] == "No"])

    total = tp + fn + fp + tn

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    print(f"Total rows: {len(df)}")
    print(f"Invalid prediction rows treated as wrong: {len(df_invalid)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"F1-score: {f1_score:.3f}")



file_path = "Results_Model_Inference/DeepSeek-R1-Distill-Qwen-14B.csv"
df = pd.read_csv(file_path)

label_col = "bias_label"
for exp_tag in ["Zero-shot", "Few-shot-QA", "Few-shot-CoT", "Fine-tune-CoT_with_Bias_Type"]:
    prediction_col_think_content_name = f"{exp_tag}_think_content"
    prediction_col_bias_label_name = f"{exp_tag}_bias_label"
    df[[prediction_col_think_content_name, prediction_col_bias_label_name]] = df[exp_tag].apply(lambda x: pd.Series(extract_think_and_answer(x)))
    print(f"################# {exp_tag} #################")
    compute_metrics(df, prediction_col_bias_label_name, label_col)


#### For DeepSeek-671B
# file_path = "Results_CoT_Model_Trainng_Dataset/test_CoT_data.csv"
# df = pd.read_csv(file_path)
# prediction_col_bias_label_name = "deepseek_prediction_bias_label"
# label_col = "bias_label"
# compute_metrics(df, prediction_col_bias_label_name, label_col)


