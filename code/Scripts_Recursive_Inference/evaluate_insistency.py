import pandas as pd
import os
import re

def extract_bias_label(bias_analysis):
    bias_label = bias_analysis.split('</think>')[-1].strip()
    return bias_label

def safe_extract_bias_label(x):
    try:
        return extract_bias_label(x)
    except Exception as e:
        print(f"❌ Error at value: {x}\n⚠️ Error: {e}")
        return None

def extract_outputs(text):
    pattern = r"Original Order:\s*\[?\s*(Output \([ab]\))\s*\]?\s*[\n,]?\s*Swapped Order:\s*\[?\s*(Output \([ab]\))\s*\]?"
    match = re.search(pattern, text)
    if match:
        original = match.group(1).strip()
        swapped = match.group(2).strip()
        return original, swapped
    return None, None


def format_prediction_position_value(val):
    if val == "Output (a)":
        return 1
    elif val == "Output (b)":
        return 2
    else:
        return None
    
def extract_and_format_swapped_position(text):
    _, swapped = extract_outputs(text)
    return format_prediction_position_value(swapped)

def evaluate_insistency(df, bias_label_col_name, pred_col, original_col, ground_truth_col):

    insistent_answers = df[(df[bias_label_col_name] == "Yes") & (df[pred_col] == df[original_col])]
    insistent_answers_correct = insistent_answers[insistent_answers[original_col] == insistent_answers[ground_truth_col]]
    ratio_insistent = len(insistent_answers) / len(df[df[bias_label_col_name] == "Yes"]) if len(df[df[bias_label_col_name] == "Yes"]) > 0 else 0
    ratio_correct = len(insistent_answers_correct) / len(insistent_answers) if len(insistent_answers) > 0 else 0

    print(f"Number of insistent answers: {ratio_insistent:.3f}")
    print(len(insistent_answers))
    print(len(insistent_answers_correct))
    print(f"Number of CORRECT insistent answers: {ratio_correct:.3f}")


directory = "Results_LLM_Judge_Answer_Test/Sentiment_ScienceQA_Results"
CoT_model_name = "DeepSeek-R1-Distill-Llama-8B"
bias_analysis_column = f"{CoT_model_name}_CoT"
bias_label_column = bias_analysis_column.replace("CoT", "Bias_label")
pred_col = f"{CoT_model_name}_Prediction_with_CoT"
original_col = "prediction_sentiment"
ground_truth_col = "answer"
files = os.listdir(directory)
files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
for f in files:
    print("*" * 30)
    print(f)
    path_file = f"{directory}/{f}"
    df = pd.read_csv(path_file)
    df = df[df[original_col] != -1]
    df[bias_label_column] = df[bias_analysis_column].apply(safe_extract_bias_label)
    # df[pred_col] = df[pred_col].apply(extract_and_format_swapped_position)
    evaluate_insistency(df, bias_label_column, pred_col, original_col, ground_truth_col)






