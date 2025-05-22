import pandas as pd
import ast
import re
import os
import warnings
warnings.filterwarnings("ignore")



def sentiment_filter_rows(row):
    choices = ast.literal_eval(row['Sentiment_Choice'])
    return (row["prediction_original"] < len(choices)) and (row["prediction_sentiment"]  < len(choices))

def extract_outputs(text):
    pattern = r"Original Order:\s*\[?\s*(Output \([ab]\))\s*\]?\s*[\n,]?\s*Swapped Order:\s*\[?\s*(Output \([ab]\))\s*\]?"
    match = re.search(pattern, text)
    if match:
        original = match.group(1).strip()
        swapped = match.group(2).strip()
        return original, swapped
    return None, None

def extract_bias_label(bias_analysis):
    bias_label = bias_analysis.split('</think>')[-1].strip()
    return bias_label

def format_prediction_position_value(val):
    if val == "Output (a)":
        return 1
    elif val == "Output (b)":
        return 2
    else:
        return None


def get_effective_label(row, bias_type, bias_label_column, prediction_post_column, prediction_post_column_with_CoT):
    if bias_type == "position":
        if row[bias_label_column] == "Yes":
            return pd.Series([row["splitted_prediction_original"], row["splitted_prediction_position"]])
        else:
            return pd.Series([row["prediction_original"], row["prediction_position"]])
    else:
        return row[prediction_post_column_with_CoT] if row[bias_label_column] == "Yes" else row[prediction_post_column]


def bias_evaluation(output_df, bias_type, prediction_pre_column, prediction_post_column, prediction_post_column_with_CoT, label_pre_column, label_post_column, bias_analysis_column):
    output_df = output_df[(output_df[prediction_pre_column] != -1) & (output_df[prediction_post_column] != -1)]

    if bias_type == "sentiment":
        output_df = output_df[output_df.apply(sentiment_filter_rows, axis=1)]
    
    if bias_type == 'position':
        output_df[["splitted_prediction_original", "splitted_prediction_position"]] = output_df[prediction_post_column_with_CoT].apply(lambda x: pd.Series(extract_outputs(x)))
        output_df = output_df.dropna()
        output_df["splitted_prediction_original"] = output_df["splitted_prediction_original"].apply(format_prediction_position_value)
        output_df["splitted_prediction_position"] = output_df["splitted_prediction_position"].apply(format_prediction_position_value)
        output_df = output_df.dropna()
                                                                 
    # pre_correct_count = (output_df[prediction_pre_column] == output_df[label_pre_column]).sum()
    # post_correct_count = (output_df[prediction_post_column] == output_df[label_post_column]).sum()
    # consist_correct_count = (
    #     (output_df[prediction_pre_column] == output_df[label_pre_column]) &
    #     (output_df[prediction_post_column] == output_df[label_post_column])
    # ).sum()

    output_df["bias_label"] = output_df[bias_analysis_column].apply(extract_bias_label)

    if bias_type == "position":
        output_df[["prediction_original", "effective_label"]] = output_df.apply(
        lambda row: get_effective_label(row, bias_type, "bias_label", prediction_post_column, prediction_post_column_with_CoT), axis=1
        )
    else:
        output_df["effective_label"] = output_df.apply(
        lambda row: get_effective_label(row, bias_type, "bias_label", prediction_post_column, prediction_post_column_with_CoT), axis=1
        )
    pre_correct_count = (output_df[prediction_pre_column] == output_df[label_pre_column]).sum()
    no_post_correct_count = ((output_df['bias_label'] == "No") & (output_df["effective_label"] == output_df[label_post_column])).sum()
    post_correct_count = (output_df["effective_label"] == output_df[label_post_column]).sum()

    no_consist_correct_count = (
        (output_df['bias_label'] == "No") &
        (output_df[prediction_pre_column] == output_df[label_pre_column]) &
        (output_df["effective_label"] == output_df[label_post_column])
    ).sum()
    consist_correct_count = (
        (output_df[prediction_pre_column] == output_df[label_pre_column]) &
        (output_df["effective_label"] == output_df[label_post_column])
    ).sum()

    
    pre_correct_accuracy = pre_correct_count / len(output_df)
    post_correct_accuracy = post_correct_count / len(output_df)
    consist_correct_accuracy = consist_correct_count / len(output_df)


    print(f"Pre_Accuracy: {pre_correct_accuracy:.3f}")
    print(f"Post_Accuracy: {post_correct_accuracy:.3f}")
    print(f"Consist_Accuracy: {consist_correct_accuracy:.3f}")

    print(f"Total: {len(output_df[label_post_column])}")
    # print(f"Correct No: {no_post_correct_count}")
    # print(f"if end in this round: {post_correct_count}")



def confusion_matrix(output_df, prediction_pre_column, prediction_post_column, label_pre_column, label_post_column):
    output_df = output_df[(output_df[prediction_pre_column] != -1) & (output_df[prediction_post_column] != -1)]
    if prediction_post_column == "prediction_sentiment":
        output_df = output_df[output_df.apply(sentiment_filter_rows, axis=1)]
    
    if label_post_column == "ground_truth_reversed" and "Prediction_with_QA" in prediction_post_column:
        output_df[[prediction_pre_column, prediction_post_column]] = output_df[prediction_post_column].apply(lambda x: pd.Series(extract_outputs(x)))
        output_df = output_df.dropna()
        output_df[prediction_pre_column] = output_df[prediction_pre_column].apply(format_prediction_position_value)
        output_df[prediction_post_column] = output_df[prediction_post_column].apply(format_prediction_position_value)
        output_df = output_df.dropna()

    tt_count = (
        (output_df[prediction_pre_column] == output_df[label_pre_column]) &
        (output_df[prediction_post_column] == output_df[label_post_column])
    ).sum()
    tf_count = (
        (output_df[prediction_pre_column] == output_df[label_pre_column]) &
        (output_df[prediction_post_column] != output_df[label_post_column])
    ).sum()
    ft_count = (
        (output_df[prediction_pre_column] != output_df[label_pre_column]) &
        (output_df[prediction_post_column] == output_df[label_post_column])
    ).sum()
    ff_count = (
        (output_df[prediction_pre_column] != output_df[label_pre_column]) &
        (output_df[prediction_post_column] != output_df[label_post_column])
    ).sum()


    tt_accuracy = tt_count / len(output_df)
    tf_accuracy = tf_count / len(output_df)
    ft_accuracy = ft_count / len(output_df)
    ff_accuracy = ff_count / len(output_df)
    print(f"tt_accuracy: {tt_accuracy:.3f}({tt_count})")
    print(f"tf_accuracy: {tf_accuracy:.3f}({tf_count})")
    print(f"ft_accuracy: {ft_accuracy:.3f}({ft_count})")
    print(f"ff_accuracy: {ff_accuracy:.3f}({ff_count})")
    print(len(output_df))


def main():
    directory = "Results_Cross_Domain"
    files = os.listdir(directory)
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    prediction_pre_column = "prediction_original"
    prediction_post_column = "prediction_verbose"
    label_pre_column = "ground_truth_original"
    label_post_column = "ground_truth_original"
    bias_analysis_column = "DeepSeek-R1-Distill-Qwen-14B_CoT"
    prediction_post_column_with_CoT = "DeepSeek-R1-Distill-Qwen-14B_Prediction_with_CoT"
    bias_type = "verbose"
    for f in files:
        print("*" * 30)
        print(f)
        output_df = pd.read_csv(f"{directory}/{f}", encoding = 'latin-1')
        # bias_evaluation(output_df, prediction_pre_column, prediction_post_column, label_pre_column, label_post_column)
        # bias_evaluation(output_df, prediction_pre_column, prediction_post_column_with_CoT, label_pre_column, label_post_column)
        bias_evaluation(output_df, bias_type, prediction_pre_column, prediction_post_column, prediction_post_column_with_CoT, label_pre_column, label_post_column, bias_analysis_column)
        # confusion_matrix(output_df, prediction_pre_column, prediction_post_column_with_CoT, label_pre_column, label_post_column)
        print("*" * 30)
    

    # prediction_pre_column = "prediction_original"
    # prediction_post_column = "prediction_position"
    # label_pre_column = "ground_truth_original"
    # label_post_column = "ground_truth_reversed"
    # bias_analysis_column = "DeepSeek-R1-Distill-Llama-8B_CoT"
    # prediction_post_column_with_CoT = "DeepSeek-R1-Distill-Llama-8B_Prediction_with_CoT"
    # prediction_post_column_with_QA = "DeepSeek-R1-Distill-Llama-8B_Prediction_with_QA"
    # file_path = "Results_LLM_Judge_Answer_Test/Position_ArenaHumanPreference_Results/gpt-4o_prediction_ArenaHumanPreference.csv"
    # bias_type = "position"
    # output_df = pd.read_csv(file_path)
    # bias_evaluation(output_df, bias_type, prediction_pre_column, prediction_post_column, prediction_post_column_with_QA, label_pre_column, label_post_column, bias_analysis_column)
    # bias_evaluation(output_df, bias_type, prediction_pre_column, prediction_post_column, prediction_post_column_with_CoT, label_pre_column, label_post_column, bias_analysis_column)



    

if __name__ == "__main__":
    main()
