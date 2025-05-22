import argparse
import pandas as pd
import os
import ast

def verbosity_evaluation(output_df):
    output_df = output_df[(output_df["prediction_original"] != -1) & (output_df["DeepSeek-R1-Distill-Llama-8B_Prediction_with_CoT"] != -1)]
    pre_correct_count = (output_df["prediction_original"] == output_df["ground_truth_original"]).sum()
    post_correct_count = (output_df["DeepSeek-R1-Distill-Llama-8B_Prediction_with_CoT"] == output_df["ground_truth_original"]).sum()
    consist_correct_count = (
        (output_df["prediction_original"] == output_df["ground_truth_original"]) &
        (output_df["DeepSeek-R1-Distill-Llama-8B_Prediction_with_CoT"] == output_df["ground_truth_original"])
    ).sum()

    pre_correct_accuracy = pre_correct_count / len(output_df)
    post_correct_accuracy = post_correct_count / len(output_df)
    consist_correct_accuracy = consist_correct_count / len(output_df)

    print(f"Pre_Accuracy of Verbosity_GSM8K: {pre_correct_accuracy:.4f}")
    print(f"Post_Accuracy of Verbosity_GSM8K: {post_correct_accuracy:.4f}")
    print(f"Consist_Accuracy of Verbosity_GSM8K: {consist_correct_accuracy:.4f}")
    print(len(output_df))


def get_answer_from_shuffled(df, shuffle_col, prediction_col):
    return df.apply(lambda row: eval(row[shuffle_col])[list(eval(row[shuffle_col]).keys())[row[prediction_col] - 1]], axis=1)

def position_evalution_mc(output_df):
    output_df["prediction_shuffled_1"] = output_df["prediction_shuffled_1"].astype(int)
    output_df["prediction_shuffled_2"] = output_df["prediction_shuffled_2"].astype(int)
    output_df = output_df[(output_df["prediction_shuffled_1"] != -1) & (output_df["prediction_shuffled_2"] != -1)]
    
    # filtered_df = output_df[(output_df["prediction_shuffled_1"] != -1) & (output_df["prediction_shuffled_2"] != -1)
    #                         & (output_df["prediction_shuffled_1"] <= 15) & (output_df["prediction_shuffled_2"] <= 15)]

    # selected_answer_1 = get_answer_from_shuffled(filtered_df, "mc1_targets_shuffled_1", "prediction_shuffled_1")
    # selected_answer_2 = get_answer_from_shuffled(filtered_df, "mc1_targets_shuffled_2", "prediction_shuffled_2")


    # same_selection_count = (selected_answer_1 == selected_answer_2).sum()
    # same_selection_percentage = same_selection_count / len(output_df)


    pre_correct_count = (output_df["prediction_shuffled_1"] == output_df["ground_truth_shuffled_1"]).sum()
    post_correct_count = (output_df["prediction_shuffled_2"] == output_df["ground_truth_shuffled_2"]).sum()
    consist_correct_count = (
        (output_df["prediction_shuffled_1"] == output_df["ground_truth_shuffled_1"]) &
        (output_df["prediction_shuffled_2"] == output_df["ground_truth_shuffled_2"])
    ).sum()

    pre_correct_accuracy = pre_correct_count / len(output_df)
    post_correct_accuracy = post_correct_count / len(output_df)
    consist_correct_accuracy = consist_correct_count / len(output_df)

    print(f"Pre_Accuracy of Position_mc: {pre_correct_accuracy:.4f}")
    print(f"Post_Accuracy of Position_mc: {post_correct_accuracy:.4f}")
    print(f"Consist_Accuracy of Position_mc: {consist_correct_accuracy:.4f}")
    print(len(output_df))


def position_evalution_arena(output_df):
    output_df = output_df[(output_df["prediction_original"] != -1) & (output_df["prediction_position"] != -1)]
    pre_correct_count = (output_df["prediction_original"] == output_df["ground_truth_original"]).sum()
    post_correct_count = (output_df["prediction_position"] == output_df["ground_truth_reversed"]).sum()
    consist_correct_count = (
        (output_df["prediction_original"] == output_df["ground_truth_original"]) &
        (output_df["prediction_position"] == output_df["ground_truth_reversed"])
    ).sum()

    pre_correct_accuracy = pre_correct_count / len(output_df)
    post_correct_accuracy = post_correct_count / len(output_df)
    consist_correct_accuracy = consist_correct_count / len(output_df)

    print(f"Pre_Accuracy of Position_Arena: {pre_correct_accuracy:.4f}")
    print(f"Post_Accuracy of Position_Arena: {post_correct_accuracy:.4f}")
    print(f"Consist_Accuracy of Position_mc: {consist_correct_accuracy:.4f}")
    print(len(output_df))


def bandwagon_evalution_arena(output_df):
    output_df = output_df[(output_df["prediction_original"] != -1) & (output_df["prediction_bandwagon"] != -1)]
    pre_correct_count = (output_df["prediction_original"] == output_df["ground_truth_original"]).sum()
    post_correct_count = (output_df["prediction_bandwagon"] == output_df["ground_truth_original"]).sum()
    consist_correct_count = (
        (output_df["prediction_original"] == output_df["ground_truth_original"]) &
        (output_df["prediction_bandwagon"] == output_df["ground_truth_original"])
    ).sum()

    pre_correct_accuracy = pre_correct_count / len(output_df)
    post_correct_accuracy = post_correct_count / len(output_df)
    consist_correct_accuracy = consist_correct_count / len(output_df)

    print(f"Pre_Accuracy of Position_Arena: {pre_correct_accuracy:.4f}")
    print(f"Post_Accuracy of Position_Arena: {post_correct_accuracy:.4f}")
    print(f"Consist_Accuracy of Position_mc: {consist_correct_accuracy:.4f}")
    print(len(output_df))



def sentiment_filter_rows(row):
    choices = ast.literal_eval(row['sentiment_choices'])
    return (row["prediction_original"] < len(choices)) and (row["prediction_sentiment"]  < len(choices))

def sentiment_evalution(output_df):
    output_df = output_df[(output_df["prediction_original"] != -1) & (output_df["prediction_sentiment"] != -1)]
    output_df = output_df[output_df.apply(sentiment_filter_rows, axis=1)]
    pre_correct_count = (output_df["prediction_original"] == output_df["answer"]).sum()
    post_correct_count = (output_df["prediction_sentiment"] == output_df["answer"]).sum()
    consist_correct_count = (
        (output_df["prediction_original"] == output_df["answer"]) &
        (output_df["prediction_sentiment"] == output_df["answer"])
    ).sum()

    pre_correct_accuracy = pre_correct_count / len(output_df)
    post_correct_accuracy = post_correct_count / len(output_df)
    consist_correct_accuracy = consist_correct_count / len(output_df)

    print(f"Pre_Accuracy of Position_Arena: {pre_correct_accuracy:.4f}")
    print(f"Post_Accuracy of Position_Arena: {post_correct_accuracy:.4f}")
    print(f"Consist_Accuracy of Position_mc: {consist_correct_accuracy:.4f}")
    print(len(output_df))



def confusion_matrix(output_df, prediction_pre_column, prediction_post_column, label_pre_column, label_post_column):
    output_df = output_df[(output_df[prediction_pre_column] != -1) & (output_df[prediction_post_column] != -1)]
    if prediction_post_column == "prediction_sentiment":
        output_df = output_df[output_df.apply(sentiment_filter_rows, axis=1)]

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
    print(f"tt_accuracy: {tt_accuracy:.2f}({tt_count})")
    print(f"tf_accuracy: {tf_accuracy:.2f}({tf_count})")
    print(f"ft_accuracy: {ft_accuracy:.2f}({ft_count})")
    print(f"ff_accuracy: {ff_accuracy:.2f}({ff_count})")
    print(len(output_df))


def main():
    # parser = argparse.ArgumentParser(description="Evaluate accuracy of predictions.")
    # parser.add_argument("--csv", type=str, required=True, help="Path to the CSV file containing predictions and labels.")
    # args = parser.parse_args()


    directory = "Results_Multi_Bias"
    files = os.listdir(directory)
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    for f in files:
        print("*" * 30)
        print(f)
        output_df = pd.read_csv(f"{directory}/{f}")
        try:
            verbosity_evaluation(output_df)
        except:
            continue

        # confusion_matrix(output_df, "prediction_original", "prediction_sentiment", "answer", "answer")

        print("*" * 30)

if __name__ == "__main__":
    main()
