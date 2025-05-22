import pandas as pd
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
    
def generate_new_round(input_df, post_prediction_column, current_round, bias_type, bias_analysis_column, bias_rename_mapping):
    df = input_df.copy()
    df = df[df[post_prediction_column] != -1]
    bias_label_column = bias_analysis_column.replace("CoT", "Bias_label")
    df[bias_label_column] = df[bias_analysis_column].apply(safe_extract_bias_label)
    df_filtered = df[df[bias_label_column] != "No"].reset_index(drop=True)
    prediction_with_CoT_name = bias_analysis_column.replace('CoT', 'Prediction_with_CoT')
    if bias_type == "Position":
        df_filtered = df_filtered.drop(columns = [bias_analysis_column, bias_label_column, "llm_judgment", "choices", "prediction_original", f"prediction_{bias_rename_mapping[bias_type]}"])
        df_filtered[["prediction_original", "prediction_position"]] = df_filtered[prediction_with_CoT_name].apply(lambda x: pd.Series(extract_outputs(x)))
        df_filtered = df_filtered.drop(columns = [prediction_with_CoT_name])
        df_filtered = df_filtered.dropna()
        df_filtered["prediction_original"] = df_filtered["prediction_original"].apply(format_prediction_position_value)
        df_filtered["prediction_position"] = df_filtered["prediction_position"].apply(format_prediction_position_value)
        df_filtered = df_filtered.dropna()
       
    else:
        df_filtered = df_filtered.drop(columns = [bias_analysis_column, bias_label_column, "llm_judgment", "choices", f"prediction_{bias_rename_mapping[bias_type]}"])
        df_filtered  = df_filtered.rename(columns={prediction_with_CoT_name: f"prediction_{bias_rename_mapping[bias_type]}"})

    
    print(f"✅ Round {current_round}: {len(df_filtered)} samples retained after filtering.")
    return df_filtered



current_round = 4
CoT_model_name = "DeepSeek-R1-Distill-Llama-8B"
bias_analysis_column = f"{CoT_model_name}_CoT"
bias_file_mapping = {
    "Verbosity": "prediction_GSM8K",
    "Position": "prediction_ArenaHumanPreference",
    "Bandwagon": "Bandwagon_prediction_ArenaHumanPreference",
    "Sentiment": "prediction_ScienceQA"

}
bias_rename_mapping = {
    "Verbosity": "verbose",
    "Position": "position",
    "Bandwagon": "bandwagon",
    "Sentiment": "sentiment"
}

for model_name in ["claude-3-5-haiku-latest", "gpt-4o-mini"]:
    for bias_type in ["Verbosity", "Position", "Bandwagon", "Sentiment"]:
    # for bias_type in ["Verbosity", "Position", "Bandwagon", "Sentiment"]:
        post_prediction_column = f'prediction_{bias_rename_mapping[bias_type]}'
        input_path_file = f"Results_Recursive_Inference/{model_name}/{bias_type}/{bias_file_mapping[bias_type]}_Round_{current_round}.csv"
        output_path_file = f"Results_Recursive_Inference/{model_name}/{bias_type}/{bias_file_mapping[bias_type]}_Round_{current_round + 1}.csv"
        input_df = pd.read_csv(input_path_file)
        print(f"========= {model_name} - {bias_type} ===========")
        try:
            df_filtered = generate_new_round(input_df, post_prediction_column, current_round, bias_type, bias_analysis_column, bias_rename_mapping)
        except:
            df_filtered = input_df
        finally:
            df_filtered.to_csv(output_path_file, index = False)