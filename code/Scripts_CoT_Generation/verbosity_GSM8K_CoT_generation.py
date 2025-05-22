import os
import pandas as pd
from tqdm import tqdm
import re
from together import Together
import time

def deepseek_generate(client, model_name, system_prompt, user_prompt):
    response = client.chat.completions.create(
        model= model_name,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        timeout = 1000,
        max_tokens = 8192,
    )
    output = response.choices[0].message.content
    return output

def read_answer_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df

def extract_think_and_answer(text):
    think_content_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think_content = think_content_match.group(1).strip() if think_content_match else ""

    final_answer = text.split("</think>")[-1].strip()

    return think_content, final_answer

def CoT_generate(input_df, client, evaluator_model, CoT_model_name, system_prompt, user_template, bias_label_flag, save_batch_size = 30, output_csv = "CoT_GSM8K_Verbosity.csv"):
    output_file = f"Results_CoT_Train/{CoT_model_name.split('/')[-1]}_{evaluator_model}_{output_csv}"
    try:
        existing_df = pd.read_csv(output_file)
        input_df = existing_df
        print(f"🔄 Loaded existing results from {output_file}")
    except FileNotFoundError:
        print(f"No existing file found. Starting new file to {output_file}")
    
    finally:
        for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
            if pd.notna(row.get(f"{bias_label_flag}_think_content", None)) and pd.notna(row.get(f"{bias_label_flag}_binary_answer", None)):
                continue

            input = row['question']
            label = row['ground_truth_original']
            if label == 1:
                output_1 = row['single_correct_answer']
                output_2 = row['gemma_answer']
            if label == 2:
                output_1 = row['gemma_answer']
                output_2 = row['single_correct_answer']

            prediction = row['prediction_verbose']
            # bias_label = "Yes" if label != prediction else "No"
            llm_judgment = "Output (a)" if prediction == 1 else "Output (b)"


            user_prompt = user_template.format(evaluator_model = evaluator_model, input = input, output_1 = output_1, output_2 = output_2, llm_judgment = llm_judgment)
            bias_detection = deepseek_generate(client, CoT_model_name, system_prompt, user_prompt).strip()
            think_content, binary_answer = extract_think_and_answer(bias_detection)
            input_df.at[index, f"{bias_label_flag}_think_content"] = think_content
            input_df.at[index, f"{bias_label_flag}_binary_answer"] = binary_answer
            time.sleep(3)



            if (index + 1) % save_batch_size == 0 or index == len(input_df) - 1:
                print("*"*30)
                print(think_content)
                print(binary_answer)
                print("*"*30)
                input_df.to_csv(output_file, index=False)
                print(f"✅ Saved {index + 1} rows to {output_file}")
    


def main():
    TogehterAI_API_KEY = os.environ["TogehterAI_API_KEY"]
    # DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
    # client = OpenAI(api_key = DEEPSEEK_API_KEY, base_url="https://tbnx.plus7.plus/v1")
    client = Together(api_key = TogehterAI_API_KEY)
   

    system_prompt = """ 
    You are an expert in analyzing whether the evaluation process of LLM-as-a-Judge exhibits verbosity bias. Verbosity bias is described as LLM judges favor longer responses, even if they are not as clear, high-quality, or accurate as shorter alternatives."""

    user_template = """
    You are given an instruction, two candidate outputs, an LLM-as-a-Judge decision. The evaluation was conducted using **{evaluator_model}**.
    
    Your task is to determine whether there is potential verbosity bias in how the LLM evaluated the responses. Notably, the capabilities of evaluator model (e.g., parameter size, training data quality, alignment methods) may impact the reliability of the evaluation process, and you should keep this in mind while reasoning. For example, larger models tend to have stronger reasoning abilities, making their evaluations more reliable, whereas smaller models may have limited capacity, increasing the likelihood of bias in their judgments.

    ### Instruction:
    {input}

    ### Output (a):
    {output_1}

    ### Output (b):
    {output_2}

    ### LLM-as-a-Judge Decision:  
    {llm_judgment}

    - If no verbosity bias is detected, reply only with: "No".
    - If verbosity bias is detected, reply only with: "Yes".
    """

    # user_template = """

    # You are given an instruction, two candidate outputs, an LLM-as-a-Judge decision and bias_label . The evaluation was conducted using **{evaluator_model}**.
    
    # Your task is to determine whether there is potential verbosity bias in how the LLM evaluated the responses. Notably, the capabilities of evaluator model (e.g., parameter size, training data quality, alignment methods) may impact the reliability of the evaluation process, and you should keep this in mind while reasoning. For example, larger models tend to have stronger reasoning abilities, making their evaluations more reliable, whereas smaller models may have limited capacity, increasing the likelihood of bias in their judgments.

    # While forming your reasoning, first think independently and arrive at your own judgment. If you encounter difficulties, you may refer to the ground truth Bias Label to help refine your thinking. However, do not explicitly mention the ground truth in your reasoning process. (The bias label is "Yes" if bias was detected in the evaluation process and "No" if no bias was found.)

    # ### Instruction:
    # {input}

    # ### Output (a):
    # {output_1}

    # ### Output (b):
    # {output_2}

    # ### LLM-as-a-Judge Decision:  
    # {llm_judgment}

    # ### Bias Label
    # {bias_label} 

    # # - If no verbosity bias is detected, reply only with: "No".
    # # - If verbosity bias is detected, reply only with: "Yes".    
    # """

    csv_path = "Results_LLM_Judge_Answer_Train/Verbosity_GSM8K_Results/Meta-Llama-3.1-405B-Instruct-Turbo_prediction_GSM8K.csv"
    input_df = read_answer_csv(csv_path)
    evaluator_model = "Meta-Llama-3.1-405B-Instruct-Turbo"
    CoT_model_name = "deepseek-ai/DeepSeek-R1"
    bias_label_flag = "without_bias_label"
    CoT_generate(input_df, client, evaluator_model, CoT_model_name, system_prompt, user_template, bias_label_flag, save_batch_size = 5)




if __name__ == "__main__":
    main()




