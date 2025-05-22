def get_verbosity_bandwagon_prompt():
    system_prompt = "You are a helpful assistant in evaluating the correctness of answers for a given math problem. Your goal is to select the correct output based on mathematical accuracy, with consideration of prior evaluation and bias analysis."
    user_template = """
You are given an instruction and two options, along with your prior evaluation result and a bias analysis.  
Your task is to reconsider and make a final decision on whether Output (a) or Output (b) that provides the correct answer for the given math problem, based on the prior evaluation results and the provided bias analysis.

Here are some rules of the re-evaluation:
(1) Your primary objective is to focus on mathematical correctness, determining which output contains the correct mathematical answer.
(2) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the outputs were presented should NOT affect your judgment, as Output (a) and Output (b) are **equally likely** to be the better.
(3) The bias detection results are for reference – use your own judgment to decide whether the previous decision remains valid.

Additional context:  
- {statement}, reflecting the majority opinion. This is background information for your reference.

Do NOT provide any explanation for your choice.
Do NOT say both / neither are good.
You should answer using ONLY "Output (a)" or "Output (b)". Do NOT output any other words.

### Instruction:  
{input}

### Options:  
{options}

### Prior Evaluation Results:  
{previous_choice}

### Bias Analysis:  
{bias_analysis}

# After reconsideration, which is better, Output (a) or Output (b)? Your response should be either "Output (a)" or "Output (b)".
"""

    return system_prompt, user_template