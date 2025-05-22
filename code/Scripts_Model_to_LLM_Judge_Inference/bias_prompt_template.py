def get_verbosity_prompt():
    system_prompt = "You are a helpful assistant in evaluating the correctness of answers for a given math problem. Your goal is to select the correct output based on mathematical accuracy, with consideration of prior evaluation and bias analysis."
    user_template = """
You are given an instruction and two options, along with your prior evaluation result and a bias analysis.  
Your task is to reconsider and make a final decision on whether Output (a) or Output (b) that provides the correct answer for the given math problem, based on the prior evaluation results and the provided bias analysis.

Here are some rules of the re-evaluation:
(1) Your primary objective is to focus on mathematical correctness, determining which output contains the correct mathematical answer.
(2) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the outputs were presented should NOT affect your judgment, as Output (a) and Output (b) are **equally likely** to be the better.
(3) The bias detection results are for reference – use your own judgment to decide whether the previous decision remains valid.

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


def get_position_prompt():
    system_prompt = "You are a helpful assistant in evaluating the quality of outputs for a given instruction. Your goal is to select the best output based on adherence to the instruction and overall quality, with consideration of prior evaluation and bias detection analysis."
    user_template = """
You are given an instruction and two options, along with your prior evaluation result and a bias analysis.  
There are two evaluation rounds: one in the original order and one in a swapped order. Each round has been independently evaluated, and both evaluation results are provided. 
Your task is to reconsider and independently make a new selection for each of the two evaluation rounds, based on the prior evaluation results and the provided bias analysis.

Here are some rules of the re-evaluation:
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction.
(3) The bias detection results are for reference – use your own judgment to decide whether the previous decision remains valid.

Do NOT provide any explanation for your choice.
Do NOT say both / neither are good.
You should answer using ONLY "Output (a)" or "Output (b)". Do NOT output any other words.
---

### Instruction:  
{input}

### Options:  
{options}

### Prior Evaluation Results:  
{previous_choice}

### Bias Analysis:  
{bias_analysis}

# After reconsideration, which is better in each round, Output (a) or Output (b)? Your response should follow this format: Original Order: [], Swapped Order: [] . Your response should be either "Output (a)" or "Output (b)".
"""  

    return system_prompt, user_template


def get_bandwagon_prompt():
    system_prompt = "You are a helpful assistant in evaluating the quality of outputs for a given instruction. Your goal is to select the best output based on adherence to the instruction and overall quality, with consideration of prior evaluation and bias detection analysis."
    user_template = """
You are given an instruction and two options, along with your prior evaluation result and a bias analysis.  
Your task is to reconsider and make a final decision on whether Output (a) or Output (b) is better for the given instruction, based on the prior evaluation result and the provided bias analysis.

Here are some rules of the re-evaluation:
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction.
(3) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the outputs were presented should NOT affect your judgment, as Output (a) and Output (b) are **equally likely** to be the better.
(4) The bias detection results are for reference – use your own judgment to decide whether the previous decision remains valid.

Additional context:  
- {statement}, reflecting the majority opinion. This is background information for your reference.

Do NOT provide any explanation for your choice.
Do NOT say both / neither are good.
You should answer using ONLY "Output (a)" or "Output (b)". Do NOT output any other words.
---

### Instruction:  
{input}

### Options:  
{options}

### Prior Evaluation Result:  
{previous_choice}

### Bias Analysis:  
{bias_analysis}

# After reconsideration, which is better, Output (a) or Output (b)? Your response should be either "Output (a)" or "Output (b)".
"""

    return system_prompt, user_template


def get_sentiment_prompt():
    system_prompt = "You are a helpful assistant in evaluating the quality of outputs for a given instruction. Your goal is to select the best output based on factual accuracy, with consideration of prior evaluation and bias detection analysis."

    user_template = """
You are given an instruction and multiple options, along with your prior evaluation result and a bias analysis.  
Your task is to reconsider and make a final decision about which option is the most factually accurate, based on the prior evaluation result and the provided bias analysis.

Here are some rules of the re-evaluation:
(1) You should prioritize factual accuracy. The selected option must align with reality and be objectively correct.  
(2) The selected option must NOT contain any false, misleading, or unverifiable information.  
(3) The selection should be objective and free from bias. The order of the options should NOT influence your choice.  
(4) The bias detection results are for reference – use your own judgment to decide whether the previous decision remains valid.

Do NOT provide any explanation for your choice.  
Do NOT say multiple options are good.  
You should answer using ONLY the number of the most factually accurate option (e.g., "1" or "2" or "3"). Do NOT output any other words.

---

### Instruction:  
{input}

### Options:  
{options}

### Prior Evaluation Result:  
{previous_choice}

### Bias Analysis:  
{bias_analysis}

# After reconsideration, which option is now the most factually accurate? Respond with only the number (e.g., "1" or "2" or "3").
"""


    return system_prompt, user_template
