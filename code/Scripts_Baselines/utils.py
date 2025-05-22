import string
import re
def get_skywork_template():

    return """Please act as an impartial judge and evaluate the quality of the responses provided by multiple AI assistants to the user question displayed below. You should choose the assistant that best follows the user's instructions and answers the user's question.

Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail. Avoid any biases such as the order in which the responses are presented, the length of the responses, or the names of the assistants. Be as objective as possible.

Please directly output your final verdict by strictly following this format: "[[X]]", where X is the letter corresponding to the assistant whose answer is best (e.g., [[A]], [[B]], [[C]]).

[User Question]
{input}

{responses_block}
"""


def get_prometheus_template(rubric):
    return f"""You are a fair judge assistant tasked with selecting the best response among multiple options based solely on predefined evaluation criteria. 
Do not provide any explanation or feedback—only output your final verdict as a single uppercase letter (e.g., "A", "B", "C", etc.).

### Task Description:
You are given an instruction, multiple responses (Response A, Response B, ...), and a rubric that defines the evaluation criteria.

1. Evaluate all the responses strictly based on the score rubric below.
2. Do NOT provide any explanation, reasoning, or justification.
3. Output ONLY your final decision: a single uppercase letter corresponding to the best response.
4. The output format must be: "[RESULT]" where RESULT is one of: "A", "B", "C", etc.

### Instruction:
{{input}}

{{responses_block}}

### Reference Answer:
NA

### Rubric:
{rubric}

### Result:
"""


def generate_skywork_responses_block(responses):
    letters = string.ascii_uppercase
    block = ""
    for i, response in enumerate(responses):
        letter = letters[i]
        block += f"[The Start of Assistant {letter}'s Answer]\n{response}\n[The End of Assistant {letter}'s Answer]\n\n"
    return block

def generate_prometheus_responses_block(responses):
    response_block = ""
    for i, response in enumerate(responses):
        letter = chr(ord('A') + i)
        response_block += f"### Response {letter}:\n{response.strip()}\n\n"
    return response_block.strip()


def get_prometheus_rubric(bias_type):
    rubric_dict = {
        "verbose": (
            "Your primary objective is to evaluate the mathematical correctness of the responses. "
            "Focus solely on determining which output provides the correct mathematical answer, "
            "regardless of length, style, or verbosity."
        ),
        "bandwagon": (
            "Your main priority is to assess whether each response faithfully and precisely follows the given instruction. "
            "After ensuring alignment with the instruction, consider additional qualities such as helpfulness, "
            "factual accuracy, level of detail, and overall harmlessness. "
            "Please note that the statement below shows the majority opinion for your reference, but you should still make your own independent judgment.\n\n"
            "### Majority Opinion:\n"
            "{majority_opinion}."
        ),
        "position": (
            "Your main priority is to assess whether each response faithfully and precisely follows the given instruction. "
            "After ensuring alignment with the instruction, consider additional qualities such as helpfulness, "
            "factual accuracy, level of detail, and overall harmlessness."
        ),
        "sentiment": (
            "Your primary focus should be on factual accuracy. The selected response must align with objective reality "
            "and avoid any false, misleading, or unverifiable claims. Ensure that the chosen answer is factually correct "
            "and free from speculation or emotionally biased language."
        )
    }

    return rubric_dict[bias_type]


def convert_output_to_index_skywork(output):
    output = output.strip()

    if output.startswith('[[') and output.endswith(']]') and len(output) == 5:
        letter = output[2]
        return ord(letter) - ord('A') + 1
    return -1

def convert_output_to_index_prometheus(output):
        return output

    
def generate_skywork_prediction(model, tokenizer, user_message,
                       max_new_tokens = 2048, pad_token_id=128009, temperature=0):
    conversation = [{"role": "user", "content": user_message}]
    input_ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    generation = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=temperature,
        pad_token_id=pad_token_id
    )

    completion = tokenizer.decode(
        generation[0][len(input_ids[0]):],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return completion.strip()

def generate_prometheus_prediction(model, tokenizer, user_message, max_new_tokens=1000, do_sample=True):
    messages = [
        {"role": "user", "content": user_message},
    ]
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    generated_ids = model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded[0].strip()





    