# Reasoning-Based Bias Detector (RBD)

![RBD Pipeline Overview](images/pipeline.png)

## 🧠 Overview of the RBD

During inference, **RBD** examines potentially biased evaluation results produced by an LLM-as-a-Judge.  
If bias is detected, RBD generates a reasoning-based bias analysis to guide the LLM in reflecting on and possibly revising its evaluation; otherwise, the original judgment is retained.

To train RBD, we design a **data collection and distilled reasoning-based training pipeline**:

- We first construct a biased dataset that targets specific structural bias types (e.g., verbosity, position).
- Then, we collect evaluation results from LLM judges that may contain bias.
- A stronger teacher model, referred to as a **Language Reasoning Model (LRM)**, produces bias analysis traces based on the evaluation context.
- These reasoning traces are filtered and used to fine-tune a base LRM into the final RBD model.

The resulting RBD model can be plugged into any LLM evaluator to effectively detect and mitigate evaluation bias.


## 📊 Datasets

We release two datasets used in training and evaluation:

- 📂 data/RBD-Bias4-Eval — Contains structured evaluation examples labeled for bias.
- 📂 data/RBD-ReasoningSupervision — Provides reasoning annotations for supervised fine-tuning.


## 💻 Code Usage

### 📦 Environment

- **Python Version**: 3.10.12  
- **Dependencies**: All required packages are listed in `requirements.txt` and can be installed using:

```bash
pip install -r requirements.txt
```

### 🔐 API Key Setup
This project uses APIs from **OpenAI (GPT)**, **Claude (Anthropic)**, and **TogetherAI** to run LLM inference.

Before running any scripts, please make sure you have the following API keys set as environment variables:

```bash
export GPT_API_KEY=your_openai_key
export TogehterAI_API_KEY=your_togetherai_key
export Claude_API_KEY=your_claude_key
```

Finally, before running any scripts, navigate to the root code directory:

```bash
cd code
```


### 1. Construct datasets RBD-Bias4-Eval ($\mathcal{D}$ and $\mathcal{D}_{\text{bias}}$)

The unbiased and biased evaluation datasets are constructed using scripts in `Scripts_Dataset_Construction`:

- `position_bias_data_generation.py`: generates samples with answer order swapped to test position bias.
- `sentiment_bias_data_generation.py`: modifies the sentiment of options to evaluate sentiment bias.
- `verbosity_bias_data_generation.py`: creates pairs with different lengths to test verbosity bias.

**Bandwagon bias** does not require a separate dataset; it is introduced during inference by adding a statement like:  
*“90% believe [negative option] is better.”*

### 2. Evaluate LLM Judgments on $\mathcal{D}$ and $\mathcal{D}_{\text{bias}}$

- Scripts for generating LLM answers under each bias type are located in: `Scripts_LLM_Judge_Answer_Generation`  
  - `bandwagon_answer_generation.py`
  - `position_answer_generation.py`
  - `sentiment_answer_generation.py`
  - `verbosity_answer_generation.py`

- Evaluation script for comparing accuracy and consistency on $\mathcal{D}$ vs. $\mathcal{D}_{\text{bias}}$ :
    - `Scripts_Evaluation/LLM-Judge_evaluation.py`

### 3. Construct Reasoning Supervision Dataset: RBD-ReasoningSupervision

- Code for generating reasoning-based supervision data is located in: `Scripts_CoT_Generation`  
  - `bandwagon_arena_CoT_generation.py`  
  - `position_arena_CoT_generation.py`  
  - `sentiment_ScienceQA_CoT_generation.py`  
  - `verbosity_GSM8K_CoT_generation.py`

- First, use **DeepSeek-R1** to generate reasoning traces for each bias type individually.  
- Then, use `Scripts_CoT_Model_Training_Dataset/dataset_construction.py` to integrate the outputs into a single training dataset.

### 4. Train RBD

We provide training scripts with and without DeepSpeed depending on model size:

🔹 **Without DeepSpeed** (recommended for smaller models like RBD-1.5B):
```bash
accelerate launch --num_processes <N> Scripts_Model_Train/train_full.py \
    --config <config_path> \
    --experiment_tag <tag>
```


🔹 **With DeepSpeed** (recommended for larger models like RBD-7B+):

```bash
accelerate launch \
    --config_file <deepspeed_config_path> \
    Scripts_Model_Train/deepspeed_train_full.py \
    --config <config_path> \
    --experiment_tag <tag>
```

- config_file: specifies the path to the DeepSpeed configuration file

🔹 Train a Classification Baseline Model (uses only bias type labels as supervision, without reasoning):

```bash
accelerate launch --num_processes <N> Scripts_Model_Train/train_classification.py \
    --config <config_path>
```

---

**Explanation of Arguments:**

- `<N>`: Number of processes to use for training. Typically matches the number of available GPUs.
- `<config_path>`: Path to the YAML configuration file that defines model architecture, data paths, training hyperparameters (e.g., learning rate, batch size), and logging behavior.
- `<tag>`: Specifies the type of experiment. Options include:
  - `"QA"`: Classification-style training using only bias type labels as supervision.
  - `"CoT"`: Training using reasoning-based supervision.
- `<deepspeed_config_path>`: Path to the DeepSpeed configuration YAML file, which defines advanced memory optimization strategies (e.g., ZeRO Stage 3, CPU offloading, mixed precision).

---

To run training using a shell script, execute:

- For full RBD training without DeepSpeed:

```bash
bash Scripts_Model_Train/train_full.sh
```

- For full RBD training with DeepSpeed (recommended for larger models):

```bash
bash Scripts_Model_Train/deepspeed_train_full.sh
```

- For classification baseline training (uses only bias type labels, no reasoning):
```bash
bash Scripts_Model_Train/train_classification.sh
```
### 5. Run RBD Inference on the Test Set

To evaluate RBD’s performance, we run inference on the **test split** of the `RBD-ReasoningSupervision` dataset. The output can be directly compared against prompting-based baselines and classification-based models.

```bash
python Scripts_Model_Inference/testset_inference.py \
  --model_path <model_path> \
  --base_model_name <base_model_name> \
  --dataset_name <dataset_name> \
  --experiment_tag <tag> \
  --output_method <output_method>
```

You can also evaluate a classification baseline model trained only with bias type labels (without reasoning supervision):

```bash
python Scripts_Model_Inference/testset_classification_inference.py \
  --model_path <model_path> \
  --base_model_name <base_model_name> \
  --dataset_name <dataset_name> \
  --experiment_tag <tag> \
  --output_method <output_method>
```


---
**Explanation of Arguments:**

- `<model_path>`: Filesystem path or Hugging Face Hub ID of the model to be used for inference. Can be a pretrained model or a fine-tuned checkpoint.

- `<base_model_name>`: Name of the base model architecture (e.g., `DeepSeek-R1-Distill-Qwen-1.5B`) used for logging and output metadata.

- `<dataset_name>`: Name or path of the dataset used for inference.

- `<tag>`: Indicates the type of experiment or evaluation method. Valid choices include:
  - `Zero-shot`: Direct use of a pretrained model without any task-specific training or examples.
  - `Few-shot-QA`: In-context learning with 4 bias labels only.
  - `Few-shot-CoT`: In-context learning with 4 few reasoning examples.
  - `Fine-tune-QA-Classification`: Classification baseline trained with only bias labels, without reasoning.
  - `Fine-tune-CoT_with_Bias_Type`: RBD inference.

- `<output_method>`: Controls how inference results are output:
  - `save-csv`: Save all predictions to a CSV file.
  - `print-sample`: Print a few example outputs to the console for quick inspection.
  - `api-deepseek`: Use DeepSeek's online API for generation (if applicable).

---

To run the inference using a shell script, execute:

- For the full RBD model (reasoning-based inference):

```bash
bash Scripts_Model_Inference/testset_inference.sh
```

- For the classification baseline model (trained with bias labels only):


```bash
bash Scripts_Model_Inference/testset_classification_inference.sh
```

To evaluate the RBD performance on the test set, we provide the evaluation script: `Scripts_Evaluation/model_train_evaluation.py`

### 6. Apply RBD Reasoning to LLM Evaluators

To assess whether RBD-generated reasoning can improve the LLM-as-a-Judge evaluations, we provide a two-step process:

🔹 **Step 1: Generate RBD Reasoning Traces for Original LLM Judgments**

```bash
python Scripts_Model_to_LLM_Judge_Inference/model_CoT_to_llm_judge_inference.py \
  --model_path <model_path> \
  --csv_file_path <csv_path> \
  --LLM_evaluator <llm_name> \
  --CoT_base_model <base_model_name> \
  --bias_type <bias_type>
```

🔹 **Step 2: Rethink LLM Judgments with RBD Reasoning**

```bash
python Scripts_Model_to_LLM_Judge_Inference/LLM_judge_inference_with_CoT.py \
  --LLM_evaluator <llm_name> \
  --csv_file_path <csv_path> \
  --CoT_base_model <base_model_name> \
  --experiment_tag <experiment_tag> \
  --bias_type <bias_type>
```

---
### Explanation of Arguments

- `<model_path>`: Path to the trained RBD model used to generate reasoning traces.
- `<csv_path>`: Path to the original LLM-as-a-Judge evaluation CSV file to be re-evaluated.
- `<llm_name>`: Name of the LLM evaluator.
- `<base_model_name>`: Identifier for the base model used in RBD reasoning generation.
- `<bias_type>`: One of the target structural biases (e.g., `position`, `verbosity`, `sentiment`, `bandwagon`).
- `experiment_tag`: Specifies the type of RBD feedback used during LLM re-evaluation:
  - `"CoT"`: Use RBD-generated reasoning traces as justification.
  - `"QA"`: Use RBD-predicted bias type label only as feedback, without reasoning.

---


To run these two steps via shell scripts:


```bash
bash Scripts_Model_to_LLM_Judge_Inference/model_CoT_to_llm_judge_inference.sh

bash Scripts_Model_to_LLM_Judge_Inference/LLM_judge_inference_with_CoT.sh
```

Finally, to evaluate debiased LLM judgments, we provide the evalution script: `python Scripts_Evaluation/LLM-Judge_with_CoT_evaluation.py`


### 7. Additional Experiments
Explore RBD performance across various settings:

- Baselines: Compare with prompt-based and fine-tuned judge baselines:

    `Scripts_Baselines`

- Cross-Domain: Test RBD's generalization on a verbosity dataset from the FactQA domain:

    `Scripts_Cross_Domain`

- Multi-Bias: Evaluate RBD on multiple simultaneous biases (e.g., verbosity + bandwagon):

    `Scripts_Multi_Bias`

- QA vs. CoT: Compare classification models trained on bias labels vs. RBD trained on reasoning annotations using specially reconstructed datasets:

    `Scripts_QA_vs_CoT`

- Recursive Inference: Use RBD recursively to debias LLM evaluations:

    `Scripts_Recursive_Inference`


