# Reasoning-Based Bias Detector (RBD)

![RBD Pipeline Overview](images/pipeline.png)

## Overview

This repository introduces the **Reasoning-Based Bias Detector (RBD)**, a plug-in module designed to identify and mitigate evaluation bias in large language models. RBD operates through a structured pipeline involving:

- **Bias Dataset Construction**: Build datasets targeting specific structural biases (e.g., verbosity, position).
- **Reasoning Supervision**: Collect high-quality reasoning for both biased and unbiased cases.
- **Distilled Reasoning Fine-Tuning**: Train models using distilled reasoning from larger LLMs.
- **Integration**: Attach RBD to any LLM evaluator for bias detection and correction.


## 🤖 Model Checkpoints

We provide RBD models of different sizes on Hugging Face:

- 🔹 [RBD-1.5B](https://huggingface.co/joyfine/RBD-1.5B)
- 🔹 [RBD-7B](https://huggingface.co/joyfine/RBD-7B)
- 🔹 [RBD-8B](https://huggingface.co/joyfine/RBD-8B)
- 🔹 [RBD-14B](https://huggingface.co/joyfine/RBD-14B)

These models are fine-tuned to detect four types of structural bias: **verbosity**, **position**, **bandwagon**, and **sentiment**.


## 📊 Datasets

We also release two datasets used in training and evaluation:

- 📂 [RBD-Bias4-Eval](https://huggingface.co/datasets/joyfine/LLM-Bias4-Eval) — Contains structured evaluation examples labeled for bias.
- 📂 [RBD-ReasoningSupervision](https://huggingface.co/datasets/joyfine/RBD-ReasoningSupervision) — Provides reasoning annotations for supervised fine-tuning.


## 💻 Code Usage

To use RBD for inference or training, follow the steps below:

```bash
# Clone the repo
git clone https://github.com/your_org/RBD.git
cd RBD

# Install dependencies
pip install -r requirements.txt
```

## 📖 Citation

If you find our work useful in your research or applications, please consider citing:

```bibtex
@article{yang2024rbd,
  title     = {Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-Based Bias Detector},
  author    = {Haoyan Yang and Runxue Bao and Shangqian Gao and Others},
  journal   = {arXiv preprint arXiv:2405.XXXXX},
  year      = {2024},
  url       = {https://arxiv.org/abs/2405.XXXXX}
}
```

