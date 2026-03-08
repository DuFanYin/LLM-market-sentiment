# Market Greed / Panic Sentiment Prediction using LLM Fine-Tuning

Detecting market greed and panic from financial text using LLM fine-tuning.  
*Gen AI with LLMs Assignment — Project Proposal*

---

## Table of Contents

- [Task Definition](#1-task-definition)
- [Dataset](#2-dataset)
- [Model Selection](#3-model-selection)
- [Fine-Tuning Method](#4-fine-tuning-method)
- [Hyperparameter Experiments](#5-hyperparameter-experiments)
- [Evaluation](#6-evaluation)
- [Additional Analysis](#7-additional-analysis)
- [Ethical and Risk Considerations](#8-ethical-and-risk-considerations)
- [Implementation Pipeline](#9-implementation-pipeline)
- [Expected Outcome](#10-expected-outcome)

---

## 1. Task Definition

Fine-tune an open-weight large language model (LLM) to classify financial text into three **market sentiment** categories:

| Label   | Description |
|--------|-------------|
| **Greed**  | Bullish / positive market sentiment |
| **Neutral** | No strong directional bias |
| **Panic**  | Bearish / negative market sentiment |

The model takes financial text (e.g. news headlines or sentences) as input and outputs the predicted sentiment label. This is formulated as a **text classification** task.

### Examples

| Input | Output |
|-------|--------|
| *Stocks rally as investors expect interest rate cuts.* | Greed |
| *Global markets plunge amid banking crisis fears.* | Panic |
| *Investors await the Federal Reserve decision next week.* | Neutral |

---

## 2. Dataset

**Dataset:** [Financial PhraseBank](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10)

- **Size:** ~4,800 financial news sentences
- **Original labels:** `positive`, `neutral`, `negative`

**Label mapping to market sentiment:**

| Original | Mapped to |
|----------|-----------|
| positive | **Greed** |
| neutral  | **Neutral** |
| negative | **Panic** |

**Splits:**

- **Training:** ~4,000 samples  
- **Evaluation:** 30–100 samples (held out, not used during training)

---

## 3. Model Selection

**Base model:** **Qwen 2.5 0.7B** (open-weight).

- Small parameter count (0.7B) to keep training feasible on limited hardware.
- Parameter-efficient fine-tuning (e.g. LoRA / QLoRA) will be used to further reduce memory and compute.

---

## 4. Fine-Tuning Method

**Approach:** **LoRA / QLoRA** (Low-Rank Adaptation) for supervised fine-tuning.

**Why LoRA:**

1. Reduces GPU memory usage.
2. Trains only low-rank adapters; full model weights stay frozen.
3. Commonly used for instruction tuning of LLMs.

**Training format (instruction-style prompt):**

```
Instruction:
Classify the market sentiment as Greed, Panic, or Neutral.

Input:
Stocks rally as investors expect lower interest rates.

Output:
Greed
```

---

## 5. Hyperparameter Experiments

At least **five** training configurations will be tested to study the effect of hyperparameters.

| Config | Learning rate | LoRA rank | Epochs |
|--------|----------------|-----------|--------|
| 1      | 2e-4           | 8         | 3      |
| 2      | 1e-4           | 8         | 3      |
| 3      | 2e-4           | 16        | 3      |
| 4      | 1e-4           | 16        | 5      |
| 5      | 5e-5           | 16        | 5      |

Training loss will be monitored for convergence and early-stopping decisions.

---

## 6. Evaluation

Evaluation compares the **base model** and the **fine-tuned model** on the held-out evaluation set.

**Metrics:**

- **Accuracy**
- **F1 Score**

**Example comparison:**

| Model            | Accuracy (example) |
|------------------|--------------------|
| Base LLM         | 55%                |
| Fine-tuned model | 82%                |

This comparison shows whether fine-tuning improves financial sentiment detection.

---

## 7. Additional Analysis

The report will include qualitative analysis of model behavior.

- **Improvement analysis**  
  e.g. better recognition of panic-related language: *crash*, *selloff*, *fear*, *market plunge*.

- **Failure cases**  
  e.g. misclassification on sarcastic/ambiguous language or mixed sentiment in one sentence.

- **Limitations**  
  Small dataset size; limited coverage of financial contexts.

---

## 8. Ethical and Risk Considerations

Financial sentiment models can influence trading decisions; incorrect predictions may amplify panic or over-optimism.

**Mitigations:**

- Use the model **only as a research tool**.
- Do **not** treat predictions as financial advice.
- Maintain **human oversight** when interpreting results.

---

## 9. Implementation Pipeline

1. Dataset loading  
2. Data preprocessing  
3. Train / evaluation split  
4. LoRA fine-tuning  
5. Training loss monitoring  
6. Base model evaluation  
7. Fine-tuned model evaluation  
8. Result analysis  

---

## Implementation & Usage

Code lives under `src/`. Run the pipeline from the **main notebook** at the repo root:

```bash
pip install -r requirements.txt
# Open and run main.ipynb (ensure kernel runs from repo root)
```

In `main.ipynb` you can set `CONFIG_NAME = "config_1"` (or another) to train a single LoRA config, or leave it `None` to run all five. Results are written to `outputs/results.json`.

---

## 10. Expected Outcome

The fine-tuned **Qwen 2.5 0.7B** model is expected to outperform the base model on sentiment classification. The project demonstrates how LLM fine-tuning can improve financial text understanding and provides a simple framework for detecting market sentiment from textual data.
