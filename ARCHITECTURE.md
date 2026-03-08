# Architecture

This document describes the system’s layered architecture, module dependencies, data flow, model structure, and training/evaluation pipeline.

---

## 1. High-level architecture

The system is split into **config**, **data**, **model**, **train**, and **evaluate** layers, orchestrated by a **Notebook** that runs them in sequence.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Orchestration (main.ipynb / all_in_one.ipynb)     │
│ prepare_splits → evaluate_base → train(×N configs) → evaluate_finetuned │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                    │                    │
         ▼                    ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ config       │    │ data         │    │ model        │    │ train        │
│ constants &  │    │ load/map/    │    │ base+LoRA    │    │ tokenize+    │
│ experiment   │    │ split        │    │              │    │ Trainer      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
         │                    │                    │                    │
         └────────────────────┴────────────────────┴────────────────────┘
                                              │
                                              ▼
                                     ┌──────────────┐
                                     │ evaluate     │
                                     │ infer+parse+ │
                                     │ metrics      │
                                     └──────────────┘
```

- **config**: Global constants and experiment configs; used by data, model, train, and evaluate.
- **data**: Depends only on config; produces train_ds and eval_ds (with `sentence`, `text`, `target_label`, etc.).
- **model**: Depends on config; produces tokenizer, base model, and PEFT model with LoRA attached.
- **train**: Depends on config and model; consumes datasets from data, runs tokenize + Trainer training + save.
- **evaluate**: Depends on config and model; consumes eval dataset from data, runs base or loads PEFT for inference and metrics.

---

## 2. Module dependencies

```
                    config.py
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
     data.py         model.py      (train.py, evaluate.py also use config)
         │               │
         │               │  get_base_model_and_tokenizer, get_tokenizer, apply_lora
         │               │
         ▼               ▼
    train.py ──────────────────► model (load base + LoRA)
         │
         │  train_dataset, eval_dataset (from data)
         │  LoRAConfig (from config)
         ▼
    Trainer (transformers) → save to outputs/<config_name>/final

    evaluate.py ─────────────────► model (base or base+PeftModel.from_pretrained)
         │
         │  eval_dataset (from data)
         ▼
    run_inference → extract_predicted_label → accuracy_score, f1_score
```

- **config** has no in-project dependencies; it is only imported by other modules.
- **data** imports only config; it does not depend on model, train, or evaluate.
- **model** imports only config and transformers/peft; it does not depend on data, train, or evaluate.
- **train** depends on config and model; it receives datasets produced by data.
- **evaluate** depends on config and model; it receives the eval dataset from data and, for fine-tuned evaluation, reads the PEFT checkpoint from disk.

---

## 3. Data flow

### 3.1 Raw data to train/eval sets

```
Hugging Face (cyrilzhang/financial_phrasebank_split)
    │
    │  load_dataset(split="train")
    ▼
raw Dataset
    columns: sentence, label (int 0/1/2 or str positive/neutral/negative)
    │
    │  map(map_labels)
    ▼
Dataset + sentiment
    sentiment ∈ {Greed, Neutral, Panic}  (from LABEL_MAP_INT / LABEL_MAP)
    │
    │  map(format_instruction)
    ▼
Dataset + text, target_label
    text = INSTRUCTION_TEMPLATE.format(input_text=sentence, label=sentiment)
    target_label = sentiment
    │
    │  shuffle(seed=42) → select(range(n-eval_size)) / select(range(n-eval_size, n))
    ▼
train_ds (~4261 rows)     eval_ds (100 rows)
    │                         │
    │  used by train()        │  used by evaluate_base_model / evaluate_finetuned_model
    ▼                         ▼
tokenize_for_training     keep original columns; use sentence + target_label at inference
    │
    ▼
train_ds tokenized: input_ids, attention_mask, labels (causal LM; labels = copy of input_ids)
```

- **Training**: Uses only `train_ds`; each sample’s `text` is tokenized to `input_ids`, `labels` equals `input_ids`, and training is causal LM maximum likelihood.
- **Evaluation**: Builds prompts from `eval_ds`’s `sentence` and uses `target_label` as gold for accuracy and macro F1.

### 3.2 Data shape during training

- **Input**: `tokenize_for_training` produces `input_ids` (max_length=256, padding="max_length"), `attention_mask`, and `labels`.
- **Collator**: `DataCollatorForLanguageModeling(mlm=False)`; no MLM masking, only padding and batching.
- **Effective batch**: `per_device_train_batch_size × gradient_accumulation_steps` (default 4×4=16). Each optimizer step sees 16 samples.

---

## 4. Model architecture

### 4.1 Base model

- **Model**: `Qwen/Qwen2.5-0.5B-Instruct` (Transformers `AutoModelForCausalLM`).
- **Structure**: Decoder-only Transformer (Qwen2), ~0.5B parameters.
- **Loading**:
  - **4-bit (QLoRA)**: When `bitsandbytes` is available and version ≥ 0.46.1, use `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)` and call `prepare_model_for_kbit_training`.
  - **Full precision**: Otherwise `torch.bfloat16` + `device_map="auto"` (GPU/CPU assigned by accelerate).
- **Tokenizer**: `AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)`; if `pad_token` is None, set it to `eos_token`.

### 4.2 LoRA adapter

- **Method**: PEFT `LoraConfig` + `get_peft_model`; base is frozen, only low-rank matrices are trained.
- **Target modules** (default):  
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`  
  (Attention Q/K/V/O and MLP gate/up/down).
- **Hyperparameters**: `r` (LoRA rank, 8 or 16 in experiments), `lora_alpha` (scale, default 16), `lora_dropout=0.05`, `bias="none"`, `task_type=TaskType.CAUSAL_LM`.
- **Forward**: Base output + LoRA residual; only LoRA parameters are updated during training; at inference the adapter is merged or loaded separately.

### 4.3 Model at training vs inference

- **Training**: Base (frozen) + LoRA (trainable); base may be 4-bit to save memory.
- **Base evaluation**: Base only, `use_4bit=False`, for consistent numerics.
- **Fine-tuned evaluation**: Base (full precision) + `PeftModel.from_pretrained(base_model, checkpoint_dir)`; only adapter weights are loaded.

---

## 5. Training flow

### 5.1 Single train() flow

```
train(train_dataset, eval_dataset, lora_config, output_dir, use_4bit)
    │
    ├─► get_base_model_and_tokenizer(MODEL_ID, use_4bit)
    │       → tokenizer, base model (4bit or bf16)
    │
    ├─► apply_lora(model, r=lora_config.lora_r, lora_alpha=lora_config.lora_alpha)
    │       → PeftModel
    │
    ├─► run_dir = output_dir / lora_config.name
    │
    ├─► tokenize_for_training(train_dataset, tokenizer, max_length=lora_config.max_seq_length)
    │       → train_ds (input_ids, attention_mask, labels)
    │
    ├─► If eval_dataset non-empty: tokenize_for_training(eval_dataset, ...) → eval_ds
    │
    ├─► Compute total_steps, warmup_steps
    │       (or use max_steps when set; then save_strategy="no")
    │       batch_size = per_device_train_batch_size × gradient_accumulation_steps
    │       total_steps from max_steps or (len(train_ds)//batch_size)*epochs
    │       warmup_steps = max(1, int(total_steps * warmup_ratio))
    │
    ├─► TrainingArguments(...)
    │       max_steps, num_train_epochs, warmup_steps, logging_steps, save_strategy,
    │       bf16, fp16, lr_scheduler_type, dataloader_pin_memory (False on MPS)
    │
    ├─► DataCollatorForLanguageModeling(tokenizer, mlm=False)
    │
    ├─► Trainer(model, args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=...)
    │
    ├─► trainer.train()
    │
    └─► trainer.save_model(run_dir/"final"); tokenizer.save_pretrained(run_dir/"final")
            → return str(run_dir/"final")
```

### 5.2 Optimizer and steps

- **Optimizer**: Trainer default AdamW (learning_rate, weight_decay, etc. from TrainingArguments).
- **LR schedule**: Linear warmup for warmup_steps, then decay (e.g. cosine when `lr_scheduler_type="cosine"`).
- **Steps per epoch**: `len(train_ds) // batch_size` (e.g. 4261 // 16 ≈ 266). When `max_steps` is set, training runs exactly that many steps.
- **Loss**: Causal LM cross-entropy over logits and `labels` per token; padding positions excluded via attention_mask or default ignore index.

---

## 6. Evaluation flow

### 6.1 Entry points

- **evaluate_base_model(eval_dataset)**: Load base (use_4bit=False) → `evaluate_model(model, tokenizer, eval_dataset)`.
- **evaluate_finetuned_model(checkpoint_dir, eval_dataset)**: Load base + `PeftModel.from_pretrained(base, checkpoint_dir)` → `evaluate_model(...)`.

### 6.2 Inside evaluate_model

```
evaluate_model(model, tokenizer, eval_dataset, sentence_key="sentence", label_key="target_label")
    │
    ├─► sentences = [ex[sentence_key] for ex in eval_dataset]
    ├─► gold = [ex[label_key] for ex in eval_dataset]
    │
    ├─► preds = run_inference(model, tokenizer, sentences)
    │       │
    │       ├─► tokenizer.padding_side = "left" (decoder left-padding)
    │       ├─► GenerationConfig(max_new_tokens=16, do_sample=True, temperature=0.3, top_p=0.9, top_k=50)
    │       ├─► Build prompts in batches of 8:
    │       │     "Instruction:\nClassify the market sentiment as Greed, Panic, or Neutral.\n\nInput:\n{s}\n\nOutput:\n"
    │       ├─► model.generate(...) → decode only the part after the prompt
    │       └─► extract_predicted_label(gen_text) for each generation → Greed/Neutral/Panic
    │
    ├─► accuracy_score(gold, preds)
    └─► f1_score(gold, preds, average="macro", zero_division=0)
```

### 6.3 Label parsing (extract_predicted_label)

- Prefer: label appears at end of string as whole word (`\b{label}\s*$`) → return that label.
- Else: first occurrence of any label in the string (case-insensitive) → return that label.
- If none: default to `Neutral` (LABELS[1]).

---

## 7. Config and experiment design

### 7.1 Global constants (config.py)

| Constant | Meaning |
|----------|---------|
| MODEL_ID | Base model ID: Qwen/Qwen2.5-0.5B-Instruct |
| DATASET_ID | Dataset: cyrilzhang/financial_phrasebank_split |
| LABELS / LABEL_MAP / LABEL_MAP_INT | Three-way Greed/Neutral/Panic and mapping from raw labels |
| EVAL_SIZE | Eval set size (default 100) |
| RANDOM_SEED | Shuffle seed (42) |
| INSTRUCTION_TEMPLATE | Instruction template for training |

### 7.2 LoRAConfig and the five experiments

| Field | Meaning | Default or experiment value |
|-------|---------|-----------------------------|
| name | Experiment name | config_1 .. config_5 |
| learning_rate | Learning rate | 2e-4, 1e-4, 5e-5 |
| lora_r | LoRA rank | 8 or 16 |
| lora_alpha | LoRA scale | 16 |
| epochs | Training epochs | 3 or 5 (ignored when max_steps set) |
| per_device_train_batch_size | Batch size per device | 4 |
| gradient_accumulation_steps | Gradient accumulation steps | 4 |
| max_seq_length | Max sequence length | 256 |
| warmup_ratio | Warmup fraction of steps | 0.1 |
| weight_decay | Weight decay | 0.01 |
| logging_steps | Logging interval | 10 |
| save_strategy | Save strategy | "epoch" or "no" when max_steps used |
| bf16 | Use bf16 | True |
| fp16 | Use fp16 | False |
| max_steps | Fixed training steps | 100 (when set, overrides epochs) |
| lr_scheduler_type | LR schedule | "cosine" |

The five experiments are returned by `get_experiment_configs()`. The notebook can set `CONFIG_NAME` to run a single config or leave it `None` to run all five.

---

## 8. Outputs and persistence

| Path | Content |
|------|---------|
| outputs/<config_name>/ | Run directory for that config (Trainer output_dir) |
| outputs/<config_name>/final/ | Final PEFT adapter + tokenizer (adapter_config.json, adapter_model.safetensors, tokenizer files) |
| outputs/results.json | Summary: base_model.accuracy/f1_macro; configs.<name>.accuracy/f1_macro |

Fine-tuned models are loaded from `outputs/<config_name>/final/` for evaluation. The notebook writes the summary to `results.json` in the final section.

---

## 9. Design choices and trade-offs

- **Single eval split**: Only train/eval split; no separate validation set. Eval is used for reporting and base vs fine-tuned comparison only; it is not used for early stopping or checkpoint selection.
- **Causal LM training**: Full instruction+output sequence; labels equal input_ids; no separate mask for “output tokens”. The model learns the format and content of the answer given the instruction.
- **Evaluation**: Generative (generate then parse label), aligned with the training objective; relies on the robustness of `extract_predicted_label`.
- **Device**: `device_map="auto"`; allocation by accelerate. On MPS, `dataloader_pin_memory` is disabled; otherwise no platform-specific branches.
- **Reproducibility**: Data shuffle and `torch.manual_seed(42)` at evaluation fix randomness; experiments are reproducible by config name and seed.
