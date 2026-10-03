# ALLaM-7B Arabic SFT

QLoRA fine-tune of [humain-ai/ALLaM-7B-Instruct-preview](https://huggingface.co/humain-ai/ALLaM-7B-Instruct-preview)
on Arabic conversational data from [SmolKalam](https://huggingface.co/datasets/AdaMLLab/smolkalam-arabic-conversational-sft),
trained on a single RTX 5080 (16 GB). Status: **training in progress.**

## Why ALLaM and not Qwen3-8B

The plan started with Qwen3-8B. Baselines on the same harness (lm-evaluation-harness, 4-bit nf4, light Arabic sets) changed it:

| Benchmark | Qwen3-8B | Qwen3-8B (chat template) | ALLaM-7B |
|---|---|---|---|
| ArabicMMLU (1,387 q) | 32.9 | 23.1 | **47.8** |
| Arabic EXAMS (53 q) | 35.8 | 20.8 | **50.9** |
| ACVA (851 q) | 43.5 | 41.1 | **77.6** |

Qwen3-8B scored lower than expected and the cause is not confirmed. ALLaM ran through the same pipeline normally.

## Pipeline

| File | Step |
|---|---|
| `run_eval.py` | benchmark any model or adapter (4-bit, so 7-8B fits 16 GB) |
| `prep_data.py` | 39,576 train / 399 val conversations, filtered for Arabic script purity and clean turn order |
| `train.py` | QLoRA r=32 on all linear layers, assistant-only loss, effective batch 16, 1 epoch, resumable |
| `merge_quantize.py` | bf16 merge on CPU, GGUF F16/Q8_0/Q5_K_M/Q4_K_M, perplexity + speed per quant |

## Windows notes (each one cost a failed run)

- **Path length**: some benchmark and dataset cache paths exceed 260 characters. Use `HF_DATASETS_CACHE=D:\hfd` and download parquet files to a short folder instead of the hub cache.
- **transformers 5** removed `load_in_4bit`, `warmup_ratio` and `group_by_length`. Use `BitsAndBytesConfig` and `warmup_steps`.
- **ALLaM's tokenizer needs `protobuf`**; without it transformers falls back to a TikToken reader and fails with a confusing parse error.
- **Batch size**: bs 4 spilled past 16 GB into system RAM (GPU at 100 W, stalled). bs 2 did not spill but was 2.7x slower than bs 1 because of padding. bs 1 x 16 accumulation: 14.6 s/step, 9.3 GB peak.

## Results

Filled in after training.
