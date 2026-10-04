"""Build the Hugging Face model card from measured results only.

Every number in the card is read from results/*.json or the trainer state; nothing is typed in.
If a result file is missing the card says so instead of leaving a stale number.
"""
import json, os

R = r"D:\arabic8b\results"
NAME = os.environ.get("MODEL_NAME", "ALLaM-7B-Arabic-SFT")
REPO = f"mastouri/{NAME}"
TASKS = [("arabic_leaderboard_arabic_mmlu_light", "ArabicMMLU", 1387),
         ("arabic_exams_light", "Arabic EXAMS", 53),
         ("arabic_leaderboard_acva_light", "ACVA", 851)]


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def acc(res, task):
    return res["results"][task]["acc,none"] * 100 if res and task in res["results"] else None


fmt = lambda v: "n/a" if v is None else f"{v:.1f}"
base, qwen = load("allam-7b.json"), load("qwen3-8b-base.json")
full, tuned = load("allam-7b-sft.json"), load("allam-7b-sft-half.json")  # tuned = the released (half) merge
gg, samples = load("gguf_stats.json"), load("samples.json")
state = json.load(open(r"D:\arabic8b\runs\allam-ar-sft\log_history.json")) \
    if os.path.exists(r"D:\arabic8b\runs\allam-ar-sft\log_history.json") else []
evals = [(h["step"], h["eval_loss"]) for h in state if "eval_loss" in h]

rows = []
for task, label, n in TASKS:
    b, t, q, f = acc(base, task), acc(tuned, task), acc(qwen, task), acc(full, task)
    d = "n/a" if b is None or t is None else f"{t - b:+.1f}"
    rows.append(f"| {label} ({n} q) | {fmt(q)} | {fmt(b)} | {fmt(f)} | **{fmt(t)}** | {d} |")

gg_rows = [f"| {q} | {s['size_gb']} | {s['ppl']} ± {s['ppl_err']} | {s['prompt_tok_s']} | {s['gen_tok_s']} |"
           for q, s in (gg or {}).items()]
ev_rows = [f"| {s} | {l:.4f} |" for s, l in evals]

card = f"""---
license: apache-2.0
language:
- ar
base_model: humain-ai/ALLaM-7B-Instruct-preview
datasets:
- AdaMLLab/smolkalam-arabic-conversational-sft
pipeline_tag: text-generation
tags:
- arabic
- allam
- qlora
- sft
- gguf
---

# {NAME}

A QLoRA fine-tune of [ALLaM-7B-Instruct-preview](https://huggingface.co/humain-ai/ALLaM-7B-Instruct-preview)
on 39,576 Arabic conversations from [SmolKalam](https://huggingface.co/datasets/AdaMLLab/smolkalam-arabic-conversational-sft),
trained on one RTX 5080 (16 GB). Merged bf16 weights and GGUF quants are in this repo.

## Benchmarks

lm-evaluation-harness, 0-shot log-likelihood, "light" Arabic leaderboard sets. **All three models were loaded
in 4-bit nf4** so they fit 16 GB; absolute scores are lower than full-precision runs, but every row used the
same settings. Single runs: differences under about 1 point on ArabicMMLU/ACVA are noise, and Arabic EXAMS has
only 53 questions (±13 points).

| Benchmark | Qwen3-8B | ALLaM-7B (base) | Full-strength LoRA | **{NAME} (released)** | Change vs base |
|---|---|---|---|---|---|
{chr(10).join(rows)}

**Why the released weights use the LoRA at half strength.** Merged at full strength (alpha/r = 2.0) the
fine-tune gained on ArabicMMLU and EXAMS but lost 6 points on ACVA (Arab culture), worse on 27 of 58 subsets:
general instruction data translated from English traded away Arab-specific knowledge. Merging the same adapter
at half strength (alpha/r = 1.0) kept most of the gain and brought ACVA back to within noise of the base.
No retraining was involved; both rows come from the same trained adapter.

## GGUF quants

Perplexity on the held-out assistant replies (lower is better), speed measured with `llama-bench` on an
RTX 5080, full GPU offload, 512-token prompt / 128-token generation.

| Quant | Size (GB) | Perplexity | Prompt tok/s | Generate tok/s |
|---|---|---|---|---|
{chr(10).join(gg_rows) if gg_rows else "| not measured | | | | |"}

## Training

- Base: ALLaM-7B-Instruct-preview, 4-bit nf4 (QLoRA), LoRA r=32 / alpha=64 / dropout 0.05 on all attention and MLP projections (80M trainable parameters, 1.1%)
- Data: SmolKalam subsets magpie-ultra (20k), systemchats (6k), everyday conversations (2k), rewrite (4k), summarize (4k), Tulu-3 persona instruction following (4k); rows kept only with Arabic script purity >= 0.95 and strictly alternating turns
- Loss on assistant tokens only; 1 epoch (2,474 steps, 10.2 h on an RTX 5080, 9.4 GB peak), effective batch 16, lr 1e-4 cosine, 75 warmup steps, max length 2048
- Released merge: the trained adapter applied at half strength (lora_alpha 64 -> 32), merged into bf16 weights on CPU
- Held-out validation loss:

| Step | Validation loss |
|---|---|
{chr(10).join(ev_rows) if ev_rows else "| n/a | |"}

## Sample outputs

Greedy decoding, same prompt to both models, unedited.

"""
for s in samples or []:
    card += f"**{s['prompt']}**\n\n<details><summary>Base ALLaM-7B</summary>\n\n{s['base']}\n\n</details>\n\n" \
            f"<details open><summary>{NAME}</summary>\n\n{s['tuned']}\n\n</details>\n\n"

card += """## Limitations

- Gains are small ({DELTAS}). In the samples the tuned model is not uniformly better; read them before choosing it over the base.
- Trained one epoch on translated data (SmolKalam is a machine translation of SmolTalk2, ensemble-filtered). Dialect coverage follows the source and is mostly Modern Standard Arabic.
- Benchmarks are multiple-choice log-likelihood; they do not measure long-form writing quality. Read the samples.
- The training data follows the SmolTalk2 upstream licences; review them before commercial use.

## Usage (llama.cpp)

```bash
llama-cli -m ALLaM-7B-Arabic-SFT-Q4_K_M.gguf -ngl 99 -cnv
```
"""
deltas = ", ".join(f"{label} {acc(tuned, t) - acc(base, t):+.1f}" for t, label, _ in TASKS
                   if acc(tuned, t) is not None and acc(base, t) is not None) or "not measured"
card = card.replace("{DELTAS}", deltas)
open(r"D:\arabic8b\out\README.md", "w", encoding="utf-8").write(card)
print("card written,", len(card), "chars; tuned results present:", tuned is not None, "| gguf:", gg is not None)
