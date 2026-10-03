"""QLoRA fine-tune of ALLaM-7B-Instruct on the SmolKalam mix.

Loss is computed on assistant tokens only. The mask comes from rendering the chat template
twice (conversation up to and including each assistant turn) and diffing character offsets,
so it follows ALLaM's own template exactly instead of hard-coding [INST] markers.

usage: python train.py [max_steps]   (max_steps > 0 = timing trial, nothing kept)
"""
import json, os, sys, time
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE = r"D:\arabic8b\models\ALLaM-7B"
OUT = r"D:\arabic8b\runs\allam-ar-sft"
MAX_LEN = 2048
max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else -1
BS = int(os.environ.get("BS", "1"))  # micro-batch; must divide 16

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.pad_token or tok.unk_token or tok.eos_token
assert tok.is_fast, "offset mapping needs a fast tokenizer"


def encode(msgs):
    text = tok.apply_chat_template(msgs, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True,
              truncation=True, max_length=MAX_LEN)
    # character spans of every assistant reply inside the rendered text
    spans = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        start = len(tok.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True)) \
            if i else 0
        end = len(tok.apply_chat_template(msgs[:i + 1], tokenize=False))
        spans.append((start, end))
    labels = [tid if any(s <= a and b <= e for s, e in spans) else -100
              for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"])]
    return enc["input_ids"], labels


class Chats(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l)["messages"] for l in open(path, encoding="utf-8")]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = encode(self.rows[i])
        return {"input_ids": ids, "labels": labels}


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda xs, v: [x + [v] * (n - len(x)) for x in xs]
    ids = torch.tensor(pad([b["input_ids"] for b in batch], tok.pad_token_id))
    lab = torch.tensor(pad([b["labels"] for b in batch], -100))
    att = torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch])
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


if __name__ == "__main__":
    # sanity check the mask on one sample before spending GPU time
    sample = Chats(r"D:\arabic8b\data\val.jsonl")[0]
    kept = [t for t, l in zip(sample["input_ids"], sample["labels"]) if l != -100]
    print("mask check: trained on", len(kept), "of", len(sample["input_ids"]), "tokens")
    print("trained text starts:", tok.decode(kept[:40]))

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                                 device_map={"": 0}, dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUT if max_steps < 0 else OUT + "-trial",
        num_train_epochs=1, max_steps=max_steps,
        # effective batch is always 16; bs 4 spilled past 16 GB, bs 1 peaked at 9.3 GB
        per_device_train_batch_size=BS, gradient_accumulation_steps=16 // BS,
        learning_rate=1e-4, lr_scheduler_type="cosine",
        warmup_steps=2 if max_steps > 0 else 75,  # ~3% of the 2,473-step epoch
        bf16=True, optim="paged_adamw_8bit", gradient_checkpointing=True,
        logging_steps=5 if max_steps > 0 else 25,
        eval_strategy="no" if max_steps > 0 else "steps", eval_steps=250,
        save_strategy="no" if max_steps > 0 else "steps", save_steps=250, save_total_limit=3,
        load_best_model_at_end=max_steps < 0, metric_for_best_model="eval_loss",
        per_device_eval_batch_size=1,
        dataloader_num_workers=0, report_to="none", remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, data_collator=collate,
                      train_dataset=Chats(r"D:\arabic8b\data\train.jsonl"),
                      eval_dataset=Chats(r"D:\arabic8b\data\val.jsonl"))
    t0 = time.time()
    has_ckpt = max_steps < 0 and os.path.isdir(OUT) and any(
        d.startswith("checkpoint-") for d in os.listdir(OUT))
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    steps = trainer.state.global_step
    print(f"steps {steps} in {secs:.0f}s = {secs / max(steps, 1):.1f} s/step; peak VRAM {peak:.1f} GB")
    if max_steps < 0:
        trainer.save_model(OUT + "/final")
        json.dump(trainer.state.log_history, open(OUT + "/log_history.json", "w"), indent=1)
    else:
        total = len(trainer.train_dataset) // 16
        print(f"full epoch = {total} steps ~ {total * secs / max(steps, 1) / 3600:.1f} h")

