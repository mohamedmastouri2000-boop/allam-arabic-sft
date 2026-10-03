"""Benchmark one model on Arabic tasks with lm-evaluation-harness.

Every model is loaded the same way (4-bit nf4, bf16 compute) because a bf16 8B (~16.4 GB)
does not fit the 16 GB card. Scores are therefore comparable between rows, but a bit
lower than full-precision scores would be.

usage: python run_eval.py <model_dir_or_id> <label> <tasks,comma,separated> [limit] [adapter_dir]
"""
import json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

model_path, label, tasks = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
limit = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "0" else None
adapter = sys.argv[5] if len(sys.argv) > 5 else None

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb,
                                             device_map={"": 0}, dtype=torch.bfloat16)
if adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter)
model.eval()
print(f"loaded {label}; VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

lm = HFLM(pretrained=model, tokenizer=tok, batch_size=8)
t0 = time.time()
chat = os.environ.get("CHAT") == "1"  # score through the model's chat template
res = simple_evaluate(model=lm, tasks=tasks, limit=limit, log_samples=False,
                      apply_chat_template=chat, fewshot_as_multiturn=chat)
secs = time.time() - t0

os.makedirs(r"D:\arabic8b\results", exist_ok=True)
out = {"label": label, "model": model_path, "adapter": adapter, "tasks": tasks, "limit": limit,
       "seconds": round(secs), "results": res["results"]}
path = rf"D:\arabic8b\results\{label}{'_smoke' if limit else ''}.json"
json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(f"saved {path} in {secs:.0f}s")
for t, r in res["results"].items():
    m = {k: round(v, 4) for k, v in r.items() if isinstance(v, float) and "stderr" not in k}
    print(t, m)
