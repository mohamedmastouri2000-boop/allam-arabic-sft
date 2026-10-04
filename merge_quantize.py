"""Merge the trained LoRA into ALLaM-7B, export GGUF, quantize, and measure each variant.

Merging happens in bf16 on the CPU (128 GB RAM) so the 4-bit training base never leaks into
the released weights. Each GGUF gets a perplexity score on held-out Arabic chat text and a
llama-bench speed test on the 5080, so the card can show what each quant costs.
"""
import json, os, re, subprocess, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = r"D:\arabic8b\models\ALLaM-7B"
ADAPTER = os.environ.get("ADAPTER", r"D:\arabic8b\runs\allam-ar-sft\final-half")  # half strength: see README
MERGED = r"D:\arabic8b\out\merged"
GG = r"D:\arabic8b\out\gguf"
LC = r"C:\Apps\llama.cpp"
PY = sys.executable
NAME = os.environ.get("MODEL_NAME", "ALLaM-7B-Arabic-SFT")
QUANTS = ["Q8_0", "Q5_K_M", "Q4_K_M"]
os.makedirs(GG, exist_ok=True)


def run(cmd):
    print(">", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"failed: {cmd[0]}")
    return r.stdout + r.stderr


if not os.path.exists(os.path.join(MERGED, "config.json")):
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": "cpu"})
    model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()
    model.save_pretrained(MERGED, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(BASE).save_pretrained(MERGED)
    print("merged ->", MERGED)

f16 = os.path.join(GG, f"{NAME}-F16.gguf")
if not os.path.exists(f16):
    run([PY, rf"{LC}\llama.cpp-b11379\convert_hf_to_gguf.py", MERGED, "--outtype", "f16", "--outfile", f16])
for q in QUANTS:
    out = os.path.join(GG, f"{NAME}-{q}.gguf")
    if not os.path.exists(out):
        run([rf"{LC}\bin\llama-quantize.exe", f16, out, q])

# held-out Arabic text for perplexity: the assistant replies of the validation split
ppl_txt = os.path.join(GG, "val_ar.txt")
with open(ppl_txt, "w", encoding="utf-8") as fh:
    for line in open(r"D:\arabic8b\data\val.jsonl", encoding="utf-8"):
        for m in json.loads(line)["messages"]:
            if m["role"] == "assistant":
                fh.write(m["content"].strip() + "\n\n")

stats = {}
for q in ["F16"] + QUANTS:
    path = os.path.join(GG, f"{NAME}-{q}.gguf")
    ppl_out = run([rf"{LC}\bin\llama-perplexity.exe", "-m", path, "-f", ppl_txt, "-ngl", "99", "-c", "2048"])
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", ppl_out)
    bench = run([rf"{LC}\bin\llama-bench.exe", "-m", path, "-ngl", "99", "-p", "512", "-n", "128", "-o", "json"])
    rows = json.loads(bench[bench.index("["): bench.rindex("]") + 1])
    stats[q] = {
        "size_gb": round(os.path.getsize(path) / 1e9, 2),
        "ppl": float(m.group(1)) if m else None, "ppl_err": float(m.group(2)) if m else None,
        "prompt_tok_s": round(next(r["avg_ts"] for r in rows if r["n_prompt"] > 0), 1),
        "gen_tok_s": round(next(r["avg_ts"] for r in rows if r["n_gen"] > 0), 1),
    }
    print(q, stats[q], flush=True)
json.dump(stats, open(r"D:\arabic8b\results\gguf_stats.json", "w"), indent=1)
print("done")
