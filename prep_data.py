"""Build the training mix from the SmolKalam subsets in D:\\sk.

Keeps rows with high Arabic script purity (SCR) and strictly alternating user/assistant turns
(ALLaM's Llama-2 chat template raises on anything else). Writes train/val JSONL of {"messages": [...]}.
"""
import glob, json, random
import pyarrow.parquet as pq

random.seed(0)
MIX = {  # subset -> rows to take
    "smoltalk_smollm3_smol_magpie_ultra_no_think": 20000,
    "smoltalk_smollm3_systemchats_30k_no_think": 6000,
    "smoltalk_smollm3_everyday_conversations_no_think": 2260,
    "smoltalk_smollm3_smol_rewrite_no_think": 4000,
    "smoltalk_smollm3_smol_summarize_no_think": 4000,
    "tulu_3_sft_personas_instruction_following_no_think": 4000,
}


def ok(msgs):
    body = msgs[1:] if msgs and msgs[0]["role"] == "system" else msgs
    if len(body) < 2 or body[-1]["role"] != "assistant":
        return False
    return all(m["role"] == ("user" if i % 2 == 0 else "assistant") and m["content"].strip()
               for i, m in enumerate(body))


rows, report = [], {}
for sub, n in MIX.items():
    pool = []
    for f in sorted(glob.glob(rf"D:\sk\{sub}\*.parquet")):
        for r in pq.read_table(f, columns=["messages", "SCR"]).to_pylist():
            msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
            if (r["SCR"] or 0) >= 0.95 and ok(msgs):
                pool.append(msgs)
    random.shuffle(pool)
    take = pool[:n]
    report[sub] = (len(pool), len(take))
    rows += [{"messages": m, "subset": sub} for m in take]

random.shuffle(rows)
n_val = max(200, len(rows) // 100)
for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
    with open(rf"D:\arabic8b\data\{name}.jsonl", "w", encoding="utf-8") as fh:
        for r in part:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
for k, (pool, take) in report.items():
    print(f"{k}: {take} of {pool} eligible")
print(f"train {len(rows) - n_val}, val {n_val}")
