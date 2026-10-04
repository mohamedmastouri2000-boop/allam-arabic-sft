"""Side-by-side Arabic answers from base ALLaM and the tuned adapter (greedy, same prompts).

Benchmarks here are multiple-choice log-likelihood; they do not show how the model writes.
This file does, and it goes into the model card unedited.
"""
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = r"D:\arabic8b\models\ALLaM-7B"
ADAPTER = os.environ.get("ADAPTER", r"D:\arabic8b\runs\allam-ar-sft\final-half")
PROMPTS = [
    "اشرح لطفل عمره عشر سنوات لماذا السماء زرقاء.",
    "اكتب رسالة بريد إلكتروني رسمية قصيرة أعتذر فيها عن التأخر في تسليم مشروع.",
    "ما الفرق بين الذكاء الاصطناعي وتعلم الآلة؟ أجب في ثلاث نقاط.",
    "لخص فوائد المشي اليومي في فقرة واحدة.",
    "أعد صياغة هذه الجملة بأسلوب أدبي: ذهبت إلى السوق واشتريت الخبز.",
    "اقترح خطة من خمس خطوات لتعلم البرمجة بلغة بايثون للمبتدئين.",
    "اكتب تغريدة حماسية عن مباراة كرة قدم انتهت بفوز في الدقيقة الأخيرة.",
    "كيف أحسب النسبة المئوية للزيادة إذا ارتفع السعر من 80 إلى 100؟",
]

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0},
                                             dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()


def answer(p):
    # transformers 5 returns a BatchEncoding here, not a bare tensor
    ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"].to(0)
    out = model.generate(ids, max_new_tokens=300, do_sample=False, repetition_penalty=1.05)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


rows = []
for p in PROMPTS:
    with model.disable_adapter():
        base = answer(p)
    tuned = answer(p)
    rows.append({"prompt": p, "base": base, "tuned": tuned})
    print("done:", p[:30], flush=True)
json.dump(rows, open(r"D:\arabic8b\results\samples.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
