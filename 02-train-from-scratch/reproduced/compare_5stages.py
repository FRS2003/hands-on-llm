import torch, warnings, numpy as np
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
from trainer.trainer_utils import setup_seed

TT = chr(60) + "/think" + chr(62)
prompts = [
    "请用Python写一个计算斐波那契数列的函数",
    "为什么天空是蓝色的",
    "解释什么是机器学习",
    "比较一下猫和狗作为宠物的优缺点",
    "如果明天下雨，我应该如何出门",
    "用三句话介绍杭州",
]
# name, mode(raw=continue / chat), backbone weight, optional lora adapter
stages = [
    ("pretrain", "raw",  "pretrain", None),
    ("sft",      "chat", "full_sft", None),
    ("lora",     "chat", "full_sft", "lora_demo"),
    ("dpo",      "chat", "dpo",      None),
    ("grpo",     "chat", "grpo",     None),
]

def load(weight, lora):
    tok = AutoTokenizer.from_pretrained("model")
    m = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=0))
    m.load_state_dict(torch.load(f"./out/{weight}_768.pth", map_location="cuda"), strict=True)
    if lora:
        apply_lora(m)
        load_lora(m, f"./out/{lora}_768.pth")
    return m.half().eval().cuda(), tok

data = {}
for name, mode, weight, lora in stages:
    m, tok = load(weight, lora)
    rows = []
    for i, p in enumerate(prompts):
        setup_seed(200 + i)
        if mode == "raw":
            text = tok.bos_token + p
        else:
            conv = [{"role": "user", "content": p}]
            text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, open_thinking=True)
        enc = tok(text, return_tensors="pt", truncation=True).to("cuda")
        with torch.no_grad():
            g = m.generate(inputs=enc["input_ids"], attention_mask=enc["attention_mask"],
                           max_new_tokens=400, do_sample=True, pad_token_id=tok.pad_token_id,
                           eos_token_id=tok.eos_token_id, top_p=0.95, temperature=0.85, repetition_penalty=1)
        rows.append(tok.decode(g[0][len(enc["input_ids"][0]):], skip_special_tokens=False))
    data[name] = rows
    del m
    torch.cuda.empty_cache()

L = []
L.append("Five-stage generation evolution (SAME prompts/seeds/sampling, max_new_tokens=400)")
L.append("pretrain=raw continuation(bos+prompt); sft/lora/dpo/grpo=chat template(open_thinking); lora = 0.78MB adapter on sft backbone")
L.append("")
for i, p in enumerate(prompts):
    L.append("=" * 84); L.append("PROMPT: " + p)
    for name, _, _, _ in stages:
        r = data[name][i]
        L.append(f"\n----- [{name}] chars={len(r)} think_tags={r.count(TT)} -----")
        L.append(r.strip())
    L.append("")
L.append("=" * 84); L.append("SUMMARY per stage")
for name, _, _, _ in stages:
    rs = data[name]
    L.append(f"{name:9s} avg_chars={np.mean([len(x) for x in rs]):.0f}  think_tags_total={sum(x.count(TT) for x in rs)}")
out = "\n".join(L)
print(out)
open("experiments/stage_evolution.txt", "w", encoding="utf-8").write(out)