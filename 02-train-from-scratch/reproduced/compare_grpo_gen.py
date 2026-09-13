import re, torch, warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed

def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    return min(cap, (len(grams)-len(set(grams)))*cap*2/len(grams)) if grams else 0.0

def rule_score(resp):
    s = 0.5 if 20 <= len(resp.strip()) <= 800 else -0.5
    if "</think>" in resp:
        think, ans = resp.split("</think>", 1)
        s += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5
        s += 0.25 if resp.count("</think>") == 1 else -0.25
    else:
        ans = resp
    s -= rep_penalty(ans.strip())
    return s

def load(weight):
    tok = AutoTokenizer.from_pretrained("model")
    m = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=0))
    m.load_state_dict(torch.load(f"./out/{weight}_768.pth", map_location="cuda"), strict=True)
    return m.half().eval().cuda(), tok

prompts = [
    "请用Python写一个计算斐波那契数列的函数",
    "为什么天空是蓝色的",
    "解释什么是机器学习",
    "比较一下猫和狗作为宠物的优缺点",
    "如果明天下雨，我应该如何出门",
]
weights = ["full_sft", "grpo"]
results = {}
for w in weights:
    model, tok = load(w)
    outs = []
    for i, p in enumerate(prompts):
        setup_seed(100+i)
        conv = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, open_thinking=True)
        enc = tok(text, return_tensors="pt", truncation=True).to("cuda")
        with torch.no_grad():
            gid = model.generate(inputs=enc["input_ids"], attention_mask=enc["attention_mask"],
                                 max_new_tokens=512, do_sample=True, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id, top_p=0.95, temperature=0.85, repetition_penalty=1)
        resp = tok.decode(gid[0][len(enc["input_ids"][0]):], skip_special_tokens=False)
        outs.append(resp)
    results[w] = outs
    del model
    torch.cuda.empty_cache()

L = []
L.append("GRPO before/after comparison (same prompts/seeds/sampling; open_thinking=True)")
L.append("weights: full_sft = before GRPO ; grpo = after 300-step rule-reward GRPO")
for i, p in enumerate(prompts):
    L.append("="*84); L.append("PROMPT: " + p)
    for w in weights:
        r = results[w][i]
        L.append(f"\n----- [{w}] rule_score={rule_score(r):.3f} chars={len(r)} think_tags={r.count(chr(60)+'/think'+chr(62))} -----")
        L.append(r.strip())
L.append("\n"+"="*84); L.append("SUMMARY (avg over 5 prompts)")
for w in weights:
    sc = [rule_score(results[w][i]) for i in range(len(prompts))]
    ln = [len(results[w][i]) for i in range(len(prompts))]
    tag = sum(1 for i in range(len(prompts)) if results[w][i].count("</think>") == 1)
    L.append(f"{w:9s}: avg_rule_score={sum(sc)/len(sc):.3f}  avg_chars={sum(ln)/len(ln):.0f}  exact-one-think-tag={tag}/{len(prompts)}")
out = "\n".join(L)
print(out)
with open("experiments_text_dump.txt", "w", encoding="utf-8") as f:
    f.write(out)