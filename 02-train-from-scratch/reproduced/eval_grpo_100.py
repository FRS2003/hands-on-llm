import re, json, csv, os, torch, warnings, numpy as np
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

def questions(path):
    q = []
    for line in open(path, encoding="utf-8"):
        c = json.loads(line)["conversations"][0]
        q.append(c.get("value") or c.get("content"))
    return q

allq = questions("dataset/rlaif.jsonl")
sub = set(questions("dataset/rlaif_sub.jsonl"))
step = len(allq)/500
cand = [allq[int(i*step)] for i in range(500)]
evalq = [q for q in cand if q not in sub][:100]
assert len(evalq) == 100, len(evalq)
print(f"held-out eval prompts = {len(evalq)} (full={len(allq)}, train_subset={len(sub)})", flush=True)

os.makedirs("experiments/grpo", exist_ok=True)
weights = ["full_sft", "grpo"]
agg, detail = {}, []
for w in weights:
    model, tok = load(w)
    scores,lens,tag1,thinkok,lenok,reps,notag = [],[],[],[],[],[],[]
    for i,p in enumerate(evalq):
        setup_seed(500+i)
        conv = [{"role":"user","content":p}]
        text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, open_thinking=True)
        enc = tok(text, return_tensors="pt", truncation=True).to("cuda")
        with torch.no_grad():
            g = model.generate(inputs=enc["input_ids"], attention_mask=enc["attention_mask"],
                               max_new_tokens=300, do_sample=True, pad_token_id=tok.pad_token_id,
                               eos_token_id=tok.eos_token_id, top_p=0.95, temperature=0.85, repetition_penalty=1)
        r = tok.decode(g[0][len(enc["input_ids"][0]):], skip_special_tokens=False)
        rs = r.strip()
        sc = rule_score(r); rp = rep_penalty(rs.split("</think>",1)[1].strip() if "</think>" in rs else rs)
        scores.append(sc); lens.append(len(r)); reps.append(rp)
        tag1.append(1 if r.count("</think>")==1 else 0); notag.append(1 if r.count("</think>")==0 else 0)
        lenok.append(1 if 20<=len(rs)<=800 else 0)
        thinkok.append(1 if ("</think>" in r and 20<=len(r.split("</think>",1)[0].strip())<=300) else 0)
        detail.append([i,w,f"{sc:.3f}",len(r),r.count("</think>"),f"{rp:.3f}",p[:40].replace("\n"," ")])
        if (i+1)%20==0: print(f"{w} {i+1}/100", flush=True)
    agg[w]=dict(n=len(evalq),score=np.mean(scores),chars=np.mean(lens),tag1=np.mean(tag1),
                thinkok=np.mean(thinkok),lenok=np.mean(lenok),rep=np.mean(reps),notag=np.mean(notag))
    del model; torch.cuda.empty_cache()

L=["GRPO quantitative eval on 100 HELD-OUT prompts (excluded from the 600 training subset)",
   "same seeds/sampling; max_new_tokens=300; rule reward identical to trainer/train_grpo_rule.py",
   f"{'weight':9s}{'n':>4}{'ruleScore':>10}{'len20-800':>10}{'thinkOK':>9}{'tag==1':>8}{'noTag':>7}{'repPen':>8}{'avgChars':>9}"]
for w in weights:
    a=agg[w]
    L.append(f"{w:9s}{a['n']:4d}{a['score']:10.3f}{a['lenok']:10.3f}{a['thinkok']:9.3f}{a['tag1']:8.3f}{a['notag']:7.3f}{a['rep']:8.3f}{a['chars']:9.0f}")
b,d=agg["full_sft"],agg["grpo"]
L+=["",f"DELTA rule_score {b['score']:.3f} -> {d['score']:.3f}  ({d['score']-b['score']:+.3f})",
    f"DELTA thinkOK     {b['thinkok']:.3f} -> {d['thinkok']:.3f}  ({d['thinkok']-b['thinkok']:+.3f})",
    f"DELTA tag==1      {b['tag1']:.3f} -> {d['tag1']:.3f}  ({d['tag1']-b['tag1']:+.3f})",
    f"DELTA repPenalty  {b['rep']:.3f} -> {d['rep']:.3f}  ({d['rep']-b['rep']:+.3f})"]
out="\n".join(L); print(out)
open("experiments/grpo/eval_100_metrics.txt","w",encoding="utf-8").write(out)
with open("experiments/grpo/eval_100_detail.csv","w",newline="",encoding="utf-8") as f:
    wr=csv.writer(f); wr.writerow(["idx","weight","rule_score","chars","think_tags","rep_penalty","prompt_head"]); wr.writerows(detail)
print("DETAIL_SAVED rows=",len(detail))