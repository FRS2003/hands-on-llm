import torch, re, random, warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed

tok = AutoTokenizer.from_pretrained("model")
m = MiniMindForCausalLM(MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=0))
m.load_state_dict(torch.load("./out/full_sft_768.pth",map_location="cuda"),strict=True)
m.half().eval().cuda()

def tasks(kind,n=32,seed=7):
    r=random.Random(seed); out=[]
    for _ in range(n):
        if kind=="add1": a,b=r.randint(1,9),r.randint(1,9); op="+"; g=a+b
        elif kind=="add2": a,b=r.randint(11,89),r.randint(11,89); op="+"; g=a+b
        elif kind=="addsub":
            a=r.randint(20,89); b=r.randint(1,a); op=r.choice(["+","-"]); g=a+b if op=="+" else a-b
        else: a,b=r.randint(2,9),r.randint(2,9); op="x"; g=a*b
        out.append((f"请口算：{a} {op} {b} = ? 直接给出最终数字，不要写过程。", g))
    return out

def last_num(s):
    z=re.findall(r"\d+", s.replace(",",""))
    return int(z[-1]) if z else None

for kind in ["add1","add2","addsub","mul1"]:
    ok=0; extract=0; ex=[]
    for i,(p,g) in enumerate(tasks(kind)):
        setup_seed(1000+i)
        conv=[{"role":"user","content":p}]
        text=tok.apply_chat_template(conv,tokenize=False,add_generation_prompt=True,open_thinking=False)
        e=tok(text,return_tensors="pt",truncation=True).to("cuda")
        with torch.no_grad():
            o=m.generate(inputs=e["input_ids"],attention_mask=e["attention_mask"],max_new_tokens=80,
                         do_sample=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,
                         top_p=0.95,temperature=0.8,repetition_penalty=1)
        r=tok.decode(o[0][len(e["input_ids"][0]):],skip_special_tokens=True)
        pred=last_num(r)
        if pred is not None: extract+=1
        if pred==g: ok+=1
        if i<3: ex.append(f"[gold={g}|pred={pred}] {r.strip()[:60]}")
    print(f"=== {kind}: acc={ok}/32={ok/32:.2f}  extractable={extract}/32")
    for s in ex: print("   ",s)