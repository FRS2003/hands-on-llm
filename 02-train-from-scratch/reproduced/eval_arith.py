import torch, re, json, argparse, warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed
ap=argparse.ArgumentParser(); ap.add_argument("--weight",default="arith_sft"); ap.add_argument("--temp",type=float,default=0.8); ap.add_argument("--seedbase",type=int,default=3000); args=ap.parse_args()
tok=AutoTokenizer.from_pretrained("model")
m=MiniMindForCausalLM(MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=0))
m.load_state_dict(torch.load(f"./out/{args.weight}_768.pth",map_location="cuda"),strict=True)
m.eval().cuda()  # fp32, avoid fp16 argmax flip
data=[json.loads(l) for l in open("dataset/arith_eval.jsonl",encoding="utf-8")]
def last_num(s):
    z=re.findall(r"\d+",s.replace(",","")); return int(z[-1]) if z else None
stat={"add":[0,0],"mul":[0,0]}; ex=[]
for i,o in enumerate(data):
    conv=o["conversations"]; q=conv[0]["content"]; gold=int(conv[1]["content"]); kind="mul" if " x " in q else "add"
    setup_seed(args.seedbase+i)
    text=tok.apply_chat_template([{"role":"user","content":q}],tokenize=False,add_generation_prompt=True,open_thinking=False)
    e=tok(text,return_tensors="pt",truncation=True).to("cuda")
    with torch.no_grad():
        g=m.generate(inputs=e["input_ids"],attention_mask=e["attention_mask"],max_new_tokens=32,do_sample=args.temp>0,
                     temperature=max(args.temp,1e-4),top_p=0.95,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    r=tok.decode(g[0][len(e["input_ids"][0]):],skip_special_tokens=True); pred=last_num(r)
    stat[kind][1]+=1
    if pred==gold: stat[kind][0]+=1
    if i<5: ex.append(f"[{kind} gold={gold} pred={pred}] {r.strip()[:40]}")
tot_ok=sum(v[0] for v in stat.values()); tot=sum(v[1] for v in stat.values())
print(f"weight={args.weight} temp={args.temp}")
for k,(ok,n) in stat.items():
    if n>0: print(f"  {k}: {ok}/{n} = {ok/n:.3f}")
print(f"  TOTAL: {tot_ok}/{tot} = {tot_ok/tot:.3f}")
for s in ex: print("   ",s)