import sys, os, re, json, math, random, copy, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("trainer"))
import torch.nn.functional as F
from contextlib import nullcontext
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed
from rollout_engine import TorchRolloutEngine

device="cuda:0"; autocast_ctx=torch.cuda.amp.autocast(dtype=torch.bfloat16)
STEPS,B,G,LR,BETA,MAXGEN = 300,8,4,4e-6,0.08,16
setup_seed(42)
tok=AutoTokenizer.from_pretrained("model")

def load(w):
    m=MiniMindForCausalLM(MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=0))
    m.load_state_dict(torch.load(f"./out/{w}_768.pth",map_location="cuda"),strict=True)
    return m.to(device)
def last_num(s):
    z=re.findall(r"\d+",s.replace(",","")); return int(z[-1]) if z else None
def chat(q):
    return tok.apply_chat_template([{"role":"user","content":q}],tokenize=False,add_generation_prompt=True,open_thinking=False)

model=load("arith_sft").train()
ref=load("arith_sft").eval().requires_grad_(False)
engine=TorchRolloutEngine(model,tok,device,autocast_ctx)
opt=torch.optim.AdamW(model.parameters(),lr=LR)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=STEPS,eta_min=LR/10)

pool=[]
for line in open("dataset/arith_sft.jsonl",encoding="utf-8"):
    c=json.loads(line)["conversations"]; pool.append((c[0]["content"],int(c[1]["content"])))
eval_set=[]
for line in open("dataset/arith_eval.jsonl",encoding="utf-8"):
    c=json.loads(line)["conversations"]; eval_set.append((c[0]["content"],int(c[1]["content"])))
eval_set=eval_set[:60]
order=list(range(len(pool))); random.shuffle(order); cursor=0
log=[]

@torch.no_grad()
def eval_acc():
    model.eval(); ok=0
    for i,(q,g) in enumerate(eval_set):
        setup_seed(7000+i)
        e=tok(chat(q),return_tensors="pt").to(device)
        o=model.generate(inputs=e["input_ids"],attention_mask=e["attention_mask"],max_new_tokens=16,do_sample=False,
                         pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        r=tok.decode(o[0][len(e["input_ids"][0]):],skip_special_tokens=True)
        if last_num(r)==g: ok+=1
    model.train(); return ok/len(eval_set)

for step in range(1,STEPS+1):
    if cursor+B>len(order): random.shuffle(order); cursor=0
    idxs=order[cursor:cursor+B]; cursor+=B
    prompts=[chat(pool[i][0]) for i in idxs]; golds=[pool[i][1] for i in idxs]
    pi=tok(prompts,return_tensors="pt",padding=True,padding_side="left",add_special_tokens=False,return_token_type_ids=False).to(device)
    rr=engine.rollout(pi["input_ids"],pi["attention_mask"],G,MAXGEN,0.8)
    outputs,cids,completions,old_lp,plens,cmask0=rr.output_ids,rr.completion_ids,rr.completions,rr.per_token_logps,rr.prompt_lens,rr.completion_mask
    gold_rep=torch.tensor(golds).repeat_interleave(G)
    rew=torch.tensor([1.0 if last_num(completions[i])==gold_rep[i].item() else 0.0 for i in range(len(completions))],device=device)
    full=(outputs!=tok.pad_token_id).long(); R=cids.size(1)
    lpos=plens.unsqueeze(1)-1+torch.arange(R,device=device).unsqueeze(0)
    full.scatter_(1,lpos+1,cmask0.to(full.dtype))
    with autocast_ctx:
        lg=model(outputs,attention_mask=full).logits[:,:-1,:]
        plp=F.log_softmax(lg,-1).gather(2,outputs[:,1:].unsqueeze(-1)).squeeze(-1).gather(1,lpos)
    with torch.no_grad(),autocast_ctx:
        rlg=ref(outputs,attention_mask=full).logits[:,:-1,:]
        rlp=F.log_softmax(rlg,-1).gather(2,outputs[:,1:].unsqueeze(-1)).squeeze(-1).gather(1,lpos)
    gr=rew.view(B,G); mean=gr.mean(1).repeat_interleave(G); std=gr.std(1,unbiased=False).repeat_interleave(G)
    adv=((rew-mean)/(std+1e-4))
    cpm=cmask0.bool(); iseos=(cids==tok.eos_token_id)&cpm
    eidx=torch.full((iseos.size(0),),R-1,dtype=torch.long,device=device)
    anye=iseos.any(1); eidx[anye]=iseos.int().argmax(1)[anye]
    cmask=((torch.arange(R,device=device).expand(iseos.size(0),-1)<=eidx.unsqueeze(1))&cpm).int()
    klv=rlp-plp; pkl=torch.exp(klv)-klv-1; ratio=torch.exp(plp-old_lp)
    cr=torch.clamp(ratio,max=5.0).detach()
    ploss=-(cr*adv.unsqueeze(1)*plp-BETA*pkl)
    loss=(ploss*cmask).sum(1)/cmask.sum(1).clamp(min=1); loss=loss.mean()
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step(); opt.zero_grad()
    if step%10==0 or step==1:
        klref=((rlp-plp)*cmask).sum().item()/max(cmask.sum().item(),1)
        line=f"step {step:3d} | rollout_acc={rew.mean().item():.3f} adv_std={adv.std().item():.3f} KL={klref:+.4f} loss={loss.item():.4f} lr={opt.param_groups[0]['lr']:.2e}"
        print(line,flush=True); log.append(line)
    if step%25==0 or step==STEPS:
        a=eval_acc(); line=f"   >> held-out greedy acc @step{step} = {a:.3f}"; print(line,flush=True); log.append(line); engine.update_policy(model)

torch.save({k:v.half().cpu() for k,v in model.state_dict().items()},"./out/arith_grpo_768.pth")
open("experiments/grpo/arith_grpo_train.log","w",encoding="utf-8").write("\n".join(log))
print("SAVED arith_grpo_768.pth")