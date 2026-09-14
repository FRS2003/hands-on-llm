import json, time, gc, os, statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL="/root/autodl-tmp/models/ms_cache/Qwen/Qwen2___5-0___5B"
EVAL ="/root/autodl-tmp/workspace/peft_lab/data/zh_instruct_eval.json"
OUT  ="/root/autodl-tmp/workspace/quant_lab"
os.makedirs(OUT, exist_ok=True)
tok=AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

def user_msg(ex):
    c=ex["instruction"] + (("\n"+ex["input"]) if ex.get("input") else "")
    return [{"role":"user","content":c}]

def build(kind):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if kind=="bf16":
        kw=dict(torch_dtype=torch.bfloat16)
    else:
        bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw=dict(quantization_config=bnb)
    m=AutoModelForCausalLM.from_pretrained(MODEL, device_map={"":0}, **kw)
    m.eval(); return m

@torch.no_grad()
def ppl_on_responses(m, data, maxn=100, cutoff=512):
    ws=0.0; ntok=0
    for ex in data[:maxn]:
        full=tok.apply_chat_template(user_msg(ex)+[{"role":"assistant","content":ex["output"]}], tokenize=False)
        prompt=tok.apply_chat_template(user_msg(ex), tokenize=False, add_generation_prompt=True)
        fid=tok(full,return_tensors="pt",truncation=True,max_length=cutoff).input_ids.cuda()
        pid=tok(prompt,return_tensors="pt",truncation=True,max_length=cutoff).input_ids.cuda()
        labels=fid.clone(); labels[:,:pid.shape[1]]=-100
        out=m(input_ids=fid,labels=labels)
        n=(labels[0]!=-100).sum().item(); ws+=out.loss.item()*n; ntok+=n
    avg=ws/ntok
    return avg, torch.tensor(avg).exp().item()

@torch.no_grad()
def gen_bench(m, prompts, warm=3, newt=64):
    tt=[];tp=[];th=[]
    for i,p in enumerate(prompts):
        ids=tok(p,return_tensors="pt").input_ids.cuda()
        torch.cuda.synchronize(); t=time.time()
        m.generate(ids,max_new_tokens=1,do_sample=False,pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize(); ttft=(time.time()-t)*1000
        t=time.time()
        y=m.generate(ids,max_new_tokens=newt,do_sample=False,pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize(); tot=time.time()-t
        ngen=max(y.shape[1]-ids.shape[1],1)
        if i>=warm:
            tt.append(ttft); tp.append((tot-ttft/1000)/max(ngen-1,1)*1000); th.append(ngen/tot)
    return dict(ttft_ms=round(statistics.mean(tt),2), tpot_ms=round(statistics.mean(tp),2),
                throughput_tok_s=round(statistics.mean(th),2), concurrency=1, n_timed=len(tt))

def main():
    data=json.load(open(EVAL,encoding="utf-8"))
    prompts=[tok.apply_chat_template(user_msg(e),tokenize=False,add_generation_prompt=True) for e in data[:23]]
    res={}
    for kind in ["bf16","nf4"]:
        m=build(kind)
        base=torch.cuda.memory_allocated()/1024**2
        nll,ppl=ppl_on_responses(m,data,100)
        torch.cuda.reset_peak_memory_stats()
        g=gen_bench(m,prompts)
        peak=torch.cuda.max_memory_allocated()/1024**2
        res[kind]=dict(bits=(16 if kind=="bf16" else 4), model_mem_MB=round(base),
                       gen_peak_MB=round(peak), avg_nll=round(nll,4), ppl=round(ppl,4), **g)
        print(kind, json.dumps(res[kind],ensure_ascii=False))
        del m
    json.dump(res,open(f"{OUT}/quant_result.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print("SAVED", f"{OUT}/quant_result.json")

if __name__=="__main__":
    main()