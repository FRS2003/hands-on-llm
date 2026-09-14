# -*- coding: utf-8 -*-
"""下载一份固定的中文指令数据集，切成可复现的 train/eval 子集（LLaMA-Factory alpaca 格式）。"""
import json, os, random
OUT="/root/autodl-tmp/workspace/peft_lab/data"
os.makedirs(OUT, exist_ok=True)
cands=["AI-ModelScope/alpaca-gpt4-data-zh","llm-wizard/alpaca-gpt4-data-zh","swift/alpaca-gpt4-data-zh"]
rows=None
from modelscope import MsDataset
for cid in cands:
    try:
        print("TRY", cid, flush=True)
        ds=MsDataset.load(cid, split="train")
        print("OK", cid, "size=", len(ds), "cols=", ds.column_names if hasattr(ds,"column_names") else "?", flush=True)
        rows=list(ds); src=cid; break
    except Exception as e:
        print("FAIL", cid, repr(e)[:160], flush=True)
if rows is None:
    raise SystemExit("所有候选数据集都失败，需要换源")
def norm(r):
    g=lambda *ks: next((str(r[k]) for k in ks if k in r and r[k] is not None), "")
    instr=g("instruction","input","question","prompt")
    inp=g("input","context")
    out=g("output","response","answer","text")
    return {"instruction":instr,"input":inp,"output":out}
data=[norm(r) for r in rows]
data=[d for d in data if d["instruction"] and d["output"]]
random.Random(42).shuffle(data)
train=data[:3000]; evals=data[3000:3300]
json.dump(train, open(os.path.join(OUT,"zh_instruct_train.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(evals, open(os.path.join(OUT,"zh_instruct_eval.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
info={"zh_instruct_3k":{"file_name":"zh_instruct_train.json","columns":{"prompt":"instruction","query":"input","response":"output"}},
      "zh_instruct_eval":{"file_name":"zh_instruct_eval.json","columns":{"prompt":"instruction","query":"input","response":"output"}}}
json.dump(info, open(os.path.join(OUT,"dataset_info.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
print("SRC=",src,"train=",len(train),"eval=",len(evals), flush=True)
print("SAMPLE=", json.dumps(train[0], ensure_ascii=False)[:200], flush=True)