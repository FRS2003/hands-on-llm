import json, random, os
from transformers import AutoTokenizer
TOK="/root/autodl-tmp/models/ms_cache/Qwen/Qwen2___5-0___5B"
SRC="/root/autodl-tmp/datasets/pref_raw/merged_dpo_zh_emoji.jsonl"
OUT="/root/autodl-tmp/workspace/align_lab/data"
os.makedirs(OUT, exist_ok=True)
tok=AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
CUT=1024; MARGIN=24
rows=[]; skip_empty=0; skip_long=0
lt=[]
for line in open(SRC, encoding="utf-8"):
    o=json.loads(line)
    q=(o.get("question") or "").strip()
    ch=(o.get("answer_zh") or "").strip()
    rj=(o.get("answer_en") or "").strip()
    if not(q and ch and rj) or len(ch)<5 or len(rj)<5:
        skip_empty+=1; continue
    n_ch=len(tok(q+ch)["input_ids"])+MARGIN
    n_rj=len(tok(q+rj)["input_ids"])+MARGIN
    if n_ch>CUT or n_rj>CUT:
        skip_long+=1; continue
    rows.append({"instruction":q,"input":"","chosen":ch,"rejected":rj})
    lt.append(max(n_ch,n_rj))
random.seed(42); random.shuffle(rows)
NE=150
ev=rows[:NE]; train=rows[NE:]
json.dump(train, open(f"{OUT}/pref_dpo_train.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(ev, open(f"{OUT}/pref_dpo_eval.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
info={
 "zh_pref_dpo":{"file_name":"pref_dpo_train.json","formatting":"alpaca","ranking":True,"columns":{"prompt":"instruction","query":"input","chosen":"chosen","rejected":"rejected"}},
 "zh_pref_dpo_eval":{"file_name":"pref_dpo_eval.json","formatting":"alpaca","ranking":True,"columns":{"prompt":"instruction","query":"input","chosen":"chosen","rejected":"rejected"}}
}
json.dump(info, open(f"{OUT}/dataset_info.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
lt.sort()
print("skip_empty",skip_empty,"skip_long",skip_long,"| kept",len(rows))
print("train",len(train),"eval",len(ev))
print("pair token-len p50/p90/max:", lt[len(lt)//2], lt[int(len(lt)*0.9)], lt[-1])