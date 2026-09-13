import json, random
random.seed(20260913)
def mk(a,b):
    return {"conversations":[
        {"role":"user","content":f"请口算：{a} + {b} = ? 直接给出最终数字，不要写过程。"},
        {"role":"assistant","content":str(a+b)}]}
with open("dataset/arith_sft.jsonl","w",encoding="utf-8") as f:
    for _ in range(4000):
        a,b=random.randint(1,9),random.randint(1,9); f.write(json.dumps(mk(a,b),ensure_ascii=False)+"\n")
with open("dataset/arith_eval.jsonl","w",encoding="utf-8") as f:
    for _ in range(120):
        a,b=random.randint(1,9),random.randint(1,9); f.write(json.dumps(mk(a,b),ensure_ascii=False)+"\n")
print("train=4000 eval=120 range[1,9] sample:",mk(3,5)["conversations"])