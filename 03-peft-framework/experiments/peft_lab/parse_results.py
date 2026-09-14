# -*- coding: utf-8 -*-
import re, os, csv, subprocess
LAB="/root/autodl-tmp/workspace/peft_lab"
methods=["full","lora","qlora"]
def grab(pat, txt, cast=float, default="NA"):
    m=re.search(pat, txt)
    try: return cast(m.group(1)) if m else default
    except Exception: return default
rows=[]
for name in methods:
    log=open(os.path.join(LAB,"logs",name+".log"),encoding="utf-8",errors="ignore").read()
    tr=re.search(r"trainable params:\s*([\d,]+)\s*\|\|\s*all params:\s*([\d,]+)\s*\|\|\s*trainable%:\s*([\d.]+)", log)
    trainable=int(tr.group(1).replace(",","")) if tr else "NA"
    allp=int(tr.group(2).replace(",","")) if tr else "NA"
    ratio=float(tr.group(3)) if tr else "NA"
    train_loss=grab(r"train_loss.?[:=]\s*([\d.]+)", log)
    eval_loss=grab(r"eval_loss.?[:=]\s*([\d.]+)", log)
    runtime=grab(r"train_runtime.?[:=]\s*([\d.]+)", log)
    wall=grab(r"WALL_SEC=(\d+)", log, int)
    gpu="NA"
    gp=os.path.join(LAB,"logs",name+"_gpu.csv")
    if os.path.exists(gp):
        vals=[int(x) for x in re.findall(r"\d+", open(gp).read())]
        if vals: gpu=max(vals)
    ckpt="NA"
    try:
        out=subprocess.run(["du","-sm",os.path.join(LAB,"runs",name)],capture_output=True,text=True).stdout
        ckpt=int(out.split()[0]) if out.strip() else "NA"
    except Exception: pass
    train_min=round(runtime/60,2) if isinstance(runtime,float) else ("NA" if wall=="NA" else round(wall/60,2))
    rows.append([name,trainable,ratio,gpu,train_min,train_loss,eval_loss,ckpt,
                 f"all_params={allp};Qwen2.5-0.5B;global_bs=16;cutoff512;1ep;seed42"])
out_csv=os.path.join(LAB,"peft_compare_result.csv")
with open(out_csv,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["method","trainable_params","param_ratio_pct","gpu_mem_peak_MB","train_min","train_loss","eval_loss","ckpt_MB","notes"])
    w.writerows(rows)
print(open(out_csv,encoding="utf-8").read())