#!/usr/bin/env python3
"""Parse MiniMind trainer log (carriage-return refreshed) into per-step loss CSV + metrics.
Usage: python parse_logs.py train.log gpu.csv out_dir
"""
import re, os, sys

def parse(train_log, gpu_csv, outdir):
    os.makedirs(outdir, exist_ok=True)
    text = open(train_log, encoding="utf-8", errors="ignore").read().replace("\r", "\n")
    stage, rows, sec = None, {"pretrain": [], "sft": []}, {}
    pat = re.compile(r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\), loss: ([\d.]+).*?lr: ([\d.eE+-]+)")
    for line in text.splitlines():
        if "PRETRAIN" in line and "START" in line: stage = "pretrain"
        elif "SFT" in line and "START" in line: stage = "sft"
        m = pat.search(line)
        if m and stage:
            e, E, s, T, loss, lr = m.groups(); rows[stage].append((e, s, T, loss, lr))
        m2 = re.search(r"(PRETRAIN|SFT)_SECONDS=(\d+)", line)
        if m2: sec[m2.group(1).lower()] = int(m2.group(2))
    for st, data in rows.items():
        with open(f"{outdir}/{st}_curve.csv", "w", encoding="utf-8") as f:
            f.write("epoch,step,total_steps,loss,lr\n")
            for e, s, T, loss, lr in data: f.write(f"{e},{s},{T},{loss},{lr}\n")
    gpu = {}
    if os.path.exists(gpu_csv):
        mem, util, temp = [], [], []
        for line in open(gpu_csv):
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 4:
                try: mem.append(int(p[1])); util.append(int(p[2])); temp.append(int(p[3]))
                except ValueError: pass
        if mem: gpu = dict(mem_peak_MB=max(mem), util_mean_pct=round(sum(util)/len(util), 1),
                          temp_peak_C=max(temp), samples=len(mem))
    with open(f"{outdir}/metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"seconds={sec}\n")
        for st, d in rows.items():
            if d: f.write(f"{st}: points={len(d)} loss_first={float(d[0][3]):.4f} loss_last={float(d[-1][3]):.4f}\n")
        f.write(f"gpu={gpu}\n")
    print("seconds", sec, "gpu", gpu)

if __name__ == "__main__":
    parse(sys.argv[1], sys.argv[2], sys.argv[3])
