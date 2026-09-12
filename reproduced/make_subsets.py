#!/usr/bin/env python3
"""Evenly sample a large jsonl into a smaller subset (covers whole-file distribution)."""
import os

D = "dataset"
# (source, destination, total_lines, target_lines) -- update total via `wc -l`
JOBS = [
    ("pretrain_t2t_mini.jsonl", "pretrain_sub.jsonl", 1_270_238, 60_000),
    ("sft_t2t_mini.jsonl", "sft_sub.jsonl", 905_718, 20_000),
    ("pretrain_t2t_mini.jsonl", "pretrain_med.jsonl", 1_270_238, 150_000),
    ("sft_t2t_mini.jsonl", "sft_med.jsonl", 905_718, 50_000),
]

def main():
    for src, dst, total, target in JOBS:
        step = max(1, total // target)
        kept = 0
        with open(os.path.join(D, src), encoding="utf-8") as fi,              open(os.path.join(D, dst), "w", encoding="utf-8") as fo:
            for i, line in enumerate(fi):
                if i % step == 0 and kept < target:
                    fo.write(line); kept += 1
        print(f"WROTE {dst}: {kept} rows (every {step})")

if __name__ == "__main__":
    main()
