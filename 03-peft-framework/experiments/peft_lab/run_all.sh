#!/bin/bash
# Run full -> lora -> qlora in sequence, then parse to csv
cd /root/autodl-tmp/workspace/peft_lab
for n in full lora qlora; do
  bash run_one.sh $n
done
source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/autodl-tmp/envs/llmfab
python parse_results.py