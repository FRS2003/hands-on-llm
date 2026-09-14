#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/llmfab
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/workspace/quant_lab
python quant_bench.py 2>&1 | tee quant_bench.log
echo QUANT_RC=${PIPESTATUS[0]}