#!/bin/bash
# Usage: bash run_dpo.sh <yaml_path> <tag>
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/llmfab
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LAB=/root/autodl-tmp/workspace/align_lab
YAML=$1; NAME=$2
mkdir -p $LAB/logs $LAB/runs
GPU=$LAB/logs/${NAME}_gpu.csv; LOG=$LAB/logs/${NAME}.log
rm -f $GPU
( for i in $(seq 1 7200); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $GPU; sleep 1; done ) &
MON=$!
S=$(date +%s)
llamafactory-cli train $YAML > $LOG 2>&1
RC=$?
E=$(date +%s); kill $MON 2>/dev/null
echo "WALL_SEC=$((E-S)) RC=$RC" | tee $LAB/logs/${NAME}_stat.txt