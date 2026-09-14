#!/bin/bash
# Usage: bash run_one.sh {full|lora|qlora}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/llmfab
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
LAB=/root/autodl-tmp/workspace/peft_lab
NAME=$1
mkdir -p $LAB/logs $LAB/runs
GPU=$LAB/logs/${NAME}_gpu.csv
LOG=$LAB/logs/${NAME}.log
echo "[$(date)] START $NAME"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 1; done > $GPU ) &
SMI=$!
T0=$(date +%s)
llamafactory-cli train $LAB/yamls/peft_${NAME}.yaml > $LOG 2>&1
RC=$?
T1=$(date +%s)
kill $SMI 2>/dev/null
echo "WALL_SEC=$((T1-T0)) RC=$RC NAME=$NAME" >> $LOG
echo "[$(date)] DONE $NAME wall=$((T1-T0))s rc=$RC"