#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/llmfab
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN TORCH_CUDA_ARCH_LIST="8.6"
LAB=/root/autodl-tmp/workspace/dist_lab
YAML=$1; TAG=$2; NPROC=$3
mkdir -p $LAB/logs $LAB/runs
[ -f $LAB/logs/gpu_topology.txt ] || nvidia-smi -L > $LAB/logs/gpu_topology.txt 2>&1
GPU=$LAB/logs/${TAG}_gpu.csv; LOG=$LAB/logs/${TAG}.log
rm -f $GPU
( for i in $(seq 1 7200); do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> $GPU; sleep 1; done ) &
MON=$!
S=$(date +%s)
/root/autodl-tmp/envs/llmfab/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NPROC $LAB/run_tuner.py $LAB/$YAML > $LOG 2>&1
RC=$?
E=$(date +%s); kill $MON 2>/dev/null
echo "WALL_SEC=$((E-S)) RC=$RC NPROC=$NPROC YAML=$YAML" | tee $LAB/logs/${TAG}_stat.txt
