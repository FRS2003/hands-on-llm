#!/bin/bash
# Usage: bash train_run.sh [small|medium]
set -e
SCALE=${1:-medium}
case $SCALE in
  small)  PRE=pretrain_sub.jsonl;  SFT=sft_sub.jsonl;  TAG=small ;;
  medium) PRE=pretrain_med.jsonl;  SFT=sft_med.jsonl;  TAG=med ;;
  *) echo "unknown scale $SCALE"; exit 1 ;;
esac
source /root/miniconda3/etc/profile.d/conda.sh && conda activate minimind
cd "$(dirname "$0")/trainer"
mkdir -p ../out
( while true; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)"; sleep 5; done ) > ../out/gpu_${TAG}.csv &
MON=$!
echo "=== PRETRAIN START $(date '+%F %T') ==="; T0=$(date +%s)
python -u train_pretrain.py --data_path ../dataset/$PRE --epochs 2 --save_weight pretrain        --log_interval 100 --save_interval 100000 --num_workers 4
T1=$(date +%s); echo "PRETRAIN_SECONDS=$((T1-T0))"
echo "=== SFT START $(date '+%F %T') ==="
python -u train_full_sft.py --data_path ../dataset/$SFT --epochs 2 --from_weight pretrain        --save_weight full_sft --log_interval 100 --save_interval 100000 --num_workers 4
T2=$(date +%s); echo "SFT_SECONDS=$((T2-T1))"
kill $MON 2>/dev/null
echo "ALL_DONE $SCALE $(date '+%F %T')"
