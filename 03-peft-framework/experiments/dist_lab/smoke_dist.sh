#!/bin/bash
# 双卡开机后先跑这个：每组只 2 步，快速验证 torchrun/NCCL/DeepSpeed 链路是否打通(约1-2分钟)
cd /root/autodl-tmp/workspace/dist_lab
echo "### 0) 卡拓扑(应看到 2 张同型号卡)"; nvidia-smi -L
echo "### 1) 0.5B 两卡 DDP 冒烟(2步)"
bash run_dp.sh yamls/ddp_0.5b.yaml smoke_ddp 2 4 "" "--max_steps 2"
grep -E "train_runtime|Error|error|Traceback|OutOfMemory" logs/smoke_ddp.log | tail -5; cat logs/smoke_ddp_stat.txt
echo "### 2) 1.5B 两卡 ZeRO-3 冒烟(2步)"
bash run_dp.sh yamls/zero_1.5b.yaml smoke_z3 2 8 ds_config/ds_zero3.json "--max_steps 2"
grep -E "train_runtime|Error|error|Traceback|OutOfMemory" logs/smoke_z3.log | tail -8; cat logs/smoke_z3_stat.txt
echo "SMOKE DONE"