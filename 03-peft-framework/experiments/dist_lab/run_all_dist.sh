#!/bin/bash
cd /root/autodl-tmp/workspace/dist_lab
bash run_dp.sh yamls/ddp1_05.yaml ddp1 1
bash run_dp.sh yamls/ddp2_05.yaml ddp2 2
bash run_dp.sh yamls/zddp_15.yaml z_ddp 2
bash run_dp.sh yamls/z1_15.yaml z1 2
bash run_dp.sh yamls/z2_15.yaml z2 2
bash run_dp.sh yamls/z3_15.yaml z3 2
echo "ALL DONE"; ls -l logs/*_stat.txt
