cd /root/autodl-tmp/workspace/dist_lab
bash run_dp.sh yamls/z1_15.yaml z1 2
sleep 6
bash run_dp.sh yamls/z2_15.yaml z2 2
sleep 6
bash run_dp.sh yamls/z3_15.yaml z3 2
echo Z_DONE
