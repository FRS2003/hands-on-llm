#!/bin/bash
SP=/root/autodl-tmp/envs/llmfab/lib/python3.10/site-packages
echo "=== g++/gcc 工具链 ==="
which gcc g++ make 2>/dev/null; gcc --version | head -1
echo "=== DeepSpeed 引擎是否默认构建 fused 算子(只在显式要求时才 load?) ==="
grep -rn "FusedAdamBuilder\|fused_adam" $SP/deepspeed/runtime/engine.py | head -20
echo "--- zero 分片里是否硬依赖 fused ---"
grep -rln "op_builder" $SP/deepspeed/runtime/zero/ 2>/dev/null
echo "=== 数据盘关键资产占用(镜像大小预估) ==="
du -sh /root/autodl-tmp/envs /root/autodl-tmp/models /root/autodl-tmp/workspace 2>/dev/null
df -h /root/autodl-tmp | tail -1