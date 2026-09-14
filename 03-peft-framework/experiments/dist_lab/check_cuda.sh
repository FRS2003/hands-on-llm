#!/bin/bash
SP=/root/autodl-tmp/envs/llmfab/lib/python3.10/site-packages
echo "=== CUDA toolkit 目录 ==="
ls -d /usr/local/cuda* 2>/dev/null
ls -l /usr/local/cuda/bin/nvcc 2>/dev/null && /usr/local/cuda/bin/nvcc --version | tail -2
echo "=== torch 自带 CUDA 版本 ==="
source /root/miniconda3/etc/profile.d/conda.sh; conda activate /root/autodl-tmp/envs/llmfab
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
echo "=== deepspeed 自带预编译 .so 数量 ==="
find $SP/deepspeed -name "*.so" 2>/dev/null | wc -l
find $SP/deepspeed -name "*.so" 2>/dev/null | head -10
echo "=== triton / ninja 版本(编译依赖) ==="
pip list 2>/dev/null | grep -iE "triton|ninja|py-cpuinfo|hjson|pydantic"
echo "=== op_builder 目录(可 JIT 的算子清单) ==="
ls $SP/deepspeed/ops/op_builder/ 2>/dev/null