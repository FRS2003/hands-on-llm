#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/llmfab
echo "=== 安装 deepspeed(清华源) ==="
pip install -q deepspeed -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -5
echo "=== 模块验证(无卡模式 cuda=False 属正常) ==="
python - <<'PY'
import importlib
for m in ["deepspeed","accelerate","torch","llamafactory.train.tuner"]:
    try:
        mod=importlib.import_module(m); print("OK ",m,getattr(mod,"__version__",""))
    except Exception as e:
        print("FAIL",m,repr(e)[:160])
import torch
print("cuda_available:",torch.cuda.is_available(),"device_count:",torch.cuda.device_count())
PY
echo "=== 模型在位 ==="
ls -d /root/autodl-tmp/models/ms_cache/Qwen/Qwen2___5-0___5B /root/autodl-tmp/models/ms_cache/Qwen/Qwen2___5-1___5B 2>&1
echo "=== peft 数据在位(dist 复用) ==="
ls /root/autodl-tmp/workspace/peft_lab/data 2>&1
echo "=== dist_lab 文件 ==="
find /root/autodl-tmp/workspace/dist_lab -type f | sort
echo "=== 磁盘 ==="
df -h /root/autodl-tmp | tail -1