#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/llmfab
export DS_ACCELERATOR=cpu
echo "=== deepspeed 版本/位置 ==="
pip show deepspeed | grep -E "Version|Location"
echo "=== CPU 模式下验证 import 链 ==="
python - <<'PY'
import deepspeed
print("deepspeed", deepspeed.__version__)
import llamafactory.train.tuner
print("llamafactory.train.tuner OK")
PY
echo "=== nvcc / CUDA toolkit ==="
which nvcc && nvcc --version | tail -2
echo "=== ds_report 摘要 ==="
ds_report 2>&1 | sed -n '1,30p'
echo "=== 预编译融合优化器算子(只需 nvcc, 不需 GPU) ==="
python - <<'PY'
for name,builder in [("FusedAdam","FusedAdamBuilder")]:
    try:
        import deepspeed.ops.op_builder as ob
        ext=getattr(ob,builder)().load(); print(name,"op BUILT OK")
    except Exception as e:
        print(name,"build note:",repr(e)[:200])
PY