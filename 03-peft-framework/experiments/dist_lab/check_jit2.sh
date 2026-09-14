#!/bin/bash
SP=/root/autodl-tmp/envs/llmfab/lib/python3.10/site-packages
echo "=== partition_parameters.py 里 op_builder 出现的上下文(判断是否 import 即编译) ==="
grep -n "op_builder\|Builder()\|\.load(" $SP/deepspeed/runtime/zero/partition_parameters.py | head -20
echo "=== 这些 builder 是否在函数内条件触发(看 AsyncIOBuilder 出现位置) ==="
grep -rn "AsyncIOBuilder\|InspectDeepSpeedComms\|class .*Builder" $SP/deepspeed/runtime/zero/partition_parameters.py | head