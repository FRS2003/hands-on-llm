# 架构组件推导（待补充）

## RMSNorm vs LayerNorm
- 公式 / 为什么去掉减均值 / 反向梯度

## RoPE
- 二维旋转矩阵 -> 为什么内积只依赖相对位置；外推性；base 频率

## GQA / MHA / MQA
- KV head 数与 KV Cache 显存公式

## SwiGLU
- GLU 形式、SiLU、FFN 中间维 8/3 的由来

## Pre-Norm + 残差为什么更稳
