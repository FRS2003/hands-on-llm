# from_scratch（手写组件）

本目录放 **不看官方实现、自己独立写出** 的模块，是体现「从零实现」的核心。
要求：每个文件标注关键张量维度；写完后再与官方实现对照并记录差异。

## 已完成：`model.py`（纯原生 PyTorch，CPU 可跑）

| 组件 | 要点 | 自检（`python test_model.py`） |
| --- | --- | --- |
| `RMSNorm` | 去均值中心化/去偏置，`x/rms(x)*gamma`，复杂度 O(n) | 归一化后 RMS≈1、与公式一致 |
| `RoPE` | `precompute_rope` 频率表 + `apply_rope`(rotate-half) | **内积只依赖相对位置**、旋转保模长 |
| `SwiGLU` | `down(SiLU(gate)·up)`，中间维 2/3·D 取整 | 形状保持、门控分支生效 |
| `GQACausalSelfAttention` | `repeat_kv` 统一 MHA/GQA/MQA、因果 mask、可挂 RoPE | **改未来 token 不影响过去输出**、头数复制正确 |
| `DecoderBlock` | Pre-Norm + 双残差 | 形状保持、梯度可回传 |

共 **12 项数值断言全部通过**（无需 GPU/训练数据，直接验证数学性质）。
维度记号：`B=batch, T=seq_len, D=hidden, H=n_heads, Hkv=n_kv_heads, Dh=D//H`。

```bash
cd from_scratch && python test_model.py    # ALL 12 CHECKS PASSED
```

## 待办
- `tokenizer_bpe.py`：BPE 分词器（merge 规则、编解码）
- `kernel_rmsnorm.py`：Triton 版 RMSNorm Kernel
- `flash_attention_tile.py`：分块 online-softmax（Flash Attention 核心）
