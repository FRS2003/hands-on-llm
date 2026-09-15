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

## 已完成：`kernel_rmsnorm.py`（Triton 融合 GPU Kernel）

用 OpenAI Triton 手写 RMSNorm 的**前向 + 反向融合 Kernel**，与 `model.RMSNorm` 同接口、可直接替换进网络：

| 组成 | 实现要点 |
| --- | --- |
| 前向 Kernel | 一个 program 处理一行，两遍扫描：分块归约 `sum(x^2)` 求 1/rms（fp32 累加）→ 融合「归一化 × gamma」写回，减少 HBM 往返 |
| 反向 Kernel | 解析梯度 `dx=(dy·γ - x̂·mean(dy·γ·x̂))/rms`、`dγ=Σ(dy·x̂)`（atomic 跨行归约），封装为 `autograd.Function`，可直接训练反传 |
| 通用性 | `BLOCK_D=next_power_of_2(D)` + mask + 循环分块，支持非 2 次幂 hidden 维度 |

- **数值自检 `python test_triton_kernel.py`（需 GPU）：10 项全过**——fp32/bf16 前向对拍 PyTorch、γ=1 时 RMS≈1、反向 `dx/dγ` 对齐、非 2 次幂维度（1000/2049）正确；无 GPU/无 triton 时自动 SKIP（CI 不报错）。
- **性能 `python bench.py`（RTX 3080 Ti, bf16）**：访存受限的大张量（M=8192,D=4096）融合 Kernel 约 **2.8×** 于「pow→mean→rsqrt→mul」的多 kernel 原生写法；小张量（D=1024）因 launch 开销约 0.85×——说明**算子融合收益随张量变大、访存占比升高而显现**，不盲目套融合。

## 待办
- `tokenizer_bpe.py`：BPE 分词器（merge 规则、编解码）
- `flash_attention_tile.py`：分块 online-softmax（Flash Attention 核心）
