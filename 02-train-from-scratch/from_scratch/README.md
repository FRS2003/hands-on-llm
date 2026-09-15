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

## 已完成：`tokenizer_bpe.py`（字节级 BPE 分词器，纯标准库）

不依赖第三方库手写 **byte-level BPE**：以 0–255 单字节为初始词表（任何中文/生僻字/emoji 都能表示、**无 OOV**），在预分词切出的词块内部逐轮合并最高频相邻符号对，编码时按学到的 merge 顺序复现合并。

| 能力 | 实现要点 |
| --- | --- |
| 训练 | 预分词切词块 → 统计词频 → 加权统计相邻对 → 最高频优先合并（同频按符号对升序，结果可复现） |
| 编码 | 按 merge 优先级反复合并；未登录词自动回退到单字节，保证不抛 OOV |
| 解码 | id→字节片段拼接后整体 UTF-8 解码，多字节中文不乱码、与原文严格可逆 |
| 持久化 | 只存有序 merge 规则即可确定性重建词表（save/load），另预留 pad/bos/eos/unk |

- **自检 `python test_tokenizer_bpe.py`：20 项全过（纯 CPU、零第三方依赖）**——中英/标点/空白编解码可逆、首条 merge 恰为最高频对、BPE token 数少于原始字节数、未见词与 emoji 无损还原、merge 词表一致、训练确定性、save/load 一致、空串边界。
- 直观效果：演示语料上 `the mat` 从 7 个 UTF-8 字节压缩为 2 个 token，而训练时没见过的 `xyz / 生僻字 / 😀` 仍能编码并原样还原。运行 `python tokenizer_bpe.py` 可看完整演示。

## 待办
- `flash_attention_tile.py`：分块 online-softmax（Flash Attention 核心）
