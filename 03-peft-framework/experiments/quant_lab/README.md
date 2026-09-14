# 量化推理对照：bf16 vs NF4-4bit（体积 / 困惑度 / 延迟实测）

本实验包用 transformers + bitsandbytes，对同一个 Qwen2.5-0.5B 分别以 **bf16 原始精度**和 **NF4 4bit 在线量化**加载，在完全相同的评测样本与生成设置下，测量**模型显存占用、困惑度 PPL、首 Token 延迟、每 Token 延迟与吞吐**，用真实数字回答"4bit 到底省多少、掉多少、快还是慢"。

## 1. 实验要回答的问题

- 4bit 量化后模型在显存里到底占多少？为什么不是简单地除以 4？
- 量化会让模型"变笨"多少？用什么客观指标衡量（这里用困惑度 PPL）？
- 4bit 推理一定更快吗？在什么条件下才会更快？
- "权重量化"和"推理引擎（vLLM）"分别解决什么问题？

## 2. 测量口径（保证两种精度可比）

- **同一模型**：Qwen2.5-0.5B；bf16 用 `torch_dtype=bfloat16`，4bit 用 bitsandbytes 的 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16)`，即 **NF4 + 双重量化 + 计算时反量化回 bf16**。
- **模型显存 model_size_MB**：模型加载完成、尚未生成时 `torch.cuda.memory_allocated()` 的常驻值（纯权重 + 量化常数）。
- **困惑度 PPL**：取 peft_lab 的 100 条中文评测样本，用 Qwen 对话模板拼好 prompt+response，把 prompt 部分 label 置 -100，**只对 response 的 token** 计算交叉熵，按 token 数加权平均得到全局 NLL，再 `exp(NLL)=PPL`。PPL 越低越好，两种精度用同一批样本。
- **延迟/吞吐**：固定 20 条 prompt、贪心解码（`do_sample=false`，结果确定）、`max_new_tokens=64`，前 3 条作 warmup 不计；
  - **TTFT 首 Token 延迟**：单独生成 1 个 token 的耗时（prefill + 解码第一个）；
  - **TPOT 每 Token 延迟**：（完整生成耗时 − TTFT）/（新生成 token 数 − 1）；
  - **吞吐**：新生成 token 数 / 总耗时。每次计时前后都 `cuda.synchronize()`，避免异步误差。
- **并发 concurrency=1**：transformers 原生逐条生成，没有连续批处理；高并发是 vLLM 那类引擎解决的问题，不在本实验范围。

## 3. 目录结构

```
quant_lab/
├── README.md            # 本文档
├── quant_bench.py       # 两种精度依次加载，测显存/PPL/延迟并写出 quant_result.json
├── run_quant.sh         # 激活环境 + 离线变量 + 运行基准
├── quant_bench.log      # 运行日志
└── quant_result.json    # 结构化实测结果
```

## 4. 复现步骤

```bash
conda activate <你的环境>          # 需 torch / transformers / bitsandbytes
bash run_quant.sh                 # 或：python quant_bench.py
# 结果打印到屏幕并写入 quant_result.json
```

脚本会先完整跑一遍 bf16、卸载并清空显存后再加载 NF4，避免两份模型互相干扰。

## 5. 实测结果（RTX 3080 Ti 12G，Qwen2.5-0.5B）

| 精度 | 模型显存 | PPL（↓越好） | TTFT | TPOT | 吞吐 | 并发 |
| --- | --- | --- | --- | --- | --- | --- |
| bf16（16bit） | 942 MB | 7.1544 | 27.21 ms | 25.20 ms | 39.61 tok/s | 1 |
| NF4（4bit） | 444 MB | 8.0397 | 61.20 ms | 47.23 ms | 21.39 tok/s | 1 |

## 6. 结果怎么读

1. **显存：942 MB → 444 MB，压到约 47%**。0.5B 参数若纯按 16bit→4bit 应降到 1/4，但实际约一半，原因是：归一化常数等量化元数据仍占空间、部分层（如 lm_head）默认不量化、还有固定的 CUDA/框架开销。模型越大，纯权重占比越高，压缩比会越接近理论的 1/4。
2. **困惑度：7.15 → 8.04，上升约 12%**。这就是 4bit 的精度代价——权重被压到 16 个离散档位，输出分布略有偏移。0.5B 小模型对量化更敏感，越大的模型通常越"抗量化"。注意 PPL 只反映语言建模层面的偏移，不等于主观对话质量。
3. **延迟：4bit 反而更慢（吞吐 39.6 → 21.4 tok/s）**。这是本实验最反直觉、也最有价值的一点：NF4 在每次前向时都要把 4bit 权重**反量化回 bf16 再算**，多出一笔开销；而 0.5B 模型权重极小，3080 Ti 的显存带宽根本不是瓶颈，省出的带宽换不来收益，反量化开销却实打实增加了。**4bit 的真正价值是"让原本放不下的模型放得下 / 省出显存开更大 batch 或更长上下文"，而不是在小模型上提速。** 当模型大到显存带宽成为瓶颈、或要靠省出的显存提高并发时，量化才会同时带来吞吐收益。

## 7. 注意点与踩坑

- 两次加载之间必须 `del model + gc.collect() + torch.cuda.empty_cache()`，并 `reset_peak_memory_stats()`，否则前一份模型的残留会污染显存读数。
- 日志里出现 "attention mask is not set" 的 warning 是因为 pad/eos 相同；本基准逐条单句推理不影响结果，工程化使用时应显式传 `attention_mask`。
- 设备映射用 `device_map={"":0}` 显式放卡；4bit 模型在 `no_grad` 下做 PPL/生成即可，只有 QLoRA 训练才需要为 LoRA 参数开梯度。
- **量化 ≠ 推理引擎**：本实验只换权重精度，仍是 HuggingFace 逐 token 解码；vLLM 的 PagedAttention、连续批处理解决的是调度与 KV Cache 显存碎片，二者正交、可叠加。

## 8. 局限性（如何正确看待这些数字）

- 仅在单张 3080 Ti、单个 0.5B 小模型上测量，**"4bit 更慢"是该场景下的结论**，不能外推到大模型/带宽受限场景；换 7B/13B 往往是另一番结果。
- GPTQ、AWQ 这类**离线**权重量化（需校准集）以及 vLLM 引擎对照未实测，`quant_infer.csv` 中对应行如实留空，未实测不臆造。
- PPL 基于 100 条样本、固定模板，数值会随评测集变化；它用于同口径横向比较，不代表绝对质量。
- 未测多并发（transformers 原生不支持连续批处理），并发>1 的服务化数字需要 vLLM/SGLang 等引擎补测。