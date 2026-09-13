# experiments · 对照实验记录表（自测模板）

## 这些表是什么

四张 csv 是**留给你自己跑实验填的空白记录表**，不是已测得的 benchmark：

| 文件 | 对照维度 | 关键列 |
| --- | --- | --- |
| `alignment_compare.csv` | SFT/DPO/PPO/KTO/ORPO/SimPO | 是否要 RM/ref、数据类型、显存、耗时、最终 loss、胜率、稳定性 |
| `peft_compare.csv` | Full/Freeze/LoRA/QLoRA/Adapter/Prefix/P-Tuning/DoRA | 可训参数、参数占比、显存、耗时、评测分、推理是否带额外结构 |
| `distributed_compare.csv` | DDP / ZeRO-1/2/3 / TP / PP | 每卡显存、吞吐 tok/s、通信占比 |
| `quant_infer.csv` | FP16/GPTQ/AWQ/vLLM | 模型体积、PPL、首 Token 延迟、TPOT、吞吐、并发 |

## 为什么留空而不是填"参考数字"

显存、延迟、吞吐**强依赖**模型规模、序列长度、batch、显卡型号和驱动版本，直接抄网上的数字既不可复现也容易误导。正确做法是在**你自己的环境**里按统一口径实测填入——这也正是"对照实验"的价值。方法原理与定性选型结论见 [`../notes/peft-distributed-quant.md`](../notes/peft-distributed-quant.md) 与 [`../notes/alignment.md`](../notes/alignment.md)。

## 填表方法（保证可比）

1. **一次只改一个变量**，其余超参/数据/步数/随机种子尽量固定；
2. 每组都记录配置文件、启动命令、框架日志与 `nvidia-smi` 峰值显存；
3. 显存统一取训练全程峰值（MB），耗时统一取纯训练时间（排除首次加载/编译）；
4. 推理指标固定输入/输出长度与并发，TTFT、TPOT、吞吐的定义见选型笔记第三部分；
5. 每组至少跑 2 次取均值，异常值（后台占用、热降频）要标注。

## 推荐最小对照路径（配合 configs 模板）

1. `sft_full` vs `sft_lora` vs `sft_qlora`：填 `peft_compare.csv`，直观看到参数与显存差异；
2. 同一 SFT 起点切 `dpo/kto/orpo`：填 `alignment_compare.csv`；
3. 单卡 vs `torchrun` 两卡 + `ds_zero2.json`：填 `distributed_compare.csv`；
4. FP16 vs GPTQ/AWQ，再用 vLLM 起服务：填 `quant_infer.csv`。

> 每次实验建议在本目录另存 `logs/`（原始日志）与 `curves/`（plot_loss 生成的曲线图），csv 里只填汇总值，做到每个数字都能回溯到日志。