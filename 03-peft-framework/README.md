# 03 · LLaMA-Factory Lab：工业级微调框架走读与选型

> 📚 **本模块属于 [Hands-on-LLM 动手学大模型](../README.md) 学习库 · 返回总览看完整学习路线**
> 以业界主流微调框架 [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)（Apache-2.0，ACL 2024）为对象，
> 目标不是"会抄命令"，而是**读懂源码主链路、会做方法选型、能套用模板与二次开发**。

## 一、这个模块怎么学

② 模块用原生 PyTorch **手写**了训练全链路，本模块换一个视角：看一个**工业框架**如何把同样的事情工程化，并横向对比各种可调方案。内容分四块：

1. **源码走读**：一份 yaml 从配置解析 → 模型加载 → 数据处理 → 训练器是怎么流动的；
2. **方法选型**：六种对齐算法、八种 PEFT、三种并行、量化与推理引擎分别怎么选；
3. **配置模板**：带逐行中文注释、复制即可改跑的 yaml / DeepSpeed 配置；
4. **二次开发**：自定义数据集注册、训练回调等"不改框架源码"的扩展示例。

## 二、学习笔记（`notes/` 与 `docs/`，均附 Word）

| 文档 | 讲什么 | Word |
| --- | --- | --- |
| [源码走读](notes/source_walkthrough.md) | 配置/模型/数据/训练器四层主链路与读码顺序 | [Word](notes/source_walkthrough.docx) |
| [对齐算法选型](notes/alignment.md) | SFT/DPO/KTO/ORPO/SimPO/PPO 的数据要求、要不要 ref/RM、怎么选、yaml 怎么配 | [Word](notes/alignment.docx) |
| [微调/分布式/量化选型](notes/peft-distributed-quant.md) | Full/LoRA/QLoRA/DoRA、DDP/ZeRO1-3/TP/PP、GPTQ/AWQ/vLLM | [Word](notes/peft-distributed-quant.docx) |
| [垂直领域微调实践方案](docs/垂直领域大模型微调实践方案.md) | 12GB 单卡从零做垂直小模型的端到端方案 | [Word](docs/垂直领域大模型微调实践方案.docx) |

> 公式推导（DPO 闭式解、GRPO 优势等）在 ② 模块 [`../02-train-from-scratch/notes/alignment.md`](../02-train-from-scratch/notes/alignment.md)，本模块只做工程选型，不重复推导。

## 三、目录结构

| 目录 | 内容 |
| --- | --- |
| `notes/` | 源码走读、对齐选型、PEFT/分布式/量化选型三篇笔记 |
| `configs/` | 可直接改跑的模板：LoRA / QLoRA / 全参 SFT / DPO + ZeRO-2，带中文注释 |
| `custom/` | 二次开发示例：数据集注册、单轮/多轮/偏好样例、训练回调 |
| `experiments/` | 对照记录表（csv）+ 填表口径；Full/LoRA/QLoRA 已在 RTX 3080 Ti 实测、完整可复现实验包见 `experiments/peft_lab/`，其余方法留空待补 |
| `docs/` | 垂直领域端到端实践方案 |

## 四、环境

```bash
conda create -n llmfactory python=3.10 -y && conda activate llmfactory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]"
# 训练（用本模块 configs 里的模板）：
llamafactory-cli train ../03-peft-framework/configs/sft_lora_qwen0.5.yaml
```

## 五、建议学习/动手路线

1. 先读 [源码走读](notes/source_walkthrough.md)，对着本地 LLaMA-Factory 走一遍四层调用关系；
2. 读两篇选型笔记，建立"数据形态 + 显存约束 → 方法"的决策框架；
3. 用 `configs/sft_lora` 跑通一个小模型基线，再只改一个变量做对照（LoRA vs 全参、r=8 vs 32）；
4. 先跑通 `experiments/peft_lab/`（Full/LoRA/QLoRA 完整实测样例），再按 `experiments/README.md` 的口径把其余方法实测补进 csv；
5. 用 `custom/` 示例接入自己的数据集、加一个 loss 记录回调，完成一次二次开发；
6. 想做完整项目时，按 `docs/垂直领域大模型微调实践方案.md` 端到端落地。

## 六、关于 experiments 里的数字口径

显存、延迟、吞吐强依赖模型、序列、batch 与显卡，抄来的数字不可复现，因此本模块不堆砌“参考 benchmark”，而是提供**选型原理（定性）+ 可复现模板 + 统一记录口径**。其中 PEFT 的 Full/LoRA/QLoRA 已在单卡 RTX 3080 Ti + Qwen2.5-0.5B 上按统一口径实测，配置、原始日志、逐秒显存采样与复现脚本都在 [`experiments/peft_lab/`](experiments/peft_lab/README.md)，每个数字可回溯；其余方法、分布式与量化表仍留空，供在自己的环境按同一口径补测。

## 七、参考与致谢

- 框架：[hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)（Apache-2.0，ACL 2024）
- 论文：DPO、KTO、ORPO、SimPO、LoRA、QLoRA、DoRA、ZeRO、PagedAttention（清单见 [`../appendix/papers.md`](../appendix/papers.md)）

> `configs/` 基于官方配置范式改写并逐行注释；`notes/`、`custom/`、`experiments/`、`docs/` 为学习笔记、示例与记录模板。字段名随框架版本演进，以本地 `src/llamafactory/hparams/*.py` 为准。