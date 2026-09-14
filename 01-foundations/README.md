# CS336 中文精讲笔记 · LLMs from Scratch

> 📚 **本模块属于 [Hands-on-LLM 动手学大模型](../README.md) 学习库 · 返回总览看完整学习路线**

> 学习 Stanford **CS336: Language Modeling from Scratch** 系统整理的**中文精讲笔记（小白友好版）**，
> 按课程讲次 L1–L17 系统梳理，用尽量通俗的方式讲清大模型从零构建的完整链路。

## 笔记目录（`notes/`）
| 讲次 | 核心主题 | Markdown | Word |
| --- | --- | --- | --- |
| L1–L3 | Tokenization 与 BPE、训练资源核算、Transformer 基础 | [在线阅读](notes/01_L1-L3_分词_资源核算_Transformer.md) | [Word](notes/01_L1-L3_分词_资源核算_Transformer.docx) |
| L4–L8 | 注意力替代方案、MoE 混合专家、GPU 系统与性能、Triton 编程、并行 | [在线阅读](notes/02_L4-L8_注意力替代_MoE_GPU系统_Triton.md) | [Word](notes/02_L4-L8_注意力替代_MoE_GPU系统_Triton.docx) |
| L9–L11 | Scaling Laws、推理优化（KV Cache / 量化 / 蒸馏）、缩放案例 | [在线阅读](notes/03_L9-L11_缩放定律_推理优化.md) | [Word](notes/03_L9-L11_缩放定律_推理优化.docx) |
| L12–L14 | 模型评估、训练数据来源与清洗处理 | [在线阅读](notes/04_L12-L14_模型评估_训练数据.md) | [Word](notes/04_L12-L14_模型评估_训练数据.docx) |
| L15–L17 | 对齐（SFT/RLHF/DPO）、推理模型与 GRPO、多模态 | [在线阅读](notes/05_L15-L17_对齐_GRPO_多模态.md) | [Word](notes/05_L15-L17_对齐_GRPO_多模态.docx) |

> 💡 每篇笔记都提供**在线 Markdown**与**可下载 Word**两个版本，Word 版已排好标题、表格与代码块样式，适合离线批注或打印。

## 知识地图
分词与资源核算 → Transformer 架构 → 注意力变体与 MoE → GPU/Triton/并行训练
→ Scaling Law 与推理优化 → 评估与数据工程 → 对齐（含 GRPO）→ 多模态。

## 说明与免责
- 本模块**只包含课程的中文精讲笔记**，不含课程官方讲义（lectures）、阅读论文（readings）与作业（assignments），这些材料的版权归 Stanford CS336 及相应作者所有。
- 笔记为学习过程中的理解，难免有疏漏，欢迎 Issue 讨论指正。
- 课程主页：[CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)
