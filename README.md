# CS336 个人中文学习笔记 · LLMs from Scratch

> 学习 Stanford **CS336: Language Modeling from Scratch** 过程中整理的**个人中文笔记（小白友好版）**，
> 按课程讲次 L1–L17 系统梳理，用尽量通俗的方式讲清大模型从零构建的完整链路。

## 笔记目录（`notes/`）
| 文件 | 讲次 | 核心主题 |
| --- | --- | --- |
| `01_L1-L3_*` | L1–L3 | Tokenization 与 BPE、训练资源核算、Transformer 基础 |
| `02_L4-L8_*` | L4–L8 | 注意力替代方案、MoE 混合专家、GPU 系统与性能、Triton 编程、并行 |
| `03_L9-L11_*` | L9–L11 | Scaling Laws、推理优化（KV Cache / 量化 / 蒸馏）、缩放案例 |
| `04_L12-L14_*` | L12–L14 | 模型评估、训练数据来源与清洗处理 |
| `05_L15-L17_*` | L15–L17 | 对齐（SFT/RLHF/DPO）、推理模型与 GRPO、多模态 |

## 知识地图
分词与资源核算 → Transformer 架构 → 注意力变体与 MoE → GPU/Triton/并行训练
→ Scaling Law 与推理优化 → 评估与数据工程 → 对齐（含 GRPO）→ 多模态。

## 说明与免责
- 本仓库**只包含本人整理的学习笔记**，不含课程官方讲义（lectures）、阅读论文（readings）与作业（assignments），这些材料的版权归 Stanford CS336 及相应作者所有。
- 笔记为个人学习理解，难免有疏漏，欢迎 Issue 讨论指正。
- 课程主页：https://stanford-cs336.github.io/
