# 学习路径与使用指南 LEARNING PATH

本仓库按 **① 理论 → ② 手搓与训练 → ③ 工业框架 → ④ Agent → ⑤ RAG 落地** 编号组织，
这是一条"先懂为什么、再会怎么做、最后能落地"的路线。每个模块都可独立阅读，但按顺序收获最大。

## 推荐顺序

### ① 01-foundations · 打地基（理论）
- **学什么**：BPE 分词、Transformer、自注意力与 GQA/MQA/MQA、位置编码 RoPE、MoE、
  GPU 体系与并行、Scaling Law、推理优化（KV Cache/量化/蒸馏）、对齐（SFT/RLHF/DPO/GRPO）、多模态。
- **怎么用**：5 篇中文笔记对应 CS336 的 L1–L17，遇到②里不懂的组件回这里查原理。
- **产出标准**：能口述一次 Transformer 前向、说清 GQA 为什么省 KV Cache、解释 Scaling Law。

### ② 02-train-from-scratch · 最核心（手写 + 训练）
- **学什么**：原生 PyTorch 手写 RMSNorm/RoPE/SwiGLU/GQA Causal Attention/Decoder Block；
  数据清洗抽样 → 预训练 → SFT → LoRA → DPO → GRPO 全链路；bf16、梯度累积、cosine 学习率。
- **怎么用**：
  1. 先读 `from_scratch/model.py`，跑 `python from_scratch/test_model.py`（12 项组件自检）
     和 `python from_scratch/test_grpo_logic.py`（14 项 GRPO 数值自检）；
  2. 读 `notes/training-pipeline-explained-zh.md`（小白友好的全流程讲解）；
  3. 对照 `experiments/` 的 loss 曲线、显存/耗时、生成样例；
  4. 用 `reproduced/` 脚本在单卡复现。
- **产出标准**：能说清每阶段"是什么/为什么学习率递减/LoRA 为何省参数/DPO loss 为何从 0.693 起步"。

### ③ 03-peft-framework · 从"能写"到"会选"（工业框架）
- **学什么**：LLaMA-Factory 的 Model Loader / Data Worker / Trainer 源码链路；
  对齐算法（SFT/PPO/DPO/KTO/ORPO/SimPO）、PEFT（Full/LoRA/QLoRA/DoRA…）、
  分布式（DDP/TP/PP/ZeRO）、量化推理（GPTQ/AWQ + vLLM）的横向选型。
- **怎么用**：读 `experiments/*.csv` 四张对照表，配合 `docs/垂直领域大模型微调实践方案.md`。
- **产出标准**：给定资源与任务约束，能说出选哪种对齐/微调/并行方案及理由。

### ④ 04-agent · 会用模型做事（Agent 工程）
- **学什么**：ReAct 循环与 Harness、Tool/Skill 分层与路由、记忆的"执行-反思-复用"闭环、
  分层上下文压缩、Supervisor-Worker 多智能体、安全过滤链。
- **怎么用**：从 `mini_code_core.py` 入口读主循环，再看 `skill_router / memory / context_manager / multi_agent / security`，`tests/` 有场景测试。
- **产出标准**：能设计一个带工具调用、记忆与安全边界的 Agent，并解释各模块解决什么问题。

### ⑤ 05-rag-application · 落到真实场景（RAG）
- **学什么**：语料采集清洗、BM25 稀疏 + 稠密向量双路召回、RRF 融合、Rerank、
  HyDE/查询扩展、带引文生成、ReAct 检索式推理、Streamlit 前端、检索效果 benchmark。
- **怎么用**：按 `retrieval → agent → app → evaluation` 的流水线顺序读，`evaluation/` 有多种召回/重排变体的对照评测。
- **产出标准**：能搭一条可评测的 RAG 流水线，并说清每一环对召回率/准确率的影响。

## 时间安排建议（参考）
| 周次 | 重点 | 对应模块 |
| --- | --- | --- |
| 第 1 周 | 补理论、跑通手写组件自检 | ① + ②/from_scratch |
| 第 2 周 | 跑通预训练/SFT/LoRA/DPO 并读懂曲线 | ② |
| 第 3 周 | 框架选型对照 + GRPO | ③ + ②/GRPO |
| 第 4 周 | Agent 与 RAG 两个工程作品复盘 | ④ + ⑤ |

## 先修知识
Python、PyTorch 基本用法、机器学习/深度学习基础概念（梯度、损失、优化器、注意力）。
缺原理时回 ①，缺工程细节时查 `appendix/` 的名词词典与论文清单。