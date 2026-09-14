# 论文清单 Papers（按模块）

> 建议顺序：先在对应模块跑出直觉，再读论文看形式化。★为优先精读。

## 01 / 02 基础结构与训练
- ★ Attention Is All You Need (Vaswani et al., 2017) — Transformer 原点。
- ★ RoFormer / RoPE (Su et al., 2021) — 旋转位置编码。
- GQA: Training Generalized Multi-Query Transformer (2023) — 分组查询注意力。
- GLU Variants Improve Transformer / SwiGLU (Shazeer, 2020)。
- Root Mean Square Layer Normalization / RMSNorm (2019)。
- ★ FlashAttention 1 & 2 (Dao et al., 2022/2023) — IO 感知的精确注意力加速。
- ★ Training Compute-Optimal LLMs / Chinchilla (Hoffmann et al., 2022) — 参数量与 token 量配比。
- LLaMA / LLaMA 2 (Meta, 2023) — 现代开源 Decoder-only 架构范式。

## 02 / 03 微调与对齐
- ★ LoRA (Hu et al., 2021) — 低秩适配。
- QLoRA (Dettmers et al., 2023) — 4bit 基座 + LoRA。
- DoRA、Adapter、Prefix/P-Tuning v2 — PEFT 其他代表。
- ★ InstructGPT / RLHF (Ouyang et al., 2022) — SFT+RM+PPO 三段式。
- ★ DPO (Rafailov et al., 2023) — 直接偏好优化，免奖励模型。
- KTO / ORPO / SimPO (2023–2024) — 无需成对偏好或参考模型的对齐变体。
- ★ DeepSeekMath / GRPO (2024)、DeepSeek-R1 (2025) — 组相对策略优化与推理模型。
- PPO (Schulman et al., 2017) — GRPO 的源头，理解 clip 目标。

## 01 / 03 系统、并行与推理
- ★ DeepSpeed ZeRO (Rajbhandari et al., 2020) — 优化器/梯度/参数分片。
- Megatron-LM (Shoeybi et al., 2019) — 张量并行。
- Megatron-LM Pipeline Parallelism (2021) — 管道并行与 1F1B 调度。
- GPTQ (2022)、AWQ (2023)、SmoothQuant — 量化。
- ★ vLLM / PagedAttention (Kwon et al., 2023) — 高吞吐推理。
- Switch Transformer / Mixtral — MoE 稀疏专家。

## 04 Agent
- ★ ReAct (Yao et al., 2022) — 推理与行动交织。
- Toolformer (2023)、Reflexion (2023)、Tree-of-Thoughts (2023) — 工具调用/反思/搜索式推理。

## 05 RAG
- ★ RAG: Retrieval-Augmented Generation (Lewis et al., 2020) — RAG 原点。
- Dense Passage Retrieval / DPR (Karpukhin et al., 2020) — 稠密双塔检索。
- Reciprocal Rank Fusion / RRF (Cormack et al., 2009) — 排名融合。
- ★ HyDE (Gao et al., 2022) — 假设文档嵌入。
- ColBERT (Khattab & Zaharia, 2020) — 后期交互式重排。
