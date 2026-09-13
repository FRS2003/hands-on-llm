# Hands-on LLM · 动手学大模型：从手搓原理到落地应用

<p align="left">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="Language" src="https://img.shields.io/badge/language-%E4%B8%AD%E6%96%87%20%7C%20Python-orange">
  <img alt="Created" src="https://img.shields.io/badge/created-2026--06-green">
  <img alt="Modules" src="https://img.shields.io/badge/modules-5-success">
</p>

> 一条**由理论到落地**的大模型自学与实战路径：先读透原理，再用原生 PyTorch 手搓并完整跑通
> **预训练 → SFT → LoRA → DPO → GRPO → 可验证奖励 RLVR**，然后读懂工业级微调框架、动手做 Agent，最后用 RAG 把模型落到真实场景。
> 所有训练实验都在**单张消费级显卡（RTX 3080 Ti 12GB）**上真实跑过，数字、曲线、生成样例与踩坑全部留档，不做"截图式学习"。

## 🧭 学习路线（建议按编号顺序）

```
① 理论基础        ② 手搓 + 训练全链路      ③ 工业框架           ④ Agent 工程        ⑤ RAG 落地
CS336 中文精讲 → 原生 PyTorch 实现 →    LLaMA-Factory  →   编程智能体     →  文献问答系统
Transformer/      Pretrain·SFT·LoRA·     对齐/PEFT/分布式/   ReAct·记忆·多智能体  混合检索·重排
MoE/Scaling/对齐   DPO·GRPO + 对照实验     量化推理 选型                        带引文生成
```

| 模块 | 目录 | 你会看到什么 |
| --- | --- | --- |
| ① 理论基础 | [`01-foundations`](01-foundations) | Stanford CS336（L1–L17）5 篇中文精讲：分词、Transformer、注意力变体、MoE、GPU/并行、Scaling Law、推理优化、对齐与 GRPO |
| ② 手搓与训练 | [`02-train-from-scratch`](02-train-from-scratch) | 原生 PyTorch 手写 RMSNorm/RoPE/GQA/SwiGLU + 26 项数值自检；单卡跑通五阶段训练 + 可验证奖励 RLVR，并做规模/资源/偏好/架构对照实验 |
| ③ 工业框架 | [`03-peft-framework`](03-peft-framework) | 源码级走读 LLaMA-Factory 四层链路；对齐/PEFT/分布式/量化选型笔记 + 带注释可改跑的 yaml 模板与二次开发示例 + 垂直领域微调方案（csv 为自测记录表） |
| ④ Agent 工程 | [`04-agent`](04-agent) | 终端编程 Agent：ReAct + Harness、Skill 路由、记忆闭环、分层上下文压缩、主从多智能体、分层安全 |
| ⑤ RAG 落地 | [`05-rag-application`](05-rag-application) | 医学文献混合检索问答：BM25+稠密双路、加权融合、Rerank、HyDE/QE、带引文生成、ReAct 编排、Streamlit |

## 🎯 适合谁 / 你能收获什么

- 想系统补齐**大模型训练与落地**、从理论走向动手实践的同学；
- 会调包但说不清"Transformer 内部 / 为什么 SFT 学习率要更小 / DPO loss 为何从 0.693 起步"的人；
- 想在**单卡**上把预训练→对齐全链路亲手跑一遍、并留下可复现实验记录的人。

收获：一套"**原理 → 手写 → 训练 → 框架 → Agent → 应用**"可运行、可复现、带真实数据的完整作品，而不是一堆零散 demo。

## 🧩 能力矩阵（想掌握的能力 → 对应模块）

| 想掌握的能力点 | 在哪里练到 |
| --- | --- |
| Transformer / 自注意力 / 位置编码 / GQA / MoE | ① + ② |
| 继续预训练、SFT、LoRA、DPO、GRPO/强化学习对齐 | ② |
| 数据清洗/抽样、Scaling Law、模型评估 | ① + ② |
| 模型/数据/管道并行、DeepSpeed ZeRO、混合精度 | ① + ③ |
| PEFT 方法选型、量化（GPTQ/AWQ）、vLLM 推理 | ③ |
| Agent / ReAct / 工具调用 / 记忆 / 多智能体 | ④ + ⑤ |
| RAG / 混合检索 / 向量库 / Rerank / 查询改写 / 检索评测 | ⑤ |
| PyTorch 原生实现与数值正确性验证 | ② |

## 🔬 部分真实实验结果（均有日志/曲线支撑，详见各模块）

- **从零训练 63.91M 小模型**（hidden 768 / 8 层 / GQA KV 头 4 / bf16）：预训练数据量 ×2.5 后，最终 loss 3.35→2.53，验证数据规模效应；
- **LoRA 只训 0.61% 参数**（0.393M/63.91M），适配器权重 132MB→0.78MB，显存 7.36GB→4.60GB；
- **DPO** 用 17,166 对偏好数据，loss 从理论值 -ln2≈0.693 起步并缓慢下降，学习率刻意取 4e-8 防止灾难性遗忘；
- **GRPO** 免 1.8B 奖励模型改纯规则奖励，单卡真实训练 300 步（21.7 min）：组内优势均值严格为 0、平均 |KL|≈0.005 未漂移，同 prompt/种子前后对比规则分 0.109→0.290；100 条未训练 held-out 题上 +0.084、3-gram 重复度下降，并做五阶段（pretrain→GRPO）同种子生成横评；另配 14 项纯 tensor 自检；
- **可验证奖励 RLVR（答案对错自动判分）**：一位加法先 SFT 冷启动到「采样能偶尔答对」的甜区（greedy 0.633），再用对错判分的 GRPO（B8×G4、lr 4e-6、β-KL 0.08、300 步）把独立 120 题 **greedy 0.633→0.792、采样三种子均值 0.503→0.603**；对照证明 RL 只放大已有能力、不注入知识，并复现了 lr 过大 / KL 过弱导致的策略崩溃；
- **架构/精度消融**：同骨架实测 MHA/GQA/MQA——训练侧 MQA 比 MHA 省约 12% 显存、快约 16%，而推理 KV cache@4096 为 100.7/50.3/12.6MB（随 KV 头数 8:4:1，长上下文这才是 GQA/MQA 主价值）；bf16 较 fp32 省 35–49% 显存、约 1.8× 吞吐；
- 全程显存峰值 4.6–7.4GB、GPU 利用率 96–99%，证明单卡可完整走通主链路。

## 🗺 怎么开始

- 第一次看：从 [`LEARNING_PATH.md`](LEARNING_PATH.md) 按顺序学，每个模块都有自己的 README 导览；
- 想直接看代码：进 [`02-train-from-scratch/from_scratch`](02-train-from-scratch/from_scratch) 看手写组件，`python test_model.py`、`python test_grpo_logic.py` 跑数值自检；
- 自测回查：[`appendix/`](appendix) 里有名词词典、论文清单，以及结合本仓库真实数字的核心知识自测；
- 离线/批注：各模块的教学笔记基本都附了排版好的 **Word 版**（与同名 `.md` 同目录的 `.docx`），导航大纲齐全，可直接下载打印或批注。

## 📂 仓库结构

```
hands-on-llm/
├── 01-foundations/          # CS336 中文精讲（理论）
├── 02-train-from-scratch/   # 手写组件 + 五阶段训练 + 对照实验
├── 03-peft-framework/       # LLaMA-Factory 源码走读、选型笔记、配置模板与二次开发示例
├── 04-agent/                # 编程智能体：ReAct/记忆/多智能体/安全
├── 05-rag-application/      # 医学文献混合检索问答 RAG（双路召回/重排/评测）
├── appendix/                # 名词词典 / 论文清单 / 核心知识自测
├── tools/                   # 辅助工具：Markdown→Word 转换器 md2docx.py
├── LEARNING_PATH.md         # 学习路径与使用指南
└── LICENSE                  # MIT
```

## 🛠 环境

- Python 3.10、PyTorch（训练/手写组件）、Transformers；Agent/RAG 通过 OpenAI 兼容接口接入模型，Key 一律走环境变量（`.env.example`），仓库不含任何密钥。
- 训练默认 bf16 混合精度；单卡 12GB 即可复现②，纯 CPU 也能跑手写组件的数值自检。

## 📌 进度与规划

- [x] 理论笔记、手写组件、Pretrain/SFT/LoRA/DPO、框架对照、Agent、RAG
- [x] GRPO 纯规则奖励改造与数值自检
- [x] GRPO 纯规则奖励 300 步全量训练，补 Reward/KL/Loss 曲线与前后生成对比
- [x] GRPO 100 条 held-out 定量评测 + 五阶段同种子生成横评（能力演进证据链）
- [x] MHA/GQA/MQA × 混合精度 × batch 架构消融（训练显存/吞吐 + 推理 KV cache 实测）
- [x] 可验证奖励 RLVR（对错判分）：SFT 甜区冷启动 + GRPO，greedy 0.633→0.792，含能力边界 / 策略崩溃对照
- [ ] Triton 自定义 Kernel、手写分块 Flash Attention、DeepSpeed ZeRO 实测
- [ ] 持续补充论文精读与自测题

## 🙏 致谢与说明

本仓库为一套面向学习者的动手学习资料，训练部分复现自 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)（MIT）、
框架实验参考 [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)（Apache-2.0）、
理论笔记对应 [Stanford CS336](https://stanford-cs336.github.io/)（笔记为学习理解，不含其受版权保护的讲义/作业），各模块 README 内有更详细出处。如有疏漏欢迎 Issue 指正。

## License

[MIT](LICENSE)